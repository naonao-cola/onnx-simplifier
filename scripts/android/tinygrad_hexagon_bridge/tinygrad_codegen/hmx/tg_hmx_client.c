/* Phone client for tg_hmx_rpc: random fp16 A (MxK), B (KxN), runs the tinygrad-generated HMX kernel and checks it bit
 * for bit against the rounding model of the accumulator-resident codegen: exact accumulation over all of K, rounded once.
 *   tg_hmx_client <uri> M K N [iters] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "remote.h"
#include "tg_hmx_rpc.h"
static float h2f(unsigned short h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static unsigned short f2h(double f) { __fp16 x = (__fp16)f; unsigned short h; memcpy(&h, &x, 2); return h; }
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s uri M K N [iters]\n", argv[0]); return 2; }
  int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 5;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  remote_handle64 h; int rc = tg_hmx_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  unsigned short *a = malloc(M * K * 2), *b = malloc(K * N * 2), *c = malloc(M * N * 2), *r = malloc(M * N * 2);
  srand(1);
  for (int i = 0; i < M * K; i++) a[i] = f2h((rand() % 2001 - 1000) / 2000.0);
  for (int i = 0; i < K * N; i++) b[i] = f2h((rand() % 2001 - 1000) / 2000.0);
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {  /* exact accumulation, one rounding (accumulator kept in HMX) */
    double s = 0; for (int k = 0; k < K; k++) s += (double)h2f(a[m * K + k]) * h2f(b[k * N + n]);
    r[m * N + n] = f2h(s);
  }
  unsigned long long t[4]; int codes[8];
  rc = tg_hmx_rpc_run(h, 1, a, M * K, b, K * N, c, M * N, t, 4, codes, 8);  /* warm-up + correctness */
  int bad = 0; for (int i = 0; i < M * N; i++) bad += c[i] != r[i];
  printf("rc %d codes power %d ctx %d hvx %d hmx %d vtcm %d thread %d; %d/%d bit mismatches vs model\n", rc, codes[0], codes[1],
         codes[2], codes[3], codes[4], codes[5], bad, M * N);
  rc = tg_hmx_rpc_run(h, iters, a, M * K, b, K * N, c, M * N, t, 4, codes, 8);
  double us = (double)t[0] / iters;
  printf("%dx%dx%d: %.1f us/call, %.3f TMAC/s (tinygrad-generated HMX, %d iters) %s\n", M, K, N, us, (double)M * K * N / us / 1e6,
         iters, bad ? "FAIL" : "PASS");
  tg_hmx_rpc_close(h);
  return bad != 0;
}
