/* hmx_gemm_client <uri> <mode> <M> <K> <N> [iters] [bias]: random fp16 GEMM on the DSP's HMX vs a double
 * reference; prints errors and TMAC/s. Env HMX_FLAGS (default 3 = turbo + HMX power vote). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmx_gemm_rpc.h"
#include "remote.h"
#include "hmx_gemm.h"
static float h2f(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
static uint16_t f2h(float f) { __fp16 x = (__fp16)f; uint16_t h; memcpy(&h, &x, 2); return h; }
int main(int argc, char** argv) {
  if (argc < 6) return 2;
  int mode = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 10, hb = argc > 7 ? atoi(argv[7]) : 0;
  struct remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof um);
  remote_handle64 h;
  int rc = hmx_gemm_rpc_open(argv[1], &h);
  if (rc) { printf("open failed %d\n", rc); return 1; }
  int prc = 0;
  hmx_gemm_rpc_perf_vote(h, getenv("HMX_FLAGS") ? atoi(getenv("HMX_FLAGS")) : 3, &prc);
  uint16_t* A = malloc((size_t)M * K * 2);
  uint16_t* W = malloc((size_t)K * N * 2);
  uint16_t* Wp = malloc((size_t)K * N * 2);
  uint16_t* B = malloc((size_t)N * 2 + 2);
  uint16_t* C = malloc((size_t)M * N * 2);
  srand(7);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = f2h((rand() % 2001 - 1000) / 1000.f);
  for (size_t i = 0; i < (size_t)K * N; i++) W[i] = f2h((rand() % 2001 - 1000) / 4000.f);
  for (int i = 0; i < N; i++) B[i] = f2h(hb ? (rand() % 200 - 100) / 50.f : 0);
  hmx_pack_w_f16(W, K, N, Wp);
  uint64 t[4];
  int codes[8];
  rc = hmx_gemm_rpc_gemm_f16(h, mode, M, K, N, iters, A, M * K, Wp, K * N, B, hb ? N : 0, C, M * N, t, 4, codes, 8);
  printf("vote %d rc %d codes ctx %d hvx %d hmx %d gemm %d vtcm %d thr %d big %d\n", prc, rc, codes[0], codes[1], codes[2],
         codes[3], codes[4], codes[5], codes[6]);
  double maxe = 0, maxr = 0;
  int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double s = hb ? h2f(B[j]) : 0;
      for (int k = 0; k < K; k++) s += (double)h2f(A[(size_t)i * K + k]) * h2f(W[(size_t)k * N + j]);
      double e = fabs(h2f(C[(size_t)i * N + j]) - s);
      if (e > maxe) maxe = e;
      if (fabs(s) > maxr) maxr = fabs(s);
      if (e > fabs(s) / 1024 + 1e-4) bad++;
    }
  double macs = (double)M * K * N * iters;
  double us = mode == 1 ? (double)t[2] : (double)t[0];
  printf("mode %d %dx%dx%d x%d: %s max abs err %.3g (max |ref| %.3g), %d beyond fp16 rounding; %.1f us/iter, %.3f TMAC/s\n",
         mode, M, K, N, iters, bad ? "FAIL" : "PASS", maxe, maxr, bad, us / iters, macs / us / 1e6);
  if (mode == 0)
    printf("  pcycles/iter: pack A %.0f, W copy %.0f, MAC+store %.0f, unpack C %.0f\n", (double)t[1] / iters,
           (double)t[2] / iters, (double)(t[3] & 0xffffffffu) / iters, (double)(t[3] >> 32) / iters);
  hmx_gemm_rpc_close(h);
  return bad ? 3 : 0;
}
