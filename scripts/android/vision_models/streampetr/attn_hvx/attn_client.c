/* Phone bench of the attn skel on emulate.py case directories: rpcmem buffers, REPS timed calls per
 * flags value, median DSP / pack time (HAP) and wall time (incl. FastRPC), output bit-exact vs contract.
 *   attn_client <uri> <reps> <turbo> <flags,flags,...> <case dir>...   (flags: threads | 256 * (rows / 4)) */
#include <string.h>
#include <time.h>

#include "attn_rpc.h"
#include "remote.h"
#include "rpcmem.h"
#include "attn_io.h"

static void* rpc_alloc(long n) { return rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)(n < 128 ? 128 : n)); }
static void* host_alloc(long n) { return malloc((size_t)(n < 128 ? 128 : n)); }
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s uri reps turbo flags,.. case...\n", argv[0]); return 2; }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (attn_rpc_open(argv[1], &h)) { printf("FAIL open\n"); return 1; }
  const int reps = atoi(argv[2]);
  int32 prc = 0;
  attn_rpc_perf_vote(h, atoi(argv[3]), &prc);
  int rc = 0;
  for (int ci = 5; ci < argc; ci++) {
    attn_case_t c;
    /* q / k / v / out in rpcmem (zero-copy), the packed buffers are the skel's own */
    if (attn_load(argv[ci], &c, host_alloc)) { printf("FAIL load %s\n", argv[ci]); return 1; }
    const attn_args_t* a = &c.a;
    const long nq = (long)a->LQ * ATTN_C, nk = (long)a->LK * ATTN_C;
    uint8_t *q = rpc_alloc(nq), *k = rpc_alloc(nk), *v = rpc_alloc(nk), *out = rpc_alloc(nq);
    memcpy(q, a->q, nq); memcpy(k, a->k, nk); memcpy(v, a->v, nk);
    int32 params[5] = {a->LQ, a->LK, a->zq, a->m16, a->dcl};
    char cfg[256];
    strncpy(cfg, argv[4], sizeof cfg - 1);
    cfg[sizeof cfg - 1] = 0;
    for (char* t = strtok(cfg, ","); t; t = strtok(NULL, ",")) {
      int flags = atoi(t);
      double dsp[64], pk[64], wall[64];
      int n = reps > 64 ? 64 : reps;
      memset(out, 0, nq);
      for (int r = 0; r < n + 2; r++) {
        uint64 us = 0, pus = 0;
        double t0 = now_ms();
        int e = attn_rpc_run(h, q, (int)nq, k, (int)nk, v, (int)nk, params, 5, flags, out, (int)nq, &us, &pus);
        double t1 = now_ms();
        if (e) { printf("FAIL run rc %d\n", e); return 1; }
        if (r >= 2) { dsp[r - 2] = us / 1000.0; pk[r - 2] = pus / 1000.0; wall[r - 2] = t1 - t0; }
      }
      qsort(dsp, n, sizeof(double), cmpd);
      qsort(pk, n, sizeof(double), cmpd);
      qsort(wall, n, sizeof(double), cmpd);
      char label[512];
      snprintf(label, sizeof label, "%s flags %d: dsp %.2f ms (pack %.2f) wall %.2f ms (min %.2f/%.2f)", argv[ci], flags,
               dsp[n / 2], pk[n / 2], wall[n / 2], dsp[0], wall[0]);
      rc |= attn_compare(label, out, c.ref, nq);
    }
  }
  attn_rpc_close(h);
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}
