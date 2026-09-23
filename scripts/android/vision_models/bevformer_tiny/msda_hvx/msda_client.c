/* Phone bench of the msda skel on split.py dump cases: rpcmem buffers (zero-copy), REPS timed calls
 * per thread count, median DSP time (HAP) and wall time (incl. FastRPC), output vs torch.
 *   msda_client <uri> <reps> <turbo> <threads,threads,...> <case dir>... */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "msda_rpc.h"
#include "remote.h"
#include "rpcmem.h"
#include "msda_io.h"

static void* rpc_alloc(long n) { return rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)(n < 128 ? 128 : n)); }
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s uri reps turbo threads,.. case...\n", argv[0]); return 2; }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (msda_rpc_open(argv[1], &h)) { printf("FAIL open\n"); return 1; }
  const int reps = atoi(argv[2]);
  int32 prc = 0;
  msda_rpc_perf_vote(h, atoi(argv[3]), &prc);
  printf("perf_vote rc %d\n", prc);
  int rc = 0;
  for (int ci = 5; ci < argc; ci++) {
    msda_case_t c;
    if (msda_load(argv[ci], &c, rpc_alloc)) { printf("FAIL load %s\n", argv[ci]); return 1; }
    const msda_args_t* a = &c.a;
    char cfg[256];
    strncpy(cfg, argv[4], sizeof cfg - 1);
    cfg[sizeof cfg - 1] = 0;
    for (char* t = strtok(cfg, ","); t; t = strtok(NULL, ",")) {
      int flags = atoi(t);
      double dsp[64], wall[64];
      int n = reps > 64 ? 64 : reps;
      memset(a->out, 0xff, c.n_out * 4); /* NaN: a missed row fails the compare */
      for (int r = 0; r < n + 2; r++) {
        uint64 us = 0;
        double t0 = now_ms();
        int e = msda_rpc_run(h, a->value, (int)c.n_value, a->ref, (int)c.n_ref, a->off, (int)c.n_off, a->attw, (int)c.n_attw,
                             a->vis, (int)c.n_vis, a->NV, a->H, a->W, a->Q, a->R, a->NO, a->P, flags, a->out, (int)c.n_out, &us);
        double t1 = now_ms();
        if (e) { printf("FAIL run rc %d\n", e); return 1; }
        if (r >= 2) { dsp[r - 2] = us / 1000.0; wall[r - 2] = t1 - t0; }
      }
      qsort(dsp, n, sizeof(double), cmpd);
      qsort(wall, n, sizeof(double), cmpd);
      char label[512];
      snprintf(label, sizeof label, "%s threads %d: dsp %.2f ms wall %.2f ms (min %.2f/%.2f)", argv[ci], flags & 0xff,
               dsp[n / 2], wall[n / 2], dsp[0], wall[0]);
      rc |= msda_compare(label, a->out, c.ref_out, c.n_out, 1e-5);
    }
  }
  msda_rpc_close(h);
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}
