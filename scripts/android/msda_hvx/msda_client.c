/* Phone bench of the msda skel on msda_ref.py case directories: rpcmem buffers (zero-copy), REPS
 * timed calls per flags value, median DSP time (HAP) and wall time (incl. FastRPC), output vs torch.
 *   msda_client <uri> <reps> <turbo> <flags,flags,...> <case dir>...   (flags: threads | 256 * (queries per job / 16)) */
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
  if (argc < 6) { fprintf(stderr, "usage: %s uri reps turbo flags,.. case...\n", argv[0]); return 2; }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (msda_rpc_open(argv[1], &h)) { printf("FAIL open\n"); return 1; }
  const int reps = atoi(argv[2]);
  int32 prc = 0;
  msda_rpc_perf_vote(h, atoi(argv[3]), &prc);
  int rc = 0;
  for (int ci = 5; ci < argc; ci++) {
    msda_case_t c;
    if (msda_load(argv[ci], &c, rpc_alloc)) { printf("FAIL load %s\n", argv[ci]); return 1; }
    const msda_args_t* a = &c.a;
    int32 shape[MSDA_SHAPE_LEN(MSDA_MAX_L)];
    const int ns = msda_shape_pack(a, shape);
    char cfg[256];
    strncpy(cfg, argv[4], sizeof cfg - 1);
    cfg[sizeof cfg - 1] = 0;
    for (char* t = strtok(cfg, ","); t; t = strtok(NULL, ",")) {
      int flags = atoi(t);
      double dsp[64], wall[64];
      int n = reps > 64 ? 64 : reps;
      memset(a->out, 0xff, msda_n_out(a) * 4); /* NaN: a missed row fails the compare */
      for (int r = 0; r < n + 2; r++) {
        uint64 us = 0;
        double t0 = now_ms();
        const int u8 = a->vdtype == MSDA_U8;
        int e = msda_rpc_run(h, a->value, u8 ? 0 : (int)msda_n_value(a), a->value_u8, u8 ? (int)msda_n_value(a) : 0,
                             a->vscale, u8 ? a->NV : 0, (const int32*)a->vzp, u8 ? a->NV : 0, a->loc, (int)msda_n_loc(a),
                             a->ref, (int)msda_n_ref(a),
                             a->attw, (int)msda_n_attw(a), a->vis, a->vis ? (int)msda_n_vis(a) : 0, shape, ns, flags, a->out,
                             (int)msda_n_out(a), &us);
        double t1 = now_ms();
        if (e) { printf("FAIL run rc %d\n", e); return 1; }
        if (r >= 2) { dsp[r - 2] = us / 1000.0; wall[r - 2] = t1 - t0; }
      }
      qsort(dsp, n, sizeof(double), cmpd);
      qsort(wall, n, sizeof(double), cmpd);
      char label[512];
      snprintf(label, sizeof label, "%s flags %d: dsp %.2f ms wall %.2f ms (min %.2f/%.2f)", argv[ci], flags, dsp[n / 2],
               wall[n / 2], dsp[0], wall[0]);
      rc |= msda_compare(label, a->out, c.ref_out, msda_n_out(a), msda_tol(a, 1));
    }
  }
  msda_rpc_close(h);
  printf(rc ? "FAIL\n" : "PASS\n");
  return rc;
}
