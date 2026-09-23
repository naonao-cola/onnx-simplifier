/* Phone-side test of the fbgather skel: loads the 4 tables, 4 LUTs and the expected volume written
 * by ../dsp_inputs.py (cwd), runs REPS calls per thread count, and prints the DSP-side time, the
 * client round trip and the exact-byte check.   ./fbgather_client URI [reps] [turbo] [threads,..] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "fbgather_rpc.h"

#define NR (200 * 200 * 4)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void* load(const char* p, long n, long extra) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n + extra);
  if (!b || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "load %s failed\n", p); exit(1); }
  memset((char*)b + n, 0, extra);
  fclose(f);
  return b;
}
static int cmp_d(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 10, turbo = argc > 3 ? atoi(argv[3]) : 0;
  const char* cfgs = argc > 4 ? argv[4] : "1,2,4,6";
  if (reps > 64) reps = 64;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (fbgather_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  if (turbo) { int vrc = -1; fbgather_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }
  const int rows = 6 * 64 * 176;
  const long tb = (rows + 1L) * 64, vb = NR * 256L;
  unsigned char* t[4]; int* l[4]; char p[32];
  for (int i = 0; i < 4; i++) { snprintf(p, sizeof p, "t%d.bin", i); t[i] = load(p, tb, 64); }
  for (int i = 0; i < 4; i++) { snprintf(p, sizeof p, "l%d.bin", i); l[i] = load(p, NR * 4L, 0); }
  unsigned char* ref = load("vol_ref.bin", vb, 0);
  unsigned char* vol = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, vb);
  for (const char* c = cfgs; *c;) {
    int flags = atoi(c);
    double rt[64], ds[64];
    int rc = 0;
    for (int r = 0; r < reps; r++) {
      unsigned long long du = 0;
      double a = now_us();
      rc |= fbgather_rpc_run(h, t[0], tb + 64, t[1], tb + 64, t[2], tb + 64, t[3], tb + 64, l[0], NR, l[1], NR,
                             l[2], NR, l[3], NR, rows, flags, vol, vb, &du);
      rt[r] = (now_us() - a) / 1e3;
      ds[r] = du / 1e3;
    }
    long ex = 0;
    for (long i = 0; i < vb; i++) ex += vol[i] == ref[i];
    qsort(rt, reps, sizeof(double), cmp_d);
    qsort(ds, reps, sizeof(double), cmp_d);
    printf("threads %d: rc %d  dsp median %.2f ms  round trip median %.2f ms (min %.2f)  exact %ld/%ld\n", flags & 0xff,
           rc, ds[reps / 2], rt[reps / 2], rt[0], ex, vb);
    while (*c && *c != ',') c++;
    if (*c) c++;
  }
  fbgather_rpc_close(h);
  return 0;
}
