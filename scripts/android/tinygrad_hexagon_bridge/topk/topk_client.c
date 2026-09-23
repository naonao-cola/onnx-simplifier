/* Native ARM64 FastRPC client (no TVM) for the TopK skel. For every real TopK call in calls.txt:
 * runs each kernel variant, checks values AND int64 indices byte-exact against ONNX Runtime's
 * captured output, and reports DSP-side kernel time (HAP_perf_get_time_us) and client round-trip
 * time. Then runs the 5 independent per-FPN-level calls (the first 5 lines) batched into one round
 * trip at several thread settings (N > 1: big collects split N ways; -1: one thread per call;
 * -N: one thread per call and the biggest call's collect split N ways).
 * Every configuration is warmed up once (untimed, and the exactness check) before timing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "topk_rpc.h"

#define NC_MAX 8
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void* load(const char* p, long n) {
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n > 0 ? n : 8);
  FILE* f = fopen(p, "rb"); if (!f || !b) { perror(p); exit(1); }
  if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(1); }
  fclose(f); return b;
}
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }
static const char* vname(int v) { return v == 3 ? "vec-reduce_or-mask" : v == 2 ? "vec-rot" : v == 1 ? "vec-reduce_or" : "scalar"; }

static remote_handle64 h;
static int reps;

/* Runs calls [c0, c0+nc) in one RPC; returns exactness; fills median dsp/roundtrip (us). */
static int run(float** x, float** rv, long long** ri, int* ns, int* ks, int c0, int nc, int variant, int nthreads,
               double* dsp_med, double* rt_med, int* surv) {
  long xn = 0, on = 0;
  for (int c = c0; c < c0 + nc; c++) { xn += ns[c]; on += ks[c]; }
  float* xb = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, xn * 4 + 4);
  float* ov = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, on * 4 + 4);
  long long* oi = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, on * 8 + 8);
  long o = 0;
  for (int c = c0; c < c0 + nc; c++) { memcpy(xb + o, x[c], ns[c] * 4L); o += ns[c]; }
  double d[64], r[64];
  unsigned ph[4 * NC_MAX];
  int ok = 1;
  for (int k = -1; k < reps; k++) {
    memset(ov, 0, on * 4); memset(oi, 0, on * 8);
    unsigned long long du = 0;
    double t0 = now_us();
    int rc = topk_rpc_run(h, xb, xn, ns + c0, nc, ks + c0, nc, variant, nthreads, ov, on, (int64*)oi, on, surv, nc, ph, 4 * nc, &du);
    double t1 = now_us();
    if (rc) { printf("rpc rc=%d\n", rc); exit(1); }
    if (k < 0) {
      long p = 0;
      for (int c = c0; c < c0 + nc; c++) {
        if (memcmp(ov + p, rv[c], ks[c] * 4L) || memcmp(oi + p, ri[c], ks[c] * 8L)) ok = 0;
        p += ks[c];
      }
      continue;
    }
    d[k] = (double)du; r[k] = t1 - t0;
  }
  if (getenv("PHASES") && nc == 1) printf("   phases(last rep) scratch_us=%u threshold_us=%u collect_us=%u sort_emit_us=%u\n", ph[0], ph[1], ph[2], ph[3]);
  qsort(d, reps, sizeof d[0], cmpd); qsort(r, reps, sizeof r[0], cmpd);
  *dsp_med = d[reps / 2]; *rt_med = r[reps / 2];
  rpcmem_free(xb); rpcmem_free(ov); rpcmem_free(oi);
  return ok;
}

int main(int argc, char** argv) {
  const char* uri = argv[1];
  reps = argc > 2 ? atoi(argv[2]) : 21;
  if (reps > 64) reps = 64;
  int turbo = argc > 3 ? atoi(argv[3]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  if (topk_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  if (turbo) { int vrc = -1; topk_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }
  FILE* cf = fopen("calls.txt", "r");
  float* x[NC_MAX]; float* rv[NC_MAX]; long long* ri[NC_MAX];
  int ns[NC_MAX], ks[NC_MAX], axis, nc = 0, bad = 0;
  long n, k;
  while (nc < NC_MAX && fscanf(cf, "%ld %ld %d", &n, &k, &axis) == 3) {
    char p[64];
    ns[nc] = (int)n; ks[nc] = (int)k;
    snprintf(p, sizeof p, "call%d_x.bin", nc); x[nc] = load(p, n * 4);
    snprintf(p, sizeof p, "call%d_vals.bin", nc); rv[nc] = load(p, k * 4);
    snprintf(p, sizeof p, "call%d_idx.bin", nc); ri[nc] = load(p, k * 8);
    nc++;
  }
  int variants[4] = {2, 0, 1, 3};
  double tot_d[4] = {0}, tot_r[4] = {0};
  for (int c = 0; c < nc; c++) {
    for (int vi = 0; vi < 4; vi++) {
      double dm, rm; int s;
      int ok = run(x, rv, ri, ns, ks, c, 1, variants[vi], 1, &dm, &rm, &s);
      if (!ok) bad = 1;
      tot_d[vi] += dm; tot_r[vi] += rm;
      printf("call%d n=%d k=%d %-18s threads=1 survivors=%d %s dsp_us(median)=%.0f roundtrip_us(median)=%.0f\n",
             c, ns[c], ks[c], vname(variants[vi]), s, ok ? "EXACT" : "MISMATCH", dm, rm);
    }
    if (ns[c] >= 32768) for (int th = 2; th <= 6; th += 2) {
      double dm, rm; int s;
      int ok = run(x, rv, ri, ns, ks, c, 1, 2, th, &dm, &rm, &s);
      if (!ok) bad = 1;
      printf("call%d n=%d k=%d %-13s threads=%d survivors=%d %s dsp_us(median)=%.0f roundtrip_us(median)=%.0f\n",
             c, ns[c], ks[c], vname(2), th, s, ok ? "EXACT" : "MISMATCH", dm, rm);
    }
  }
  for (int vi = 0; vi < 4; vi++)
    printf("TOTAL per-call (%d RPCs, 1 thread) %-18s dsp_us=%.0f roundtrip_us=%.0f\n", nc, vname(variants[vi]), tot_d[vi], tot_r[vi]);
  if (nc >= 7) {
    int sv[NC_MAX];
    double tail_d = 0, tail_r = 0;
    for (int c = 5; c < 7; c++) { double dm, rm; int s; run(x, rv, ri, ns, ks, c, 1, 2, 1, &dm, &rm, &s); tail_d += dm; tail_r += rm; }
    int ths[8] = {1, 2, 4, 6, -1, -2, -3, -4};
    for (int vi = 0; vi < 2; vi++) for (int q = 0; q < 8; q++) {
      int th = ths[q];
      if (vi == 1 && th != 1 && th != 4 && th != -1) continue;
      double dm, rm;
      int ok = run(x, rv, ri, ns, ks, 0, 5, variants[vi], th, &dm, &rm, sv);
      if (!ok) bad = 1;
      printf("BATCH 5 levels in 1 RPC %-13s threads=%-2d %s dsp_us(median)=%.0f roundtrip_us(median)=%.0f"
             " | +2 later calls: all 7 dsp_us=%.0f roundtrip_us=%.0f (3 RPCs)\n",
             vname(variants[vi]), th, ok ? "EXACT" : "MISMATCH", dm, rm, dm + tail_d, rm + tail_r);
    }
  }
  topk_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
