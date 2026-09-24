/* Native ARM64 FastRPC client (no TVM) for the proposal-decode skel. Loads the 5 captured levels
 * (capture_proposal_decode.py + pd_host_check), uploads the model constants once, then per config runs the
 * whole 5-level decode in ONE call, checks every box against ONNX Runtime's output exactly, and
 * reports DSP-side kernel time and client round-trip time. Also runs bb_layout (the backbone's
 * full-map NCHW->per-anchor dequant+transpose, which the nchw source makes unnecessary) and checks
 * it against rest.onnx's real per-anchor delta input. Buffers come from rpcmem (mapped, not copied).
 *   pd_client URI reps turbo */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "pd_kernel.h"
#include "pd_rpc.h"

#define NLVL 5
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void* ralloc(long n) { void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n > 0 ? n : 16); if (!b) { puts("rpcmem_alloc failed"); exit(1); } return b; }
static void loadf(const char* name, int lvl, void* dst, long bytes) {
  char p[64]; snprintf(p, sizeof p, "l%d_%s.bin", lvl, name);
  FILE* f = fopen(p, "rb");
  if (!f || fread(dst, 1, bytes, f) != (size_t)bytes) { printf("load %s failed\n", p); exit(1); }
  fclose(f);
}
static int cmp_u64(const void* a, const void* b) { unsigned long long x = *(const unsigned long long*)a, y = *(const unsigned long long*)b; return x < y ? -1 : x > y; }
static int cmp_d(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 15;
  int turbo = argc > 3 ? atoi(argv[3]) : 0;
  int A[NLVL], K[NLVL], H[NLVL], W[NLVL], meta[8 * NLVL];
  pd_params P[NLVL];
  long totA = 0, totK = 0, totN = 0;
  FILE* lf = fopen("levels.txt", "r");
  for (int l = 0; l < NLVL; l++) {
    pd_params* q = &P[l];
    if (fscanf(lf, "%d %d %d %d %f %d %f %d %f %f %f %f %d %f %d", &A[l], &K[l], &H[l], &W[l], &q->s1, &q->z1, &q->s2, &q->z2,
               &q->exp_clip, &q->clip_x, &q->clip_y, &q->box_s, &q->box_z, &q->bb_s, &q->bb_z) != 15) { puts("levels.txt"); return 1; }
    int* m = meta + 8 * l;
    m[0] = A[l]; m[1] = K[l]; m[2] = H[l]; m[3] = W[l];
    m[4] = 4 * totA; m[5] = 4 * totA; m[6] = totN; m[7] = 4 * totK;
    totA += A[l]; totK += K[l]; totN += 12L * H[l] * W[l];
  }
  float* anchors = ralloc(16 * totA); float* deltas = ralloc(16 * totA); unsigned char* nchw = ralloc(totN);
  int* idx = ralloc(4 * totK); float* ref = ralloc(16 * totK); float* out = ralloc(16 * totK);
  float* lay = ralloc(16 * totA);
  int* rmeta = ralloc(sizeof meta); unsigned char* rparams = ralloc(sizeof P);
  memcpy(rmeta, meta, sizeof meta); memcpy(rparams, P, sizeof P);
  for (int l = 0; l < NLVL; l++) {
    int* m = meta + 8 * l;
    loadf("anchors", l, anchors + m[4], 16L * A[l]);
    loadf("deltas", l, deltas + m[5], 16L * A[l]);
    loadf("nchw_q", l, nchw + m[6], 12L * H[l] * W[l]);
    loadf("idx", l, idx + m[7] / 4, 4L * K[l]);
    loadf("ref", l, ref + m[7], 16L * K[l]);
  }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (pd_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  if (turbo) { int rc = -1; pd_rpc_perf_vote(h, 1, &rc); printf("perf_vote(TURBO) rc=%d\n", rc); }
  double t0 = now_us();
  if (pd_rpc_set_model(h, anchors, 4 * totA, rmeta, 8 * NLVL, rparams, sizeof P)) { puts("set_model failed"); return 1; }
  printf("set_model (once: %ld anchors = %.1f MB, 5 levels' params + LUTs): %.0f us\n", totA, 16.0 * totA / 1e6, now_us() - t0);
  int gm = 0; pd_rpc_anchor_grid(h, &gm);
  printf("anchor tables verified as exact grids on the DSP (computed, not gathered): mask=0x%x\n", gm);
  int bad = 0;
  int thr[] = {1, 4, 6};
  for (int v = 0; v < 12; v++) {
    int src = v / 6, fast = (v / 3) % 2, nt = thr[v % 3];
    unsigned long long d[64]; double rt[64];
    for (int r = 0; r < reps; r++) {
      memset(out, 0, 16 * totK);
      double a = now_us();
      int rc = pd_rpc_run(h, idx, totK, deltas, src ? 0 : 4 * totA, nchw, src ? totN : 0, src, fast, nt, out, 4 * totK, &d[r]);
      rt[r] = now_us() - a;
      if (rc) { printf("run rc=%d\n", rc); return 1; }
    }
    long nd = 0; for (long i = 0; i < 4 * totK; i++) nd += out[i] != ref[i];
    bad |= nd != 0;
    qsort(d, reps, sizeof d[0], cmp_u64); qsort(rt, reps, sizeof rt[0], cmp_d);
    printf("decode src=%-10s path=%-9s threads=%d  dsp_us median=%llu min=%llu  roundtrip_us median=%.0f min=%.0f  boxes=%ld diff_elems=%ld\n",
           src ? "nchw_u8" : "deltas_f32", fast ? "fast" : "reference", nt, d[reps / 2], d[0], rt[reps / 2], rt[0], totK, nd);
  }
  for (int ti = 0; ti < 2; ti++) {
    unsigned long long d[64]; double rt[64];
    for (int r = 0; r < reps; r++) {
      double a = now_us();
      int rc = pd_rpc_bb_layout(h, nchw, totN, thr[ti], lay, 4 * totA, &d[r]);
      rt[r] = now_us() - a;
      if (rc) { printf("bb_layout rc=%d\n", rc); return 1; }
    }
    long nd = 0; for (long i = 0; i < 4 * totA; i++) nd += lay[i] != deltas[i];
    bad |= nd != 0;
    qsort(d, reps, sizeof d[0], cmp_u64); qsort(rt, reps, sizeof rt[0], cmp_d);
    printf("bb_layout (full-map NCHW u8 -> per-anchor f32, %ld anchors) threads=%d  dsp_us median=%llu min=%llu  roundtrip_us median=%.0f  diff_vs_rest_input=%ld\n",
           totA, thr[ti], d[reps / 2], d[0], rt[reps / 2], nd);
  }
  pd_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
