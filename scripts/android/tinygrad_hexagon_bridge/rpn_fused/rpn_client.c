/* Native ARM64 FastRPC client (no TVM) for the fused RPN skel. Uploads the model once, then per
 * image dir (argv[4..]): for each delta source (fp32 per-anchor / backbone NCHW uint8) and
 * threading mode, one untimed run whose proposal list must be byte-exact with ORT's (and whose
 * per-level NMS kept counts must match), then REPS timed runs: median DSP time
 * (HAP_perf_get_time_us around the whole span) and median client round trip.
 *   rpn_client URI REPS TURBO IMG_DIR...     (cwd: levels.txt, model.txt, lN_anchors.bin) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "rpn_rpc.h"
#include "rpn_model_io.h"

#define RPN_STATS (2 + 3 * RPN_NLVL + 4 * RPN_NLVL + 1 + RPN_NLVL + 4)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static long fsize(const char* p) { FILE* f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }
static void loadinto(const char* p, void* b, long n) {
  FILE* f = fopen(p, "rb");
  if (!f || fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "load %s\n", p); exit(1); }
  fclose(f);
}
static void* rpcbuf(long n) {
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n > 0 ? n : 8);
  if (!b) { fprintf(stderr, "rpcmem_alloc %ld\n", n); exit(1); }
  return b;
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s URI REPS TURBO IMG_DIR...\n", argv[0]); return 2; }
  int reps = atoi(argv[2]); if (reps > 64) reps = 64; if (reps < 1) reps = 1;
  int turbo = atoi(argv[3]);
  static rpn_model M;
  float* an[RPN_NLVL];
  rpn_load_model(".", ".", &M, an);
  long atot = 0, stot = 0, ntot = 0;
  for (int l = 0; l < RPN_NLVL; l++) { atot += 4L * M.lv[l].A; stot += M.lv[l].A; ntot += 12L * M.lv[l].H * M.lv[l].W; }
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (rpn_rpc_open(argv[1], &h)) { puts("open failed"); return 1; }
  if (turbo) { int vrc = -1; rpn_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }
  float* anc = rpcbuf(4 * atot);
  for (long o = 0, l = 0; l < RPN_NLVL; o += 4L * M.lv[l].A, l++) memcpy(anc + o, an[l], 16L * M.lv[l].A);
  int grid = -1;
  int rc = rpn_rpc_set_model(h, anc, atot, (const unsigned char*)&M, sizeof M, &grid);
  printf("set_model rc=%d anchor grid mask=0x%x\n", rc, grid);
  if (rc) return 1;
  float* sc = rpcbuf(4 * stot);
  float* dl = rpcbuf(16 * stot);
  unsigned char* nq = rpcbuf(ntot);
  float* out = rpcbuf(16L * M.post_cap);
  int st[RPN_STATS];
  int bad = 0;
  struct { int src, mode, nsplit, nms; const char* name; } cfg[] = {
      {0, 0, 1, 1, "fp32 deltas, 1 thread"},
      {0, 1, 1, 1, "fp32 deltas, thread/level"},
      {0, 3, 4, 1, "fp32 deltas, phased (P2 split 4)"},
      {0, 4, 4, 1, "fp32 deltas, phased, decode per level"},
      {1, 0, 1, 1, "nchw u8, 1 thread"},
      {1, 1, 1, 1, "nchw u8, thread/level"},
      {1, 2, 4, 1, "nchw u8, thread/level + P2 split 4"},
      {1, 3, 4, 1, "nchw u8, phased (P2 split 4)"},
      {1, 3, 2, 1, "nchw u8, phased (P2 split 2)"},
      {1, 4, 4, 1, "nchw u8, phased, decode per level"},
      {1, 3, 4, 0, "nchw u8, phased, nms_scalar"},
  };
  int ncfg = sizeof cfg / sizeof cfg[0];
  for (int im = 4; im < argc; im++) {
    const char* d = argv[im];
    char p[512];
    long so = 0, dof = 0, no = 0;
    for (int l = 0; l < RPN_NLVL; l++) {
      const rpn_level_cfg* L = &M.lv[l];
      snprintf(p, sizeof p, "%s/l%d_scores.bin", d, l); loadinto(p, sc + so, 4L * L->A);
      snprintf(p, sizeof p, "%s/l%d_deltas.bin", d, l); loadinto(p, dl + dof, 16L * L->A);
      snprintf(p, sizeof p, "%s/l%d_nchw_q.bin", d, l); loadinto(p, nq + no, 12L * L->H * L->W);
      so += L->A; dof += 4L * L->A; no += 12L * L->H * L->W;
    }
    int rns[RPN_NLVL];
    for (int l = 0; l < RPN_NLVL; l++) { snprintf(p, sizeof p, "%s/l%d_nms_sel.bin", d, l); rns[l] = (int)(fsize(p) / 4); }
    snprintf(p, sizeof p, "%s/proposals.bin", d);
    long nref = fsize(p) / 16;
    float* ref = malloc(16 * nref);
    loadinto(p, ref, 16 * nref);
    for (int c = 0; c < ncfg; c++) {
      double dd[64], rr[64];
      int ok = 1;
      for (int k = -1; k < reps; k++) {
        memset(out, 0, 16L * M.post_cap);
        unsigned long long du = 0;
        double t0 = now_us();
        rc = rpn_rpc_run(h, sc, stot, dl, 4 * stot, nq, ntot, cfg[c].src, cfg[c].mode, cfg[c].nsplit, cfg[c].nms,
                         out, 4 * M.post_cap, st, RPN_STATS, &du);
        double t1 = now_us();
        if (rc) { printf("%s: rpc rc=%d\n", cfg[c].name, rc); return 1; }
        if (k < 0) {
          ok = st[0] == nref && !memcmp(out, ref, 16 * nref);
          for (int l = 0; l < RPN_NLVL; l++) ok &= st[2 + 3 * l + 1] == rns[l] && st[2 + 3 * l] == M.lv[l].k;
          continue;
        }
        dd[k] = (double)du; rr[k] = t1 - t0;
      }
      qsort(dd, reps, sizeof dd[0], cmpd); qsort(rr, reps, sizeof rr[0], cmpd);
      bad |= !ok;
      printf("%s  %-40s %s  proposals %d of %d  dsp_us(median)=%.0f roundtrip_us(median)=%.0f\n", d, cfg[c].name,
             ok ? "EXACT" : "MISMATCH", st[0], st[1], dd[reps / 2], rr[reps / 2]);
      if (getenv("PHASES")) {
        const int* ph = st + 2 + 3 * RPN_NLVL;
        for (int l = 0; l < RPN_NLVL; l++)
          printf("    level%d topk=%d decode=%d filter=%d nms=%d us, thread %d us (last rep)\n", l, ph[4 * l],
                 ph[4 * l + 1], ph[4 * l + 2], ph[4 * l + 3], st[2 + 3 * RPN_NLVL + 4 * RPN_NLVL + 1 + l]);
        const int* pw = st + 2 + 3 * RPN_NLVL + 4 * RPN_NLVL + 1 + RPN_NLVL;
        printf("    merge=%d us; phase walls (last rep, modes 3/4): A topk=%d B decode=%d C filter+nms=%d D merge=%d us\n",
               st[2 + 3 * RPN_NLVL + 4 * RPN_NLVL], pw[0], pw[1], pw[2], pw[3]);
      }
    }
    free(ref);
  }
  rpn_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
