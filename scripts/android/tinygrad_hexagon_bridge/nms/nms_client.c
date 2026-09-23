/* Native ARM64 FastRPC client (no TVM) for the NMS skel. For each group (level: 5 RPN calls,
 * class: 80 box-head calls) it runs the whole group in one RPC with the scalar and HVX kernels at
 * 1 and 4 threads, checks every call's selected indices exactly against ONNX Runtime's, and reports
 * DSP-side time (HAP_perf_get_time_us) and client round-trip time. It also runs each call as its
 * own RPC (one per graph node, like ORT runs them), to show what per-node dispatch would cost. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "nms_rpc.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void* load(const char* p, long* bytes) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, *bytes + 128);
  if (!b || fread(b, 1, *bytes, f) != (size_t)*bytes) { fprintf(stderr, "load %s\n", p); exit(1); }
  fclose(f); return b;
}
static int cmp_d(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* uri = argv[1];
  int reps = argc > 2 ? atoi(argv[2]) : 7;
  int turbo = argc > 3 ? atoi(argv[3]) : 0;
  struct remote_rpc_control_unsigned_module up = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof up);
  remote_handle64 h;
  if (nms_rpc_open(uri, &h)) { puts("open failed"); return 1; }
  int vrc = -1;
  if (turbo) { nms_rpc_perf_vote(h, 1, &vrc); printf("perf_vote(TURBO) rc=%d\n", vrc); }
  const char* groups[2] = {"level", "class"};
  int bad = 0;
  for (int g = 0; g < 2; g++) {
    char p[64]; long nb;
    snprintf(p, sizeof p, "%s_boxes.bin", groups[g]); float* boxes = load(p, &nb);
    snprintf(p, sizeof p, "%s_scores.bin", groups[g]); float* scores = load(p, &nb); int tot = nb / 4;
    snprintf(p, sizeof p, "%s_ref.bin", groups[g]); int* ref = load(p, &nb);
    snprintf(p, sizeof p, "%s_thr.bin", groups[g]); float* thr = load(p, &nb); int nc = nb / 4;
    int* n = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4 * nc);
    int* mo = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4 * nc);
    int* nr = malloc(4 * nc);
    snprintf(p, sizeof p, "%s_calls.txt", groups[g]); FILE* cf = fopen(p, "r");
    float t_; for (int c = 0; c < nc; c++) if (fscanf(cf, "%d %f %d %d", &n[c], &t_, &mo[c], &nr[c]) != 4) return 1;
    fclose(cf);
    int* sel = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4 * tot + 128);
    int* nsel = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4 * nc);
    int cfg[6][2] = {{0, 1}, {0, 4}, {1, 1}, {1, 4}, {1, 5}, {1, 6}};
    for (int k = 0; k < 6; k++) {
      double d[32], rt[32]; int ok = 0;
      for (int r = 0; r < reps; r++) {
        unsigned long long du; memset(sel, 0xff, 4 * tot); memset(nsel, 0, 4 * nc);
        double t0 = now_us();
        int rc = nms_rpc_run(h, boxes, 4 * tot, scores, tot, n, nc, thr, nc, mo, nc, cfg[k][0], cfg[k][1], sel, tot, nsel, nc, &du);
        rt[r] = now_us() - t0; d[r] = du;
        if (rc) { printf("%s rc=%d\n", groups[g], rc); return 1; }
      }
      long bo = 0, ro = 0;
      for (int c = 0; c < nc; c++) {
        int good = nsel[c] == nr[c] && !memcmp(sel + bo, ref + ro, 4 * nr[c]);
        if (!good) {
          int j = 0; while (j < nsel[c] && j < nr[c] && sel[bo + j] == ref[ro + j]) j++;
          printf("  MISMATCH %s call%d n=%d got=%d want=%d first_diff_pos=%d got_idx=%d want_idx=%d\n", groups[g], c, n[c], nsel[c], nr[c], j,
                 j < nsel[c] ? sel[bo + j] : -1, j < nr[c] ? ref[ro + j] : -1);
        }
        ok += good; bo += n[c]; ro += nr[c];
      }
      if (ok != nc) bad = 1;
      qsort(d, reps, sizeof d[0], cmp_d); qsort(rt, reps, sizeof rt[0], cmp_d);
      printf("%s batched %s threads=%d exact=%d/%d dsp_us(median)=%.0f roundtrip_us(median)=%.0f roundtrip_us(min)=%.0f\n",
             groups[g], cfg[k][0] ? "hvx" : "scalar", cfg[k][1], ok, nc, d[reps / 2], rt[reps / 2], rt[0]);
    }
    /* one RPC per call (per graph node), hvx kernel, 1 thread */
    double rts[32];
    for (int r = 0; r < reps; r++) {
      double t0 = now_us(); long bo = 0;
      for (int c = 0; c < nc; c++) {
        unsigned long long du; int ns1 = 0;
        int rc = nms_rpc_run(h, boxes + 4 * bo, 4 * n[c], scores + bo, n[c], n + c, 1, thr + c, 1, mo + c, 1, 1, 1, sel + bo, n[c], &ns1, 1, &du);
        if (rc) { printf("%s per-call rc=%d\n", groups[g], rc); return 1; }
        bo += n[c];
      }
      rts[r] = now_us() - t0;
    }
    qsort(rts, reps, sizeof rts[0], cmp_d);
    printf("%s per-node RPCs (%d calls) hvx threads=1 total_roundtrip_us(median)=%.0f (min)=%.0f\n", groups[g], nc, rts[reps / 2], rts[0]);
    rpcmem_free(boxes); rpcmem_free(scores); rpcmem_free(ref); rpcmem_free(thr); rpcmem_free(n); rpcmem_free(mo); rpcmem_free(sel); rpcmem_free(nsel); free(nr);
  }
  nms_rpc_close(h);
  puts(bad ? "FAIL" : "PASS");
  return bad;
}
