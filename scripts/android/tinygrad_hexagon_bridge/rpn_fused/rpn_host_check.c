/* Host check of the whole fused RPN span (rpn_kernel.h) against ONNX Runtime, per image:
 * every intermediate (per-level TopK values+indices, decoded boxes, NMS selections, post-NMS TopK
 * indices) and the final proposal list, byte for byte. Runs both delta sources (fp32 per-anchor,
 * backbone NCHW uint8), both NMS kernels (nms_hvx's portable path, nms_scalar), and decode's
 * reference (division) path once.
 *   cc -O2 -ffp-contract=off -o rpn_host_check rpn_host_check.c
 *   ./rpn_host_check PD_DIR DATA IMG_DIR... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rpn_model_io.h"

static void* lv(const char* dir, int l, const char* name, long bytes) {
  char p[512];
  snprintf(p, sizeof p, "%s/l%d_%s.bin", dir, l, name);
  return rpn_load(p, bytes);
}
static long fsize(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fclose(f);
  return n;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s PD_DIR DATA IMG_DIR...\n", argv[0]); return 2; }
  static rpn_model M;
  float* anchors[RPN_NLVL];
  rpn_load_model(argv[1], argv[2], &M, anchors);
  { /* the parsed model as a raw struct, for the freestanding qemu harness (no strtof there) */
    char p[512];
    snprintf(p, sizeof p, "%s/model.bin", argv[2]);
    FILE* f = fopen(p, "wb");
    fwrite(&M, sizeof M, 1, f);
    fclose(f);
    for (int l = 0; l < RPN_NLVL; l++) {
      snprintf(p, sizeof p, "%s/l%d_anchors.bin", argv[2], l);
      f = fopen(p, "wb");
      fwrite(anchors[l], 16, M.lv[l].A, f);
      fclose(f);
    }
  }
  pd_prep prep[RPN_NLVL];
  for (int l = 0; l < RPN_NLVL; l++) {
    pd_prepare(&M.lv[l].P, &prep[l]);
    pd_prepare_anchors(anchors[l], M.lv[l].H, M.lv[l].W, &prep[l]);
  }
  static rpn_level_ws ws[RPN_NLVL];
  rpn_level_ws* wp[RPN_NLVL];
  for (int l = 0; l < RPN_NLVL; l++) wp[l] = &ws[l];
  static rpn_merge_ws mw;
  uint32_t* tks = malloc(4 * TK_SCRATCH_WORDS(M.lv[0].A));
  int bad = 0;
  for (int im = 3; im < argc; im++) {
    const char* d = argv[im];
    float *sc[RPN_NLVL], *dl[RPN_NLVL], *rv[RPN_NLVL], *rb[RPN_NLVL];
    uint8_t* nq[RPN_NLVL];
    int64_t* ri[RPN_NLVL];
    int32_t* rs[RPN_NLVL];
    int rns[RPN_NLVL];
    char p[512];
    for (int l = 0; l < RPN_NLVL; l++) {
      const rpn_level_cfg* L = &M.lv[l];
      sc[l] = lv(d, l, "scores", 4L * L->A);
      dl[l] = lv(d, l, "deltas", 16L * L->A);
      nq[l] = lv(d, l, "nchw_q", 12L * L->H * L->W);
      rv[l] = lv(d, l, "topk_vals", 4L * L->k);
      ri[l] = lv(d, l, "topk_idx", 8L * L->k);
      rb[l] = lv(d, l, "boxes", 16L * L->k);
      snprintf(p, sizeof p, "%s/l%d_nms_sel.bin", d, l);
      rns[l] = (int)(fsize(p) / 4);
      rs[l] = rpn_load(p, 4L * rns[l]);
    }
    snprintf(p, sizeof p, "%s/proposals.bin", d);
    long nprop = fsize(p) / 16;
    float* rprop = rpn_load(p, 16 * nprop);
    snprintf(p, sizeof p, "%s/post_idx.bin", d);
    int64_t* rpost = rpn_load(p, 8 * nprop);
    float* out = malloc(16 * RPN_MERGE_MAX);
    /* v: 0 deltas/hvx, 1 nchw/hvx, 2 deltas/scalar, 3 nchw/scalar, 4 deltas/hvx/decode-reference,
     * 5 deltas/hvx with the (provably no-op) filter actually computed */
    const char* vname[6] = {"deltas_f32 nms_hvx", "nchw_u8 nms_hvx", "deltas_f32 nms_scalar", "nchw_u8 nms_scalar",
                            "deltas_f32 nms_hvx decode-ref", "deltas_f32 nms_hvx exact-filter"};
    for (int v = 0; v < 6; v++) {
      int src = v == 1 || v == 3, hvx = v != 2 && v != 3;
      rpn_exact_filter = v == 5;
      int lvl_bad = 0;
      for (int l = 0; l < RPN_NLVL; l++) {
        const rpn_level_cfg* L = &M.lv[l];
        if (rpn_level(L, v == 4 ? NULL : &prep[l], anchors[l], sc[l], src ? NULL : dl[l], src ? nq[l] : NULL,
                      &ws[l], tks, rpn_collect_plain, 0, hvx)) { puts("rpn_level failed"); return 1; }
        int e_tk = memcmp(ws[l].vals, rv[l], 4L * L->k) || memcmp(ws[l].idx64, ri[l], 8L * L->k);
        int e_bx = memcmp(ws[l].boxes, rb[l], 16L * L->k);
        int e_nm = ws[l].ns != rns[l] || memcmp(ws[l].sel, rs[l], 4L * rns[l]);
        if (e_tk || e_bx || e_nm || ws[l].nf != L->k)
          printf("  %s level%d: topk %s, boxes %s, nms %s (kept %d vs %d), filter kept %d/%d\n", vname[v], l,
                 e_tk ? "DIFF" : "ok", e_bx ? "DIFF" : "ok", e_nm ? "DIFF" : "ok", ws[l].ns, rns[l], ws[l].nf, L->k);
        lvl_bad |= e_tk || e_bx || e_nm;
      }
      int total, kp = rpn_merge(&M, wp, &mw, out, &total);
      int e_post = kp != nprop || memcmp(mw.idx, rpost, 8 * kp);
      int e_out = kp != nprop || memcmp(out, rprop, 16 * kp);
      printf("%s  %-30s levels %s  post-NMS TopK idx %s  proposals %s (%d of %d kept boxes -> %d)\n", d, vname[v],
             lvl_bad ? "DIFF" : "exact", e_post ? "DIFF" : "exact", e_out ? "DIFF" : "exact", kp, total, (int)nprop);
      bad |= lvl_bad || e_post || e_out;
    }
    for (int l = 0; l < RPN_NLVL; l++) { free(sc[l]); free(dl[l]); free(nq[l]); free(rv[l]); free(ri[l]); free(rb[l]); free(rs[l]); }
    free(rprop); free(rpost); free(out);
  }
  puts(bad ? "FAIL" : "PASS (every intermediate and the final proposals byte-exact vs ORT, all images, all variants)");
  return bad;
}
