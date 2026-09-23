/* Loads the fused span's model constants for the host check and the phone client:
 *   PD_DIR/levels.txt  one line per level, ../proposal_decode's format:
 *                      A k H W s1 z1 s2 z2 exp_clip clip_x clip_y box_s box_z bb_s bb_z
 *   PD_DIR/lN_anchors.bin
 *   DATA/model.txt     one line per level: k iou max_out cap filt_s filt_z; then post_cap
 * Thresholds and scales are printed by the capture as the float32 value's shortest double repr,
 * so strtof-style parsing gets back the exact float32 the graph holds. */
#ifndef RPN_MODEL_IO_H
#define RPN_MODEL_IO_H
#include <stdio.h>
#include <stdlib.h>
#include "rpn_kernel.h"

static void* rpn_load(const char* path, long bytes) {
  void* b = malloc(bytes ? bytes : 1);
  FILE* f = fopen(path, "rb");
  if (!f || fread(b, 1, bytes, f) != (size_t)bytes) { fprintf(stderr, "load %s (%ld bytes)\n", path, bytes); exit(1); }
  fclose(f);
  return b;
}

/* anchors[l] gets level l's [A,4] table. Returns 0 or exits. */
static int rpn_load_model(const char* pd_dir, const char* data_dir, rpn_model* M, float** anchors) {
  char p[512];
  snprintf(p, sizeof p, "%s/levels.txt", pd_dir);
  FILE* lf = fopen(p, "r");
  if (!lf) { fprintf(stderr, "open %s\n", p); exit(1); }
  for (int l = 0; l < RPN_NLVL; l++) {
    rpn_level_cfg* L = &M->lv[l];
    pd_params* P = &L->P;
    if (fscanf(lf, "%d %d %d %d %f %d %f %d %f %f %f %f %d %f %d", &L->A, &L->k, &L->H, &L->W, &P->s1, &P->z1,
               &P->s2, &P->z2, &P->exp_clip, &P->clip_x, &P->clip_y, &P->box_s, &P->box_z, &P->bb_s, &P->bb_z) != 15) {
      fprintf(stderr, "levels.txt line %d\n", l);
      exit(1);
    }
    snprintf(p, sizeof p, "%s/l%d_anchors.bin", pd_dir, l);
    anchors[l] = rpn_load(p, 16L * L->A);
  }
  fclose(lf);
  snprintf(p, sizeof p, "%s/model.txt", data_dir);
  FILE* mf = fopen(p, "r");
  if (!mf) { fprintf(stderr, "open %s\n", p); exit(1); }
  for (int l = 0; l < RPN_NLVL; l++) {
    rpn_level_cfg* L = &M->lv[l];
    int k;
    if (fscanf(mf, "%d %f %d %d %f %d", &k, &L->iou, &L->max_out, &L->cap, &L->filt_s, &L->filt_z) != 6 || k != L->k) {
      fprintf(stderr, "model.txt line %d (k %d vs %d)\n", l, k, L->k);
      exit(1);
    }
  }
  if (fscanf(mf, "%d", &M->post_cap) != 1) { fprintf(stderr, "model.txt post_cap\n"); exit(1); }
  fclose(mf);
  return 0;
}
#endif
