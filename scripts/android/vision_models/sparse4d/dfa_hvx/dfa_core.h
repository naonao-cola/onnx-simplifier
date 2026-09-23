/* Sparse4D v3's deformable 4D aggregation (DFA) as 24 calls of the generic MSDA kernel
 * (../../../msda_hvx/msda_kernel.h, header-only, used unmodified).
 *
 * One decoder layer: for Q anchors (900), 8 groups x 32 channels, 6 cameras, 4 FPN levels,
 * 13 keypoints (padded to 16 with zero weights):
 *
 *   out[q, g*32:(g+1)*32] = sum over cam c, level l, point p of
 *       w[c, l, q, g, p] * bilinear(level l of camera c, group g's channels, at pts[c, q, p])
 *
 * which is model.dfa_upstream (grid_sample, align_corners=False, zero padding) summed over cameras
 * and points. Per (cam, level) this is exactly one msda call with NV = 1, L = 1, P = 16, M = 8, D = 32,
 * mode MSDA_REF_PIX with all-zero offsets and pts as the 16 per-query reference points (R = 16,
 * RD = 2): 8 * 1 * 1 * 16 = 128 points per query, inside the HVX body's limits. A per-camera
 * visibility mask (some point of the anchor inside [0, 1]^2 for that camera) skips the anchors a
 * camera can't see; msda writes zeros for them, the same as the zero padding would give.
 *
 * Inputs (all channels-last, rpcmem on the phone):
 *   val[l]   uint8 (6, H_l, W_l, 256), real = (u8 - zp[l]) * scale[l]  (the HTP backbone's output)
 *   pts      float (6, Q, 16, 2)       normalized image coordinates, pad points anywhere
 *   w        float (6, 4, Q, 8, 16)    softmaxed weights, 0 on the 3 pad points
 * Output: out float (Q, 256). */
#ifndef DFA_CORE_H
#define DFA_CORE_H
#include <stdint.h>
#include <string.h>

#include "msda_kernel.h"

#define DFA_CAMS 6
#define DFA_LEVELS 4
#define DFA_M 8
#define DFA_D 32
#define DFA_C (DFA_M * DFA_D)
#define DFA_P 16

typedef struct {
  int Q;
  int H[DFA_LEVELS], W[DFA_LEVELS];
  const uint8_t* val[DFA_LEVELS];
  float scale[DFA_LEVELS];
  int32_t zp[DFA_LEVELS];
  const float* pts;   /* (6, Q, 16, 2) */
  const float* w;     /* (6, 4, Q, 8, 16) */
  const float* zeros; /* (Q, 8, 16, 2) zeros: the msda offsets */
  uint8_t* vis;       /* (6, Q) scratch, filled by dfa_visibility */
  float* tmp;         /* (Q, 256) scratch: one (cam, level) call's result */
  float* out;         /* (Q, 256) */
} dfa_args_t;

/* per camera: 1 if any of the anchor's 13 real points (the pads are skipped) can touch the image.
 * A bilinear tap reaches half a pixel outside [0, 1] at its level, so the margin is a pixel of the
 * *coarsest* level (8 x 22 at 256 x 704; a level-0 margin dropped real contributions). */
static inline void dfa_visibility(const dfa_args_t* a, int q0, int q1) {
  const float mx = 1.0f / a->W[DFA_LEVELS - 1], my = 1.0f / a->H[DFA_LEVELS - 1];
  for (int c = 0; c < DFA_CAMS; c++)
    for (int q = q0; q < q1; q++) {
      const float* p = a->pts + ((long)c * a->Q + q) * DFA_P * 2;
      int v = 0;
      for (int k = 0; k < 13 && !v; k++)
        v = p[2 * k] > -mx && p[2 * k] < 1 + mx && p[2 * k + 1] > -my && p[2 * k + 1] < 1 + my;
      a->vis[c * a->Q + q] = (uint8_t)v;
    }
}

static inline void dfa_msda_args(const dfa_args_t* a, int c, int l, msda_args_t* m) {
  memset(m, 0, sizeof *m);
  m->NV = 1; m->L = 1; m->H[0] = a->H[l]; m->W[0] = a->W[l]; m->start[0] = 0; m->S = a->H[l] * a->W[l];
  m->M = DFA_M; m->D = DFA_D; m->P = DFA_P; m->Q = a->Q; m->NO = 1;
  m->mode = MSDA_REF_PIX; m->NVR = 1; m->RL = 1; m->R = DFA_P; m->RD = 2;
  m->vdtype = MSDA_U8;
  m->value_u8 = a->val[l] + (long)c * m->S * DFA_C;
  m->vscale = &a->scale[l];
  m->vzp = &a->zp[l];
  m->loc = a->zeros;
  m->ref = a->pts + (long)c * a->Q * DFA_P * 2;
  m->attw = a->w + ((long)c * DFA_LEVELS + l) * a->Q * DFA_M * DFA_P;
  m->vis = a->vis + c * a->Q;
  m->out = a->tmp;
}

static inline int dfa_check(const dfa_args_t* a) {
  msda_args_t m;
  for (int l = 0; l < DFA_LEVELS; l++) {
    dfa_msda_args(a, 0, l, &m);
    if (msda_check(&m)) return -1;
  }
  return 0;
}

/* rows [q0, q1) of out += tmp */
static inline void dfa_accumulate(float* out, const float* tmp, int q0, int q1) {
#if defined(MSDA_HVX)
  HVX_Vector* o = (HVX_Vector*)(out + (long)q0 * DFA_C);
  const HVX_Vector* t = (const HVX_Vector*)(tmp + (long)q0 * DFA_C);
  for (long i = 0; i < (long)(q1 - q0) * DFA_C / 32; i++)
    o[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(o[i], t[i]));
#else
  for (long i = (long)q0 * DFA_C; i < (long)q1 * DFA_C; i++) out[i] += tmp[i];
#endif
}

/* queries [q0, q1): visibility, then 24 msda calls accumulated into out. Thread-safe on disjoint ranges. */
static inline void dfa_run(const dfa_args_t* a, int q0, int q1) {
  dfa_visibility(a, q0, q1);
  memset(a->out + (long)q0 * DFA_C, 0, (size_t)(q1 - q0) * DFA_C * sizeof(float));
  msda_args_t m;
  for (int c = 0; c < DFA_CAMS; c++)
    for (int l = 0; l < DFA_LEVELS; l++) {
      dfa_msda_args(a, c, l, &m);
      msda_run(&m, q0, q1);
      dfa_accumulate(a->out, a->tmp, q0, q1);
    }
}
#endif
