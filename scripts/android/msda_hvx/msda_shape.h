/* msda_rpc.idl's packed `shape` <-> msda_args_t, shared by the skel and every client. */
#ifndef MSDA_SHAPE_H
#define MSDA_SHAPE_H

#include "msda_kernel.h"

#define MSDA_SHAPE_HDR 15
#define MSDA_SHAPE_LEN(L) (MSDA_SHAPE_HDR + 3 * (L))

static inline int msda_shape_pack(const msda_args_t* a, int32_t* s) {
  const int32_t h[MSDA_SHAPE_HDR] = {a->NV, a->L, a->S, a->M, a->D, a->P, a->Q, a->NO, a->mode, a->NVR, a->RL, a->R, a->RD,
                                     a->vis != 0, a->vdtype};
  for (int i = 0; i < MSDA_SHAPE_HDR; i++) s[i] = h[i];
  for (int l = 0; l < a->L; l++) {
    s[MSDA_SHAPE_HDR + 3 * l] = a->H[l];
    s[MSDA_SHAPE_HDR + 3 * l + 1] = a->W[l];
    s[MSDA_SHAPE_HDR + 3 * l + 2] = a->start[l];
  }
  return MSDA_SHAPE_LEN(a->L);
}

/* returns has_vis, or -1 if malformed */
static inline int msda_shape_unpack(const int32_t* s, int n, msda_args_t* a) {
  if (n < MSDA_SHAPE_HDR || s[1] < 1 || s[1] > MSDA_MAX_L || n < MSDA_SHAPE_LEN(s[1])) return -1;
  a->NV = s[0]; a->L = s[1]; a->S = s[2]; a->M = s[3]; a->D = s[4]; a->P = s[5]; a->Q = s[6]; a->NO = s[7];
  a->mode = s[8]; a->NVR = s[9]; a->RL = s[10]; a->R = s[11]; a->RD = s[12]; a->vdtype = s[14];
  for (int l = 0; l < a->L; l++) {
    a->H[l] = s[MSDA_SHAPE_HDR + 3 * l];
    a->W[l] = s[MSDA_SHAPE_HDR + 3 * l + 1];
    a->start[l] = s[MSDA_SHAPE_HDR + 3 * l + 2];
  }
  return s[13] != 0;
}

/* element counts of every buffer */
static inline long msda_n_value(const msda_args_t* a) { return (long)a->NV * a->S * a->M * a->D; } /* elements */
static inline long msda_n_loc(const msda_args_t* a) { return (long)a->Q * a->M * a->NO * a->L * a->P * 2; }
static inline long msda_n_ref(const msda_args_t* a) {
  return a->mode == MSDA_LOC ? 0 : (long)a->NVR * a->Q * a->RL * a->R * a->RD;
}
static inline long msda_n_attw(const msda_args_t* a) { return (long)a->Q * a->M * a->NO * a->L * a->P; }
static inline long msda_n_vis(const msda_args_t* a) { return (long)a->NV * a->Q; }
static inline long msda_n_out(const msda_args_t* a) { return (long)a->Q * a->M * a->D; }

#endif
