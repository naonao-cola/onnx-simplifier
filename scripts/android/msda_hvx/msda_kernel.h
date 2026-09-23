/* Multi-scale deformable attention (MSDA) on the Hexagon CDSP -- model-agnostic, header-only.
 * See README.md for the full contract; in short, per query q and head h:
 *
 *   out[q, h*D:(h+1)*D] = 1/max(1, #visible maps) * sum_{visible map v} sum_{level l, point p}
 *                         attw[q, h, o, l, p] * bilinear(value[v, level l, :, h*D:(h+1)*D], loc)
 *
 * with o = v when offsets are per map (NO = NV) else 0, `bilinear` = grid_sample(mode=bilinear,
 * padding_mode=zeros, align_corners=False) at pixel (loc_x * W_l - 0.5, loc_y * H_l - 0.5) -- mmcv's
 * multi_scale_deformable_attn_pytorch for NV = 1. The normalized location `loc` is given directly
 * (MSDA_LOC) or formed from a reference point and a raw offset:
 *   MSDA_REF_PIX: loc = ref.xy + off / (W_l, H_l)            (Deformable DETR 2-d refs, BEVFormer)
 *   MSDA_REF_BOX: loc = ref.xy + off / P * ref.wh * 0.5      (Deformable DETR / RT-DETR box refs)
 * Several value maps (NV > 1) cover BEVFormer: the SCA's cameras (shared offsets, a per-(camera,
 * query) visibility mask, camera average) and the TSA's 2-frame queue (per-frame offsets, mean).
 *
 * Two bodies: a scalar one (any shape; the host reference msda_host_check.c compares it with torch)
 * and, on 128-byte HVX V68+, an HVX qf32 one (D = 32, M <= 32; M * NO * L * P a multiple of 32,
 * <= 256; every level H * W < 65536; ~24 KB of stack): per query, 32 points per vector compute location, floor, bilinear weights
 * and tap addresses (floor without V73's vector float->int convert: + 1.5 * 2^23 in qf32 -> sf puts
 * the nearest integer in the mantissa, a +-1 fix-up makes it the floor), then 4 independent qf32
 * accumulators multiply-accumulate each head's taps, one 128-byte vector per tap. qf32 differs from
 * IEEE fp32 only by rounding. */
#ifndef MSDA_KERNEL_H
#define MSDA_KERNEL_H

#include <stdint.h>
#include <string.h>

#define MSDA_MAX_L 8
#define MSDA_MAX_NV 8
#define MSDA_MAX_D 256
enum { MSDA_LOC = 0, MSDA_REF_PIX = 1, MSDA_REF_BOX = 2 };

typedef struct {
  int NV;                                                /* value maps averaged over (1: plain MSDA) */
  int L;                                                 /* levels */
  int H[MSDA_MAX_L], W[MSDA_MAX_L], start[MSDA_MAX_L];   /* level l = rows [start, start + H*W) of a map */
  int S;                                                 /* rows per value map */
  int M, D;                                              /* heads, head dim: a row has C = M * D channels */
  int P;                                                 /* points per (head, level) */
  int Q;                                                 /* queries */
  int NO;                                                /* offsets/weights: 1 = shared by all maps, NV = per map */
  int mode;                                              /* MSDA_LOC / MSDA_REF_PIX / MSDA_REF_BOX */
  int NVR, RL, R, RD;                                    /* ref (NVR, Q, RL, R, RD): NVR 1 or NV, RL 1 or L,
                                                            point p uses entry p % R, RD 2 (x, y) or 4 (cx, cy, w, h) */
  const float* value;                                    /* (NV, S, C) channels-last */
  const float* loc;                                      /* (Q, M, NO, L, P, 2): locations (MSDA_LOC) or raw offsets */
  const float* ref;                                      /* see NVR..RD; unused for MSDA_LOC */
  const float* attw;                                     /* (Q, M, NO, L, P), softmaxed */
  const uint8_t* vis;                                    /* (NV, Q) or NULL = all visible */
  float* out;                                            /* (Q, C) */
} msda_args_t;

/* 0 if the shape is valid (sizes are the callers' to check against their buffers). */
static inline int msda_check(const msda_args_t* A) {
  if (A->NV < 1 || A->NV > MSDA_MAX_NV || A->L < 1 || A->L > MSDA_MAX_L || A->M < 1 || A->D < 1 || A->D > MSDA_MAX_D ||
      A->P < 1 || A->Q < 1 || (A->NO != 1 && A->NO != A->NV) || A->mode < MSDA_LOC || A->mode > MSDA_REF_BOX)
    return -1;
  if (A->mode != MSDA_LOC &&
      ((A->NVR != 1 && A->NVR != A->NV) || (A->RL != 1 && A->RL != A->L) || A->R < 1 || (A->RD != 2 && A->RD != 4) ||
       (A->mode == MSDA_REF_BOX && A->RD != 4)))
    return -1;
  for (int l = 0; l < A->L; l++)
    if (A->H[l] < 1 || A->W[l] < 1 || A->start[l] < 0 || (long)A->start[l] + (long)A->H[l] * A->W[l] > A->S) return -1;
  return 0;
}

/* Pixel coordinates of one point: x = ax + off_x * sx (and y), per the location mode. */
static inline void msda_affine(const msda_args_t* A, int v, int q, int l, int p, float* ax, float* ay, float* sx,
                               float* sy) {
  const float W = (float)A->W[l], H = (float)A->H[l];
  if (A->mode == MSDA_LOC) {
    *ax = -0.5f; *ay = -0.5f; *sx = W; *sy = H;
    return;
  }
  const float* r = A->ref + ((((long)(A->NVR > 1 ? v : 0) * A->Q + q) * A->RL + (A->RL > 1 ? l : 0)) * A->R + p % A->R) * A->RD;
  *ax = r[0] * W - 0.5f;
  *ay = r[1] * H - 0.5f;
  if (A->mode == MSDA_REF_PIX) {
    *sx = 1.0f; *sy = 1.0f;
  } else {
    *sx = r[2] * 0.5f / (float)A->P * W;
    *sy = r[3] * 0.5f / (float)A->P * H;
  }
}

/* Scalar body, queries [q0, q1). Plain C: any shape; the reference the HVX body is checked against. */
static void msda_run_scalar(const msda_args_t* A, int q0, int q1) {
  const int NV = A->NV, L = A->L, M = A->M, D = A->D, P = A->P, Q = A->Q, NO = A->NO, C = M * D;
  for (int q = q0; q < q1; q++) {
    int maps[MSDA_MAX_NV], n = 0;
    for (int v = 0; v < NV; v++)
      if (!A->vis || A->vis[(long)v * Q + q]) maps[n++] = v;
    const float inv = 1.0f / (float)(n > 1 ? n : 1);
    for (int h = 0; h < M; h++) {
      float acc[MSDA_MAX_D];
      for (int c = 0; c < D; c++) acc[c] = 0.0f;
      for (int k = 0; k < n; k++) {
        const int v = maps[k], o = NO > 1 ? v : 0;
        const float* vmap = A->value + (long)v * A->S * C + h * D;
        for (int l = 0; l < L; l++) {
          const int Wl = A->W[l], Hl = A->H[l];
          const float* vl = vmap + (long)A->start[l] * C;
          const long pt = ((((long)q * M + h) * NO + o) * L + l) * P;
          for (int p = 0; p < P; p++) {
            float ax, ay, sx, sy;
            msda_affine(A, v, q, l, p, &ax, &ay, &sx, &sy);
            const float x = ax + A->loc[(pt + p) * 2] * sx, y = ay + A->loc[(pt + p) * 2 + 1] * sy;
            if (!((x > -1.0f) & (x < (float)Wl) & (y > -1.0f) & (y < (float)Hl))) continue; /* also NaN */
            const int x0 = (int)(x + 1.0f) - 1, y0 = (int)(y + 1.0f) - 1;             /* floor: x + 1 > 0 */
            const float fx = x - (float)x0, fy = y - (float)y0, a = A->attw[pt + p] * inv;
            const float w[4] = {(1.0f - fy) * (1.0f - fx) * a, (1.0f - fy) * fx * a, fy * (1.0f - fx) * a, fy * fx * a};
            const int xs[4] = {x0, x0 + 1, x0, x0 + 1}, ys[4] = {y0, y0, y0 + 1, y0 + 1};
            for (int t = 0; t < 4; t++) {
              if (xs[t] < 0 || xs[t] >= Wl || ys[t] < 0 || ys[t] >= Hl) continue;
              const float* row = vl + ((long)ys[t] * Wl + xs[t]) * C;
              for (int c = 0; c < D; c++) acc[c] += row[c] * w[t];
            }
          }
        }
      }
      for (int c = 0; c < D; c++) A->out[(long)q * C + h * D + c] = acc[c];
    }
  }
}

#if defined(__HVX__) && __HVX_LENGTH__ == 128 && defined(__HVX_ARCH__) && __HVX_ARCH__ >= 68
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define MSDA_HVX 1
#define MSDA_VPTS 256    /* points per iteration: M * NO * L * P <= 256 (8 vectors) */
#define MSDA_MAX_K 64    /* distinct (o, level, ref entry) per iteration: NO * L * R */
#define MSDA_HVX_MAX_M 32

static inline int32_t msda_bits(float f) { int32_t i; memcpy(&i, &f, 4); return i; }
static inline HVX_Vector msda_sf(HVX_Vector qf) { return Q6_Vsf_equals_Vqf32(qf); }
static inline HVX_Vector msda_splatf(float f) { return Q6_V_vsplat_R(msda_bits(f)); }
#define MSDA_TAP(a, p, w) \
  (a) = Q6_Vqf32_vadd_Vqf32Vqf32((a), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector*)(p), Q6_V_vsplat_R(msda_bits(w))))

typedef struct {
  int npv, k;
  HVX_Vector epat[MSDA_VPTS / 32];   /* per lane: entry (o * L + l) * R + p % R */
  /* per lane, fixed for the whole call (they depend on the lane's head / o / level only) */
  HVX_Vector base[MSDA_VPTS / 32];   /* head * D * 4 (+ map o's byte offset when NO > 1) */
  HVX_Vector wm1[MSDA_VPTS / 32], hm1[MSDA_VPTS / 32], wi[MSDA_VPTS / 32], st[MSDA_VPTS / 32];
  HVX_Vector wf[MSDA_VPTS / 32], hf[MSDA_VPTS / 32];
  HVX_Vector sx[MSDA_VPTS / 32], sy[MSDA_VPTS / 32];  /* MSDA_LOC / MSDA_REF_PIX: x = ax + off * sx */
} msda_lanes_t;

static int msda_hvx_ok(const msda_args_t* A) {
  const long np = (long)A->M * A->NO * A->L * A->P, C4 = (long)A->M * A->D * 4;
  if (A->D != 32 || A->M > MSDA_HVX_MAX_M || np % 32 || np > MSDA_VPTS || (long)A->NO * A->L * (A->mode == MSDA_LOC ? 1 : A->R) > MSDA_MAX_K ||
      C4 >= 32768 || (long long)A->NV * A->S * C4 >= (1LL << 31))
    return 0;
  for (int l = 0; l < A->L; l++)
    if ((long)A->H[l] * A->W[l] >= 65536) return 0;
  return 1;
}

static void msda_lanes_init(const msda_args_t* A, msda_lanes_t* L) {
  int32_t ep[MSDA_VPTS] __attribute__((aligned(128))), b[MSDA_VPTS] __attribute__((aligned(128)));
  int32_t wm1[MSDA_VPTS] __attribute__((aligned(128))), hm1[MSDA_VPTS] __attribute__((aligned(128)));
  int32_t st[MSDA_VPTS] __attribute__((aligned(128)));
  float wf[MSDA_VPTS] __attribute__((aligned(128))), hf[MSDA_VPTS] __attribute__((aligned(128)));
  float sx[MSDA_VPTS] __attribute__((aligned(128))), sy[MSDA_VPTS] __attribute__((aligned(128)));
  const int R = A->mode == MSDA_LOC ? 1 : A->R, per_head = A->NO * A->L * A->P, np = A->M * per_head;
  const long long C4 = (long long)A->M * A->D * 4;
  L->npv = np / 32;
  L->k = A->NO * A->L * R;
  for (int g = 0; g < np; g++) {
    const int o = (g / (A->L * A->P)) % A->NO, l = (g / A->P) % A->L, p = g % A->P;
    ep[g] = (o * A->L + l) * R + p % R;
    b[g] = (g / per_head) * A->D * 4 + (A->NO > 1 ? (int32_t)(o * A->S * C4) : 0);
    wm1[g] = A->W[l] - 1;
    hm1[g] = A->H[l] - 1;
    st[g] = A->start[l];
    wf[g] = (float)A->W[l];
    hf[g] = (float)A->H[l];
    sx[g] = A->mode == MSDA_LOC ? (float)A->W[l] : 1.0f;
    sy[g] = A->mode == MSDA_LOC ? (float)A->H[l] : 1.0f;
  }
  for (int j = 0; j < L->npv; j++) {
    L->epat[j] = ((HVX_Vector*)ep)[j];
    L->base[j] = ((HVX_Vector*)b)[j];
    L->wm1[j] = ((HVX_Vector*)wm1)[j];
    L->hm1[j] = ((HVX_Vector*)hm1)[j];
    L->wi[j] = Q6_Vw_vadd_VwVw(L->wm1[j], Q6_V_vsplat_R(1));
    L->st[j] = ((HVX_Vector*)st)[j];
    L->wf[j] = ((HVX_Vector*)wf)[j];
    L->hf[j] = ((HVX_Vector*)hf)[j];
    L->sx[j] = ((HVX_Vector*)sx)[j];
    L->sy[j] = ((HVX_Vector*)sy)[j];
  }
}

/* Per-entry tables -> one lane vector: a splat when the field is uniform, else a vmux chain. */
static inline HVX_Vector msda_lanevec(const int32_t* t, int k, HVX_Vector epat) {
  HVX_Vector r = Q6_V_vsplat_R(t[0]);
  for (int e = 1; e < k; e++)
    if (t[e] != t[0]) {
      for (e = 1; e < k; e++) r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(epat, Q6_V_vsplat_R(e)), Q6_V_vsplat_R(t[e]), r);
      break;
    }
  return r;
}

/* One iteration's points -> 4 taps each: to[t][g] byte offsets into value, tw[t][g] sf weights. */
static void msda_taps_hvx(const msda_args_t* A, const msda_lanes_t* LN, int q, const int* vmap, float inv,
                          int32_t (*to)[MSDA_VPTS], float (*tw)[MSDA_VPTS]) {
  const int L = A->L, P = A->P, Q = A->Q, NO = A->NO, R = A->mode == MSDA_LOC ? 1 : A->R, k = LN->k;
  const long C4 = (long)A->M * A->D * 4;
  int32_t ax[MSDA_MAX_K], ay[MSDA_MAX_K], sx[MSDA_MAX_K], sy[MSDA_MAX_K], sc[MSDA_MAX_K];
  for (int e = 0; e < k; e++) { /* entry e = (o, level, ref entry) */
    const int o = e / (L * R), l = (e / R) % L, v = vmap[o];
    float fax, fay, fsx, fsy;
    msda_affine(A, v, q, l, e % R, &fax, &fay, &fsx, &fsy);
    ax[e] = msda_bits(fax); ay[e] = msda_bits(fay); sx[e] = msda_bits(fsx); sy[e] = msda_bits(fsy);
    sc[e] = msda_bits(!A->vis || A->vis[(long)v * Q + q] ? inv : 0.0f);
  }
  /* with shared offsets (NO = 1) this iteration is one map: its byte offset is uniform */
  const HVX_Vector vb = Q6_V_vsplat_R(NO > 1 ? 0 : (int32_t)((long long)vmap[0] * A->S * C4));
  const float* locq = A->loc + (long)q * LN->npv * 32 * 2;
  const float* awq = A->attw + (long)q * LN->npv * 32;
  const HVX_Vector zero = Q6_V_vzero(), one = msda_splatf(1.0f), m1 = msda_splatf(-1.0f);
  const HVX_Vector magic = msda_splatf(12582912.0f), magic_i = Q6_V_vsplat_R(0x4B400000);
  const HVX_Vector i_m1 = Q6_V_vsplat_R(-1), i_1 = Q6_V_vsplat_R(1);
  const int c4h = (int)((C4 & 0xffff) | (C4 << 16));
  (void)P; (void)NO;
  for (int j = 0; j < LN->npv; j++) {
    const HVX_Vector ep = LN->epat[j];
    const HVX_Vector AX = msda_lanevec(ax, k, ep), AY = msda_lanevec(ay, k, ep), SC = msda_lanevec(sc, k, ep);
    const HVX_Vector SX = A->mode == MSDA_REF_BOX ? msda_lanevec(sx, k, ep) : LN->sx[j];
    const HVX_Vector SY = A->mode == MSDA_REF_BOX ? msda_lanevec(sy, k, ep) : LN->sy[j];
    const HVX_Vector WM1 = LN->wm1[j], HM1 = LN->hm1[j], WF = LN->wf[j], HF = LN->hf[j], WI = LN->wi[j], ST = LN->st[j];
    const HVX_VectorPair xy = Q6_W_vdeal_VVR(((const HVX_Vector*)locq)[2 * j + 1], ((const HVX_Vector*)locq)[2 * j], -4);
    HVX_Vector x = msda_sf(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(xy), SX), AX));
    HVX_Vector y = msda_sf(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(xy), SY), AY));
    const HVX_VectorPred ok = Q6_Q_and_QQ(Q6_Q_and_QQ(Q6_Q_vcmp_gt_VsfVsf(x, m1), Q6_Q_vcmp_gt_VsfVsf(WF, x)),
                                          Q6_Q_and_QQ(Q6_Q_vcmp_gt_VsfVsf(y, m1), Q6_Q_vcmp_gt_VsfVsf(HF, y)));
    x = Q6_V_vmux_QVV(ok, x, zero);
    y = Q6_V_vmux_QVV(ok, y, zero);
    const HVX_Vector a = Q6_V_vmux_QVV(ok, msda_sf(Q6_Vqf32_vmpy_VsfVsf(((const HVX_Vector*)awq)[j], SC)), zero);
    /* nearest integer, then the floor: fx in [0, 1) */
    HVX_Vector x0 = Q6_Vw_vsub_VwVw(msda_sf(Q6_Vqf32_vadd_VsfVsf(x, magic)), magic_i);
    HVX_Vector y0 = Q6_Vw_vsub_VwVw(msda_sf(Q6_Vqf32_vadd_VsfVsf(y, magic)), magic_i);
    HVX_Vector fx = msda_sf(Q6_Vqf32_vsub_VsfVsf(x, msda_sf(Q6_Vqf32_vsub_VsfVsf(Q6_Vw_vadd_VwVw(x0, magic_i), magic))));
    HVX_Vector fy = msda_sf(Q6_Vqf32_vsub_VsfVsf(y, msda_sf(Q6_Vqf32_vsub_VsfVsf(Q6_Vw_vadd_VwVw(y0, magic_i), magic))));
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VsfVsf(zero, fx);
    x0 = Q6_V_vmux_QVV(neg, Q6_Vw_vadd_VwVw(x0, i_m1), x0);
    fx = Q6_V_vmux_QVV(neg, msda_sf(Q6_Vqf32_vadd_VsfVsf(fx, one)), fx);
    neg = Q6_Q_vcmp_gt_VsfVsf(zero, fy);
    y0 = Q6_V_vmux_QVV(neg, Q6_Vw_vadd_VwVw(y0, i_m1), y0);
    fy = Q6_V_vmux_QVV(neg, msda_sf(Q6_Vqf32_vadd_VsfVsf(fy, one)), fy);
    HVX_Vector wy1 = msda_sf(Q6_Vqf32_vmpy_VsfVsf(fy, a));
    HVX_Vector wy0 = msda_sf(Q6_Vqf32_vsub_VsfVsf(a, wy1));
    HVX_Vector wx1 = fx, wx0 = msda_sf(Q6_Vqf32_vsub_VsfVsf(one, fx));
    wy0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(y0, i_m1), wy0, zero); /* y0 >= 0 */
    wy1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(HM1, y0), wy1, zero);  /* y0 + 1 < H */
    wx0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(x0, i_m1), wx0, zero);
    wx1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(WM1, x0), wx1, zero);
    /* rows start + y * W + x: y * W per lane with a 16-bit multiply (every level H * W < 65536) */
    const HVX_Vector ya = Q6_Vh_vmpyi_VhVh(Q6_Vw_vmax_VwVw(y0, zero), WI);
    const HVX_Vector yb = Q6_Vh_vmpyi_VhVh(Q6_Vw_vmin_VwVw(Q6_Vw_vadd_VwVw(y0, i_1), HM1), WI);
    const HVX_Vector xa = Q6_Vw_vadd_VwVw(Q6_Vw_vmax_VwVw(x0, zero), ST);
    const HVX_Vector xb = Q6_Vw_vadd_VwVw(Q6_Vw_vmin_VwVw(Q6_Vw_vadd_VwVw(x0, i_1), WM1), ST);
    const HVX_Vector base = Q6_Vw_vadd_VwVw(vb, LN->base[j]);
    ((HVX_Vector*)to[0])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_Vw_vadd_VwVw(ya, xa), c4h), base);
    ((HVX_Vector*)to[1])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_Vw_vadd_VwVw(ya, xb), c4h), base);
    ((HVX_Vector*)to[2])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_Vw_vadd_VwVw(yb, xa), c4h), base);
    ((HVX_Vector*)to[3])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_Vw_vadd_VwVw(yb, xb), c4h), base);
    ((HVX_Vector*)tw[0])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy0, wx0));
    ((HVX_Vector*)tw[1])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy0, wx1));
    ((HVX_Vector*)tw[2])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy1, wx0));
    ((HVX_Vector*)tw[3])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy1, wx1));
  }
}

/* HVX body, queries [q0, q1). Work items = (query, value-map iteration): with shared offsets
 * (NO = 1) one iteration per visible map, else one covering every map. Software-pipelined by one
 * item: item i + 1's taps are built (HVX stores) before item i's multiply-accumulate, which
 * dcfetches them into L1 after its first head so the scalar tap reads don't miss. */
static void msda_run_hvx(const msda_args_t* A, const msda_lanes_t* LN, int q0, int q1) {
  const int NV = A->NV, Q = A->Q, NO = A->NO, M = A->M, per_head = NO * A->L * A->P, np = M * per_head;
  int32_t to[2][4][MSDA_VPTS] __attribute__((aligned(128)));
  float tw[2][4][MSDA_VPTS] __attribute__((aligned(128)));
  HVX_Vector part[MSDA_HVX_MAX_M];
  const char* vbase = (const char*)A->value;
  int cq = q0, ck = 0, cn = 0, cmaps[MSDA_MAX_NV];
  float cinv = 1.0f;
#define MSDA_ITEM_START()                                          \
  do {                                                             \
    cn = 0;                                                        \
    for (int v = 0; v < NV; v++)                                   \
      if (!A->vis || A->vis[(long)v * Q + cq]) cmaps[cn++] = v;    \
    cinv = 1.0f / (float)(cn > 1 ? cn : 1);                        \
    ck = 0;                                                        \
  } while (0)
#define MSDA_BUILD(b)                                              \
  ({                                                               \
    int vmap_[MSDA_MAX_NV], has_ = NO > 1 ? 1 : ck < cn;           \
    if (has_) {                                                    \
      if (NO > 1)                                                  \
        for (int o = 0; o < NO; o++) vmap_[o] = o;                 \
      else                                                         \
        vmap_[0] = cmaps[ck];                                      \
      msda_taps_hvx(A, LN, cq, vmap_, cinv, to[b], tw[b]);         \
    }                                                              \
    has_;                                                          \
  })
  if (q0 >= q1) return;
  MSDA_ITEM_START();
  int buf = 0, have = MSDA_BUILD(0), q = cq;
  for (int h = 0; h < M; h++) part[h] = Q6_V_vzero();
  for (;;) {
    const int iters = NO > 1 ? 1 : cn;
    int last = 0, nhave = 0;
    if (have && ck + 1 < iters) {
      ck++;
      nhave = MSDA_BUILD(buf ^ 1);
    } else {
      last = 1;
      if (cq + 1 < q1) {
        cq++;
        MSDA_ITEM_START();
        nhave = MSDA_BUILD(buf ^ 1);
      }
    }
    if (have) {
      for (int h = 0; h < M; h++) {
        if (h == 1 && nhave)
          for (int t = 0; t < 4; t++)
            for (int g = 0; g < np; g += 8) {
              __builtin_HEXAGON_Y2_dcfetch(&to[buf ^ 1][t][g]);
              __builtin_HEXAGON_Y2_dcfetch(&tw[buf ^ 1][t][g]);
            }
        HVX_Vector a0 = part[h], a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        const int32_t *o0 = to[buf][0], *o1 = to[buf][1], *o2 = to[buf][2], *o3 = to[buf][3];
        const float *w0 = tw[buf][0], *w1 = tw[buf][1], *w2 = tw[buf][2], *w3 = tw[buf][3];
        for (int g = h * per_head; g < (h + 1) * per_head; g++) {
          MSDA_TAP(a0, vbase + o0[g], w0[g]);
          MSDA_TAP(a1, vbase + o1[g], w1[g]);
          MSDA_TAP(a2, vbase + o2[g], w2[g]);
          MSDA_TAP(a3, vbase + o3[g], w3[g]);
        }
        part[h] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a0, a1), Q6_Vqf32_vadd_Vqf32Vqf32(a2, a3));
      }
    }
    if (last) {
      for (int h = 0; h < M; h++) {
        *(HVX_Vector*)(A->out + (long)q * M * 32 + h * 32) = msda_sf(part[h]);
        part[h] = Q6_V_vzero();
      }
      if (q + 1 >= q1) break;
      q++;
    }
    have = nhave;
    buf ^= 1;
  }
#undef MSDA_ITEM_START
#undef MSDA_BUILD
}

#endif

/* Queries [q0, q1) with the fastest body that handles the shape. The HVX body needs value, loc,
 * attw and out 128-byte aligned. */
static inline void msda_run(const msda_args_t* A, int q0, int q1) {
#ifdef MSDA_HVX
  if (msda_hvx_ok(A) && !(((uintptr_t)A->value | (uintptr_t)A->loc | (uintptr_t)A->attw | (uintptr_t)A->out) & 127)) {
    msda_lanes_t LN;
    msda_lanes_init(A, &LN);
    msda_run_hvx(A, &LN, q0, q1);
    return;
  }
#endif
  msda_run_scalar(A, q0, q1);
}

#endif
