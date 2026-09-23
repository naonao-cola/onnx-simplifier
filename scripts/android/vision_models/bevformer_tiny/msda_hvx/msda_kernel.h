/* Fused deformable sampling for BEVFormer's TSA and SCA (split.py's msda_fused): per query and
 * head, bilinear-sample the head's 32 channels at every point, weight by the attention weight,
 * sum over points and value maps (cameras / queue frames), divide by the number of visible maps.
 * Replaces the HTP's "grid build -> GridSample -> Mul -> ReduceSum (-> camera average)" span.
 *
 *   value (NV, H*W, 256) f32 channels-last, 128-byte aligned (a head's 32 channels = one HVX vector)
 *   ref   (NV, Q, R, 2)  f32 in [0, 1]; point p uses ref[nv, q, p % R]
 *   off   (Q, 8, NO, P, 2) f32 pixel offsets (NO = 1: shared by all maps, NO = NV: per map)
 *   attw  (Q, 8, NO, P)  f32 softmaxed weights
 *   vis   (NV, Q) u8     0 = skip this (map, query) pair entirely
 *   out   (Q, 256) f32
 * Sample position x = ref_x * W + off_x - 0.5 (grid_sample, align_corners=False); zero padding per
 * tap, so a point contributes only if -1 < x < W and -1 < y < H.
 *
 * The coordinates and the four tap weights (x attention weight / #visible) are scalar fp32 (the
 * scalar core is IEEE); the 32-channel multiply-accumulate is one HVX qf32 vmpy + vadd per tap on
 * 128-byte HVX (V68+), or plain C on the host (the exact-semantics reference msda_host_check.c
 * compares against torch). qf32 differs from IEEE fp32 only by rounding (checked against the same
 * torch reference on the phone and in hexagon-sim). Header-only. */
#ifndef MSDA_KERNEL_H
#define MSDA_KERNEL_H

#include <stdint.h>
#include <string.h>

#define MSDA_C 256
#define MSDA_M 8
#define MSDA_D 32
#define MSDA_MAX_NV 8
#define MSDA_MAX_P 16

typedef struct {
  const float* value;
  const float* ref;
  const float* off;
  const float* attw;
  const uint8_t* vis;
  float* out;
  int NV, H, W, Q, R, NO, P;
} msda_args_t;

#if defined(__HVX__) && __HVX_LENGTH__ == 128 && defined(__HVX_ARCH__) && __HVX_ARCH__ >= 68
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define MSDA_HVX 1
static inline int32_t msda_bits(float f) { int32_t i; memcpy(&i, &f, 4); return i; }
typedef HVX_Vector msda_acc_t;
#define MSDA_ACC_ZERO(a) (a) = Q6_V_vzero()
#define MSDA_TAP(a, p, w) \
  (a) = Q6_Vqf32_vadd_Vqf32Vqf32((a), Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector*)(p), Q6_V_vsplat_R(msda_bits(w))))
#define MSDA_STORE(dst, a) *(HVX_Vector*)(dst) = Q6_Vsf_equals_Vqf32(a)
#define MSDA_SUM4(a, b, c, d) Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b), Q6_Vqf32_vadd_Vqf32Vqf32(c, d))
#else
typedef float msda_f32x32 __attribute__((vector_size(128)));
typedef msda_f32x32 msda_acc_t;
#define MSDA_ACC_ZERO(a) (a) = (msda_f32x32){0}
#define MSDA_TAP(a, p, w) (a) += *(const msda_f32x32*)(p) * (w)
#define MSDA_STORE(dst, a) *(msda_f32x32*)(dst) = (a)
#define MSDA_SUM4(a, b, c, d) (((a) + (b)) + ((c) + (d)))
#endif

#define MSDA_MAX_TAPS (MSDA_MAX_NV * 16 * 4) /* maps x points x 4 */

/* Queries [q0, q1) of heads [h0, h1), head-major (one head's 128-byte slice of the value pixels at a
 * time). Per (query, head) two phases: the scalar core turns every (map, point) into 4 taps (byte
 * offset, weight = bilinear x attention / #visible; out-of-range taps get weight 0 at a clamped
 * address, so there are no branches), then HVX streams the tap list into 4 independent qf32
 * accumulators, so the vector loads are not each waited on before the next is issued. */
static void msda_run_heads(const msda_args_t* A, int q0, int q1, int h0, int h1) {
  const int NV = A->NV, H = A->H, W = A->W, Q = A->Q, R = A->R, NO = A->NO, P = A->P;
  const long HWC = (long)H * W * MSDA_C, rowC = (long)W * MSDA_C;
  const float fW = (float)W, fH = (float)H;
  int32_t toff[MSDA_MAX_TAPS + 4];
  float tw[MSDA_MAX_TAPS + 4];
  for (int h = h0; h < h1; h++) {
    const char* base = (const char*)(A->value + h * MSDA_D);
    for (int q = q0; q < q1; q++) {
      int maps[MSDA_MAX_NV], n = 0, nt = 0;
      for (int v = 0; v < NV; v++)
        if (A->vis[(long)v * Q + q]) maps[n++] = v;
      const float inv = 1.0f / (float)(n > 1 ? n : 1);
      for (int k = 0; k < n; k++) {
        const int v = maps[k], vo = NO > 1 ? v : 0;
        const float* rf = A->ref + ((long)v * Q + q) * R * 2;
        const float* of = A->off + (((long)q * MSDA_M + h) * NO + vo) * P * 2;
        const float* aw = A->attw + (((long)q * MSDA_M + h) * NO + vo) * P;
        const long vbase = v * HWC;
        /* point p uses anchor p % R, stepped instead of computed: `%` is a libcall on Hexagon */
        for (int p = 0, ri = 0; p < P; p++, ri = ri + 1 == R ? 0 : ri + 1) {
          const float* r = rf + ri * 2;
          float x = r[0] * fW + of[2 * p] - 0.5f;
          float y = r[1] * fH + of[2 * p + 1] - 0.5f;
          const int ok = (x > -1.0f) & (x < fW) & (y > -1.0f) & (y < fH); /* also drops NaN */
          x = ok ? x : 0.0f;
          y = ok ? y : 0.0f;
          const int x0 = (int)(x + 1.0f) - 1, y0 = (int)(y + 1.0f) - 1; /* floor: x + 1 > 0 */
          const float fx = x - (float)x0, fy = y - (float)y0, a = ok ? aw[p] * inv : 0.0f;
          const float wy0 = y0 >= 0 ? (1.0f - fy) * a : 0.0f, wy1 = y0 + 1 < H ? fy * a : 0.0f;
          const float wx0 = x0 >= 0 ? 1.0f - fx : 0.0f, wx1 = x0 + 1 < W ? fx : 0.0f;
          const long xa = x0 >= 0 ? x0 : 0, xb = x0 + 1 < W ? x0 + 1 : W - 1;
          const long ya = vbase + (y0 >= 0 ? y0 : 0) * rowC, yb = vbase + (y0 + 1 < H ? y0 + 1 : H - 1) * rowC;
          toff[nt] = (int32_t)((ya + xa * MSDA_C) * 4); tw[nt++] = wy0 * wx0;
          toff[nt] = (int32_t)((ya + xb * MSDA_C) * 4); tw[nt++] = wy0 * wx1;
          toff[nt] = (int32_t)((yb + xa * MSDA_C) * 4); tw[nt++] = wy1 * wx0;
          toff[nt] = (int32_t)((yb + xb * MSDA_C) * 4); tw[nt++] = wy1 * wx1;
        }
      }
      while (nt & 3) { toff[nt] = 0; tw[nt++] = 0.0f; } /* pad to a multiple of 4 with zero taps */
      msda_acc_t a0, a1, a2, a3;
      MSDA_ACC_ZERO(a0); MSDA_ACC_ZERO(a1); MSDA_ACC_ZERO(a2); MSDA_ACC_ZERO(a3);
      for (int t = 0; t < nt; t += 4) {
        MSDA_TAP(a0, base + toff[t], tw[t]);
        MSDA_TAP(a1, base + toff[t + 1], tw[t + 1]);
        MSDA_TAP(a2, base + toff[t + 2], tw[t + 2]);
        MSDA_TAP(a3, base + toff[t + 3], tw[t + 3]);
      }
      MSDA_STORE(A->out + (long)q * MSDA_C + h * MSDA_D, MSDA_SUM4(a0, a1, a2, a3));
    }
  }
}

#ifdef MSDA_HVX
/* HVX path. Per query, the scalar version's per-point work (117 of its 168 cycles per point in
 * hexagon-sim: a serial chain of dependent float ops) runs 32 points per vector instead:
 * the M * NO * P points of one query and value-map iteration (TSA: both queue frames at once,
 * NO = 2; SCA: one visible camera per iteration, NO = 1) are laid out (head, o, point) exactly like
 * the offsets / weights in memory, so they load as whole vectors (x/y de-interleaved with vdeal).
 * floor() without a vector float->int convert (V73+): x + 1.5 * 2^23 in qf32 -> sf puts the nearest
 * integer (whatever the conversion's rounding) in the mantissa, and one +-1 fix-up on the fraction
 * makes it the floor. The tap list (byte offset, weight) then feeds the same HVX multiply-accumulate. */
#define MSDA_VPTS 128 /* points per iteration: M * NO * P <= 128 (4 vectors) */
typedef struct {
  int npv, k;                        /* vectors per iteration; distinct ref entries K = NO * R */
  HVX_Vector apat[MSDA_VPTS / 32];   /* per lane: ref entry o * R + p % R */
  HVX_Vector hbyte[MSDA_VPTS / 32];  /* per lane: head * 128 (byte offset of its channels) */
} msda_lanes_t;

static void msda_lanes_init(const msda_args_t* A, msda_lanes_t* L) {
  int32_t ap[MSDA_VPTS] __attribute__((aligned(128))), hb[MSDA_VPTS] __attribute__((aligned(128)));
  const int np = MSDA_M * A->NO * A->P;
  L->npv = np / 32;
  L->k = A->NO * A->R;
  for (int g = 0; g < np; g++) {
    ap[g] = ((g / A->P) % A->NO) * A->R + (g % A->P) % A->R;
    hb[g] = (g / (A->NO * A->P)) * MSDA_D * 4;
  }
  for (int j = 0; j < L->npv; j++) {
    L->apat[j] = ((HVX_Vector*)ap)[j];
    L->hbyte[j] = ((HVX_Vector*)hb)[j];
  }
}

static inline HVX_Vector msda_sf(HVX_Vector qf) { return Q6_Vsf_equals_Vqf32(qf); }
static inline HVX_Vector msda_splatf(float f) { return Q6_V_vsplat_R(msda_bits(f)); }

/* One iteration's points -> 4 taps each into to[t][g] (byte offsets) / tw[t][g] (sf weights). */
static inline void msda_taps_hvx(const msda_args_t* A, const msda_lanes_t* L, int q, const int* vmap, float scale,
                                 int32_t (*to)[MSDA_VPTS], float (*tw)[MSDA_VPTS]) {
  const int H = A->H, W = A->W, Q = A->Q, R = A->R, NO = A->NO, P = A->P;
  const long HW = (long)H * W;
  float bx[MSDA_MAX_NV * 16], by[MSDA_MAX_NV * 16], sc[MSDA_MAX_NV * 16];
  int32_t vb[MSDA_MAX_NV * 16];
  for (int e = 0; e < L->k; e++) { /* entry e = (o, anchor) */
    const int o = e / R, v = vmap[o];
    const float* r = A->ref + (((long)v * Q + q) * R + e % R) * 2;
    bx[e] = r[0] * (float)W - 0.5f;
    by[e] = r[1] * (float)H - 0.5f;
    sc[e] = A->vis[(long)v * Q + q] ? scale : 0.0f;
    vb[e] = (int32_t)(v * HW * MSDA_C * 4);
  }
  const float* offq = A->off + (long)q * MSDA_M * NO * P * 2;
  const float* awq = A->attw + (long)q * MSDA_M * NO * P;
  const HVX_Vector zero = Q6_V_vzero(), one = msda_splatf(1.0f), m1 = msda_splatf(-1.0f);
  const HVX_Vector fW = msda_splatf((float)W), fH = msda_splatf((float)H);
  const HVX_Vector magic = msda_splatf(12582912.0f), magic_i = Q6_V_vsplat_R(0x4B400000);
  const HVX_Vector i_m1 = Q6_V_vsplat_R(-1), i_1 = Q6_V_vsplat_R(1);
  const HVX_Vector i_wm1 = Q6_V_vsplat_R(W - 1), i_hm1 = Q6_V_vsplat_R(H - 1);
  const int wh = (W & 0xffff) | (W << 16);
  for (int j = 0; j < L->npv; j++) {
    HVX_Vector BX = msda_splatf(bx[0]), BY = msda_splatf(by[0]), SC = msda_splatf(sc[0]), VB = Q6_V_vsplat_R(vb[0]);
    for (int e = 1; e < L->k; e++) {
      const HVX_VectorPred qe = Q6_Q_vcmp_eq_VwVw(L->apat[j], Q6_V_vsplat_R(e));
      BX = Q6_V_vmux_QVV(qe, msda_splatf(bx[e]), BX);
      BY = Q6_V_vmux_QVV(qe, msda_splatf(by[e]), BY);
      SC = Q6_V_vmux_QVV(qe, msda_splatf(sc[e]), SC);
      VB = Q6_V_vmux_QVV(qe, Q6_V_vsplat_R(vb[e]), VB);
    }
    const HVX_VectorPair xy = Q6_W_vdeal_VVR(((const HVX_Vector*)offq)[2 * j + 1], ((const HVX_Vector*)offq)[2 * j], -4);
    HVX_Vector x = msda_sf(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(xy), BX));
    HVX_Vector y = msda_sf(Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(xy), BY));
    const HVX_VectorPred ok = Q6_Q_and_QQ(Q6_Q_and_QQ(Q6_Q_vcmp_gt_VsfVsf(x, m1), Q6_Q_vcmp_gt_VsfVsf(fW, x)),
                                          Q6_Q_and_QQ(Q6_Q_vcmp_gt_VsfVsf(y, m1), Q6_Q_vcmp_gt_VsfVsf(fH, y)));
    x = Q6_V_vmux_QVV(ok, x, zero);
    y = Q6_V_vmux_QVV(ok, y, zero);
    const HVX_Vector a = Q6_V_vmux_QVV(ok, msda_sf(Q6_Vqf32_vmpy_VsfVsf(((const HVX_Vector*)awq)[j], SC)), zero);
    /* nearest integer, then fix up to the floor: fx in [0, 1) */
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
    wy0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(y0, i_m1), wy0, zero);  /* y0 >= 0 */
    wy1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(i_hm1, y0), wy1, zero); /* y0 + 1 < H */
    wx0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(x0, i_m1), wx0, zero);
    wx1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(i_wm1, x0), wx1, zero);
    const HVX_Vector ya = Q6_Vw_vmpyi_VwRh(Q6_Vw_vmax_VwVw(y0, zero), wh);
    const HVX_Vector yb = Q6_Vw_vmpyi_VwRh(Q6_Vw_vmin_VwVw(Q6_Vw_vadd_VwVw(y0, i_1), i_hm1), wh);
    const HVX_Vector xa = Q6_Vw_vmax_VwVw(x0, zero), xb = Q6_Vw_vmin_VwVw(Q6_Vw_vadd_VwVw(x0, i_1), i_wm1);
    const HVX_Vector base = Q6_Vw_vadd_VwVw(VB, L->hbyte[j]);
    ((HVX_Vector*)to[0])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(ya, xa), 10), base);
    ((HVX_Vector*)to[1])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(ya, xb), 10), base);
    ((HVX_Vector*)to[2])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(yb, xa), 10), base);
    ((HVX_Vector*)to[3])[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(yb, xb), 10), base);
    ((HVX_Vector*)tw[0])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy0, wx0));
    ((HVX_Vector*)tw[1])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy0, wx1));
    ((HVX_Vector*)tw[2])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy1, wx0));
    ((HVX_Vector*)tw[3])[j] = msda_sf(Q6_Vqf32_vmpy_VsfVsf(wy1, wx1));
  }
}

/* Work items = (query, value-map iteration). Software-pipelined by one item: the taps of item i + 1
 * are built (vector stores) before item i's multiply-accumulate, which dcfetches them into L1 after
 * its first head, so reading them back with scalar loads later doesn't miss L1 (HVX stores go to L2
 * and invalidate L1; a dcfetch right behind the stores is too early). */
static void msda_run_hvx(const msda_args_t* A, const msda_lanes_t* L, int q0, int q1) {
  const int NV = A->NV, Q = A->Q, NO = A->NO, per_head = NO * A->P, np = MSDA_M * NO * A->P;
  int32_t to[2][4][MSDA_VPTS] __attribute__((aligned(128)));
  float tw[2][4][MSDA_VPTS] __attribute__((aligned(128)));
  HVX_Vector part[MSDA_M];
  const char* vbase = (const char*)A->value;
  /* item cursor */
  int cq = q0, ck = 0, cn = 0, cmaps[MSDA_MAX_NV];
  float cinv = 1.0f;
#define MSDA_ITEM_START()                                                \
  do {                                                                   \
    cn = 0;                                                              \
    for (int v = 0; v < NV; v++)                                         \
      if (A->vis[(long)v * Q + cq]) cmaps[cn++] = v;                     \
    cinv = 1.0f / (float)(cn > 1 ? cn : 1);                              \
    ck = 0;                                                              \
  } while (0)
  /* builds the taps of the item at the cursor into buffer `b`; returns 0 if the query has none */
#define MSDA_BUILD(b)                                                    \
  ({                                                                     \
    int vmap_[MSDA_MAX_NV], has_ = NO > 1 ? 1 : ck < cn;                 \
    if (has_) {                                                          \
      if (NO > 1) for (int o = 0; o < NO; o++) vmap_[o] = o;             \
      else vmap_[0] = cmaps[ck];                                         \
      msda_taps_hvx(A, L, cq, vmap_, cinv, to[b], tw[b]);                \
    }                                                                    \
    has_;                                                                \
  })
  if (q0 >= q1) return;
  MSDA_ITEM_START();
  int buf = 0, have = MSDA_BUILD(0), q = cq, last;
  for (int h = 0; h < MSDA_M; h++) part[h] = Q6_V_vzero();
  for (;;) {
    /* advance the cursor to the next item and build it into the other buffer */
    const int iters = NO > 1 ? 1 : cn;
    last = 0;
    int nhave = 0;
    if (have && ck + 1 < iters) {
      ck++;
      nhave = MSDA_BUILD(buf ^ 1);
    } else {
      last = 1; /* this item is the query's last */
      if (cq + 1 < q1) {
        cq++;
        MSDA_ITEM_START();
        nhave = MSDA_BUILD(buf ^ 1);
      }
    }
    if (have) {
      for (int h = 0; h < MSDA_M; h++) {
        if (h == 1 && nhave) /* the next item's taps, into L1 now that its HVX stores have landed */
          for (int t = 0; t < 4; t++)
            for (int g = 0; g < np; g += 8) {
              __builtin_HEXAGON_Y2_dcfetch(&to[buf ^ 1][t][g]);
              __builtin_HEXAGON_Y2_dcfetch(&tw[buf ^ 1][t][g]);
            }
        HVX_Vector a0 = part[h], a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        const int32_t* o0 = to[buf][0];
        const int32_t* o1 = to[buf][1];
        const int32_t* o2 = to[buf][2];
        const int32_t* o3 = to[buf][3];
        const float* w0 = tw[buf][0];
        const float* w1 = tw[buf][1];
        const float* w2 = tw[buf][2];
        const float* w3 = tw[buf][3];
        for (int g = h * per_head; g < (h + 1) * per_head; g++) {
          MSDA_TAP(a0, vbase + o0[g], w0[g]);
          MSDA_TAP(a1, vbase + o1[g], w1[g]);
          MSDA_TAP(a2, vbase + o2[g], w2[g]);
          MSDA_TAP(a3, vbase + o3[g], w3[g]);
        }
        part[h] = MSDA_SUM4(a0, a1, a2, a3);
      }
    }
    if (last) {
      for (int h = 0; h < MSDA_M; h++) {
        MSDA_STORE(A->out + (long)q * MSDA_C + h * MSDA_D, part[h]);
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

/* Shapes the HVX path handles; others use the scalar path. */
static int msda_hvx_ok(const msda_args_t* A) {
  const int np = MSDA_M * A->NO * A->P;
  return np % 32 == 0 && np <= MSDA_VPTS && 32 % (A->NO * A->P) == 0 && A->NO * A->R <= MSDA_MAX_NV * 16;
}
#endif

/* Queries [q0, q1): the vectorized HVX path where it applies, else the scalar one. */
static void msda_run(const msda_args_t* A, int q0, int q1) {
#ifdef MSDA_HVX
  if (msda_hvx_ok(A)) {
    msda_lanes_t L;
    msda_lanes_init(A, &L);
    msda_run_hvx(A, &L, q0, q1);
    return;
  }
#endif
  msda_run_heads(A, q0, q1, 0, MSDA_M);
}

#endif
