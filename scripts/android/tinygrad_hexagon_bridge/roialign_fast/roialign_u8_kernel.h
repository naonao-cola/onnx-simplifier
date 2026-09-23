/* RoiAlign over the backbone's uint8 NHWC FPN maps, written straight into the merged, quantized head
 * input -- the whole "dq x4 -> RoiAlign x4 -> ScatterND merge -> QuantizeLinear" span of the e2e
 * pipeline (PR #1841) as one integer kernel.
 *
 * Semantics (ONNX RoiAlign opset < 16, mode=avg, output_half_pixel), per job j (one RoI):
 *   out[row_j] = QuantizeLinear(RoiAlign(DequantizeLinear(map[level_j]), roi_j), s_out, z_out)
 * with out laid out as NHWC rows (row, OH, OW, C) -- the layout box_head_1000_nhwc / mask_head_*_nhwc
 * consume. Sample positions and the bilinear weights are computed in fp32 with exactly ORT's
 * expressions (so y_low/x_low and the clamping match bit for bit), then each weight is quantized to
 * Q14 with the 4th absorbing the rounding so every valid sample's weights sum to exactly 1<<14.
 * Accumulation is exact int32 (u8 taps x Q14 weights); the input zero point is subtracted once per
 * bin; one fixed-point requant maps the bin to the output's uint8 scale. The only deviations from
 * the fp32 reference are the Q14 weights and the requant rounding: both far below one output LSB, so
 * results differ from QuantizeLinear(fp32 RoiAlign) only where the fp32 value sits within a hair of a
 * rounding boundary (measured by roialign_u8_host_check.c).
 *
 * Header-only. The multiply-accumulate has two bodies producing identical bytes: plain C vector
 * extensions (host, exact semantic check) and, when built for 128-byte HVX, u8 -> u16 zero-extend +
 * vmpyuh_acc (u16 x scalar u16 -> u32 accumulate, 3 ops per tap per 128 channels instead of ~12 for
 * generic 32-bit lanes). The HVX body is integer-only, so it runs under qemu 8.2 as well as on the
 * CDSP. */
#ifndef ROIALIGN_U8_KERNEL_H
#define ROIALIGN_U8_KERNEL_H

#include <stdint.h>

/* The sample positions must round exactly like ORT's (x86, unfused): no FMA contraction, which
 * hexagon-clang would otherwise do for y1 + ph * bin_h. */
#pragma STDC FP_CONTRACT OFF

typedef uint8_t ru8_u8x128 __attribute__((vector_size(128)));
typedef int32_t ru8_i32x128 __attribute__((vector_size(512)));

#ifndef RU8_WBITS
#define RU8_WBITS 14
#endif
#define RU8_ONE (1 << RU8_WBITS)
#define RU8_MAX_HALVES 2 /* C <= 256, C % 128 == 0 */
#ifndef RU8_PRESHIFT
#define RU8_PRESHIFT 7 /* acc >> PRESHIFT before the multiplier requant keeps the product in int32 */
#endif

typedef struct {
  const uint8_t* map; /* (H, W, C) uint8, 128-byte aligned rows (C % 128 == 0) */
  int H, W;
  float spatial_scale;
  int z_in; /* its DequantizeLinear zero point (scale folds into the requant below) */
  /* requant for this level: q_out = clamp(((acc_shifted * mult) + (1 << (shift-1))) >> shift + z_out) */
  int32_t mult, shift;
} ru8_level_t;

/* Fixed-point multiplier for level scale s_in -> output scale s_out (count = sr*sr samples/bin):
 *   value(out LSBs) = s_in * acc / (count * ONE * s_out) = (acc >> PRESHIFT) * mult / 2^shift */
static inline void ru8_requant_params(float s_in, float s_out, int count, int32_t* mult, int32_t* shift) {
  double M = (double)s_in / ((double)count * (double)RU8_ONE * (double)s_out) * (double)(1 << RU8_PRESHIFT);
  int sh = 0;
  /* largest mult with (max |acc| >> PRESHIFT) * mult < 2^31; max |acc| = 4 samples * 255 * ONE */
  const double lim = 2147483647.0 / (double)((4L * 255L * RU8_ONE >> RU8_PRESHIFT) + 1);
  while (M * 2.0 < lim && sh < 60) { M *= 2.0; sh++; }
  *mult = (int32_t)(M + 0.5);
  *shift = sh;
}

static inline float ru8_maxf(float a, float b) { return a > b ? a : b; }

#ifdef __hexagon__
static inline void ru8_l2fetch(const void* p, unsigned bytes) {
  unsigned long long ctl = ((unsigned long long)bytes << 32) | ((unsigned long long)bytes << 16) | 1ull;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}
#else
static inline void ru8_l2fetch(const void* p, unsigned bytes) { (void)p; (void)bytes; }
#endif

#define RU8_MAX_SR 4
#define RU8_MAX_AXIS 256 /* OH * sr and OW * sr */

/* ORT's pre_calc_for_bilinear_interpolate is separable: the y part of a sample depends only on
 * (ph, iy), the x part only on (pw, ix). Computing each axis once per RoI (OH*sr + OW*sr evaluations
 * instead of OH*OW*sr*sr) gives the identical y_low/x_low/ly/lx, and the weights are then formed as
 * hy*hx, hy*lx, ly*hx in fp32 exactly like ORT. */
typedef struct {
  int valid, lo, hi;
  float l, h;
} ru8_axis_t;

static inline void ru8_axis(float v, int n, ru8_axis_t* a) {
  if (v < -1.0f || v > (float)n) { a->valid = 0; return; }
  if (v <= 0.0f) v = 0.0f;
  int lo = (int)v, hi;
  if (lo >= n - 1) { hi = lo = n - 1; v = (float)lo; } else { hi = lo + 1; }
  a->valid = 1; a->lo = lo; a->hi = hi;
  a->l = v - (float)lo; a->h = 1.0f - a->l;
}

static inline void ru8_weights(const ru8_axis_t* y, const ru8_axis_t* x, int32_t w[4]) {
  const float f1 = y->h * x->h, f2 = y->h * x->l, f3 = y->l * x->h;
  w[0] = (int32_t)(f1 * (float)RU8_ONE + 0.5f);
  w[1] = (int32_t)(f2 * (float)RU8_ONE + 0.5f);
  w[2] = (int32_t)(f3 * (float)RU8_ONE + 0.5f);
  w[3] = RU8_ONE - w[0] - w[1] - w[2];
  if (w[3] < 0) { w[0] += w[3]; w[3] = 0; } /* only reachable through rounding; keeps the sum exact */
}

#if defined(__HVX__) && __HVX_LENGTH__ == 128
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define RU8_HVX 1
typedef int32_t ru8_i32x32 __attribute__((vector_size(128)));

/* acc_e/acc_o += w * (the even / odd bytes of the 128 channels at p), zero-extended to u16; vmpyuh
 * then splits each again into even/odd halfwords (lo/hi of the pair). All in registers. */
#define RU8_MAC(acc_e, acc_o, p, ww)                                              \
  do {                                                                            \
    const HVX_VectorPair v_ = Q6_Wuh_vzxt_Vub(*(const HVX_Vector*)(p));            \
    acc_e = Q6_Wuw_vmpyacc_WuwVuhRuh(acc_e, Q6_V_lo_W(v_), (ww));                  \
    acc_o = Q6_Wuw_vmpyacc_WuwVuhRuh(acc_o, Q6_V_hi_W(v_), (ww));                  \
  } while (0)

static inline ru8_i32x32 ru8_rq(ru8_i32x32 a, int32_t zsum, int32_t rnd_pre, int32_t mult, int32_t rnd, int32_t shift,
                                int32_t z_out) {
  a = (a - zsum + rnd_pre) >> RU8_PRESHIFT;
  a = ((a * mult + rnd) >> shift) + z_out;
  return __builtin_elementwise_min(__builtin_elementwise_max(a, (ru8_i32x32){0} + 0), (ru8_i32x32){0} + 255);
}

/* Requantize one 128-channel group and restore channel order: lo(e) = ch 4j, hi(e) = 4j+2,
 * lo(o) = 4j+1, hi(o) = 4j+3 (each value in the low byte of its word). */
static inline HVX_Vector ru8_rq_pack(HVX_VectorPair e, HVX_VectorPair o, int32_t zsum, int32_t rnd_pre, int32_t mult,
                                     int32_t rnd, int32_t shift, int32_t z_out) {
  const ru8_i32x32 q0 = ru8_rq((ru8_i32x32)Q6_V_lo_W(e), zsum, rnd_pre, mult, rnd, shift, z_out);
  const ru8_i32x32 q1 = ru8_rq((ru8_i32x32)Q6_V_hi_W(e), zsum, rnd_pre, mult, rnd, shift, z_out);
  const ru8_i32x32 q2 = ru8_rq((ru8_i32x32)Q6_V_lo_W(o), zsum, rnd_pre, mult, rnd, shift, z_out);
  const ru8_i32x32 q3 = ru8_rq((ru8_i32x32)Q6_V_hi_W(o), zsum, rnd_pre, mult, rnd, shift, z_out);
  const HVX_Vector ev = Q6_Vh_vshuffe_VhVh((HVX_Vector)q1, (HVX_Vector)q0); /* h[i] = ch 2i */
  const HVX_Vector od = Q6_Vh_vshuffe_VhVh((HVX_Vector)q3, (HVX_Vector)q2); /* h[i] = ch 2i+1 */
  return Q6_Vb_vshuffe_VbVb(od, ev);
}
#endif

/* One RoI into one output row. prefetch: before bin b, l2fetch the rows bin b+1 will
 * read -- per sample row one l2fetch over the x span of that bin's samples (y_low and y_high rows).
 * (A row-ahead variant -- a whole bin row of lead time -- measured no better on the phone.)
 * `halves` (C / 128) is a compile-time constant in every instantiation below, so the accumulators
 * and the tap loop stay in registers instead of being indexed through the stack. */
static inline __attribute__((always_inline)) void ru8_roi_h(const ru8_level_t* L, const int halves, const float* roi,
                                                            int OH, int OW, int sr, int z_out, int prefetch,
                                                            uint8_t* out_row) {
  const int C = 128 * halves;
  const float x1 = roi[0] * L->spatial_scale, y1 = roi[1] * L->spatial_scale;
  const float x2 = roi[2] * L->spatial_scale, y2 = roi[3] * L->spatial_scale;
  const float roi_w = ru8_maxf(x2 - x1, 1.0f), roi_h = ru8_maxf(y2 - y1, 1.0f);
  const float bin_w = roi_w / (float)OW, bin_h = roi_h / (float)OH;
  const uint8_t* map = L->map;
  const int H = L->H, W = L->W;
  ru8_axis_t ya[RU8_MAX_AXIS], xa[RU8_MAX_AXIS];
  for (int ph = 0; ph < OH; ph++)
    for (int iy = 0; iy < sr; iy++)
      ru8_axis(y1 + ph * bin_h + ((float)iy + 0.5f) * bin_h / (float)sr, H, &ya[ph * sr + iy]);
  for (int pw = 0; pw < OW; pw++)
    for (int ix = 0; ix < sr; ix++)
      ru8_axis(x1 + pw * bin_w + ((float)ix + 0.5f) * bin_w / (float)sr, W, &xa[pw * sr + ix]);
  const int32_t rnd_pre = 1 << (RU8_PRESHIFT - 1), rnd = (int32_t)1 << (L->shift - 1);
  const long rowC = (long)W * C;
  for (int ph = 0, b = 0; ph < OH; ph++) {
    for (int pw = 0; pw < OW; pw++, b++) {
      if (prefetch && b + 1 < OH * OW) {
        const int nph = (b + 1) / OW, npw = (b + 1) % OW;
        int xl = W, xh = -1;
        for (int ix = 0; ix < sr; ix++) {
          const ru8_axis_t* x = &xa[npw * sr + ix];
          if (x->valid) { if (x->lo < xl) xl = x->lo; if (x->hi > xh) xh = x->hi; }
        }
        if (xh >= xl)
          for (int iy = 0; iy < sr; iy++) {
            const ru8_axis_t* y = &ya[nph * sr + iy];
            if (!y->valid) continue;
            ru8_l2fetch(map + y->lo * rowC + (long)xl * C, (unsigned)(xh - xl + 1) * C);
            ru8_l2fetch(map + y->hi * rowC + (long)xl * C, (unsigned)(xh - xl + 1) * C);
          }
      }
      int nvalid = 0;
      uint8_t* o = out_row + (long)b * C;
#ifdef RU8_HVX
      const HVX_VectorPair zero = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
      HVX_VectorPair a0e = zero, a0o = zero, a1e = zero, a1o = zero;
#else
      ru8_i32x128 acc[RU8_MAX_HALVES];
      for (int h = 0; h < halves; h++) acc[h] = (ru8_i32x128){0};
#endif
      for (int iy = 0; iy < sr; iy++) {
        const ru8_axis_t* y = &ya[ph * sr + iy];
        if (!y->valid) continue;
        const uint8_t* rlo = map + y->lo * rowC;
        const uint8_t* rhi = map + y->hi * rowC;
        for (int ix = 0; ix < sr; ix++) {
          const ru8_axis_t* x = &xa[pw * sr + ix];
          if (!x->valid) continue;
          nvalid++;
          int32_t w[4];
          ru8_weights(y, x, w);
          const long xl = (long)x->lo * C, xh = (long)x->hi * C;
          const uint8_t *p1 = rlo + xl, *p2 = rlo + xh, *p3 = rhi + xl, *p4 = rhi + xh;
#ifdef RU8_HVX
          const int32_t w1 = w[0] | (w[0] << 16), w2 = w[1] | (w[1] << 16);
          const int32_t w3 = w[2] | (w[2] << 16), w4 = w[3] | (w[3] << 16);
          RU8_MAC(a0e, a0o, p1, w1); RU8_MAC(a0e, a0o, p2, w2);
          RU8_MAC(a0e, a0o, p3, w3); RU8_MAC(a0e, a0o, p4, w4);
          if (halves == 2) {
            RU8_MAC(a1e, a1o, p1 + 128, w1); RU8_MAC(a1e, a1o, p2 + 128, w2);
            RU8_MAC(a1e, a1o, p3 + 128, w3); RU8_MAC(a1e, a1o, p4 + 128, w4);
          }
#else
          const uint8_t* p[4] = {p1, p2, p3, p4};
          for (int h = 0; h < halves; h++) {
            ru8_i32x128 a = acc[h];
            for (int t = 0; t < 4; t++)
              a += __builtin_convertvector(*(const ru8_u8x128*)(p[t] + 128 * h), ru8_i32x128) * w[t];
            acc[h] = a;
          }
#endif
        }
      }
      const int32_t zsum = L->z_in * nvalid * RU8_ONE;
#ifdef RU8_HVX
      *(HVX_Vector*)o = ru8_rq_pack(a0e, a0o, zsum, rnd_pre, L->mult, rnd, L->shift, z_out);
      if (halves == 2) *(HVX_Vector*)(o + 128) = ru8_rq_pack(a1e, a1o, zsum, rnd_pre, L->mult, rnd, L->shift, z_out);
#else
      for (int h = 0; h < halves; h++) {
        ru8_i32x128 a = (acc[h] - zsum + rnd_pre) >> RU8_PRESHIFT;
        a = ((a * L->mult + rnd) >> L->shift) + z_out;
        a = __builtin_elementwise_min(__builtin_elementwise_max(a, (ru8_i32x128){0} + 0), (ru8_i32x128){0} + 255);
        *(ru8_u8x128*)(o + 128 * h) = __builtin_convertvector(a, ru8_u8x128);
      }
#endif
    }
  }
}

static void ru8_roi_c128(const ru8_level_t* L, const float* roi, int OH, int OW, int sr, int z_out, int prefetch,
                         uint8_t* out_row) {
  ru8_roi_h(L, 1, roi, OH, OW, sr, z_out, prefetch, out_row);
}
static void ru8_roi_c256(const ru8_level_t* L, const float* roi, int OH, int OW, int sr, int z_out, int prefetch,
                         uint8_t* out_row) {
  ru8_roi_h(L, 2, roi, OH, OW, sr, z_out, prefetch, out_row);
}
static void ru8_roi(const ru8_level_t* L, int C, const float* roi, int OH, int OW, int sr, int z_out, int prefetch,
                    uint8_t* out_row) {
  (C == 256 ? ru8_roi_c256 : ru8_roi_c128)(L, roi, OH, OW, sr, z_out, prefetch, out_row);
}

/* A job = one RoI: its level, its box (4 floats, image coords), its destination row. */
typedef struct {
  int level, row;
  float box[4];
} ru8_job_t;

static void ru8_run_jobs(const ru8_level_t* lv, int C, const ru8_job_t* jobs, int njobs, int OH, int OW,
                         int sr, int z_out, int prefetch, uint8_t* out) {
  const long row_bytes = (long)OH * OW * C;
  for (int j = 0; j < njobs; j++)
    ru8_roi(&lv[jobs[j].level], C, jobs[j].box, OH, OW, sr, z_out, prefetch, out + jobs[j].row * row_bytes);
}

/* Optional locality order: within each level, sort jobs by (y1, x1) of the box (insertion sort on a
 * key array is fine for n <= a few thousand; stable so equal boxes keep their order). */
static void ru8_sort_jobs(ru8_job_t* jobs, int n) {
  for (int i = 1; i < n; i++) {
    ru8_job_t t = jobs[i];
    int j = i - 1;
    while (j >= 0 && (jobs[j].level > t.level ||
                      (jobs[j].level == t.level && (jobs[j].box[1] > t.box[1] ||
                                                    (jobs[j].box[1] == t.box[1] && jobs[j].box[0] > t.box[0]))))) {
      jobs[j + 1] = jobs[j];
      j--;
    }
    jobs[j + 1] = t;
  }
}

#endif
