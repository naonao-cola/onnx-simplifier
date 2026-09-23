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
 * Header-only, plain C + clang vector extensions (no HVX intrinsics): the same code builds for the
 * host (exact semantic check), qemu (Hexagon lowering check, integer HVX only -- no qfloat) and the
 * CDSP skel. */
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

typedef struct {
  int valid;
  long p1, p2, p3, p4;     /* byte offsets of the 4 taps (pixel * C) */
  int32_t w1, w2, w3, w4;  /* Q14, summing to RU8_ONE when valid */
} ru8_sample_t;

/* ORT's pre_calc_for_bilinear_interpolate for one sample, fp32, same expression order. */
static inline void ru8_sample(float y, float x, int H, int W, int C, ru8_sample_t* s) {
  if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) { s->valid = 0; return; }
  if (y <= 0.0f) y = 0.0f;
  if (x <= 0.0f) x = 0.0f;
  int y_low = (int)y, x_low = (int)x, y_high, x_high;
  if (y_low >= H - 1) { y_high = y_low = H - 1; y = (float)y_low; } else { y_high = y_low + 1; }
  if (x_low >= W - 1) { x_high = x_low = W - 1; x = (float)x_low; } else { x_high = x_low + 1; }
  const float ly = y - (float)y_low, lx = x - (float)x_low, hy = 1.0f - ly, hx = 1.0f - lx;
  const float f1 = hy * hx, f2 = hy * lx, f3 = ly * hx;
  int32_t w1 = (int32_t)(f1 * (float)RU8_ONE + 0.5f), w2 = (int32_t)(f2 * (float)RU8_ONE + 0.5f);
  int32_t w3 = (int32_t)(f3 * (float)RU8_ONE + 0.5f);
  int32_t w4 = RU8_ONE - w1 - w2 - w3;
  if (w4 < 0) { w1 += w4; w4 = 0; }  /* can't happen for f4 >= 0 beyond rounding; keep the sum exact */
  s->valid = 1;
  s->w1 = w1; s->w2 = w2; s->w3 = w3; s->w4 = w4;
  s->p1 = ((long)y_low * W + x_low) * C;
  s->p2 = ((long)y_low * W + x_high) * C;
  s->p3 = ((long)y_high * W + x_low) * C;
  s->p4 = ((long)y_high * W + x_high) * C;
}

#ifdef __hexagon__
static inline void ru8_l2fetch(const void* p, unsigned bytes) {
  unsigned long long ctl = ((unsigned long long)bytes << 32) | ((unsigned long long)bytes << 16) | 1ull;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}
#else
static inline void ru8_l2fetch(const void* p, unsigned bytes) { (void)p; (void)bytes; }
#endif

#define RU8_MAX_SR 4

/* One RoI into one output row. prefetch: before bin b, l2fetch the 2-pixel row segments bin b+1
 * will read (same scheme as roialign_kernel.h's roialign_hwc_pf, 2*C bytes per segment). */
static void ru8_roi(const ru8_level_t* L, int C, const float* roi, int OH, int OW, int sr,
                    int z_out, int prefetch, uint8_t* out_row) {
  const int halves = C / 128;
  const float x1 = roi[0] * L->spatial_scale, y1 = roi[1] * L->spatial_scale;
  const float x2 = roi[2] * L->spatial_scale, y2 = roi[3] * L->spatial_scale;
  const float roi_w = ru8_maxf(x2 - x1, 1.0f), roi_h = ru8_maxf(y2 - y1, 1.0f);
  const float bin_w = roi_w / (float)OW, bin_h = roi_h / (float)OH;
  const uint8_t* map = L->map;
  const int H = L->H, W = L->W;
  for (int b = 0; b < OH * OW; b++) {
    const int ph = b / OW, pw = b % OW;
    if (prefetch && b + 1 < OH * OW) {
      const int nph = (b + 1) / OW, npw = (b + 1) % OW;
      for (int iy = 0; iy < sr; iy++)
        for (int ix = 0; ix < sr; ix++) {
          ru8_sample_t s;
          ru8_sample(y1 + nph * bin_h + ((float)iy + 0.5f) * bin_h / (float)sr,
                     x1 + npw * bin_w + ((float)ix + 0.5f) * bin_w / (float)sr, H, W, C, &s);
          if (s.valid) { ru8_l2fetch(map + s.p1, 2u * C); ru8_l2fetch(map + s.p3, 2u * C); }
        }
    }
    ru8_i32x128 acc[RU8_MAX_HALVES];
    for (int h = 0; h < halves; h++) acc[h] = (ru8_i32x128){0};
    int32_t wsum = 0;
    for (int iy = 0; iy < sr; iy++) {
      const float y = y1 + ph * bin_h + ((float)iy + 0.5f) * bin_h / (float)sr;
      for (int ix = 0; ix < sr; ix++) {
        const float x = x1 + pw * bin_w + ((float)ix + 0.5f) * bin_w / (float)sr;
        ru8_sample_t s;
        ru8_sample(y, x, H, W, C, &s);
        if (!s.valid) continue;
        wsum += RU8_ONE;
        for (int h = 0; h < halves; h++) {
          const ru8_u8x128 v1 = *(const ru8_u8x128*)(map + s.p1 + 128 * h);
          const ru8_u8x128 v2 = *(const ru8_u8x128*)(map + s.p2 + 128 * h);
          const ru8_u8x128 v3 = *(const ru8_u8x128*)(map + s.p3 + 128 * h);
          const ru8_u8x128 v4 = *(const ru8_u8x128*)(map + s.p4 + 128 * h);
          acc[h] += __builtin_convertvector(v1, ru8_i32x128) * s.w1 + __builtin_convertvector(v2, ru8_i32x128) * s.w2 +
                    __builtin_convertvector(v3, ru8_i32x128) * s.w3 + __builtin_convertvector(v4, ru8_i32x128) * s.w4;
        }
      }
    }
    const int32_t zsum = L->z_in * wsum;
    const int32_t rnd_pre = 1 << (RU8_PRESHIFT - 1), rnd = (int32_t)1 << (L->shift - 1);
    uint8_t* o = out_row + (long)b * C;
    for (int h = 0; h < halves; h++) {
      ru8_i32x128 a = (acc[h] - zsum + rnd_pre) >> RU8_PRESHIFT;
      a = ((a * L->mult + rnd) >> L->shift) + z_out;
      a = __builtin_elementwise_min(__builtin_elementwise_max(a, (ru8_i32x128){0} + 0), (ru8_i32x128){0} + 255);
      *(ru8_u8x128*)(o + 128 * h) = __builtin_convertvector(a, ru8_u8x128);
    }
  }
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
