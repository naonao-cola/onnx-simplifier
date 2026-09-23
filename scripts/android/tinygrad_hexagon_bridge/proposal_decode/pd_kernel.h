/* RPN proposal decode for one FPN level of the real Mask R-CNN graph, header-only so the host
 * check, qemu and the DSP skel all compile the exact same code.
 *
 * Replaces, per level, rest.onnx's post-TopK chain: Gather anchors/deltas by the top-k indices ->
 * Q/DQ the deltas twice -> Detectron decode (+1 width/height, dw/dh clipped at log(1000/16), -1 on
 * x2/y2) -> clip to the image -> Q/DQ onto the uint8 box grid. The two Q/DQs on the deltas are a
 * genuine requantization for P2-P4 (the backbone's output grid differs from rest's), so they are
 * reproduced, not skipped. fp32 op order follows the graph node by node (build with
 * -ffp-contract=off: no FMA) so the result is bit-exact with ONNX Runtime.
 *
 * Two delta sources:
 *   deltas != NULL : per-anchor fp32 [A,4], exactly rest.onnx's input today.
 *   nchw   != NULL : the backbone's pre-transpose uint8 conv output [1, 3*4, H, W]; the kernel
 *                    gathers the k rows it needs straight from it, so the backbone's full-map
 *                    Reshape/Transpose/Q/DQ over all A anchors is no longer needed.
 *
 * Speed without giving up exactness (see pd_prepare / pd_prepare_anchors):
 *   - the delta requant chain is a function of the backbone's uint8 value alone, so it's a
 *     256-entry per-level LUT (fp32 deltas map back to their uint8 index with an exact on-grid
 *     check, falling back to the full chain if a value is ever off-grid);
 *   - the box-grid quantize multiplies by 1/scale and only does the real division when the
 *     quotient is within 1e-3 of a .5 tie, the only case where the two can round differently.
 *   - anchors are computed from 3 base anchors + stride when the real table is verified to be
 *     exactly that grid (pd_prepare_anchors), instead of a gather from a multi-MB table;
 *   - boxes are processed in blocks with branch-free math, and the next block's gather addresses
 *     are prefetched (dcfetch) -- on the DSP the random gathers, not the fp32 math, dominate.
 * Passing Q = NULL runs the plain, division-per-element, table-anchor reference path.
 *
 * All literals are float (f suffix) and nothing calls libm: sigmoid/RoiAlign showed that a stray
 * double or libm call pulls in compiler-rt symbols the freestanding DSP link doesn't have. */
#ifndef PD_KERNEL_H
#define PD_KERNEL_H
#include <stdint.h>

typedef struct {
  float s1, s2;       /* rest.onnx delta Q/DQ scales (applied in this order) */
  int32_t z1, z2;     /* ...and zero points (uint8) */
  float exp_clip;     /* dw/dh upper clip, log(1000/16) */
  float clip_x, clip_y;
  float box_s;        /* box-grid Q/DQ scale, zero point box_z (uint8) */
  int32_t box_z;
  float bb_s;         /* backbone conv-output dequant (nchw source, and the fp32 on-grid check) */
  int32_t bb_z;
} pd_params;

typedef struct {
  float lut[256];     /* lut[q] = Q/DQ(Q/DQ((q - bb_z) * bb_s, s1, z1), s2, z2) */
  float inv_bb_s, inv_box_s;
  int grid;           /* anchors verified == base[i%3] + (w, h, w, h) * stride (pd_prepare_anchors) */
  float base[3][4], stride;
  uint64_t wmagic;    /* hw / W as (hw * wmagic) >> 32: no integer-divide helper (Hexagon has no divide instruction) */
} pd_prep;

/* round half to even for |x| < 2^22, default rounding mode; no libm. Needs no fast-math. */
static inline float pd_rint(float x) {
  const float m = 12582912.0f; /* 1.5 * 2^23 */
  float t = x + m;
  return t - m;
}
static inline float pd_clampq(float q) { return q < 0.0f ? 0.0f : (q > 255.0f ? 255.0f : q); }
static inline float pd_qdq(float x, float s, int32_t z) {
  float q = pd_clampq(pd_rint(x / s) + (float)z); /* ORT QuantizeLinear: divide, round-half-even, +zp, saturate */
  return (q - (float)z) * s;
}
/* same result as pd_qdq, division only near a rounding tie */
static inline float pd_qdq_fast(float x, float s, float inv, int32_t z) {
  float t = x * inv, n = pd_rint(t), d = t - n;
  d = d < 0.0f ? -d : d;
  if (d > 0.499f) n = pd_rint(x / s);
  float q = pd_clampq(n + (float)z);
  return (q - (float)z) * s;
}
/* expf, Cephes-style: n = round(x*log2e), r = x - n*ln2 (two-part), degree-6 poly, scale by 2^n.
 * Measured <= 1 ulp from correctly rounded exp over every float in [-10, log(1000/16)]. */
static inline float pd_expf(float x) {
  if (x > 88.0f) x = 88.0f;
  if (x < -87.0f) return 0.0f;
  float n = pd_rint(x * 1.44269504088896341f);
  float r = x - n * 0.693359375f;
  r = r - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.0f;
  union { float f; int32_t i; } s;
  s.i = ((int32_t)n + 127) << 23;
  return p * s.f;
}

static inline void pd_prepare(const pd_params* P, pd_prep* Q) {
  for (int q = 0; q < 256; q++)
    Q->lut[q] = pd_qdq(pd_qdq(((float)q - (float)P->bb_z) * P->bb_s, P->s1, P->z1), P->s2, P->z2);
  Q->inv_bb_s = 1.0f / P->bb_s;
  Q->inv_box_s = 1.0f / P->box_s;
  Q->grid = 0; /* table anchors until pd_prepare_anchors verifies the grid */
}

/* hw / W == (hw * magic) >> 32, exact for hw < 2^16 and W < 2^16 with magic = floor((2^32-1)/W) + 1
 * (overshoot per unit < 2^-32, times hw < 2^-16 < 1/W). */
static inline int pd_divw(int hw, uint64_t magic) { return (int)(((uint64_t)(uint32_t)hw * magic) >> 32); }
/* floor((2^32-1)/W) + 1 by shift-subtract long division: no divide instruction or helper needed */
static inline uint64_t pd_magic(uint32_t W) {
  uint64_t rem = 0, q = 0;
  for (int b = 31; b >= 0; b--) { rem = (rem << 1) | 1u; if (rem >= W) { rem -= W; q |= 1ull << b; } }
  return q + 1u;
}

/* The real anchor tables are an exact grid: anchor i = base[i % 3] + (w, h, w, h) * stride with
 * (h, w) = divmod(i / 3, W), all small integers, so fp32 reproduces them exactly. Verify that
 * against the actual table once; if it holds, the kernel computes anchors instead of gathering
 * 16 bytes per box from a multi-MB table (one likely cache miss per box). */
static inline void pd_prepare_anchors(const float* anchors, int H, int W, pd_prep* Q) {
  Q->wmagic = pd_magic((uint32_t)W);
  for (int a = 0; a < 3; a++)
    for (int c = 0; c < 4; c++) Q->base[a][c] = anchors[4 * a + c];
  Q->stride = W > 1 ? anchors[12] - anchors[0] : (H > 1 ? anchors[12 * W + 1] - anchors[1] : 0.0f);
  Q->grid = 1;
  for (int i = 0; i < 3 * H * W && Q->grid; i++) {
    int a = i % 3, hw = i / 3, h = pd_divw(hw, Q->wmagic);
    float fx = (float)(hw - h * W) * Q->stride, fy = (float)h * Q->stride;
    float e[4] = {Q->base[a][0] + fx, Q->base[a][1] + fy, Q->base[a][2] + fx, Q->base[a][3] + fy};
    for (int c = 0; c < 4; c++) Q->grid &= e[c] == anchors[4 * i + c];
  }
}

static inline float pd_delta(float x, const pd_params* P, const pd_prep* Q) {
  if (Q) { /* fp32 delta on the backbone's uint8 grid? then it's the LUT entry for that index */
    float q = pd_rint(x * Q->inv_bb_s) + (float)P->bb_z;
    if (q >= 0.0f && q <= 255.0f && (q - (float)P->bb_z) * P->bb_s == x) return Q->lut[(int)q];
  }
  return pd_qdq(pd_qdq(x, P->s1, P->z1), P->s2, P->z2);
}

/* Branch-free expf for the blocked fast path: same arithmetic as pd_expf, clamps as selects. */
static inline float pd_expf_nb(float x) {
  x = x > 88.0f ? 88.0f : x;
  float xl = x < -87.0f ? -87.0f : x;
  float n = pd_rint(xl * 1.44269504088896341f);
  float r = xl - n * 0.693359375f;
  r = r - n * -2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.0f;
  union { float f; int32_t i; } s;
  s.i = ((int32_t)n + 127) << 23;
  float e = p * s.f;
  return x < -87.0f ? 0.0f : e;
}

#define PD_BLK 64
/* L1/L2 prefetch hint for the next block's gather addresses: the top-k indices are known up
 * front, and each box's gather is a random access into a multi-MB map (the real bottleneck on the
 * DSP -- the math is software-pipelined and cheap). A no-op off-DSP. */
#if defined(__hexagon__)
#define PD_PREFETCH(p) __builtin_HEXAGON_Y2_dcfetch((void*)(p))
#else
#define PD_PREFETCH(p) ((void)(p))
#endif
static inline void pd_prefetch_block(const int32_t* idx, int n, const float* anchors, const float* deltas,
                                     const uint8_t* nchw, int HW, int grid) {
  for (int j = 0; j < n; j++) {
    int i = idx[j];
    if (nchw) {
      const uint8_t* src = nchw + ((i % 3) * 4) * HW + i / 3;
      PD_PREFETCH(src); PD_PREFETCH(src + HW); PD_PREFETCH(src + 2 * HW); PD_PREFETCH(src + 3 * HW);
    } else {
      PD_PREFETCH(deltas + 4 * i);
    }
    if (!grid) PD_PREFETCH(anchors + 4 * i);
  }
}
/* Fast path: gather a block of boxes into local SoA arrays, run the math as branch-free loops over
 * independent boxes (so the compiler can software-pipeline them instead of stalling on each box's
 * dependent fp32 latency chain), then redo the rare near-tie box-grid quantizations and off-grid
 * fp32 deltas exactly in a separate pass. Same values as the per-box path, bit for bit. */
static inline void pd_decode_blocked(const float* anchors, const int32_t* idx, int k, const float* deltas,
                                     const uint8_t* nchw, int H, int W, const pd_params* P,
                                     const pd_prep* Q, float* out) {
  const int HW = H * W;
  const float zb = (float)P->bb_z, sb = P->bb_s, ib = Q->inv_bb_s, ec = P->exp_clip;
  const float cxl = P->clip_x, cyl = P->clip_y, bs = P->box_s, ibs = Q->inv_box_s, bz = (float)P->box_z;
  pd_prefetch_block(idx, k < PD_BLK ? k : PD_BLK, anchors, deltas, nchw, HW, Q->grid);
  for (int j0 = 0; j0 < k; j0 += PD_BLK) {
    int n = k - j0 < PD_BLK ? k - j0 : PD_BLK;
    float a0[PD_BLK], a1[PD_BLK], a2[PD_BLK], a3[PD_BLK], d[4][PD_BLK], b[4][PD_BLK];
    int redo = 0;
    if (j0 + PD_BLK < k) /* next block's gathers, overlapped with this block's math */
      pd_prefetch_block(idx + j0 + PD_BLK, k - j0 - PD_BLK < PD_BLK ? k - j0 - PD_BLK : PD_BLK,
                        anchors, deltas, nchw, HW, Q->grid);
    if (Q->grid) {
      for (int j = 0; j < n; j++) {
        int i = idx[j0 + j], a = i % 3, hw = i / 3, h = pd_divw(hw, Q->wmagic);
        float fx = (float)(hw - h * W) * Q->stride, fy = (float)h * Q->stride;
        a0[j] = Q->base[a][0] + fx; a1[j] = Q->base[a][1] + fy;
        a2[j] = Q->base[a][2] + fx; a3[j] = Q->base[a][3] + fy;
      }
    } else {
      for (int j = 0; j < n; j++) {
        const float* a = anchors + 4 * idx[j0 + j];
        a0[j] = a[0]; a1[j] = a[1]; a2[j] = a[2]; a3[j] = a[3];
      }
    }
    if (nchw) {
      for (int j = 0; j < n; j++) {
        int i = idx[j0 + j];
        const uint8_t* src = nchw + ((i % 3) * 4) * HW + i / 3;
        d[0][j] = Q->lut[src[0]]; d[1][j] = Q->lut[src[HW]];
        d[2][j] = Q->lut[src[2 * HW]]; d[3][j] = Q->lut[src[3 * HW]];
      }
    } else {
      for (int c = 0; c < 4; c++)
        for (int j = 0; j < n; j++) {
          float x = deltas[4 * idx[j0 + j] + c];
          float q = pd_rint(x * ib) + zb;
          int ok = q >= 0.0f && q <= 255.0f && (q - zb) * sb == x;
          int qi = ok ? (int)q : 0;
          d[c][j] = Q->lut[qi];
          redo |= !ok;
        }
      if (redo) /* never on this graph (deltas are dequantized on the backbone grid), kept exact */
        for (int c = 0; c < 4; c++)
          for (int j = 0; j < n; j++) d[c][j] = pd_delta(deltas[4 * idx[j0 + j] + c], P, Q);
    }
    /* branch-free per-stage loops (splitting the stages vs one fused loop measured no different) */
    float wd[PD_BLK], ht[PD_BLK], pcx[PD_BLK], pcy[PD_BLK], ex[2 * PD_BLK];
    for (int j = 0; j < n; j++) {
      float w = (a2[j] - a0[j]) + 1.0f, h = (a3[j] - a1[j]) + 1.0f;
      float cx = a0[j] + 0.5f * w, cy = a1[j] + 0.5f * h;
      float px = d[0][j] * w, py = d[1][j] * h;
      wd[j] = w; ht[j] = h; pcx[j] = px + cx; pcy[j] = py + cy;
      ex[j] = d[2][j] < ec ? d[2][j] : ec;
      ex[PD_BLK + j] = d[3][j] < ec ? d[3][j] : ec;
    }
    for (int j = 0; j < n; j++) ex[j] = pd_expf_nb(ex[j]);
    for (int j = 0; j < n; j++) ex[PD_BLK + j] = pd_expf_nb(ex[PD_BLK + j]);
    for (int j = 0; j < n; j++) {
      float hw2 = 0.5f * (ex[j] * wd[j]), hh2 = 0.5f * (ex[PD_BLK + j] * ht[j]);
      float v0 = pcx[j] - hw2, v1 = pcy[j] - hh2, v2 = (pcx[j] + hw2) - 1.0f, v3 = (pcy[j] + hh2) - 1.0f;
      b[0][j] = v0 < 0.0f ? 0.0f : (v0 > cxl ? cxl : v0);
      b[1][j] = v1 < 0.0f ? 0.0f : (v1 > cyl ? cyl : v1);
      b[2][j] = v2 < 0.0f ? 0.0f : (v2 > cxl ? cxl : v2);
      b[3][j] = v3 < 0.0f ? 0.0f : (v3 > cyl ? cyl : v3);
    }
    int tie = 0;
    for (int c = 0; c < 4; c++)
      for (int j = 0; j < n; j++) {
        float v = b[c][j], t = v * ibs, r = pd_rint(t), e = t - r;
        e = e < 0.0f ? -e : e;
        tie |= e > 0.499f;
        float q = pd_clampq(r + bz);
        out[4 * (j0 + j) + c] = (q - bz) * bs;
      }
    if (tie) /* ~0.02% of values land within 1e-3 of a .5 tie: redo those with the real division */
      for (int c = 0; c < 4; c++)
        for (int j = 0; j < n; j++) out[4 * (j0 + j) + c] = pd_qdq_fast(b[c][j], bs, ibs, P->box_z);
  }
}

static inline void pd_decode(const float* anchors, const int32_t* idx, int k, const float* deltas,
                             const uint8_t* nchw, int H, int W, const pd_params* P, const pd_prep* Q,
                             float* out) {
  if (Q) { pd_decode_blocked(anchors, idx, k, deltas, nchw, H, W, P, Q, out); return; }
  const int HW = H * W;
  for (int j = 0; j < k; j++) {
    int i = idx[j];
    const float* a = anchors + 4 * i;
    float d[4];
    if (deltas) {
      for (int c = 0; c < 4; c++) d[c] = pd_delta(deltas[4 * i + c], P, 0);
    } else { /* [1,3,4,H,W] -> transpose(0,3,4,1,2) -> [-1,4]: anchor i = (h*W + w)*3 + a */
      const uint8_t* src = nchw + ((i % 3) * 4) * HW + i / 3;
      for (int c = 0; c < 4; c++) d[c] = pd_delta(((float)src[c * HW] - (float)P->bb_z) * P->bb_s, P, 0);
    }
    float w = (a[2] - a[0]) + 1.0f, h = (a[3] - a[1]) + 1.0f;
    float cx = a[0] + 0.5f * w, cy = a[1] + 0.5f * h;
    float pcx = d[0] * w; pcx = pcx + cx;
    float pcy = d[1] * h; pcy = pcy + cy;
    float dw = d[2] < P->exp_clip ? d[2] : P->exp_clip;
    float dh = d[3] < P->exp_clip ? d[3] : P->exp_clip;
    float pw = pd_expf(dw) * w, ph = pd_expf(dh) * h;
    float hw2 = 0.5f * pw, hh2 = 0.5f * ph;
    float b[4] = {pcx - hw2, pcy - hh2, (pcx + hw2) - 1.0f, (pcy + hh2) - 1.0f};
    for (int c = 0; c < 4; c++) {
      float lim = (c & 1) ? P->clip_y : P->clip_x;
      float v = b[c] < 0.0f ? 0.0f : (b[c] > lim ? lim : b[c]);
      out[4 * j + c] = pd_qdq(v, P->box_s, P->box_z);
    }
  }
}
#endif
