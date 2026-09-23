/* NonMaxSuppression (ONNX, center_point_box=0, one batch, one class per call -- how all 85 real
 * calls in MaskRCNN-12-qdq look), bit-for-bit ONNX Runtime's CPU semantics:
 *   - candidates visited in ORT's priority_queue order: score descending, ties by lower index;
 *   - a candidate is kept unless SuppressByIOU(candidate, kept) is true for any already-kept box;
 *   - SuppressByIOU is onnxruntime/core/providers/cpu/object_detection/non_max_suppression_helper.h,
 *     copied operation for operation (the same fp32 ops in the same order, no FMA contraction --
 *     build with -ffp-contract=off) so ties at the threshold resolve identically.
 *
 * Two kernels, same output:
 *   nms_scalar -- the greedy loop, scalar, exact.
 *   nms_hvx    -- same greedy loop, but each candidate is tested against 32 kept boxes per HVX
 *                 vector (kept boxes stored structure-of-arrays). The test phone's CDSP is Hexagon
 *                 V69: its HVX has no IEEE fp32, only qfloat, and a qfloat op on IEEE (.sf) inputs,
 *                 converted back to .sf, is off by up to 2^-23 * max(|operand|) (measured on the
 *                 phone; chaining qf32 x qf32 without renormalizing -- what clang emits for plain
 *                 vector-extension C -- is far worse, ~500 ulp). So the vector path is written with
 *                 explicit intrinsics, renormalizing to .sf after every op, and only decides the
 *                 clear-cut pairs. It computes d = inter*(1+thr) - (thr*area1 + thr*area2), which
 *                 is inter - thr*union in exact arithmetic (thr*area per kept box is precomputed in
 *                 scalar IEEE fp32); its accumulated error is bounded by 2^-23 * (2*M*(w+h) +
 *                 7*(area1+area2)) with M = largest |coordinate| in the call. The band half-width
 *                 is t = 4x that bound, using the candidate's own w+h (>= the intersection's): a
 *                 pair is "surely suppressed" if d > t, "surely not" if d < -t, and anything in
 *                 between is re-decided with the exact scalar SuppressByIOU.
 *                 Box-overlap tests (max/min/compare of input coordinates) are exact in HVX.
 * Header-only so the host check, the qemu harness and the FastRPC skel compile the same code. */
#ifndef NMS_KERNEL_H
#define NMS_KERNEL_H

/* uncertainty band: t = NMS_K1M * M * (w_cand + h_cand) + NMS_K2 * (area1 + area2), 4x the bound */
#define NMS_EPS (1.0f / 8388608.0f) /* 2^-23 */
#define NMS_K1M (8.0f * NMS_EPS)
#define NMS_K2 (28.0f * NMS_EPS)
#define NMS_SOA 7 /* kept-box SoA arrays: x0, x1, y0, y1, area, thr*area, NMS_K2*area */


static inline void nms_maxmin(float lhs, float rhs, float* mn, float* mx) {
  if (lhs >= rhs) { *mn = rhs; *mx = lhs; } else { *mn = lhs; *mx = rhs; }
}
static inline float nms_fmax(float a, float b) { return a < b ? b : a; } /* std::max(a, b) */
static inline float nms_fmin(float a, float b) { return b < a ? b : a; } /* std::min(a, b) */

/* ORT's SuppressByIOU for center_point_box == 0 (boxes are [y1, x1, y2, x2]). */
static int nms_suppress_exact(const float* box1, const float* box2, float iou_threshold) {
  float x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max;
  nms_maxmin(box1[1], box1[3], &x1_min, &x1_max);
  nms_maxmin(box2[1], box2[3], &x2_min, &x2_max);
  float ix_min = nms_fmax(x1_min, x2_min), ix_max = nms_fmin(x1_max, x2_max);
  if (ix_max <= ix_min) return 0;
  nms_maxmin(box1[0], box1[2], &y1_min, &y1_max);
  nms_maxmin(box2[0], box2[2], &y2_min, &y2_max);
  float iy_min = nms_fmax(y1_min, y2_min), iy_max = nms_fmin(y1_max, y2_max);
  if (iy_max <= iy_min) return 0;
  const float intersection_area = (ix_max - ix_min) * (iy_max - iy_min);
  if (intersection_area <= .0f) return 0;
  const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const float union_area = area1 + area2 - intersection_area;
  if (area1 <= .0f || area2 <= .0f || union_area <= .0f) return 0;
  const float iou = intersection_area / union_area;
  return iou > iou_threshold;
}

/* Visit order: indices sorted by score descending, ties by index ascending (a stable merge sort
 * over 0..n-1 by score descending gives exactly that). tmp: n ints of scratch. */
static void nms_order(const float* scores, int n, int* order, int* tmp) {
  for (int i = 0; i < n; i++) order[i] = i;
  for (int w = 1; w < n; w *= 2) {
    for (int lo = 0; lo < n; lo += 2 * w) {
      int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n;
      int a = lo, b = mid, k = lo;
      while (a < mid && b < hi) tmp[k++] = scores[order[b]] > scores[order[a]] ? order[b++] : order[a++];
      while (a < mid) tmp[k++] = order[a++];
      while (b < hi) tmp[k++] = order[b++];
    }
    for (int i = 0; i < n; i++) order[i] = tmp[i];
  }
}

/* Scratch per call: order/tmp need n ints each; nms_hvx additionally needs NMS_SOA * round_up(n, 64)
 * 128-byte-aligned floats for the kept boxes' SoA. Returns the number of kept boxes written to sel
 * (original box indices, in selection order -- ORT's output order within a class). */
static int nms_scalar(const float* boxes, const float* scores, int n, float thr, int max_out,
                      int* sel, int* order, int* tmp) {
  nms_order(scores, n, order, tmp);
  int ns = 0;
  for (int k = 0; k < n && ns < max_out; k++) {
    const int c = order[k];
    int keep = 1;
    for (int j = 0; j < ns; j++)
      if (nms_suppress_exact(boxes + 4 * c, boxes + 4 * sel[j], thr)) { keep = 0; break; }
    if (keep) sel[ns++] = c;
  }
  return ns;
}

/* One candidate vs 32 or 64 kept boxes (SoA slices, 128-byte aligned; two independent HVX vectors
 * per call for instruction-level parallelism, both flags folded into one cross-lane OR-reduction). *sure: some lane surely
 * suppresses the candidate; *maybe: some lane is inside the uncertainty band. Padding lanes hold
 * x0=y0=+inf, x1=y1=-inf, so they never overlap. */
#if defined(__hexagon__) && defined(__HVX__)
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
static inline HVX_Vector nms_splat(float f) { union { float f; int i; } u = {f}; return Q6_V_vsplat_R(u.i); }
/* qf32 -> IEEE sf, and keep it that way: without the empty asm, LLVM folds the conversion into the
 * next qfloat op (emitting e.g. vmpy(qf32,qf32)), which is exactly the imprecise form this avoids --
 * measured on the phone: 3.50012207 * 3.00006104 came out 10.5625 that way, 10.5006 with the barrier. */
static inline HVX_Vector nms_sf(HVX_Vector qf32) {
  HVX_Vector r = Q6_Vsf_equals_Vqf32(qf32);
  __asm__("" : "+v"(r));
  return r;
}
static inline void nms_vec(const float* kx0, const float* kx1, const float* ky0, const float* ky1,
                           const float* kta, const float* kka, HVX_Vector vx0, HVX_Vector vx1,
                           HVX_Vector vy0, HVX_Vector vy1, HVX_Vector vc1, HVX_Vector vtca,
                           HVX_Vector vtc, HVX_Vector* flags) {
  const HVX_Vector x0 = *(const HVX_Vector*)kx0, x1 = *(const HVX_Vector*)kx1;
  const HVX_Vector y0 = *(const HVX_Vector*)ky0, y1 = *(const HVX_Vector*)ky1;
  const HVX_Vector ix0 = Q6_Vsf_vmax_VsfVsf(x0, vx0), ix1 = Q6_Vsf_vmin_VsfVsf(x1, vx1);
  const HVX_Vector iy0 = Q6_Vsf_vmax_VsfVsf(y0, vy0), iy1 = Q6_Vsf_vmin_VsfVsf(y1, vy1);
  const HVX_VectorPred overlap = Q6_Q_vcmp_gtand_QVsfVsf(Q6_Q_vcmp_gt_VsfVsf(ix1, ix0), iy1, iy0);
  const HVX_Vector w = nms_sf(Q6_Vqf32_vsub_VsfVsf(ix1, ix0)), h = nms_sf(Q6_Vqf32_vsub_VsfVsf(iy1, iy0));
  const HVX_Vector inter = nms_sf(Q6_Vqf32_vmpy_VsfVsf(w, h));
  const HVX_Vector tsum = nms_sf(Q6_Vqf32_vadd_VsfVsf(vtca, *(const HVX_Vector*)kta)); /* thr*(area1+area2) */
  const HVX_Vector d = nms_sf(Q6_Vqf32_vsub_VsfVsf(nms_sf(Q6_Vqf32_vmpy_VsfVsf(inter, vc1)), tsum));
  const HVX_Vector t = nms_sf(Q6_Vqf32_vadd_VsfVsf(vtc, *(const HVX_Vector*)kka));
  const HVX_Vector negt = nms_sf(Q6_Vqf32_vsub_VsfVsf(Q6_V_vzero(), t));
  /* bit 0: surely suppresses; bit 1: inside the uncertainty band (overlap & !(d < -t)) */
  *flags = Q6_V_vor_VV(*flags, Q6_V_vor_VV(Q6_V_vand_QR(Q6_Q_vcmp_gtand_QVsfVsf(overlap, d, t), 1),
                                           Q6_V_vand_QR(Q6_Q_and_QQn(overlap, Q6_Q_vcmp_gt_VsfVsf(negt, d)), 2)));
}
/* nlanes = 32 or 64 kept boxes starting at the given SoA slices. */
static inline void nms_chunk(const float* kx0, const float* kx1, const float* ky0, const float* ky1,
                             const float* kta, const float* kka, int nlanes, float cx0, float cx1,
                             float cy0, float cy1, float c1, float tca, float tc, int* sure, int* maybe) {
  const HVX_Vector vx0 = nms_splat(cx0), vx1 = nms_splat(cx1), vy0 = nms_splat(cy0), vy1 = nms_splat(cy1);
  const HVX_Vector vc1 = nms_splat(c1), vtca = nms_splat(tca), vtc = nms_splat(tc);
  HVX_Vector f = Q6_V_vzero();
  nms_vec(kx0, kx1, ky0, ky1, kta, kka, vx0, vx1, vy0, vy1, vc1, vtca, vtc, &f);
  if (nlanes > 32) nms_vec(kx0 + 32, kx1 + 32, ky0 + 32, ky1 + 32, kta + 32, kka + 32, vx0, vx1, vy0, vy1, vc1, vtca, vtc, &f);
  for (int sh = 64; sh >= 4; sh >>= 1) f = Q6_V_vor_VV(f, Q6_V_vror_VR(f, sh));
  const int r = Q6_R_vextract_VR(f, 0);
  *sure = r & 1; *maybe = (r >> 1) & 1;
}
#else /* portable version (host check, qemu): same classification in IEEE fp32 */
static inline void nms_chunk(const float* kx0, const float* kx1, const float* ky0, const float* ky1,
                             const float* kta, const float* kka, int nlanes, float cx0, float cx1,
                             float cy0, float cy1, float c1, float tca, float tc, int* sure, int* maybe) {
  *sure = *maybe = 0;
  for (int l = 0; l < nlanes; l++) {
    const float ix0 = nms_fmax(kx0[l], cx0), ix1 = nms_fmin(kx1[l], cx1);
    const float iy0 = nms_fmax(ky0[l], cy0), iy1 = nms_fmin(ky1[l], cy1);
    if (!(ix1 > ix0 && iy1 > iy0)) continue;
    const float inter = (ix1 - ix0) * (iy1 - iy0);
    const float d = inter * c1 - (tca + kta[l]), t = tc + kka[l];
    if (d > t) *sure = 1;
    if (!(-t > d)) *maybe = 1;
  }
}
#endif

static int nms_hvx(const float* boxes, const float* scores, int n, float thr, int max_out,
                   int* sel, int* order, int* tmp, float* soa) {
  nms_order(scores, n, order, tmp);
  const int cap = (n + 63) & ~63;
  float *kx0 = soa, *kx1 = soa + cap, *ky0 = soa + 2 * cap, *ky1 = soa + 3 * cap;
  float *kta = soa + 4 * cap, *kka = soa + 5 * cap;
  int* kidx = tmp; /* order[] is fully built, tmp is free again: original index per SoA slot */
  const float inf = __builtin_inff();
  for (int i = 0; i < cap; i++) { kx0[i] = inf; kx1[i] = -inf; ky0[i] = inf; ky1[i] = -inf; kta[i] = kka[i] = 0.0f; }
  float M = 0.0f; /* largest |coordinate| in this call, for the error bound */
  for (int i = 0; i < 4 * n; i++) { float a = boxes[i] < 0 ? -boxes[i] : boxes[i]; if (a > M) M = a; }
  const float k1 = NMS_K1M * M, c1 = 1.0f + thr;
  int ns = 0, nk = 0; /* ns: kept overall; nk: kept boxes with area > 0 (the only ones that can suppress) */
  for (int k = 0; k < n && ns < max_out; k++) {
    const int c = order[k];
    const float* b = boxes + 4 * c;
    float cx0, cx1, cy0, cy1;
    nms_maxmin(b[1], b[3], &cx0, &cx1);
    nms_maxmin(b[0], b[2], &cy0, &cy1);
    const float carea = (cx1 - cx0) * (cy1 - cy0);
    int keep = 1;
    if (carea > .0f && nk) {
      int maybe_any = 0;
      const float tca = thr * carea, tc = k1 * ((cx1 - cx0) + (cy1 - cy0)) + NMS_K2 * carea;
      for (int j = 0; j < nk && keep; j += 64) {
        int sure, maybe;
        nms_chunk(kx0 + j, kx1 + j, ky0 + j, ky1 + j, kta + j, kka + j, nk - j > 32 ? 64 : 32, cx0, cx1, cy0, cy1, c1, tca, tc, &sure, &maybe);
        if (sure) { keep = 0; break; }
        maybe_any |= maybe;
      }
      if (keep && maybe_any) { /* rare: decide the near-threshold pairs exactly, like ORT */
        for (int j = 0; j < nk; j++)
          if (nms_suppress_exact(b, boxes + 4 * kidx[j], thr)) { keep = 0; break; }
      }
    }
    if (keep) {
      sel[ns++] = c;
      if (carea > .0f) { kx0[nk] = cx0; kx1[nk] = cx1; ky0[nk] = cy0; ky1[nk] = cy1; kta[nk] = thr * carea; kka[nk] = NMS_K2 * carea; kidx[nk] = c; nk++; }
    }
  }
  return ns;
}

#endif
