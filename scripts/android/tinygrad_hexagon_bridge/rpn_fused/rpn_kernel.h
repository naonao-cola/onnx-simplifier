/* Mask R-CNN's whole RPN post-processing span as one computation, from the per-level objectness
 * scores and box deltas (today's backbone/rest boundary) to the proposal list that feeds the ROI
 * heads: exactly the 604 non-Constant rest.onnx nodes between those 10 inputs and tensor `2527`
 * (see capture_rpn_fused.py, which reads every constant below out of the graph).
 *
 * Per FPN level (independent, so the DSP runs one thread per level):
 *   1. TopK(objectness, k)                 -- ../topk/topk_kernel.h (exact, ORT's tie order)
 *   2. proposal decode of the k boxes      -- ../proposal_decode/pd_kernel.h (bit-exact)
 *   3. the "min-size" filter: w = (x2-x1)+1, h = (y2-y1)+1, Q/DQ'd onto their own uint8 grid,
 *      keep !(w<0) && !(h<0), then compact boxes and TopK values (NonZero + Gather)
 *   4. NonMaxSuppression(iou 0.7)          -- ../nms/nms_kernel.h (exact, qfloat + exact recheck)
 *   5. the first min(kept, cap=1000) selections, their boxes and scores
 * Then: concat the 5 levels (P2..P6 order), TopK(min(1000, N)) on the scores (ORT's tie order:
 * value desc, then position in the concat asc), gather the boxes.
 *
 * Exactness notes, all checked end to end against ORT's real `2527` on 5 images:
 *   - Every box after decode is on the uint8 box grid (scale 5.0157, zero point 0). The graph's
 *     later Q/DQs onto that same grid (after the Split, the filter Gather, the per-level Gather,
 *     the Concat, the final Gather) are identities on grid values, so they are not recomputed.
 *   - The filter's w/h grid is uint8 with zero point 0 at every level, so the quantized w/h are
 *     never negative and the filter can never drop a box in this model, for any input. With
 *     zero point 0 the kernel therefore reads the boxes in place (saves ~90 us per level on the
 *     DSP); rpn_exact_filter = 1 computes it exactly as the graph does instead, and the host check
 *     and qemu verify both paths against ORT.
 *   - NMS gets the boxes in rest.onnx's [x1,y1,x2,y2] order and treats them as [y1,x1,y2,x2]
 *     (center_point_box=0), exactly like ORT does with the same array; IoU is symmetric in that.
 *
 * Header-only, so the host check, the qemu harness and the FastRPC skel compile the same code.
 * Build with -ffp-contract=off (pd/nms exactness). */
#ifndef RPN_KERNEL_H
#define RPN_KERNEL_H
#include <stdint.h>
#include <string.h>
#include "../proposal_decode/pd_kernel.h"
#include "../topk/topk_kernel.h"
#include "../nms/nms_kernel.h"

#define RPN_NLVL 5
#define RPN_KMAX 1024 /* >= every level's k (1000) and < nms_kernel.h's per-call limit */
#ifndef RPN_NOW /* the DSP skel defines this as HAP_perf_get_time_us() for per-phase timing */
#define RPN_NOW() 0ull
#endif

typedef struct {
  int A, k, H, W;   /* anchors, TopK k, conv-map size (deltas are [A,4], nchw source [12,H,W]) */
  float iou;        /* NMS iou_threshold */
  int max_out, cap; /* NMS max_output_boxes_per_class; kept selections taken per level */
  float filt_s;     /* the w/h filter's Q/DQ grid */
  int filt_z;
  pd_params P;      /* decode constants (../proposal_decode) */
} rpn_level_cfg;

typedef struct {
  rpn_level_cfg lv[RPN_NLVL];
  int post_cap;     /* final TopK k = min(post_cap, total kept) */
} rpn_model;

/* Per-level working set (reused across calls; the DSP keeps one per level thread). */
typedef struct {
  float soa[NMS_SOA * RPN_KMAX] __attribute__((aligned(128))); /* nms_hvx kept-box SoA */
  float vals[RPN_KMAX], boxes[4 * RPN_KMAX], fboxes[4 * RPN_KMAX], fscores[RPN_KMAX];
  int64_t idx64[RPN_KMAX];
  int32_t idx[RPN_KMAX];
  int sel[RPN_KMAX], order[RPN_KMAX], tmp[RPN_KMAX];
  const float *kb, *ksc; /* the filtered boxes/scores NMS and the merge read: fboxes/fscores, or
                          * boxes/vals themselves when the filter provably keeps everything */
  int nf, ns, m;    /* after filter, kept by NMS, taken (min(ns, cap)) */
  unsigned ph[4];   /* RPN_NOW() deltas: topk, decode, filter, nms */
} rpn_level_ws;

/* The TopK collect pass is the only step worth splitting across threads (P2: 163,200 scores);
 * the DSP skel passes its own splitter, everyone else uses the plain one. Contract is
 * tk_collect_range's: survivors of x[0, n) with key >= t, in index order; returns the count. */
typedef int (*rpn_collect_fn)(void* ctx, const float* x, int n, uint32_t t, uint32_t* ck, uint32_t* ci);
static int rpn_collect_plain(void* ctx, const float* x, int n, uint32_t t, uint32_t* ck, uint32_t* ci) {
  (void)ctx;
  return tk_collect_range(x, 0, n, t, ck, ci, TK_VEC_ROT);
}

/* 1. TopK -- topk_desc() with a pluggable collect. scratch: TK_SCRATCH_WORDS(n) uint32. */
static int rpn_topk(const float* x, int n, int k, float* ov, int64_t* oi, uint32_t* scratch,
                    rpn_collect_fn collect, void* cctx) {
  if (k < 0 || k > n) return -1;
  uint32_t *ck = scratch, *ci = scratch + n, *tk = scratch + 2 * (long)n, *ti = scratch + 3 * (long)n;
  uint32_t *sk = scratch + 4 * (long)n, *tmp = sk + TK_SAMPLE, *hs = tmp + TK_SAMPLE;
  int r, m, c;
  uint32_t t = tk_threshold(x, n, k, sk, tmp, &r, &m);
  for (;;) {
    c = collect(cctx, x, n, t, ck, ci);
    if (c < 0) return c;
    if (c >= k || t == 0) break;
    t = tk_retry(sk, tmp, m, &r);
  }
  tk_emit(x, ck, ci, tk, ti, hs, c, k, ov, oi);
  return c;
}

/* The level chain as separate steps (the DSP skel schedules them in phases across threads; the
 * host check and qemu run them through rpn_level). */

/* Step 1: TopK. scratch: TK_SCRATCH_WORDS(A) uint32. */
static int rpn_level_topk(const rpn_level_cfg* L, const float* scores, rpn_level_ws* w, uint32_t* tk_scratch,
                          rpn_collect_fn collect, void* cctx) {
  const int k = L->k;
  if (k > RPN_KMAX || k > L->A) return -1;
  unsigned long long t0 = RPN_NOW();
  if (rpn_topk(scores, L->A, k, w->vals, w->idx64, tk_scratch, collect, cctx) < 0) return -1;
  for (int j = 0; j < k; j++) w->idx[j] = (int32_t)w->idx64[j];
  w->ph[0] = (unsigned)(RPN_NOW() - t0);
  return 0;
}

/* Step 2: decode boxes [j0, j1) of the level's k. Boxes are independent, so any split of [0, k)
 * across threads gives the same values (../proposal_decode/pd_impl.c splits the same way). */
static void rpn_level_decode(const rpn_level_cfg* L, const pd_prep* pq, const float* anchors, const float* deltas,
                             const uint8_t* nchw, rpn_level_ws* w, int j0, int j1) {
  if (j0 < j1) pd_decode(anchors, w->idx + j0, j1 - j0, deltas, nchw, L->H, L->W, &L->P, pq, w->boxes + 4 * j0);
}

/* Steps 3-5: filter + compaction, NMS, cap. exact_filter = 1 computes the filter even when it's
 * provably a no-op (host check / qemu use it to test both paths). */
static void rpn_level_filter_nms_x(const rpn_level_cfg* L, rpn_level_ws* w, int nms_mode, int exact_filter);
static int rpn_exact_filter; /* host check / qemu set this to also verify the computed filter */
#define rpn_level_filter_nms(L, w, nms_mode) rpn_level_filter_nms_x(L, w, nms_mode, rpn_exact_filter)

/* One level, steps 1-5. scores: [A]; exactly one of deltas ([A,4] fp32) / nchw ([12,H,W] uint8).
 * pq: the level's pd_prep (NULL = decode's division-per-element reference path). nms_mode: 1 =
 * nms_hvx (portable IEEE classification off-DSP), 0 = nms_scalar. anchors: the level's [A,4] table
 * (only read if pq is NULL or its grid check failed). Returns 0, or <0 on bad arguments. */
static int rpn_level(const rpn_level_cfg* L, const pd_prep* pq, const float* anchors, const float* scores,
                     const float* deltas, const uint8_t* nchw, rpn_level_ws* w, uint32_t* tk_scratch,
                     rpn_collect_fn collect, void* cctx, int nms_mode) {
  if (rpn_level_topk(L, scores, w, tk_scratch, collect, cctx)) return -1;
  unsigned long long t1 = RPN_NOW();
  rpn_level_decode(L, pq, anchors, deltas, nchw, w, 0, L->k);
  w->ph[1] = (unsigned)(RPN_NOW() - t1);
  rpn_level_filter_nms(L, w, nms_mode);
  return 0;
}

static void rpn_level_filter_nms_x(const rpn_level_cfg* L, rpn_level_ws* w, int nms_mode, int exact_filter) {
  const int k = L->k;
  unsigned long long t2 = RPN_NOW();
  /* 3. filter: graph order is Sub(x2, x1) then Add(., 1.0); the Q/DQs onto the w/h grid are
   * applied as the graph does (repeats are idempotent, one suffices); Less(q, 0) -> Not -> And.
   * The grid is uint8 (capture_rpn_fused.py asserts it): with zero point 0 a dequantized value is
   * (q - 0) * s with q in [0, 255], never < 0 (a NaN compares false too), so nothing is ever
   * dropped -- then NMS reads the boxes/TopK values in place, no copy. */
  int nf = 0;
  if (L->filt_z == 0 && !exact_filter) {
    nf = k;
    w->kb = w->boxes;
    w->ksc = w->vals;
  } else {
    const float fs = L->filt_s, fi = 1.0f / fs;
    for (int j = 0; j < k; j++) {
      const float* b = w->boxes + 4 * j;
      float bw = (b[2] - b[0]) + 1.0f, bh = (b[3] - b[1]) + 1.0f;
      float qw = pd_qdq_fast(bw, fs, fi, L->filt_z), qh = pd_qdq_fast(bh, fs, fi, L->filt_z);
      if (!(qw < 0.0f) && !(qh < 0.0f)) {
        memcpy(w->fboxes + 4 * nf, b, 16);
        w->fscores[nf] = w->vals[j];
        nf++;
      }
    }
    w->kb = w->fboxes;
    w->ksc = w->fscores;
  }
  w->nf = nf;
  unsigned long long t3 = RPN_NOW();
  /* 4. NMS (per-call limit is well above RPN_KMAX) */
  w->ns = nms_mode ? nms_hvx(w->kb, w->ksc, nf, L->iou, L->max_out, w->sel, w->order, w->tmp, w->soa)
                   : nms_scalar(w->kb, w->ksc, nf, L->iou, L->max_out, w->sel, w->order, w->tmp);
  unsigned long long t4 = RPN_NOW();
  w->ph[2] = (unsigned)(t3 - t2); w->ph[3] = (unsigned)(t4 - t3);
  /* 5. Gather(sel, 2) -> Slice[0:cap]: the first cap selections, in selection order */
  w->m = w->ns < L->cap ? w->ns : L->cap;
}

/* Merge scratch: concat scores/boxes (5 * RPN_KMAX), TopK outputs and scratch for n = total. */
#define RPN_MERGE_MAX (RPN_NLVL * RPN_KMAX)
typedef struct {
  float scores[RPN_MERGE_MAX], boxes[4 * RPN_MERGE_MAX], vals[RPN_MERGE_MAX];
  int64_t idx[RPN_MERGE_MAX];
  uint32_t tk_scratch[TK_SCRATCH_WORDS(RPN_MERGE_MAX)];
} rpn_merge_ws;

/* Concat the 5 levels' taken selections (P2..P6), TopK(min(post_cap, N)), gather boxes.
 * Writes out[4 * kpost]; returns kpost, and *total = N. */
static int rpn_merge(const rpn_model* M, rpn_level_ws* const* w, rpn_merge_ws* mw, float* out, int* total) {
  int N = 0;
  for (int l = 0; l < RPN_NLVL; l++) {
    const rpn_level_ws* v = w[l];
    for (int j = 0; j < v->m; j++) {
      int s = v->sel[j];
      mw->scores[N] = v->ksc[s];
      memcpy(mw->boxes + 4 * N, v->kb + 4 * s, 16);
      N++;
    }
  }
  *total = N;
  int kp = M->post_cap < N ? M->post_cap : N;
  if (rpn_topk(mw->scores, N, kp, mw->vals, mw->idx, mw->tk_scratch, rpn_collect_plain, 0) < 0) return -1;
  for (int j = 0; j < kp; j++) memcpy(out + 4 * j, mw->boxes + 4 * mw->idx[j], 16);
  return kp;
}
#endif
