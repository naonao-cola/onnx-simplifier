/* Exact TopK (largest=1, sorted=1, 1-D) matching ONNX Runtime's CPU output bit-for-bit, including
 * its tie order: value descending, then index ascending (checked on all 7 real Mask R-CNN TopK
 * calls, whose inputs are dequantized int8 -- only 51..213 distinct values, so ties at the k-th
 * value are the common case, not an edge case).
 *
 * Algorithm (select, then sort only the survivors):
 *   1. Map fp32 to a monotone uint32 key (bigger float -> bigger key; -0 canonicalized to +0 so it
 *      ties with +0, as a float compare would).
 *   2. Threshold t: the right rank of a 1024-key sample (128 evenly spaced 8-element blocks, one
 *      quickselect), aimed at ~1.5k survivors (tk_threshold). Small inputs skip this and take
 *      everything (t = 0).
 *   3. One streaming pass (tk_collect_range): HVX 32-lane `key >= t` compare, turned into a 32-bit
 *      lane bitmask (AND with per-lane bit weights, then a rotate/OR tree); only set bits are
 *      visited (ctz), appending survivors in index order. Ranges are independent, so the DSP side
 *      splits big inputs across hardware threads. If fewer than k survive, lower t (tk_retry) and
 *      redo -- never needed on the real data; the last resort t = 0 takes everything, so the
 *      result is exact regardless of the sample.
 *   4. Emit (tk_emit): the survivors' distinct keys (hash set) sorted descending, then a single
 *      stable counting-sort pass by key rank -- stability keeps index order within a key, which is
 *      exactly ORT's tie order. Falls back to a stable LSD radix sort when there are more than 256
 *      distinct keys. Values are recovered from the keys; indices int64.
 * Correctness does not depend on t: every top-k element has key >= k-th key >= t whenever at
 * least k elements pass, so the survivor set always contains the whole answer.
 *
 * Portable C + clang vector extensions: the same file builds for the host check, for qemu, and for
 * the CDSP (where clang lowers the u32x32 ops to HVX). NaN inputs are not handled (none occur). */
#ifndef TOPK_KERNEL_H
#define TOPK_KERNEL_H
#include <stdint.h>
#include <string.h>

typedef uint32_t tk_u32x32 __attribute__((vector_size(128)));
typedef int32_t tk_i32x32 __attribute__((vector_size(128)));

static inline uint32_t tk_key(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if (u == 0x80000000u) u = 0;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline tk_u32x32 tk_keyv(tk_u32x32 u) {
  u &= ~(tk_u32x32)(u == 0x80000000u);                  /* -0 -> +0 */
  tk_u32x32 s = (tk_u32x32)((tk_i32x32)u >> 31);        /* all-ones where negative */
  return u ^ (s | 0x80000000u);
}

/* Stable sort of (key, idx) by key descending. a/b are the data, ta/tb scratch of the same size.
 * Returns 0 if the result is in (a, b), 1 if it ended in (ta, tb). */
static int tk_radix_desc(uint32_t* a, uint32_t* b, uint32_t* ta, uint32_t* tb, int m) {
  int where = 0;
  for (int sh = 0; sh < 32; sh += 8) {
    uint32_t cnt[256];
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < m; i++) cnt[((~a[i]) >> sh) & 255]++;
    int single = 0;
    for (int d = 0; d < 256; d++) if (cnt[d] == (uint32_t)m) { single = 1; break; }
    if (single) continue;
    uint32_t pos = 0;
    for (int d = 0; d < 256; d++) { uint32_t c = cnt[d]; cnt[d] = pos; pos += c; }
    for (int i = 0; i < m; i++) {
      uint32_t p = cnt[((~a[i]) >> sh) & 255]++;
      ta[p] = a[i];
      tb[p] = b[i];
    }
    uint32_t* s;
    s = a; a = ta; ta = s;
    s = b; b = tb; tb = s;
    where ^= 1;
  }
  return where;
}

/* OR of all 32 lanes by explicit rotate-and-OR (lowers to valign/vor). Default instead of
 * __builtin_reduce_or: on a plain 0/-1 compare mask, the builtin lowers (hexagon-clang 19) to a
 * vrmpy + vdeal/vdeal/vdeal/vshuff sequence that qemu 8.2 executes wrongly (misses single-lane
 * hits) while the real DSP gets it right -- TK_VEC_REDUCE_OR_MASK reproduces it. See README. */
#define TK_ROT(v, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, w, x, y, z, A, B, C, D, E, F, G) \
  __builtin_shufflevector(v, v, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, w, x, y, z, A, B, C, D, E, F, G)
static inline uint32_t tk_or_rot(tk_i32x32 m) {
  m |= TK_ROT(m, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  m |= TK_ROT(m, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7);
  m |= TK_ROT(m, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3);
  m |= TK_ROT(m, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1);
  m |= TK_ROT(m, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0);
  return (uint32_t)m[0];
}

enum { TK_SCALAR = 0, TK_VEC_REDUCE_OR = 1, TK_VEC_ROT = 2, TK_VEC_REDUCE_OR_MASK = 3 };

#ifdef __hexagon__
/* l2fetch box prefetch (Rtt = dir|stride|width|height, 16 bits each; same encoding as the SDK's
 * qhl_hvx hvx_internal.h helper and ../roialign_fast/roialign_kernel.h). Non-blocking. */
static inline void tk_l2fetch(const void* p, unsigned stride, unsigned width, unsigned height) {
  unsigned long long ctl = ((unsigned long long)stride << 32) | ((unsigned long long)width << 16) | height;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(ctl));
}
#else
static inline void tk_l2fetch(const void* p, unsigned stride, unsigned width, unsigned height) {
  (void)p; (void)stride; (void)width; (void)height;
}
#endif
#define TK_PF 4096 /* collect prefetches this many elements (16 KB) ahead, one l2fetch per block */

/* Survivors (key >= t) of x[i0, i1) in index order, written to ck/ci (room for i1 - i0 entries);
 * returns their count. i0 must be a multiple of 32 for the vector variants (alignment of x+i0).
 * variant: TK_VEC_ROT (default), TK_SCALAR, TK_VEC_REDUCE_OR (__builtin_reduce_or on the weighted
 * bitmask), TK_VEC_REDUCE_OR_MASK (__builtin_reduce_or on the plain 0/-1 mask as an any-test, then
 * rotate/OR for the bits -- the lowering qemu 8.2 gets wrong; kept only as the A/B reproducer). */
static int tk_collect_range(const float* x, int i0, int i1, uint32_t t, uint32_t* ck, uint32_t* ci, int variant) {
  int c = 0, i = i0;
  if (variant != TK_SCALAR) {
    const tk_i32x32 bitw = {1, 2, 4, 8, 16, 32, 64, 128, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15,
                            1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25,
                            1 << 26, 1 << 27, 1 << 28, 1 << 29, 1 << 30, (int32_t)0x80000000};
    const tk_u32x32 iota = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    tk_l2fetch(x + i, TK_PF * 4, TK_PF * 4, 1);
    for (; i + 32 <= i1; i += 32) {
      if (((i - i0) & (TK_PF - 1)) == 0 && i + TK_PF < i1) {
        int left = i1 - (i + TK_PF);
        tk_l2fetch(x + i + TK_PF, TK_PF * 4, (left < TK_PF ? left : TK_PF) * 4, 1);
      }
      tk_u32x32 u;
      memcpy(&u, x + i, 128);
      tk_u32x32 k = tk_keyv(u);
      tk_i32x32 cmp = (tk_i32x32)(k >= t), m = cmp & bitw;
      uint32_t bits;
      if (variant == TK_VEC_REDUCE_OR_MASK) bits = __builtin_reduce_or(cmp) ? tk_or_rot(m) : 0;
      else bits = variant == TK_VEC_ROT ? tk_or_rot(m) : (uint32_t)__builtin_reduce_or(m);
      if (bits == 0xFFFFFFFFu) {       /* every lane survives (always, when t == 0): whole vectors */
        tk_u32x32 iv = iota + (uint32_t)i;
        memcpy(ck + c, &k, 128);
        memcpy(ci + c, &iv, 128);
        c += 32;
        continue;
      }
      while (bits) {
        int l = __builtin_ctz(bits);
        bits &= bits - 1;
        ck[c] = k[l];
        ci[c] = (uint32_t)(i + l);
        c++;
      }
    }
  }
  for (; i < i1; i++) {
    uint32_t k = tk_key(x[i]);
    if (k >= t) { ck[c] = k; ci[c] = (uint32_t)i; c++; }
  }
  return c;
}

#define TK_SAMPLE 1024
#define TK_BLOCK 8                     /* contiguous elements per sample block (32 B) */
#define TK_BLOCKS (TK_SAMPLE / TK_BLOCK)

/* r-th largest (0-based) of a[0..m), permuting a. Quickselect with a 3-way (<, ==, >) partition,
 * so the heavy ties of this data (dequantized int8: tens of distinct values) cost nothing. */
static uint32_t tk_select_desc(uint32_t* a, int m, int r) {
  int lo = 0, hi = m - 1;
  while (lo < hi) {
    uint32_t p = a[lo + (hi - lo) / 2];
    int lt = lo, i = lo, gt = hi;               /* a[lo..lt) > p, a[lt..i) == p, a(gt..hi] < p */
    while (i <= gt) {
      uint32_t v = a[i];
      if (v > p) { a[i] = a[lt]; a[lt] = v; lt++; i++; }
      else if (v < p) { a[i] = a[gt]; a[gt] = v; gt--; }
      else i++;
    }
    if (r < lt) hi = lt - 1;
    else if (r > gt) lo = gt + 1;
    else return p;
  }
  return a[r];
}

/* Threshold from TK_BLOCKS evenly spaced blocks of TK_BLOCK contiguous elements (128 spatial
 * locations; 32 wider blocks were too spatially correlated and caused a retry pass on the real
 * data, and grouping blocks onto fewer pages made the estimate worse without making it faster).
 * sk (TK_SAMPLE entries) receives the sample keys (unsorted, kept for tk_retry); tmp is a
 * TK_SAMPLE-entry temporary. Returns t (0 = take everything) and sets *r (sample rank of t) and
 * *m (sample size, 0 if none). Only one rank is needed, so a quickselect replaces the full sort
 * (on the DSP, sorting the 1024-entry sample cost ~100 us -- as much as sorting the survivors). */
static uint32_t tk_threshold(const float* x, int n, int k, uint32_t* sk, uint32_t* tmp, int* r, int* m) {
  *r = 0;
  *m = 0;
  if (!(n > 4 * k && n > 4 * TK_SAMPLE)) return 0;
  int step = (n / TK_BLOCKS) & ~(TK_BLOCK - 1);
  if (step * 4L < 65536) tk_l2fetch(x, step * 4, TK_BLOCK * 4, TK_BLOCKS); /* all sample lines, one box */
  for (int b = 0; b < TK_BLOCKS; b++)
    for (int e = 0; e < TK_BLOCK; e++) sk[TK_BLOCK * b + e] = tk_key(x[(long)b * step + e]);
  /* Aim for ~1.5k survivors (sample is TK_SAMPLE/n of the input). Integer arithmetic: no
   * soft-float helpers needed on the DSP. */
  int rr = (int)((3L * k * TK_SAMPLE) / (2L * n));
  if (rr >= TK_SAMPLE) rr = TK_SAMPLE - 1;
  *r = rr;
  *m = TK_SAMPLE;
  memcpy(tmp, sk, TK_SAMPLE * 4);
  return tk_select_desc(tmp, TK_SAMPLE, rr);
}

/* Next, lower threshold after a collect found fewer than k survivors (sk: the unsorted sample,
 * tmp: TK_SAMPLE-entry temporary). */
static uint32_t tk_retry(const uint32_t* sk, uint32_t* tmp, int m, int* r) {
  if (*r >= m - 1) return 0;
  *r = 2 * *r + 1;
  if (*r >= m) *r = m - 1;
  memcpy(tmp, sk, m * 4);
  return tk_select_desc(tmp, m, *r);
}

static inline float tk_unkey(uint32_t k) {
  uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

#define TK_MAX_DISTINCT 256
#define TK_HASH 1024                   /* open-addressing slots (> 2 * TK_MAX_DISTINCT) */

/* Emit the first k of c survivors (ck/ci, index order) in (key desc, index asc) order. The real
 * inputs have few distinct keys (51..213 in the whole array), so instead of a 3-4-pass radix sort:
 * collect the distinct keys (small open-addressing hash set), sort those descending, then one
 * stable counting-sort pass over the survivors by the rank of their key -- stability keeps index
 * order within a key, which is exactly ORT's tie order. More than TK_MAX_DISTINCT distinct keys ->
 * stable LSD radix sort of all survivors. (A per-distinct-key HVX `==` scan was tried first: fine
 * for a handful of keys, but 2x slower than radix on the calls with ~90-200 distinct keys.)
 * Values come back from the keys (the map is invertible), not from a random-access gather of x --
 * except key 0x80000000, which +0 and -0 share, where x is read to keep -0's sign bit.
 * tk/ti: c-entry temporaries; hs: TK_EMIT_WORDS of scratch (hash keys, hash ranks, distinct keys,
 * per-rank counts). */
#define TK_EMIT_WORDS (2 * TK_HASH + 2 * (TK_MAX_DISTINCT + 1))
static void tk_emit(const float* x, uint32_t* ck, uint32_t* ci, uint32_t* tk, uint32_t* ti, uint32_t* hs, int c, int k, float* ov, int64_t* oi) {
  uint32_t *hr = hs + TK_HASH, *dk = hr + TK_HASH, *cnt = dk + TK_MAX_DISTINCT + 1;
  int nd = 0;
  memset(hs, 0, TK_HASH * 4);                   /* key 0 is never produced (it would be a NaN) */
  for (int i = 0; i < c && nd <= TK_MAX_DISTINCT; i++) {
    uint32_t v = ck[i], h = (v * 2654435761u) >> 22;
    while (hs[h] && hs[h] != v) h = (h + 1) & (TK_HASH - 1);
    if (!hs[h]) { hs[h] = v; dk[nd++] = v; }
    tk[i] = h;                                  /* remember the slot: no second lookup below */
  }
  if (nd > TK_MAX_DISTINCT) {
    int in_t = tk_radix_desc(ck, ci, tk, ti, c);
    const uint32_t *rk = in_t ? tk : ck, *ri = in_t ? ti : ci;
    for (int j = 0; j < k; j++) {
      ov[j] = rk[j] == 0x80000000u ? x[ri[j]] : tk_unkey(rk[j]);
      oi[j] = (int64_t)ri[j];
    }
    return;
  }
  for (int i = 1; i < nd; i++) {                /* insertion sort of the distinct keys, descending */
    uint32_t v = dk[i];
    int j = i - 1;
    while (j >= 0 && dk[j] < v) { dk[j + 1] = dk[j]; j--; }
    dk[j + 1] = v;
  }
  for (int r = 0; r < nd; r++) {                /* slot -> rank, and zero the per-rank counts */
    uint32_t h = (dk[r] * 2654435761u) >> 22;
    while (hs[h] != dk[r]) h = (h + 1) & (TK_HASH - 1);
    hr[h] = (uint32_t)r;
    cnt[r] = 0;
  }
  for (int i = 0; i < c; i++) { tk[i] = hr[tk[i]]; cnt[tk[i]]++; }
  uint32_t pos = 0;
  for (int r = 0; r < nd; r++) { uint32_t n_r = cnt[r]; cnt[r] = pos; pos += n_r; }
  for (int i = 0; i < c; i++) {                 /* stable scatter; only the first k are needed */
    uint32_t p = cnt[tk[i]]++;
    if (p < (uint32_t)k) ti[p] = ci[i];
  }
  int out = 0;
  for (int r = 0; r < nd && out < k; r++) {
    uint32_t v = dk[r], end = cnt[r] < (uint32_t)k ? cnt[r] : (uint32_t)k;
    float fv = tk_unkey(v);
    for (; (uint32_t)out < end; out++) {
      ov[out] = v == 0x80000000u ? x[ti[out]] : fv;
      oi[out] = (int64_t)ti[out];
    }
  }
}

static int tk_passes; /* collect passes in the last topk_desc call (1 unless a retry happened) */

/* Scratch layout (uint32 words): ck, ci, tk, ti (n each), sample + temporary (TK_SAMPLE each),
 * emit hash set + distinct list (TK_EMIT_WORDS). */
#define TK_SCRATCH_WORDS(n) (4L * (n) + 2L * TK_SAMPLE + TK_EMIT_WORDS)

/* Single-thread entry. Scratch: TK_SCRATCH_WORDS(n) uint32. Returns the survivor count, or -1 on
 * bad arguments. */
static int topk_desc(const float* x, int n, int k, float* ov, int64_t* oi, uint32_t* scratch, int variant) {
  if (k < 0 || k > n) return -1;
  uint32_t *ck = scratch, *ci = scratch + n, *tk = scratch + 2 * (long)n, *ti = scratch + 3 * (long)n;
  uint32_t *sk = scratch + 4 * (long)n, *tmp = sk + TK_SAMPLE, *hs = tmp + TK_SAMPLE;
  int r, m;
  uint32_t t = tk_threshold(x, n, k, sk, tmp, &r, &m);
  int c;
  tk_passes = 0;
  for (;;) {
    tk_passes++;
    c = tk_collect_range(x, 0, n, t, ck, ci, variant);
    if (c >= k || t == 0) break;
    t = tk_retry(sk, tmp, m, &r);
  }
  tk_emit(x, ck, ci, tk, ti, hs, c, k, ov, oi);
  return c;
}
#endif
