/* QDQ 3x3 convolution (pad 1, stride 1 or 2) on V69 HMX, requantized exactly as hmx_qconv.h (QC_FAST / QC_EXACT).
 *
 * Activations live in a *flat padded* crouton layout: pixel (y, x) of an H x W map is flat pixel
 *   p = M0 + (y + 1) * Wp + x,   Wp = roundup(W + 1, 4),  M0 = (-Wp) mod 64 (64 if that is 0)
 * i.e. rows of Wp pixels (columns W..Wp-1 are padding), one padding row above (y = -1) and below, and M0 margin
 * pixels in front so that output pixel o = y * Wp + x sits at flat pixel (M0 + Wp) + o with M0 + Wp a multiple of
 * 64: an output tile block is an input block of the next layer. Pixels are grouped 64 per crouton (the :cm row
 * dimension); crouton (block b, channel block kb) is at (b * kt + kb) * 2 KB, pixel row r at byte 32 * r.
 * Padding pixels hold the activation zero point (zx), so padding is exact (xq - zx = 0).
 *
 * A 3x3 tap (dy, dx) of output block ob reads the 64 input pixels starting at flat (M0 + Wp) + ob*64 + dy*Wp + dx.
 *  - dy*Wp is a multiple of 4 pixels: `activation.ub = mxmem(Rs, Rt):single:cm` reads a 64-row window starting
 *    4 * Rs[10:7] rows into the crouton at Rs and continuing into the crouton at Rs + Rt[31:11] (hexagon-sim map,
 *    full HMX rate); Rt[31:11] = kt * 2 KB = the next pixel block of the same channel block.
 *  - dx = -1 / +1 read one-pixel-shifted copies of the input (HVX valign by 32 bytes, 2 extra copies).
 * So a 3x3 conv is 9 * kt :single instructions per 64 x 64 output tile pair, with no im2col.
 * Stride 2: HVX splits the input into row/column phases (in the output's flat geometry): tap (dy, dx) reads
 * phase (dy != 0, dx != 0) at row offset -1 for dy = -1 and column offset -1 for dx = -1 (a shifted copy of the
 * odd-column phase), so the same stride-1 machinery runs on quarter-size maps. */
#ifndef HMX_QCONV3_H
#define HMX_QCONV3_H
#include "hmx_qconv.h"

typedef struct {
  int H, W, Wp, M0, nblk; /* nblk: pixel blocks of the whole flat buffer (incl. margins) */
} qc_geom_t;

static inline qc_geom_t qc_geom(int H, int W) {
  qc_geom_t g;
  g.H = H, g.W = W, g.Wp = (W + 1 + 3) & ~3;
  g.M0 = (64 - g.Wp % 64) % 64;
  if (g.M0 == 0) g.M0 = 64;
  int last = g.M0 + (H + 2) * g.Wp + 64; /* + a margin block for the one-pixel shifts and :single windows */
  g.nblk = (last + 63) / 64 + 1;
  return g;
}
static inline int qc_geom_pix(const qc_geom_t* g, int y, int x) { return g->M0 + (y + 1) * g->Wp + x; }
static inline size_t qc_geom_bytes(const qc_geom_t* g, int kt) { return (size_t)g->nblk * kt * 2048; }
/* first pixel block of the output region (o = 0) */
static inline int qc_geom_oblk(const qc_geom_t* g) { return (g->M0 + g->Wp) / 64; }
static inline int qc_geom_nob(const qc_geom_t* g) { return (g->H * g->Wp + 63) / 64; }

/* host: NHWC uint8 [H, W, C] -> flat padded crouton buffer (zx everywhere else) */
static inline void qc_flat_pack(const uint8_t* x, int H, int W, int C, int zx, const qc_geom_t* g, uint8_t* out) {
  int kt = C / 32;
  memset(out, zx, qc_geom_bytes(g, kt));
  for (int y = 0; y < H; y++)
    for (int xx = 0; xx < W; xx++) {
      int p = qc_geom_pix(g, y, xx);
      for (int kb = 0; kb < kt; kb++) memcpy(out + ((size_t)(p / 64) * kt + kb) * 2048 + 32 * (p % 64), x + ((size_t)y * W + xx) * C + 32 * kb, 32);
    }
}
/* host: output region (flat o = y*Wp + x from block qc_geom_oblk) -> NHWC-rows [H*W, N] */
static inline void qc_flat_unpack(const uint8_t* buf, const qc_geom_t* g, int N, uint8_t* y) {
  int nt = N / 32;
  for (int yy = 0; yy < g->H; yy++)
    for (int x = 0; x < g->W; x++) {
      int p = qc_geom_pix(g, yy, x);
      for (int j = 0; j < nt; j++) memcpy(y + ((size_t)yy * g->W + x) * N + 32 * j, buf + ((size_t)(p / 64) * nt + j) * 2048 + 32 * (p % 64), 32);
    }
}

/* host: ONNX weights [N, C, 3, 3] -> tap-major k-major [9*C, N] (for qc_pack_params) and the HMX packing
 * [N/64 groups][9 taps][kt][2 KB :deep block] */
static inline void qc_pack_w3(const int8_t* w, int N, int C, int8_t* wk, int8_t* wp) {
  for (int t = 0; t < 9; t++)
    for (int c = 0; c < C; c++)
      for (int n = 0; n < N; n++) wk[((size_t)t * C + c) * N + n] = w[((size_t)n * C + c) * 9 + t];
  int kt = C / 32;
  for (int g = 0; g < N / 64; g++)
    for (int t = 0; t < 9; t++)
      for (int kb = 0; kb < kt; kb++) {
        int8_t* d = wp + (((size_t)g * 9 + t) * kt + kb) * 2048;
        for (int h = 0; h < 2; h++)
          for (int k = 0; k < 32; k++)
            for (int cc = 0; cc < 32; cc++)
              d[1024 * h + 128 * (k / 4) + 4 * cc + k % 4] = wk[((size_t)t * C + 32 * kb + k) * N + 64 * g + 32 * h + cc];
      }
}

#ifdef __hexagon__
static inline void qc_fill(uint8_t* buf, size_t n, int v) {
  HVX_Vector z = Q6_Vb_vsplat_R(v);
  for (size_t i = 0; i < n; i += 128) *(HVX_Vector*)(buf + i) = z;
}

/* one-pixel shifted copies of a flat buffer: m1[p] = in[p - 1], p1[p] = in[p + 1] (zx beyond the ends). Per channel
 * block the pixel blocks form one stream of vectors (4 pixels each) at stride kt * 2 KB between blocks. */
static inline void qc_shift_copies(const uint8_t* __restrict in, uint8_t* __restrict m1, uint8_t* __restrict p1, int nblk, int kt,
                                   int zx) {
  HVX_Vector z = Q6_Vb_vsplat_R(zx);
  size_t bs = (size_t)kt * 2048;
  for (int kb = 0; kb < kt; kb++) {
    HVX_Vector prev = z;
    for (int b = 0; b < nblk; b++) {
      const HVX_Vector* v = (const HVX_Vector*)(in + b * bs + (size_t)kb * 2048);
      HVX_Vector* om = (HVX_Vector*)(m1 + b * bs + (size_t)kb * 2048);
      HVX_Vector* op = (HVX_Vector*)(p1 + b * bs + (size_t)kb * 2048);
      HVX_Vector next = b + 1 < nblk ? *(const HVX_Vector*)(in + (b + 1) * bs + (size_t)kb * 2048) : z;
      HVX_Vector cur = v[0];
#pragma unroll 4
      for (int j = 0; j < 15; j++) {
        HVX_Vector nv = v[j + 1];
        om[j] = Q6_V_vlalign_VVR(cur, prev, 32);
        op[j] = Q6_V_valign_VVR(nv, cur, 32);
        prev = cur, cur = nv;
      }
      om[15] = Q6_V_vlalign_VVR(cur, prev, 32);
      op[15] = Q6_V_valign_VVR(next, cur, 32);
      prev = cur;
    }
  }
}

/* stride-2 phase split: out[ph] (ph = 2*py + px, flat buffers in the output geometry go) holds input pixel
 * (2i + py, 2j + px) at output position (i, j); rows i = -1 and columns j >= Wo are zx (so is any input pixel
 * outside the map). gi: input geometry. 4 pixels (one vector) at a time: vdeal by 32 bytes splits a pair of
 * vectors (8 consecutive input pixels) into even and odd pixels. */
static inline void qc_phase_split(const uint8_t* in, const qc_geom_t* gi, uint8_t* const out[4], const qc_geom_t* go, int kt,
                                  int zx) {
  HVX_Vector z = Q6_Vb_vsplat_R(zx);
  for (int ph = 0; ph < 4; ph++) qc_fill(out[ph], qc_geom_bytes(go, kt), zx);
  for (int py = 0; py < 2; py++)
    for (int i = 0; i < go->H; i++) {
      int r = 2 * i + py;
      if (r >= gi->H) continue; /* stays zx */
      int p0 = qc_geom_pix(gi, r, 0), q0 = qc_geom_pix(go, i, 0);
      for (int j = 0; j < go->W; j += 4) {
        int p = p0 + 2 * j, q = q0 + j; /* p, q multiples of 4 */
        /* pixels j + n >= Wo are padding: keep zx there (the last vector of a row) */
        HVX_VectorPred keep = Q6_Q_vsetq2_R(j + 4 <= go->W ? 128 : 32 * (go->W - j));
        for (int kb = 0; kb < kt; kb++) {
          HVX_Vector a = *(const HVX_Vector*)(in + ((size_t)(p / 64) * kt + kb) * 2048 + 32 * (p % 64));
          HVX_Vector b = *(const HVX_Vector*)(in + ((size_t)((p + 4) / 64) * kt + kb) * 2048 + 32 * ((p + 4) % 64));
          HVX_VectorPair d = Q6_W_vdeal_VVR(b, a, -32); /* lo: pixels 0,2,4,6; hi: 1,3,5,7 */
          size_t o = ((size_t)(q / 64) * kt + kb) * 2048 + 32 * (q % 64);
          *(HVX_Vector*)(out[2 * py] + o) = Q6_V_vmux_QVV(keep, Q6_V_lo_W(d), z);
          *(HVX_Vector*)(out[2 * py + 1] + o) = Q6_V_vmux_QVV(keep, Q6_V_hi_W(d), z);
        }
      }
    }
}

/* tap sources for one 3x3 conv in the output geometry go: src[t] = buffer, drow[t] = row offset (-1/0/+1) */
typedef struct {
  const uint8_t* src[9];
  int drow[9];
} qc_taps_t;

/* The HMX faults when one instruction's operands span a 256 KB VTCM boundary (phone only; hexagon-sim does not
 * model it). A :single window whose two croutons (kt * 2 KB apart) straddle a boundary is therefore stitched into a
 * side crouton by HVX (16 vector moves: window offsets are multiples of 4 rows = 128 bytes) and read with a
 * one-crouton instruction. qc_conv3x3_plan (once per layer and buffer set) fills atab[(ob * 9 + t) * kt + kb] with
 * the Rs of every instruction (offset bits included; offset 0 = no second crouton) and stitch[] with the straddling
 * windows (src Rs, side crouton); returns their number, or -1 if side_cap is too small. qc_conv3x3_stitch copies
 * them (every run, after the tap sources are written). */
typedef struct {
  uint32_t rs;         /* the original window (address | offset bits) */
  uint8_t* side;       /* its side crouton */
} qc_stitch_t;

static inline int qc_conv3x3_plan(const qc_taps_t* tp, const qc_geom_t* go, int kt, uint32_t* atab, qc_stitch_t* stitch,
                                  uint8_t* side, int side_cap) {
  int ob0 = qc_geom_oblk(go), nob = qc_geom_nob(go), ns = 0;
  for (int t = 0; t < 9; t++) {
    int s = tp->drow[t] * go->Wp; /* pixels, multiple of 4, may be negative */
    int sb = s >= 0 ? s / 64 : -((-s + 63) / 64), off = s - 64 * sb, o4 = off / 4;
    for (int ob = 0; ob < nob; ob++)
      for (int kb = 0; kb < kt; kb++) {
        const uint8_t* a0 = tp->src[t] + ((size_t)(ob0 + sb + ob) * kt + kb) * 2048;
        const uint8_t* a1 = a0 + (size_t)kt * 2048;
        uint32_t rs = (uint32_t)(uintptr_t)a0 | ((uint32_t)o4 << 7);
#ifndef QC_STITCH_ALL /* test hook: stitch every offset window */
        if (o4 && ((uintptr_t)a0 >> 18) != ((uintptr_t)a1 >> 18)) {
#else
        if (o4) {
#endif
          if (ns >= side_cap) return -1;
          stitch[ns].rs = rs, stitch[ns].side = side + (size_t)ns * 2048;
          rs = (uint32_t)(uintptr_t)stitch[ns].side, ns++;
        }
        atab[((size_t)ob * 9 + t) * kt + kb] = rs;
      }
  }
  return ns;
}
static inline void qc_conv3x3_stitch(const qc_stitch_t* st, int ns, int kt) {
  for (int i = 0; i < ns; i++) {
    int o4 = (st[i].rs >> 7) & 15;
    const HVX_Vector* v0 = (const HVX_Vector*)(uintptr_t)(st[i].rs & ~2047u);
    const HVX_Vector* v1 = (const HVX_Vector*)((uintptr_t)(st[i].rs & ~2047u) + (size_t)kt * 2048);
    HVX_Vector* d = (HVX_Vector*)st[i].side;
    for (int j = 0; j < 16; j++) d[j] = j + o4 < 16 ? v0[j + o4] : v1[j + o4 - 16];
  }
}

/* 3x3 conv: Y = output flat buffer in geometry go (tiles for blocks oblk .. oblk+nob-1; the rest of Y untouched),
 * W3 from qc_pack_w3, blk/h from qc_pack_params(K = 9*C), atab from qc_conv3x3_prep. */
static inline int qc_conv3x3(const uint32_t* atab, const qc_geom_t* go, uint8_t* Y, const uint8_t* W3, const qc_blk_t* blk,
                             const qc_hdr_t* h, int kt, int mode, uint8_t* scratch) {
  int nt = h->n / 32, nfix = 0, ob0 = qc_geom_oblk(go), nob = qc_geom_nob(go);
  unsigned rt2 = ((unsigned)kt * 2048) | 0x7ff;
  for (int g = 0; g < h->n / 64; g++)
    for (int ob = 0; ob < nob; ob++) {
      const uint8_t* w = W3 + (size_t)g * 9 * kt * 2048;
      const uint32_t* at = atab + (size_t)ob * 9 * kt;
      for (int i = 0; i < 9 * kt; i++, w += 2048) {
        uint32_t rs = at[i];
        __asm__ volatile("{ activation.ub = mxmem(%0,%1):single:cm\n weight.b = mxmem(%2,%3):deep }" ::"r"(rs),
                         "r"(rs & 0x780 ? rt2 : 0x7ffu), "r"(w), "r"(0x7ff)
                         : "memory");
      }
      for (int hh = 0; hh < 2; hh++) {
        const qc_blk_t* b = &blk[2 * g + hh];
        uint8_t* yt = Y + ((size_t)(ob0 + ob) * nt + 2 * g + hh) * 2048;
        if (mode == QC_FAST) {
          hmx_blk_set_table2(b->tbl_fast);
          hmx_blk_store_u8cm(yt);
        } else {
          for (int p = 0; p < 4; p++) {
            hmx_blk_set_table2(b->tbl_plane[p]);
            if (p < 3)
              __asm__ volatile("mxmem(%0,%1):after:retain:cm.ub = acc" ::"r"(scratch + 2048 * p), "r"(0) : "memory");
            else
              __asm__ volatile("mxmem(%0,%1):after:cm.ub = acc" ::"r"(scratch + 2048 * p), "r"(0) : "memory");
          }
          nfix += qc_requant_tile(scratch, yt, b, h);
        }
      }
    }
  if (mode == QC_FAST && h->relu && h->lo > 0) {
    HVX_Vector lo = Q6_Vb_vsplat_R(h->lo);
    for (size_t i = (size_t)ob0 * nt * 2048; i < (size_t)(ob0 + nob) * nt * 2048; i += 128)
      *(HVX_Vector*)(Y + i) = Q6_Vub_vmax_VubVub(*(HVX_Vector*)(Y + i), lo);
  }
  return nfix;
}

/* taps of a stride-1 conv: X = input, xm1/xp1 = qc_shift_copies of it (all in geometry go = the input's) */
static inline qc_taps_t qc_taps_s1(const uint8_t* X, const uint8_t* xm1, const uint8_t* xp1) {
  qc_taps_t tp;
  for (int t = 0; t < 9; t++) tp.src[t] = t % 3 == 0 ? xm1 : t % 3 == 1 ? X : xp1, tp.drow[t] = t / 3 - 1;
  return tp;
}
/* taps of a stride-2 conv from qc_phase_split's ph[4] and ph1m1 = the m1 shift of ph[1] (odd columns, x - 1) and
 * ph3m1 = the m1 shift of ph[3] */
static inline qc_taps_t qc_taps_s2(uint8_t* const ph[4], const uint8_t* ph1m1, const uint8_t* ph3m1) {
  qc_taps_t tp;
  for (int t = 0; t < 9; t++) {
    int dy = t / 3 - 1, dx = t % 3 - 1, py = dy != 0, px = dx != 0;
    const uint8_t* s = ph[2 * py + px];
    if (dx == -1) s = py ? ph3m1 : ph1m1;
    tp.src[t] = s, tp.drow[t] = dy == -1 ? -1 : 0;
  }
  return tp;
}
#endif /* __hexagon__ */
#endif
