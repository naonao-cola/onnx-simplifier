/* HMX (Hexagon matrix unit) GEMM for V69, from an unsigned FastRPC skel or hexagon-sim.
 *
 * Layouts (derived on hexagon-sim -mv69 --mhmx 1, confirmed on the Xiaomi 12S; see README):
 *   fp16 32x32 tile, element (i, j) at halfword 64*(i/2) + 2*j + i%2 -- the same "2x1" interleave for the
 *   activation A(r, k), the weight W(k, c) and the output C(r, c). K > 32 = consecutive 2 KB tiles for
 *   both operands, streamed by one `activation.hf = mxmem(A, Rt):deep; weight.hf = mxmem(W, Rt)` with
 *   Rt = (K/32)*2048 - 1; accumulation is exact, rounded to fp16 once at `mxmem(C, 0):after.hf = acc`.
 *   Output column table (`bias = mxmem(T)`, 256 B): word c (c < 32) high 16 bits = fp16 bias of column c.
 *
 * Callers must already hold HVX + HMX (HAP_power_set_HMX power_up, HAP_compute_res with hmx param,
 * HAP_compute_res_hmx_lock on this thread); `vtcm` must be VTCM (HMX only reads/writes VTCM). */
#ifndef HMX_GEMM_H
#define HMX_GEMM_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define HMX_TILE 32
#define HMX_TILE_BYTES 2048
#define HMX_IDX(i, j) (64 * ((i) / 2) + 2 * (j) + ((i) % 2))

/* Pack a K x N row-major fp16 weight into tile-major blocks: for each 32-column block nb, K/32 tiles
 * of W(k, c) (host or DSP; done once per weight). out holds (N/32) * (K/32) * 1024 halfwords. */
static inline void hmx_pack_w_f16(const uint16_t* W, int K, int N, uint16_t* out) {
  for (int nb = 0; nb < N / 32; nb++)
    for (int kb = 0; kb < K / 32; kb++) {
      uint16_t* t = out + ((size_t)nb * (K / 32) + kb) * 1024;
      for (int k = 0; k < 32; k++)
        for (int c = 0; c < 32; c++) t[HMX_IDX(k, c)] = W[(size_t)(kb * 32 + k) * N + nb * 32 + c];
    }
}

/* One row block of A (rows m0..m0+31, zero past M) into K/32 activation tiles. */
static inline void hmx_pack_a_f16(const uint16_t* A, int M, int K, int m0, uint16_t* out) {
  for (int kb = 0; kb < K / 32; kb++) {
    uint16_t* t = out + (size_t)kb * 1024;
    for (int r = 0; r < 32; r++) {
      const uint16_t* row = m0 + r < M ? A + (size_t)(m0 + r) * K + kb * 32 : NULL;
      for (int k = 0; k < 32; k++) t[HMX_IDX(r, k)] = row ? row[k] : 0;
    }
  }
}

#ifdef __hexagon__
/* One :deep load pair covers at most 32 K-tiles (K <= 1024 fp16, Rt <= 65535; K = 1056 is already wrong
 * on hexagon-sim), so longer K is issued as consecutive <= 32-tile load pairs that keep accumulating
 * until the store. */
#define HMX_MAX_KTILES 32
static inline void hmx_mac_f16(const void* a, const void* w, int ktiles) {
  const uint8_t *pa = (const uint8_t*)a, *pw = (const uint8_t*)w;
  for (int k0 = 0; k0 < ktiles; k0 += HMX_MAX_KTILES) {
    int n = ktiles - k0 < HMX_MAX_KTILES ? ktiles - k0 : HMX_MAX_KTILES, lim = n * HMX_TILE_BYTES - 1;
    __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" ::"r"(pa + (size_t)k0 * HMX_TILE_BYTES),
                     "r"(lim), "r"(pw + (size_t)k0 * HMX_TILE_BYTES), "r"(lim)
                     : "memory");
  }
}
static inline void hmx_store_f16(void* c) { __asm__ volatile("mxmem(%0,%1):after.hf = acc" ::"r"(c), "r"(0) : "memory"); }
static inline void hmx_set_table(const void* t) { __asm__ volatile("bias = mxmem(%0)" ::"r"(t) : "memory"); }

/* On the phone an HMX operand span (A or W of one mxmem, Rt+1 bytes) must not cross a 256 KB VTCM
 * boundary: such a load takes a user-PD page fault at the boundary (hexagon-sim does not model this).
 * hmx_valloc places each span inside one 256 KB window; one span is at most 256 KB (K <= 4096 fp16). */
#define HMX_VTCM_WINDOW (256 * 1024)
static inline size_t hmx_valloc(size_t* off, size_t n) {
  if (*off % HMX_VTCM_WINDOW + n > HMX_VTCM_WINDOW) *off = (*off + HMX_VTCM_WINDOW - 1) / HMX_VTCM_WINDOW * HMX_VTCM_WINDOW;
  size_t at = *off;
  *off += (n + 127) & ~(size_t)127;
  return at;
}

/* VTCM layout of hmx_gemm_f16 for M, K: returns the bytes needed and fills the offsets (may be NULL). */
static inline size_t hmx_gemm_f16_layout(int M, int K, size_t* a_off, size_t* w_off, size_t* c_off) {
  size_t mt = (M + 31) / 32, span = (size_t)(K / 32) * HMX_TILE_BYTES, off = 0;
  for (size_t mb = 0; mb < mt; mb++) {
    size_t at = hmx_valloc(&off, span);
    if (a_off) a_off[mb] = at;
  }
  for (int h = 0; h < 2; h++) {
    size_t at = hmx_valloc(&off, span);
    if (w_off) w_off[h] = at;
  }
  size_t at = hmx_valloc(&off, 2 * HMX_TILE_BYTES + 512);
  if (c_off) *c_off = at;
  return off;
}
static inline size_t hmx_gemm_f16_vtcm(int M, int K) { return hmx_gemm_f16_layout(M, K, NULL, NULL, NULL); }

#include <hexagon_types.h>
#include <hexagon_protos.h>
typedef HVX_Vector __attribute__((aligned(1))) hmx_uvec;

/* HVX row-block pack: rows 2i, 2i+1 of A (K fp16 each) -> the row-pair vector of every K block at once.
 * vshuff(row1, row0, -2) interleaves halfwords: lo = K block 2j (row0 k, row1 k, ...), hi = block 2j+1. */
static inline void hmx_pack_a_f16_hvx(const uint16_t* A, int M, int K, int m0, uint16_t* out) {
  int kt = K / 32;
  HVX_Vector z = Q6_V_vzero();
  for (int p = 0; p < 16; p++) {
    int r0 = m0 + 2 * p, r1 = r0 + 1;
    const uint16_t* a0 = r0 < M ? A + (size_t)r0 * K : NULL;
    const uint16_t* a1 = r1 < M ? A + (size_t)r1 * K : NULL;
    for (int j = 0; j < kt; j += 2) {
      HVX_Vector v0 = a0 ? *(const hmx_uvec*)(a0 + 32 * j) : z, v1 = a1 ? *(const hmx_uvec*)(a1 + 32 * j) : z;
      HVX_VectorPair s = Q6_W_vshuff_VVR(v1, v0, -2);
      *(HVX_Vector*)(out + (size_t)j * 1024 + 64 * p) = Q6_V_lo_W(s);
      if (j + 1 < kt) *(HVX_Vector*)(out + (size_t)(j + 1) * 1024 + 64 * p) = Q6_V_hi_W(s);
    }
  }
}

/* Background DDR -> L2 prefetch of n linear bytes (l2fetch box: 128-byte rows, <= 255 rows per op). */
static inline void hmx_l2fetch(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t off = 0; off < n; off += 255 * 128) {
    size_t rows = (n - off + 127) / 128;
    if (rows > 255) rows = 255;
    unsigned rt = (128u << 16) | (128u << 8) | (unsigned)rows;
    __asm__ volatile("l2fetch(%0, %1)" ::"r"(b + off), "r"(rt));
  }
}

/* HVX copy (dst VTCM, 128-aligned; src any alignment), n a multiple of 128. */
static inline void hmx_copy_hvx(void* dst, const void* src, size_t n) {
  HVX_Vector* d = (HVX_Vector*)dst;
  const hmx_uvec* s = (const hmx_uvec*)src;
  const size_t chunk = 16 * 1024 / 128; /* vectors per prefetch chunk; prefetch runs 2 chunks ahead */
  size_t nv = n / 128;
  hmx_l2fetch(src, n < 2 * chunk * 128 ? n : 2 * chunk * 128);
  for (size_t i = 0; i < nv; i += chunk) {
    if (i + 2 * chunk < nv) hmx_l2fetch(s + i + 2 * chunk, (nv - i - 2 * chunk < chunk ? nv - i - 2 * chunk : chunk) * 128);
    size_t e = i + chunk < nv ? i + chunk : nv;
    for (size_t k = i; k < e; k++) d[k] = s[k];
  }
}

/* Unpack two adjacent 32x32 output tiles (column blocks nb, nb+1) into 64 columns of rows m0..: each tile
 * row-pair vector deals (vdeal h) into [row 2p | row 2p+1] halves; vmux + vror pair the halves of both tiles. */
static inline void hmx_unpack2_f16_hvx(const uint16_t* t0, const uint16_t* t1, uint16_t* C, int ldc, int rows) {
  HVX_VectorPred lo64 = Q6_Q_vsetq_R(64);
  for (int p = 0; p < 16 && 2 * p < rows; p++) {
    HVX_Vector x = Q6_Vh_vdeal_Vh(*(const HVX_Vector*)(t0 + 64 * p));
    HVX_Vector y = Q6_Vh_vdeal_Vh(*(const HVX_Vector*)(t1 + 64 * p));
    HVX_Vector r0 = Q6_V_vmux_QVV(lo64, x, Q6_V_vror_VR(y, 64)), r1 = Q6_V_vmux_QVV(lo64, Q6_V_vror_VR(x, 64), y);
    uint16_t *c0 = C + (size_t)(2 * p) * ldc, *c1 = c0 + ldc;
    if (!(((uintptr_t)c0 | (size_t)ldc * 2) & 127)) {
      *(HVX_Vector*)c0 = r0;
      if (2 * p + 1 < rows) *(HVX_Vector*)c1 = r1;
    } else {
      *(hmx_uvec*)c0 = r0;
      if (2 * p + 1 < rows) *(hmx_uvec*)c1 = r1;
    }
  }
}

/* C[M, N] = A[M, K] . W[K, N] (+ bias[N]), fp16 in/out, row-major A and C, Wp from hmx_pack_w_f16.
 * K a multiple of 32, N a multiple of 64; M any. Returns 0, or -1 if vtcm is too small. */
#ifndef HMX_NOW
#define HMX_NOW() 0ull
#endif
/* prof (may be NULL): += pack A, weight copy, MAC + store, C unpack (HMX_NOW units). */
static inline int hmx_gemm_f16_prof(const uint16_t* A, const uint16_t* Wp, const uint16_t* bias, uint16_t* C, int M,
                                    int K, int N, uint8_t* vtcm, size_t vtcm_bytes, unsigned long long* prof) {
  unsigned long long pr[4] = {0, 0, 0, 0}, t0 = HMX_NOW(), t1;
  int mt = (M + 31) / 32, kt = K / 32, nt = N / 32;
  size_t a_off[128], w_off[2], c_off;
  if (mt > 128 || K > 4096 || N % 64 || hmx_gemm_f16_layout(M, K, a_off, w_off, &c_off) > vtcm_bytes) return -1;
  uint16_t* ct = (uint16_t*)(vtcm + c_off);
  uint32_t* tbl = (uint32_t*)((uint8_t*)ct + 2 * HMX_TILE_BYTES);
  hmx_l2fetch(A, (size_t)(M < 32 ? M : 32) * K * 2);
  for (int mb = 0; mb < mt; mb++) {
    if (mb + 1 < mt) hmx_l2fetch(A + (size_t)(mb + 1) * 32 * K, (size_t)((M - (mb + 1) * 32) < 32 ? M - (mb + 1) * 32 : 32) * K * 2);
    hmx_pack_a_f16_hvx(A, M, K, mb * 32, (uint16_t*)(vtcm + a_off[mb]));
  }
  t1 = HMX_NOW(), pr[0] += t1 - t0, t0 = t1;
  for (int nb = 0; nb < nt; nb += 2) {
    for (int h = 0; h < 2; h++)
      hmx_copy_hvx(vtcm + w_off[h], Wp + (size_t)(nb + h) * kt * 1024, (size_t)kt * HMX_TILE_BYTES);
    t1 = HMX_NOW(), pr[1] += t1 - t0, t0 = t1;
    for (int h = 0; h < 2; h++) {
      for (int c = 0; c < 32; c++) tbl[32 * h + c] = bias ? (uint32_t)bias[(nb + h) * 32 + c] << 16 : 0;
    }
    for (int mb = 0; mb < mt; mb++) {
      for (int h = 0; h < 2; h++) {
        hmx_set_table(tbl + 32 * h);
        hmx_mac_f16(vtcm + a_off[mb], vtcm + w_off[h], kt);
        hmx_store_f16(ct + (size_t)h * 1024);
      }
      t1 = HMX_NOW(), pr[2] += t1 - t0, t0 = t1;
      int rows = M - mb * 32 < 32 ? M - mb * 32 : 32;
      hmx_unpack2_f16_hvx(ct, ct + 1024, C + (size_t)mb * 32 * N + nb * 32, N, rows);
      t1 = HMX_NOW(), pr[3] += t1 - t0, t0 = t1;
    }
  }
  if (prof)
    for (int i = 0; i < 4; i++) prof[i] += pr[i];
  return 0;
}
static inline int hmx_gemm_f16(const uint16_t* A, const uint16_t* Wp, const uint16_t* bias, uint16_t* C, int M, int K,
                               int N, uint8_t* vtcm, size_t vtcm_bytes) {
  return hmx_gemm_f16_prof(A, Wp, bias, C, M, K, N, vtcm, vtcm_bytes, NULL);
}
#endif /* __hexagon__ */
#endif
