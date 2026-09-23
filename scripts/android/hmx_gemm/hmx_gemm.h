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

static inline void hmx_mac_f16(const void* a, const void* w, int ktiles) {
  int lim = ktiles * HMX_TILE_BYTES - 1;
  __asm__ volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%3) }" ::"r"(a), "r"(lim), "r"(w), "r"(lim)
                   : "memory");
}
static inline void hmx_store_f16(void* c) { __asm__ volatile("mxmem(%0,%1):after.hf = acc" ::"r"(c), "r"(0) : "memory"); }
static inline void hmx_set_table(const void* t) { __asm__ volatile("bias = mxmem(%0)" ::"r"(t) : "memory"); }

/* VTCM bytes hmx_gemm_f16 needs for M, K. */
static inline size_t hmx_gemm_f16_vtcm(int M, int K) {
  size_t mt = (M + 31) / 32, kt = K / 32;
  return mt * kt * HMX_TILE_BYTES + 2 * kt * HMX_TILE_BYTES + HMX_TILE_BYTES + 256;
}

/* C[M, N] = A[M, K] . W[K, N] (+ bias[N]), fp16 in/out, row-major A and C, Wp from hmx_pack_w_f16.
 * K and N must be multiples of 32; M any. Returns 0, or -1 if vtcm is too small. */
static inline int hmx_gemm_f16(const uint16_t* A, const uint16_t* Wp, const uint16_t* bias, uint16_t* C, int M, int K,
                               int N, uint8_t* vtcm, size_t vtcm_bytes) {
  int mt = (M + 31) / 32, kt = K / 32, nt = N / 32;
  if (hmx_gemm_f16_vtcm(M, K) > vtcm_bytes) return -1;
  uint8_t* ap = vtcm;
  uint8_t* wb = ap + (size_t)mt * kt * HMX_TILE_BYTES;
  uint16_t* ct = (uint16_t*)(wb + 2 * (size_t)kt * HMX_TILE_BYTES);
  uint32_t* tbl = (uint32_t*)((uint8_t*)ct + HMX_TILE_BYTES);
  for (int mb = 0; mb < mt; mb++) hmx_pack_a_f16(A, M, K, mb * 32, (uint16_t*)(ap + (size_t)mb * kt * HMX_TILE_BYTES));
  memset(tbl, 0, 256);
  for (int nb = 0; nb < nt; nb++) {
    uint8_t* w = wb + (size_t)(nb & 1) * kt * HMX_TILE_BYTES;
    memcpy(w, Wp + (size_t)nb * kt * 1024, (size_t)kt * HMX_TILE_BYTES);
    for (int c = 0; c < 32; c++) tbl[c] = bias ? (uint32_t)bias[nb * 32 + c] << 16 : 0;
    hmx_set_table(tbl);
    for (int mb = 0; mb < mt; mb++) {
      hmx_mac_f16(ap + (size_t)mb * kt * HMX_TILE_BYTES, w, kt);
      hmx_store_f16(ct);
      int rows = M - mb * 32 < 32 ? M - mb * 32 : 32;
      for (int r = 0; r < rows; r++) {
        uint16_t* crow = C + (size_t)(mb * 32 + r) * N + nb * 32;
        for (int c = 0; c < 32; c++) crow[c] = ct[HMX_IDX(r, c)];
      }
    }
  }
  return 0;
}
#endif
