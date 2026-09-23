/* Fast-BEV M0's view transform as a byte gather, straight into the BEV net's uint8 NHWC input.
 *
 * For every voxel row r (200 x 200 x 4 voxels in (x, y, z) order) and time step t (4):
 *   out[(r * 4 + t) * 64 .. +64] = table_t[min(lut_t[r], rows) * 64 .. +64]
 * i.e. the (1, 200, 200, 1024) volume with channel z*256 + t*64 + c (Fast-BEV's space-to-channel
 * order): each t's LUT row (geometry.m0_lut) picks one camera pixel's 64-channel uint8 feature row,
 * or row `rows` -- the zero-point row the host keeps after each table ("no camera").
 * Tables are (rows + 1) * 64 bytes plus 64 bytes of padding (the HVX body loads 128 bytes per row);
 * out must be 128-byte aligned (each r writes 256 bytes, two aligned vectors).
 *
 * Header-only, two bodies producing identical bytes: plain C (host / qemu reference) and 128-byte
 * HVX: two unaligned row loads, a rotate and a mux per output vector (vmux(q_first64, rowA,
 * vror(rowB, 64)) = [rowA[0:64] | rowB[0:64]]). */
#ifndef FBGATHER_KERNEL_H
#define FBGATHER_KERNEL_H

#include <stdint.h>

#define FBG_T 4
#define FBG_C 64

static inline int32_t fbg_clamp(int32_t i, int32_t rows) { return (uint32_t)i > (uint32_t)rows ? rows : i; }

static inline void fbg_run_c(const uint8_t* const tab[FBG_T], const int32_t* const lut[FBG_T], int32_t rows,
                             int32_t r0, int32_t r1, uint8_t* out) {
  for (int32_t r = r0; r < r1; r++)
    for (int t = 0; t < FBG_T; t++)
      __builtin_memcpy(out + ((long)r * FBG_T + t) * FBG_C, tab[t] + (long)fbg_clamp(lut[t][r], rows) * FBG_C, FBG_C);
}

#if defined(__hexagon__) && defined(__HVX__)
#include <hexagon_types.h>
#include <hexagon_protos.h>
typedef long HVX_UVector __attribute__((__vector_size__(128), __aligned__(1)));

static inline void fbg_run_hvx(const uint8_t* const tab[FBG_T], const int32_t* const lut[FBG_T], int32_t rows,
                               int32_t r0, int32_t r1, uint8_t* out) {
  const HVX_VectorPred lo = Q6_Q_vsetq_R(64);
  HVX_Vector* o = (HVX_Vector*)(out + (long)r0 * FBG_T * FBG_C);
  const uint8_t *t0 = tab[0], *t1 = tab[1], *t2 = tab[2], *t3 = tab[3];
  const int32_t *l0 = lut[0], *l1 = lut[1], *l2 = lut[2], *l3 = lut[3];
  for (int32_t r = r0; r < r1; r++) {
    HVX_Vector a = *(const HVX_UVector*)(t0 + (long)fbg_clamp(l0[r], rows) * FBG_C);
    HVX_Vector b = *(const HVX_UVector*)(t1 + (long)fbg_clamp(l1[r], rows) * FBG_C);
    HVX_Vector c = *(const HVX_UVector*)(t2 + (long)fbg_clamp(l2[r], rows) * FBG_C);
    HVX_Vector d = *(const HVX_UVector*)(t3 + (long)fbg_clamp(l3[r], rows) * FBG_C);
    *o++ = Q6_V_vmux_QVV(lo, a, Q6_V_vror_VR(b, 64));
    *o++ = Q6_V_vmux_QVV(lo, c, Q6_V_vror_VR(d, 64));
  }
}
#define fbg_run fbg_run_hvx
#else
#define fbg_run fbg_run_c
#endif

#endif
