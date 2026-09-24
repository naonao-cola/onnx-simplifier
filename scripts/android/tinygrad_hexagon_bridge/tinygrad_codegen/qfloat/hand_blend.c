/* Hand-written qf32 4-tap blend for comparison with the plain-tinygrad one: out[m,:] = sum_t tap_t[m,:] * w[m,t],
 * M=4096, C=256. Products stay qf32 and are summed as qf32 (vadd_qf32), one qf32->sf conversion at the end. Argument
 * order matches the captured tinygrad kernel: out, a, w, b, c, d. Same prefetch as the generated code: one dcfetch per
 * 128-byte line of each tap, 2048 bytes ahead. */
#include <hexagon_types.h>
#include <hexagon_protos.h>
static inline HVX_Vector splat(float f) { union { float f; int i; } u = {f}; return Q6_V_vsplat_R(u.i); }
void hand_blend(float* out, const float* a, const float* w, const float* b, const float* c, const float* d) {
  for (int m = 0; m < 4096; m++) {
    const HVX_Vector w0 = splat(w[4*m]), w1 = splat(w[4*m+1]), w2 = splat(w[4*m+2]), w3 = splat(w[4*m+3]);
    const HVX_Vector *pa = (const HVX_Vector*)(a + 256*m), *pb = (const HVX_Vector*)(b + 256*m);
    const HVX_Vector *pc = (const HVX_Vector*)(c + 256*m), *pd = (const HVX_Vector*)(d + 256*m);
    HVX_Vector* po = (HVX_Vector*)(out + 256*m);
    for (int j = 0; j < 8; j++) {
      Q6_dcfetch_A((char*)(pa + j) + 2048); Q6_dcfetch_A((char*)(pb + j) + 2048);
      Q6_dcfetch_A((char*)(pc + j) + 2048); Q6_dcfetch_A((char*)(pd + j) + 2048);
      HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(pa[j], w0), Q6_Vqf32_vmpy_VsfVsf(pb[j], w1));
      s = Q6_Vqf32_vadd_Vqf32Vqf32(s, Q6_Vqf32_vmpy_VsfVsf(pc[j], w2));
      s = Q6_Vqf32_vadd_Vqf32Vqf32(s, Q6_Vqf32_vmpy_VsfVsf(pd[j], w3));
      po[j] = Q6_Vsf_equals_Vqf32(s);
    }
  }
}
