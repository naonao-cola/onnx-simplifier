/* Fused single-.so driver for ResNet-50 stage1/block1 (the first bottleneck block after the
 * stem+maxpool subgraph, PR #1804 / ../README.md's "Chaining a real subgraph" section, block1
 * follow-up):
 *   main path:     conv10(1x1 reduce) -> bias -> requantize(->s15) -> conv17(3x3) -> bias ->
 *                   requantize(->s22) -> conv24(1x1 expand) -> bias  [kept RAW int32]
 *   shortcut path: conv30(1x1 downsample, from the same maxpool input) -> bias  [kept RAW int32]
 *   merge:         rescale both raw int32 branches to a shared scale domain (s37, the real
 *                   backbone's block-output scale) without an intermediate uint8 clamp, add,
 *                   relu, then one final clamp to uint8 -- see
 *                   ../backbone_subgraph/extract_and_verify_block1.py's own numpy verification
 *                   (max abs diff 3/255, mean 0.4/255 vs a real ONNX Runtime reference) for why
 *                   this rescale-then-add is necessary: the two branches' raw int32 accumulators
 *                   do NOT share an implicit common scale (confirmed: ratio ~0.51, a naive
 *                   un-rescaled add measured max diff 65/mean diff 4.36 -- decisively worse),
 *                   unlike this project's earlier FPN-lateral `add` finding.
 * Weights/biases/quant params are REAL values extracted from backbone.onnx (see
 * ../backbone_subgraph/extract_and_verify_block1.py, which writes gen_block1_weight_consts.c).
 * Every arithmetic kernel body below is verbatim tinygrad-generated code
 * (../hex_gemm_signed_kernel.py, ../hex_conv3x3_kernel.py, ../hex_bias_add_kernel.py,
 * ../hex_requantize_kernel.py's own build_kernel(), captured via ClangRenderer at this block's
 * real shapes and pasted in unmodified) -- only the layout/padding glue and the
 * rescale-add-relu-clamp merge step (new numeric code this block needed, not covered by any
 * existing kernel, in block1_glue.c) are hand-written. All intermediate buffers stay on-device
 * (static, never cross the RPC boundary), matching native_transport/'s "one fixed function, one
 * RPC call" design. */

#include "gen_block1_weight_consts.c"


__attribute__((noinline)) void sg_conv10(int* restrict __attribute__((align_value(128))) data0_3481600, unsigned char* restrict __attribute__((align_value(128))) data1_3481600, unsigned char* restrict __attribute__((align_value(128))) data2_4096) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 54400; Lidx0++) {
    int alu0 = (Lidx0<<6);
    for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {
      *(buf0+0) = 0;
      *(buf0+1) = 0;
      *(buf0+2) = 0;
      *(buf0+3) = 0;
      *(buf0+4) = 0;
      *(buf0+5) = 0;
      *(buf0+6) = 0;
      *(buf0+7) = 0;
      *(buf0+8) = 0;
      *(buf0+9) = 0;
      *(buf0+10) = 0;
      *(buf0+11) = 0;
      *(buf0+12) = 0;
      *(buf0+13) = 0;
      *(buf0+14) = 0;
      *(buf0+15) = 0;
      *(buf0+16) = 0;
      *(buf0+17) = 0;
      *(buf0+18) = 0;
      *(buf0+19) = 0;
      *(buf0+20) = 0;
      *(buf0+21) = 0;
      *(buf0+22) = 0;
      *(buf0+23) = 0;
      *(buf0+24) = 0;
      *(buf0+25) = 0;
      *(buf0+26) = 0;
      *(buf0+27) = 0;
      *(buf0+28) = 0;
      *(buf0+29) = 0;
      *(buf0+30) = 0;
      *(buf0+31) = 0;
      for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
        *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), (int __attribute__((vector_size(128)))){*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2))),*(unsigned int*)(data1_3481600+(alu0+(Ridx2<<2)))}, *(unsigned char __attribute__((vector_size(128)))*)(data2_4096+((Lidx1<<11)+(Ridx2<<7))));
      }
      *(int __attribute__((vector_size(128)))*)(data0_3481600+(alu0+(Lidx1<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
    }
  }
}


__attribute__((noinline)) void sg_bias64(int* restrict __attribute__((align_value(128))) data0_3481600, int* restrict __attribute__((align_value(128))) data1_3481600, int* restrict __attribute__((align_value(128))) data2_64) {
  for (int Lidx0 = 0; Lidx0 < 54400; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {
      int alu0 = ((Lidx0<<6)+Lidx1);
      *(int*)(data0_3481600+alu0) = *(int*)(data1_3481600+alu0) + *(int*)(data2_64+Lidx1);
    }
  }
}


__attribute__((noinline)) void sg_requant_x15(unsigned char* restrict __attribute__((align_value(128))) data0_3481600, int* restrict __attribute__((align_value(128))) data1_3481600) {
  for (int Lidx0 = 0; Lidx0 < 3481600; Lidx0++) {
    { long long t = (long long)(*(int*)(data1_3481600+Lidx0)) - 0; t = (t << 0) * 1311432633LL; t = (t + 34359738368LL) >> 36; t += 0; if (t < 0) t = 0; if (t > 255) t = 255; *(unsigned char*)(data0_3481600+Lidx0) = (unsigned char)t; }
  }
}


__attribute__((noinline)) void sg_conv17(int* restrict __attribute__((align_value(128))) data0_3481600, unsigned char* restrict __attribute__((align_value(128))) data1_3542272, unsigned char* restrict __attribute__((align_value(128))) data2_36864) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 200; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 272; Lidx1++) {
      int alu0 = (Lidx1<<6);
      for (int Lidx2 = 0; Lidx2 < 2; Lidx2++) {
        *(buf0+0) = 0;
        *(buf0+1) = 0;
        *(buf0+2) = 0;
        *(buf0+3) = 0;
        *(buf0+4) = 0;
        *(buf0+5) = 0;
        *(buf0+6) = 0;
        *(buf0+7) = 0;
        *(buf0+8) = 0;
        *(buf0+9) = 0;
        *(buf0+10) = 0;
        *(buf0+11) = 0;
        *(buf0+12) = 0;
        *(buf0+13) = 0;
        *(buf0+14) = 0;
        *(buf0+15) = 0;
        *(buf0+16) = 0;
        *(buf0+17) = 0;
        *(buf0+18) = 0;
        *(buf0+19) = 0;
        *(buf0+20) = 0;
        *(buf0+21) = 0;
        *(buf0+22) = 0;
        *(buf0+23) = 0;
        *(buf0+24) = 0;
        *(buf0+25) = 0;
        *(buf0+26) = 0;
        *(buf0+27) = 0;
        *(buf0+28) = 0;
        *(buf0+29) = 0;
        *(buf0+30) = 0;
        *(buf0+31) = 0;
        for (int Ridx3_0_0 = 0; Ridx3_0_0 < 3; Ridx3_0_0++) {
          for (int Ridx3_0_1 = 0; Ridx3_0_1 < 3; Ridx3_0_1++) {
            for (int Ridx3_1 = 0; Ridx3_1 < 16; Ridx3_1++) {
              unsigned char __attribute__((vector_size(128))) __wv = *(unsigned char __attribute__((vector_size(128)))*)(data2_36864+((Ridx3_0_0*12288)+(Ridx3_0_1<<12)+(Lidx2<<11)+(Ridx3_1<<7))); *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), (int __attribute__((vector_size(128)))){*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2))),*(unsigned int*)(data1_3542272+((Lidx0*17536)+(Ridx3_0_0*17536)+alu0+(Ridx3_0_1<<6)+(Ridx3_1<<2)))}, __wv);
            }
          }
        }
        *(int __attribute__((vector_size(128)))*)(data0_3481600+((Lidx0*17408)+alu0+(Lidx2<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
      }
    }
  }
}


__attribute__((noinline)) void sg_conv2430(int* restrict __attribute__((align_value(128))) data0_13926400, unsigned char* restrict __attribute__((align_value(128))) data1_3481600, unsigned char* restrict __attribute__((align_value(128))) data2_16384) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 54400; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {
      *(buf0+0) = 0;
      *(buf0+1) = 0;
      *(buf0+2) = 0;
      *(buf0+3) = 0;
      *(buf0+4) = 0;
      *(buf0+5) = 0;
      *(buf0+6) = 0;
      *(buf0+7) = 0;
      *(buf0+8) = 0;
      *(buf0+9) = 0;
      *(buf0+10) = 0;
      *(buf0+11) = 0;
      *(buf0+12) = 0;
      *(buf0+13) = 0;
      *(buf0+14) = 0;
      *(buf0+15) = 0;
      *(buf0+16) = 0;
      *(buf0+17) = 0;
      *(buf0+18) = 0;
      *(buf0+19) = 0;
      *(buf0+20) = 0;
      *(buf0+21) = 0;
      *(buf0+22) = 0;
      *(buf0+23) = 0;
      *(buf0+24) = 0;
      *(buf0+25) = 0;
      *(buf0+26) = 0;
      *(buf0+27) = 0;
      *(buf0+28) = 0;
      *(buf0+29) = 0;
      *(buf0+30) = 0;
      *(buf0+31) = 0;
      for (int Ridx2 = 0; Ridx2 < 16; Ridx2++) {
        *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), (int __attribute__((vector_size(128)))){*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2))),*(unsigned int*)(data1_3481600+((Lidx0<<6)+(Ridx2<<2)))}, *(unsigned char __attribute__((vector_size(128)))*)(data2_16384+((Lidx1<<11)+(Ridx2<<7))));
      }
      *(int __attribute__((vector_size(128)))*)(data0_13926400+((Lidx0<<8)+(Lidx1<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
    }
  }
}


__attribute__((noinline)) void sg_bias256(int* restrict __attribute__((align_value(128))) data0_13926400, int* restrict __attribute__((align_value(128))) data1_13926400, int* restrict __attribute__((align_value(128))) data2_256) {
  for (int Lidx0 = 0; Lidx0 < 54400; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 256; Lidx1++) {
      int alu0 = ((Lidx0<<8)+Lidx1);
      *(int*)(data0_13926400+alu0) = *(int*)(data1_13926400+alu0) + *(int*)(data2_256+Lidx1);
    }
  }
}


__attribute__((noinline)) void sg_requant_x22(unsigned char* restrict __attribute__((align_value(128))) data0_3481600, int* restrict __attribute__((align_value(128))) data1_3481600) {
  for (int Lidx0 = 0; Lidx0 < 3481600; Lidx0++) {
    { long long t = (long long)(*(int*)(data1_3481600+Lidx0)) - 0; t = (t << 0) * 1886846189LL; t = (t + 549755813888LL) >> 40; t += 0; if (t < 0) t = 0; if (t > 255) t = 255; *(unsigned char*)(data0_3481600+Lidx0) = (unsigned char)t; }
  }
}
