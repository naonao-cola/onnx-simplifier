/* Fused single-.so driver for the ResNet-50 stem subgraph: conv7x7(s2,p3) -> bias_add ->
 * requantize -> layout_transform(NHWC-flat -> TVM's packed NCHWc) -> maxpool(3x3,s2,p1).
 * Every arithmetic kernel body below is verbatim tinygrad-generated code (hex_stem7x7_kernel.py,
 * hex_bias_add_kernel.py, hex_requantize_kernel.py, hex_maxpool_kernel.py's own build_kernel(),
 * captured via ClangRenderer at the real subgraph scale and pasted in unmodified -- only the
 * layout-transform glue and the top-level wrapper below are hand-written new code). See
 * ../README.md's "Chaining a real subgraph" section for how the real weight/bias/scale data and
 * the zero-point-folded bias were derived from backbone.onnx.
 *
 * All intermediate buffers stay entirely on-device (static, never round-tripped over RPC) --
 * only the padded quantized image goes in and the final packed-NCHWc maxpool output comes out,
 * matching native_transport/'s "one fixed function, one RPC call" design and avoiding this
 * project's own previously-found ~32-48MB/buffer RPC transfer wall for every intermediate. */

#include "gen_weight_bias_consts.c"

/* ---- sg_hex_stem7x7 (hex_stem7x7_kernel.py, cin=3,cout=64,ih=800,iw=1088) ---- */
__attribute__((noinline)) void sg_hex_stem7x7(int* restrict __attribute__((align_value(128))) data0_13926400, unsigned char* restrict __attribute__((align_value(128))) data1_3527056, unsigned char* restrict __attribute__((align_value(128))) data2_12544) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 400; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 544; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 2; Lidx2++) {
        *(buf0+0) = 0; *(buf0+1) = 0; *(buf0+2) = 0; *(buf0+3) = 0; *(buf0+4) = 0; *(buf0+5) = 0; *(buf0+6) = 0; *(buf0+7) = 0;
        *(buf0+8) = 0; *(buf0+9) = 0; *(buf0+10) = 0; *(buf0+11) = 0; *(buf0+12) = 0; *(buf0+13) = 0; *(buf0+14) = 0; *(buf0+15) = 0;
        *(buf0+16) = 0; *(buf0+17) = 0; *(buf0+18) = 0; *(buf0+19) = 0; *(buf0+20) = 0; *(buf0+21) = 0; *(buf0+22) = 0; *(buf0+23) = 0;
        *(buf0+24) = 0; *(buf0+25) = 0; *(buf0+26) = 0; *(buf0+27) = 0; *(buf0+28) = 0; *(buf0+29) = 0; *(buf0+30) = 0; *(buf0+31) = 0;
        for (int Ridx3_0_0 = 0; Ridx3_0_0 < 7; Ridx3_0_0++) {
          for (int Ridx3_0_1 = 0; Ridx3_0_1 < 7; Ridx3_0_1++) {
            *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), (int __attribute__((vector_size(128)))){*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_3527056+((Lidx0*8752)+(Ridx3_0_0*4376)+(Lidx1<<3)+(Ridx3_0_1<<2)))}, *(unsigned char __attribute__((vector_size(128)))*)(data2_12544+((Ridx3_0_0*1792)+(Ridx3_0_1<<8)+(Lidx2<<7))));
          }
        }
        *(int __attribute__((vector_size(128)))*)(data0_13926400+((Lidx0*34816)+(Lidx1<<6)+(Lidx2<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
      }
    }
  }
}

/* ---- sg_hex_bias_add (hex_bias_add_kernel.py, pos=217600,cout=64) ---- */
__attribute__((noinline)) void sg_hex_bias_add(int* restrict __attribute__((align_value(128))) data0_13926400, int* restrict __attribute__((align_value(128))) data1_13926400, const int* restrict __attribute__((align_value(128))) data2_64) {
  for (int Lidx0 = 0; Lidx0 < 217600; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {
      int alu0 = ((Lidx0<<6)+Lidx1);
      *(int*)(data0_13926400+alu0) = *(int*)(data1_13926400+alu0) + *(int*)(data2_64+Lidx1);
    }
  }
}

/* ---- sg_hex_requantize (hex_requantize_kernel.py, n=13926400, in_scale=img_scale*w_scale,
 * out_scale=7_scale (real backbone.onnx values), in_zp=0, out_zp=0 -- multiplier/shift baked
 * from compute_multiplier_shift(0.00019681659035583263 / 0.1038491278886795)) ---- */
__attribute__((noinline)) void sg_hex_requantize(unsigned char* restrict __attribute__((align_value(128))) data0_13926400, int* restrict __attribute__((align_value(128))) data1_13926400) {
  for (int Lidx0 = 0; Lidx0 < 13926400; Lidx0++) {
    { long long t = (long long)(*(int*)(data1_13926400+Lidx0)) - 0; t = (t << 0) * 2083812681LL; t = (t + 549755813888LL) >> 40; t += 0; if (t < 0) t = 0; if (t > 255) t = 255; *(unsigned char*)(data0_13926400+Lidx0) = (unsigned char)t; }
  }
}

/* ---- sg_hex_maxpool (hex_maxpool_kernel.py, cin=64,ih=400,iw=544, kernel=3,stride=2,pad=1) ---- */
__attribute__((noinline)) void sg_hex_maxpool(unsigned char* restrict __attribute__((align_value(128))) data0_3481600, unsigned char* restrict __attribute__((align_value(128))) data1_14047488) {
  unsigned char buf0[32];
  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 200; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 272; Lidx2++) {
        *(buf0+0) = ((unsigned char)(0u)); *(buf0+1) = ((unsigned char)(0u)); *(buf0+2) = ((unsigned char)(0u)); *(buf0+3) = ((unsigned char)(0u));
        *(buf0+4) = ((unsigned char)(0u)); *(buf0+5) = ((unsigned char)(0u)); *(buf0+6) = ((unsigned char)(0u)); *(buf0+7) = ((unsigned char)(0u));
        *(buf0+8) = ((unsigned char)(0u)); *(buf0+9) = ((unsigned char)(0u)); *(buf0+10) = ((unsigned char)(0u)); *(buf0+11) = ((unsigned char)(0u));
        *(buf0+12) = ((unsigned char)(0u)); *(buf0+13) = ((unsigned char)(0u)); *(buf0+14) = ((unsigned char)(0u)); *(buf0+15) = ((unsigned char)(0u));
        *(buf0+16) = ((unsigned char)(0u)); *(buf0+17) = ((unsigned char)(0u)); *(buf0+18) = ((unsigned char)(0u)); *(buf0+19) = ((unsigned char)(0u));
        *(buf0+20) = ((unsigned char)(0u)); *(buf0+21) = ((unsigned char)(0u)); *(buf0+22) = ((unsigned char)(0u)); *(buf0+23) = ((unsigned char)(0u));
        *(buf0+24) = ((unsigned char)(0u)); *(buf0+25) = ((unsigned char)(0u)); *(buf0+26) = ((unsigned char)(0u)); *(buf0+27) = ((unsigned char)(0u));
        *(buf0+28) = ((unsigned char)(0u)); *(buf0+29) = ((unsigned char)(0u)); *(buf0+30) = ((unsigned char)(0u)); *(buf0+31) = ((unsigned char)(0u));
        for (int Ridx3_0 = 0; Ridx3_0 < 3; Ridx3_0++) {
          for (int Ridx3_1 = 0; Ridx3_1 < 3; Ridx3_1++) {
            *(unsigned char __attribute__((vector_size(32)))*)(buf0+0) = __builtin_elementwise_max(*(unsigned char __attribute__((vector_size(32)))*)(buf0+0), *(unsigned char __attribute__((vector_size(32)))*)(data1_14047488+((Lidx1*34944)+(Ridx3_0*17472)+(Lidx2<<6)+(Ridx3_1<<5)+(Lidx0*7023744))));
          }
        }
        *(unsigned char __attribute__((vector_size(32)))*)(data0_3481600+((Lidx0*1740800)+(Lidx1*8704)+(Lidx2<<5))) = *(unsigned char __attribute__((vector_size(32)))*)(buf0+0);
      }
    }
  }
}

/* ---- Hand-written glue: NHWC-flat (pos=oh*ow, c=64) uint8 -> TVM's packed, padded NCHWc
 * (ic_chunk, h+pad, w+pad, 32) uint8, matching hex_maxpool_kernel.py's pad_nchwc()/pack_nchwc()
 * convention exactly. This is the one real composition gap this task's own investigation found:
 * hex_stem7x7_kernel.py's conv output and hex_maxpool_kernel.py's expected input are two
 * genuinely different physical layouts (confirmed by reading each kernel's own docstring/build_
 * kernel() body, not assumed) -- every kernel in this project was individually correct, but nothing
 * had ever tested whether two of them compose, and this is exactly the kind of gap that only shows
 * up when they do. Plain, unvectorized copy loop -- correctness first, matching this project's
 * established precedent for every new kernel's first pass. */
static unsigned char g_layout_buf[14047488] __attribute__((aligned(128)));
static void nhwc_to_padded_nchwc(const unsigned char* nhwc, unsigned char* out_padded) {
  const int OH = 400, OW = 544, C = 64, IC_CHUNKS = 2, PAD = 1;
  const int H_PAD = OH + 2 * PAD, W_PAD = OW + 2 * PAD;
  for (int i = 0; i < IC_CHUNKS * H_PAD * W_PAD * 32; i++) out_padded[i] = 0;
  for (int h = 0; h < OH; h++) {
    for (int w = 0; w < OW; w++) {
      const unsigned char* src = nhwc + (h * OW + w) * C;
      for (int c = 0; c < C; c++) {
        int ic_chunk = c / 32, ic_block = c % 32;
        int dst = ic_chunk * (H_PAD * W_PAD * 32) + (h + PAD) * (W_PAD * 32) + (w + PAD) * 32 + ic_block;
        out_padded[dst] = src[c];
      }
    }
  }
}

/* ---- Top-level fused entry point: image in, packed-NCHWc maxpool output out. Every
 * intermediate buffer below is static (on-device only, never crosses the RPC boundary). ---- */
static int g_stem_acc[13926400] __attribute__((aligned(128)));         /* stem conv raw accumulate, then bias-added in place */
static unsigned char g_requant_out[13926400] __attribute__((aligned(128)));

void run_stem_subgraph(const unsigned char* image_padded, unsigned char* maxpool_out) {
  sg_hex_stem7x7(g_stem_acc, (unsigned char*)image_padded, (unsigned char*)g_stem_weight_packed);
  sg_hex_bias_add(g_stem_acc, g_stem_acc, g_stem_bias_folded);
  sg_hex_requantize(g_requant_out, g_stem_acc);
  nhwc_to_padded_nchwc(g_requant_out, g_layout_buf);
  sg_hex_maxpool(maxpool_out, g_layout_buf);
}
