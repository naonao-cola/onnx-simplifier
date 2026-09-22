/* Diagnostic-only: a tiny-scale (ih=iw=32) copy of subgraph_driver.c's fused pipeline, used to
 * test whether the full-scale version's crash is a memory-size limit or a logic bug -- see
 * ../README.md's "Chaining a real subgraph" section for the outcome. Reuses g_stem_weight_packed
 * / g_stem_bias_folded from gen_weight_bias_consts.c (weight/bias depend only on cin/cout, not
 * spatial size, so they're identical at any scale). */

__attribute__((noinline)) void sg_small_hex_stem7x7(int* restrict __attribute__((align_value(128))) data0_16384, unsigned char* restrict __attribute__((align_value(128))) data1_5776, unsigned char* restrict __attribute__((align_value(128))) data2_12544) {
  int buf0[32];
  for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 16; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 2; Lidx2++) {
        for (int z = 0; z < 32; z++) buf0[z] = 0;
        for (int Ridx3_0_0 = 0; Ridx3_0_0 < 7; Ridx3_0_0++) {
          for (int Ridx3_0_1 = 0; Ridx3_0_1 < 7; Ridx3_0_1++) {
            *(int __attribute__((vector_size(128)))*)(buf0+0) = __builtin_HEXAGON_V6_vrmpybusv_acc_128B(*(int __attribute__((vector_size(128)))*)(buf0+0), (int __attribute__((vector_size(128)))){*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2))),*(unsigned int*)(data1_5776+((Lidx0*304)+(Ridx3_0_0*152)+(Lidx1<<3)+(Ridx3_0_1<<2)))}, *(unsigned char __attribute__((vector_size(128)))*)(data2_12544+((Ridx3_0_0*1792)+(Ridx3_0_1<<8)+(Lidx2<<7))));
          }
        }
        *(int __attribute__((vector_size(128)))*)(data0_16384+((Lidx0<<10)+(Lidx1<<6)+(Lidx2<<5))) = *(int __attribute__((vector_size(128)))*)(buf0+0);
      }
    }
  }
}
__attribute__((noinline)) void sg_small_hex_bias_add(int* restrict __attribute__((align_value(128))) data0_16384, int* restrict __attribute__((align_value(128))) data1_16384, const int* restrict __attribute__((align_value(128))) data2_64) {
  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {
      int alu0 = ((Lidx0<<6)+Lidx1);
      *(int*)(data0_16384+alu0) = *(int*)(data1_16384+alu0) + *(int*)(data2_64+Lidx1);
    }
  }
}
__attribute__((noinline)) void sg_small_hex_requantize(unsigned char* restrict __attribute__((align_value(128))) data0_16384, int* restrict __attribute__((align_value(128))) data1_16384) {
  for (int Lidx0 = 0; Lidx0 < 16384; Lidx0++) {
    { long long t = (long long)(*(int*)(data1_16384+Lidx0)) - 0; t = (t << 0) * 2083812681LL; t = (t + 549755813888LL) >> 40; t += 0; if (t < 0) t = 0; if (t > 255) t = 255; *(unsigned char*)(data0_16384+Lidx0) = (unsigned char)t; }
  }
}
__attribute__((noinline)) void sg_small_hex_maxpool(unsigned char* restrict __attribute__((align_value(128))) data0_4096, unsigned char* restrict __attribute__((align_value(128))) data1_20736) {
  unsigned char buf0[32];
  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {
    for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {
      for (int Lidx2 = 0; Lidx2 < 8; Lidx2++) {
        for (int z = 0; z < 32; z++) buf0[z] = 0;
        for (int Ridx3_0 = 0; Ridx3_0 < 3; Ridx3_0++) {
          for (int Ridx3_1 = 0; Ridx3_1 < 3; Ridx3_1++) {
            *(unsigned char __attribute__((vector_size(32)))*)(buf0+0) = __builtin_elementwise_max(*(unsigned char __attribute__((vector_size(32)))*)(buf0+0), *(unsigned char __attribute__((vector_size(32)))*)(data1_20736+((Lidx1*1152)+(Ridx3_0*576)+(Lidx2<<6)+(Ridx3_1<<5)+(Lidx0*10368))));
          }
        }
        *(unsigned char __attribute__((vector_size(32)))*)(data0_4096+((Lidx0<<11)+(Lidx1<<8)+(Lidx2<<5))) = *(unsigned char __attribute__((vector_size(32)))*)(buf0+0);
      }
    }
  }
}

static unsigned char sg_small_layout_buf[20736] __attribute__((aligned(128)));
static void sg_small_nhwc_to_padded_nchwc(const unsigned char* nhwc, unsigned char* out_padded) {
  const int OH = 16, OW = 16, C = 64, IC_CHUNKS = 2, PAD = 1;
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

static int sg_small_acc[16384] __attribute__((aligned(128)));
static unsigned char sg_small_requant_out[16384] __attribute__((aligned(128)));

void run_small_stem_subgraph(const unsigned char* image_padded, unsigned char* maxpool_out) {
  sg_small_hex_stem7x7(sg_small_acc, (unsigned char*)image_padded, (unsigned char*)g_stem_weight_packed);
  sg_small_hex_bias_add(sg_small_acc, sg_small_acc, g_stem_bias_folded);
  sg_small_hex_requantize(sg_small_requant_out, sg_small_acc);
  sg_small_nhwc_to_padded_nchwc(sg_small_requant_out, sg_small_layout_buf);
  sg_small_hex_maxpool(maxpool_out, sg_small_layout_buf);
}
