
/* ---- Hand-written glue ---- */

/* Inverse of Stage 2's nhwc_to_padded_nchwc(): maxpool's packed, unpadded NCHWc output
 * (ic_chunk=2, h=200, w=272, 32) uint8 -> NHWC-flat (pos=h*w, c=64) uint8, the layout every
 * hex_gemm_signed_kernel.py/hex_bias_add_kernel.py/hex_requantize_kernel.py call in this block
 * expects. Plain unvectorized copy loop, correctness first (same precedent as Stage 2's own
 * layout glue). */
static void packed_nchwc_to_nhwc(const unsigned char* packed, unsigned char* nhwc) {
  const int OH = 200, OW = 272, C = 64, IC_CHUNKS = 2;
  for (int h = 0; h < OH; h++) {
    for (int w = 0; w < OW; w++) {
      unsigned char* dst = nhwc + (h * OW + w) * C;
      for (int c = 0; c < C; c++) {
        int ic_chunk = c / 32, ic_block = c % 32;
        int src = ic_chunk * (OH * OW * 32) + (h * OW + w) * 32 + ic_block;
        dst[c] = packed[src];
      }
    }
  }
}

/* NHWC-flat (pos=h*w, c=64) uint8 -> zero-padded (h+2, w+2, c=64) uint8, matching
 * hex_conv3x3_kernel.py's pad_input() exactly (literal-0 pad is correct here since every
 * activation in this block has zero_point=0, confirmed from backbone.onnx -- unlike the stem
 * conv's real image, which needed the real image zero-point). */
static void pad_nhwc_3x3(const unsigned char* nhwc, unsigned char* padded) {
  const int H = 200, W = 272, C = 64, HP = H + 2, WP = W + 2;
  for (int i = 0; i < HP * WP * C; i++) padded[i] = 0;
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      const unsigned char* src = nhwc + (h * W + w) * C;
      unsigned char* dst = padded + ((h + 1) * WP + (w + 1)) * C;
      for (int c = 0; c < C; c++) dst[c] = src[c];
    }
  }
}

/* Q31 fixed-point rescale (same formula as hex_requantize_kernel.py's kernel body) WITHOUT the
 * final clip(0,255)/uint8 cast -- brings a raw int32 accumulator into a shared scale domain,
 * staying int32 (may be negative), so two differently-scaled branches can be added correctly
 * before the one real final requantize. See this file's own header comment / extract_block1.py
 * for why this step is necessary (a raw, un-rescaled int32+int32 add is measurably wrong here). */
static void rescale_inplace_noclamp(int* data, int n, long long multiplier, int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift < 0 ? -shift : 0;
  int total_shift = right_shift + 31;
  long long rounding = 1LL << (total_shift - 1);
  for (int i = 0; i < n; i++) {
    long long t = (long long)data[i];
    t = (t << left_shift) * multiplier;
    t = (t + rounding) >> total_shift;
    data[i] = (int)t;
  }
}

/* rescaled_main + rescaled_shortcut -> relu -> clamp[0,255] -> uint8. Both inputs already share
 * the s37 (block output) scale domain after rescale_inplace_noclamp(), so this step needs no
 * further multiply -- just the add, relu, and the one real final clamp. */
static void add_relu_clamp(const int* main_br, const int* shortcut_br, unsigned char* out, int n) {
  for (int i = 0; i < n; i++) {
    long long t = (long long)main_br[i] + (long long)shortcut_br[i];
    if (t < 0) t = 0;
    if (t > 255) t = 255;
    out[i] = (unsigned char)t;
  }
}

/* ---- Top-level fused entry point ---- */
static unsigned char g_x0_nhwc[3481600] __attribute__((aligned(128)));
static int g_raw10[3481600] __attribute__((aligned(128)));
static unsigned char g_x15[3481600] __attribute__((aligned(128)));
static unsigned char g_x15_pad[202 * 274 * 64] __attribute__((aligned(128)));
static int g_raw17[3481600] __attribute__((aligned(128)));
static unsigned char g_x22[3481600] __attribute__((aligned(128)));
static int g_raw24[13926400] __attribute__((aligned(128)));
static int g_raw30[13926400] __attribute__((aligned(128)));

void run_block1(const unsigned char* maxpool_out_packed, unsigned char* block1_out) {
  packed_nchwc_to_nhwc(maxpool_out_packed, g_x0_nhwc);

  /* main path */
  sg_conv10(g_raw10, g_x0_nhwc, g_stem_conv10_weight_packed);
  sg_bias64(g_raw10, g_raw10, g_conv10_bias);
  sg_requant_x15(g_x15, g_raw10);
  pad_nhwc_3x3(g_x15, g_x15_pad);
  sg_conv17(g_raw17, g_x15_pad, g_conv17_weight_packed);
  sg_bias64(g_raw17, g_raw17, g_conv17_bias);
  sg_requant_x22(g_x22, g_raw17);
  sg_conv2430(g_raw24, g_x22, g_conv24_weight_packed);
  sg_bias256(g_raw24, g_raw24, g_conv24_bias);

  /* shortcut path (downsample), from the same maxpool input as conv10 */
  sg_conv2430(g_raw30, g_x0_nhwc, g_conv30_weight_packed);
  sg_bias256(g_raw30, g_raw30, g_conv30_bias);

  /* merge: rescale both branches to the shared (s37) domain, add, relu, one final clamp */
  rescale_inplace_noclamp(g_raw24, 13926400, 1219494715LL, -6);
  rescale_inplace_noclamp(g_raw30, 13926400, 1193971754LL, -5);
  add_relu_clamp(g_raw24, g_raw30, block1_out, 13926400);
}
