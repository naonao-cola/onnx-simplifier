/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dequantization for GGML's MXFP4 block format (gguf_dtype.h's IsMxfp4):
 * decodes a tensor's native, still-packed GGML block bytes (see
 * gguf_dtype.h's Mxfp4BlockBytes) into plain host-order float32 values.
 *
 * Pure, dependency-free (no protobuf, no onnx headers) like gguf_dtype.h and
 * ggml_kquant.h -- operates on raw bytes and the same integer ggml_type
 * codes, so this builds and unit-tests standalone.
 *
 * Block layout and dequantization formula transcribed verbatim from GGML's
 * own reference implementation (https://github.com/ggml-org/ggml --
 * ggml-common.h's block_mxfp4 struct and kvalues_fp4/kvalues_mxfp4 lookup
 * table, ggml-quants.c's dequantize_row_mxfp4, and ggml-impl.h's
 * ggml_e8m0_to_fp32_half). MXFP4 (the OCP Microscaling FP4 format --
 * https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
 * is structurally unrelated to the K-quant family ggml_kquant.h covers:
 * instead of a linear affine (scale * code - min) reconstruction over
 * consecutive-pair-packed nibbles, each 32-element block carries one shared
 * power-of-two E8M0 exponent byte and 16 bytes of packed 4-bit codes
 * indexing a small fixed lookup table of signed e2m1-style magnitudes, with
 * elements j and j+16 of the block packed into the low and high nibble of
 * byte j (not elements 2j/2j+1, as K-quant packs). This is gpt-oss-20b's
 * real, shipping GGUF quantization for its MoE expert weights
 * (`general.architecture=gpt-oss`).
 */
#ifndef ONNXSIM_GGML_MXFP4_H_
#define ONNXSIM_GGML_MXFP4_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "gguf_dtype.h"

namespace onnxsim {
namespace tensor_pool {
namespace gguf {

// GGML's kvalues_fp4/kvalues_mxfp4 table (shared with NVFP4): the 16
// possible decoded magnitudes a 4-bit MXFP4 code indexes, already doubled
// (matching GGML's own table verbatim, and GgmlE8m0ToFloat32Half's halved
// scale below) -- codes 0-7 are the non-negative e2m1 magnitudes {0, 0.5, 1,
// 1.5, 2, 3, 4, 6} times 2, codes 8-15 their negated counterparts.
inline constexpr int8_t kMxfp4Values[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                            0, -1, -2, -3, -4, -6, -8, -12};

// GGML's ggml_e8m0_to_fp32_half: decodes one E8M0 (8-bit unsigned power-of-
// two) exponent byte to *half* the float32 value it nominally represents --
// GGML halves here because kMxfp4Values above is already doubled, so the
// product of the two reproduces the true e2m1 magnitude without needing a
// fractional entry in either table. `x < 2` has no normal float32
// representation of 2^(x-127), so those two smallest cases are written
// directly as denormal float32 bit patterns; `x >= 2` computes the normal-
// float32 exponent field directly. NaNs are not handled, mirroring GGML's
// own implementation (an E8M0 byte of 0xFF, GGML's own only NaN encoding,
// never appears in a real MXFP4 tensor's per-block scale).
inline float GgmlE8m0ToFloat32Half(uint8_t x) {
  uint32_t bits;
  if (x == 0) {
    bits = 0x00200000u;  // 2^(-128)
  } else if (x == 1) {
    bits = 0x00400000u;  // 2^(-127)
  } else {
    bits = static_cast<uint32_t>(x - 1) << 23;  // 0.5 * 2^(x-127) = 2^(x-128)
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Decodes one 17-byte MXFP4 block (32 elements) into `out`.
inline void DequantizeMxfp4Block(const uint8_t* block, float* out) {
  const float d = GgmlE8m0ToFloat32Half(block[0]);
  const uint8_t* qs = block + 1;  // 16 bytes, 2 elements packed per byte.
  for (int j = 0; j < 16; ++j) {
    out[j] = static_cast<float>(kMxfp4Values[qs[j] & 0xF]) * d;
    out[j + 16] = static_cast<float>(kMxfp4Values[qs[j] >> 4]) * d;
  }
}

// Decodes `raw` (an IsMxfp4(ggml_type) tensor's native block bytes, exactly
// `numel / 32 * 17` bytes long) into `numel` host-order float32 values,
// appended to `out` (not cleared first). Returns false, leaving `out`
// untouched, if `ggml_type` is not MXFP4, `numel` is not a multiple of 32,
// or `raw_size` does not match the expected byte length for `numel`
// elements (a corrupt/truncated buffer).
inline bool DequantizeGgmlMxfp4(const uint8_t* raw, size_t raw_size,
                                uint32_t ggml_type, int64_t numel,
                                std::vector<float>* out) {
  if (!IsMxfp4(ggml_type) || numel < 0) {
    return false;
  }
  const size_t block_elems = Mxfp4BlockElements(ggml_type);
  const size_t block_bytes = Mxfp4BlockBytes(ggml_type);
  const uint64_t unumel = static_cast<uint64_t>(numel);
  if (unumel % block_elems != 0) {
    return false;
  }
  const uint64_t num_blocks = unumel / block_elems;
  if (num_blocks * block_bytes != raw_size) {
    return false;
  }

  const size_t out_start = out->size();
  out->resize(out_start + unumel);
  float* dst = out->data() + out_start;
  const uint8_t* src = raw;
  for (uint64_t b = 0; b < num_blocks; ++b) {
    DequantizeMxfp4Block(src, dst);
    src += block_bytes;
    dst += block_elems;
  }
  return true;
}

}  // namespace gguf
}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_GGML_MXFP4_H_
