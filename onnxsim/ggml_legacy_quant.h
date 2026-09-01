/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dequantization for GGML's legacy Q4_0/Q4_1/Q5_0/Q5_1 block formats
 * (gguf_dtype.h's IsLegacyQuant): decodes a tensor's native, still-packed
 * GGML block bytes (see gguf_dtype.h's LegacyQuantBlockBytes) into plain
 * host-order float32 values.
 *
 * Pure, dependency-free (no protobuf, no onnx headers) like gguf_dtype.h and
 * ggml_kquant.h -- operates on raw bytes and the same integer ggml_type
 * codes, so this builds and unit-tests standalone.
 *
 * Every block layout and dequantization formula here is transcribed
 * verbatim from GGML's own reference implementation
 * (https://github.com/ggml-org/ggml -- ggml-common.h's
 * block_q4_0/block_q4_1/block_q5_0/block_q5_1 struct layouts, ggml-quants.c's
 * dequantize_row_q4_0/q4_1/q5_0/q5_1). Byte order: like onnx::TensorProto::
 * raw_data and every GGUF file (see tensor_pool.h's file comment), a
 * block's multi-byte fields are little-endian on disk regardless of host
 * byte order -- this file's Float16BitsToFloat32 argument is always
 * reconstructed via explicit byte-at-a-time reads (ReadLE16/ReadLE32), never
 * a reinterpret_cast of the block struct, so decoding is correct on a
 * big-endian host too (this repo tests on s390x). A block's single-byte
 * fields (the packed quant codes) need no such care -- one byte has no
 * endianness.
 *
 * Unlike the K-quant family (ggml_kquant.h), each of these four types packs
 * a single, plain 32-element block with no super-block scale/min table:
 * Q4_0/Q5_0 reconstruct as `code*d` (with a fixed per-format zero-point bias
 * subtracted from the 4/5-bit code before scaling), Q4_1/Q5_1 as
 * `code*d + m` (an explicit per-block min, no bias needed since the code is
 * used unsigned). Q5_0/Q5_1's 5th bit lives in a separate 4-byte `qh`
 * bitfield, one bit per element, rather than packed alongside the other 4
 * bits.
 *
 * Reuses ggml_kquant.h's ReadLE16/Float16BitsToFloat32 (every one of these
 * four formats' scale/min fields is the same fp16 GGML `ggml_half` K-quant
 * already needs to decode) rather than duplicating that conversion --
 * unlike gguf_dtype.h's own choice to duplicate small wire-format constants
 * across codecs, an fp16-to-fp32 conversion is intricate enough (denormal/
 * inf/nan handling) that a second, independently-maintained copy would risk
 * silently drifting out of sync with the original.
 */
#ifndef ONNXSIM_GGML_LEGACY_QUANT_H_
#define ONNXSIM_GGML_LEGACY_QUANT_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "ggml_kquant.h"
#include "gguf_dtype.h"

namespace onnxsim {
namespace tensor_pool {
namespace gguf {

// Reconstructs a little-endian uint32_t from four bytes, regardless of host
// byte order -- Q5_0/Q5_1's `qh` field, mirroring ggml_kquant.h's ReadLE16
// for the 2-byte case.
inline uint32_t ReadLE32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

// Decodes one 18-byte Q4_0 block (32 elements) into `out`. Each 4-bit code
// is biased by -8 (Q4_0's codes are unsigned 0..15, representing signed
// -8..7) before scaling -- no separate min value stored.
inline void DequantizeQ4_0Block(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const uint8_t* qs = block + 2;
  for (int j = 0; j < 16; ++j) {
    const int x0 = (qs[j] & 0x0F) - 8;
    const int x1 = (qs[j] >> 4) - 8;
    out[j] = static_cast<float>(x0) * d;
    out[j + 16] = static_cast<float>(x1) * d;
  }
}

// Decodes one 20-byte Q4_1 block (32 elements) into `out`. Codes are used
// unsigned (0..15) with an explicit per-block min `m` added after scaling.
inline void DequantizeQ4_1Block(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const float m = Float16BitsToFloat32(ReadLE16(block + 2));
  const uint8_t* qs = block + 4;
  for (int j = 0; j < 16; ++j) {
    const int x0 = qs[j] & 0x0F;
    const int x1 = qs[j] >> 4;
    out[j] = static_cast<float>(x0) * d + m;
    out[j + 16] = static_cast<float>(x1) * d + m;
  }
}

// Decodes one 22-byte Q5_0 block (32 elements) into `out`. Each element's
// 5th (high) bit lives in the block's 4-byte `qh` bitfield rather than
// alongside the other 4 bits in `qs`; the resulting 5-bit unsigned code
// (0..31) is biased by -16 before scaling -- no separate min value stored.
inline void DequantizeQ5_0Block(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const uint32_t qh = ReadLE32(block + 2);
  const uint8_t* qs = block + 6;
  for (int j = 0; j < 16; ++j) {
    const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
    const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
    const int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
    const int x1 = ((qs[j] >> 4) | xh_1) - 16;
    out[j] = static_cast<float>(x0) * d;
    out[j + 16] = static_cast<float>(x1) * d;
  }
}

// Decodes one 24-byte Q5_1 block (32 elements) into `out`. Same 5th-bit
// scheme as Q5_0, but the resulting 5-bit code is used unsigned (0..31)
// with an explicit per-block min `m` added after scaling, like Q4_1.
inline void DequantizeQ5_1Block(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const float m = Float16BitsToFloat32(ReadLE16(block + 2));
  const uint32_t qh = ReadLE32(block + 4);
  const uint8_t* qs = block + 8;
  for (int j = 0; j < 16; ++j) {
    const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
    const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
    const int x0 = (qs[j] & 0x0F) | xh_0;
    const int x1 = (qs[j] >> 4) | xh_1;
    out[j] = static_cast<float>(x0) * d + m;
    out[j + 16] = static_cast<float>(x1) * d + m;
  }
}

// Decodes `raw` (an IsLegacyQuant(ggml_type) tensor's native block bytes,
// exactly `numel / 32 * LegacyQuantBlockBytes(ggml_type)` bytes long) into
// `numel` host-order float32 values, appended to `out` (not cleared first).
// Returns false, leaving `out` untouched, if `ggml_type` is not one of the
// four IsLegacyQuant types, `numel` is not a multiple of 32, or `raw_size`
// does not match the expected byte length for `numel` elements (a corrupt/
// truncated buffer).
inline bool DequantizeGgmlLegacyQuant(const uint8_t* raw, size_t raw_size,
                                      uint32_t ggml_type, int64_t numel,
                                      std::vector<float>* out) {
  if (!IsLegacyQuant(ggml_type) || numel < 0) {
    return false;
  }
  const size_t block_elems = LegacyQuantBlockElements(ggml_type);
  const size_t block_bytes = LegacyQuantBlockBytes(ggml_type);
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
    switch (ggml_type) {
      case GGML_TYPE_Q4_0:
        DequantizeQ4_0Block(src, dst);
        break;
      case GGML_TYPE_Q4_1:
        DequantizeQ4_1Block(src, dst);
        break;
      case GGML_TYPE_Q5_0:
        DequantizeQ5_0Block(src, dst);
        break;
      case GGML_TYPE_Q5_1:
        DequantizeQ5_1Block(src, dst);
        break;
      default:
        return false;  // Unreachable: IsLegacyQuant already filtered this.
    }
    src += block_bytes;
    dst += block_elems;
  }
  return true;
}

}  // namespace gguf
}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_GGML_LEGACY_QUANT_H_
