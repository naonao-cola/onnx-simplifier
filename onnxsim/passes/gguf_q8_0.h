// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's GGUF "Q8_0" block quant format -- C++ port of
// gguf_q8_0.py's own apply_gguf_q8_0_quantization. See that module's
// docstring for the full rationale and encoder-provenance honesty note.
// A Q8_0 block is a single flat 32-element block sharing one fp16 scale
// `d`, with each element's code a full signed 8-bit integer
// ([-128, 127]): `dequant = code * d`. No bias, no explicit min, no
// sub-block requantization, no codebook -- structurally the same plain
// 32-element block scheme as gguf_legacy_quant.h's own Q4_0/Q4_1 and
// gguf_legacy_quant_5bit.h's own Q5_0/Q5_1, just with a wider (and here,
// signed rather than biased-unsigned) code range and no offset. Reuses
// gguf_legacy_quant.h's own GgufLegacyQuantBase template outright.
//
// Before/after diagram, scope, and the same ACCEPTED, PERMANENT
// DIVERGENCE FROM gguf_q8_0.py (no accumulation/iterative-refinement
// step, so this port tracks the Python port's own float64 numpy
// implementation closely but not bit-identically, up to float16
// round-trip and floating-point summation order differences): see
// gguf_legacy_quant.h's own top-of-file comment, which applies here with
// one addition -- unlike gguf_q8_0.py's own apply_gguf_q8_0_quantization,
// this port does not add an include_conv option and only matches a
// constant 2-D float32 weight, exactly like gguf_q6_k.h's own established
// scope decision for this repo's *_cpp ports.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "passes/gguf_legacy_quant.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace gguf_q8_0_detail {

constexpr int64_t kMaxCode = 127;
constexpr int64_t kMinCode = -128;

// Quantize-dequantize round trip for one Q8_0 block of `count` (<=
// gguf_legacy_quant_detail::kBlockSize) contiguous elements, written back
// in place. Mirrors gguf_q8_0.py's own quantize_dequantize_q8_0:
// d = max(|block|) / 127 (round-tripped through float16),
// code = round(value / d) clamped to [-128, 127], dequant = code * d.
inline void QuantizeDequantizeQ8_0Block(double* data, int64_t count) {
  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  const double d = gguf_legacy_quant_detail::RoundTripFloat16(
      std::max(max_abs, 1e-12) / kMaxCode);
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round(data[i] / d);
    code = std::min(std::max(code, static_cast<double>(kMinCode)),
                    static_cast<double>(kMaxCode));
    data[i] = code * d;
  }
}

}  // namespace gguf_q8_0_detail

inline constexpr char kGgufQ8_0Name[] = "gguf_q8_0";

using GgufQ8_0 =
    GgufLegacyQuantBase<gguf_q8_0_detail::QuantizeDequantizeQ8_0Block,
                        kGgufQ8_0Name>;

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
