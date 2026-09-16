// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's legacy GGUF "Q5_0"/"Q5_1" block quant formats -- C++ port of
// gguf_legacy_quant_5bit.py's own apply_gguf_q5_0_quantization/
// apply_gguf_q5_1_quantization. See that module's docstring for the full
// rationale and the exact reconstruction formulas this port targets
// (transcribed from, and kept consistent with, this repo's own verified
// onnxsim/ggml_legacy_quant.h decoder): Q5_0 is symmetric with no separate
// min (`dequant = (code - 16) * d`, code in [0, 31]); Q5_1 is asymmetric
// with an explicit per-block min (`dequant = code * d + m`, code in
// [0, 31]) -- exactly gguf_legacy_quant.h's own Q4_0/Q4_1 scheme, one bit
// wider. Reuses that header's own GgufLegacyQuantBase template outright
// (same 32-element plain block, same MatMul/vanilla-Gemm-only match, same
// ragged-last-block handling, same "no include_conv option" scope
// decision) rather than duplicating it -- only the two block functions and
// their bias/max-code constants differ.
//
// Before/after diagram, scope, and the same ACCEPTED, PERMANENT DIVERGENCE
// FROM gguf_legacy_quant_5bit.py (no accumulation/iterative-refinement
// step, so this port tracks the Python port's own float64 numpy
// implementation closely but not bit-identically, up to float16
// round-trip and floating-point summation order differences): see
// gguf_legacy_quant.h's own top-of-file comment, which applies here
// unchanged with s/Q4/Q5/ and s/4-bit/5-bit/.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/gguf_legacy_quant.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace gguf_legacy_quant_5bit_detail {

constexpr int64_t kMaxCode = 31;
constexpr int64_t kQ5_0Bias = 16;

// Quantize-dequantize round trip for one Q5_0 block of `count` (<=
// gguf_legacy_quant_detail::kBlockSize) contiguous elements, written back
// in place. Mirrors gguf_legacy_quant_5bit.py's own
// quantize_dequantize_q5_0: d = max(|block|) / 16 (round-tripped through
// float16), code = round(value / d) + 16 clamped to [0, 31],
// dequant = (code - 16) * d.
inline void QuantizeDequantizeQ5_0Block(double* data, int64_t count) {
  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  const double d = gguf_legacy_quant_detail::RoundTripFloat16(
      std::max(max_abs, 1e-12) / kQ5_0Bias);
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round(data[i] / d) + static_cast<double>(kQ5_0Bias);
    code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
    data[i] = (code - static_cast<double>(kQ5_0Bias)) * d;
  }
}

// Quantize-dequantize round trip for one Q5_1 block, written back in
// place. Mirrors gguf_legacy_quant_5bit.py's own
// quantize_dequantize_q5_1: m = min(block), d = (max(block) - m) / 31
// (both round-tripped through float16), code = round((value - m) / d)
// clamped to [0, 31], dequant = code * d + m.
inline void QuantizeDequantizeQ5_1Block(double* data, int64_t count) {
  double lo = data[0];
  double hi = data[0];
  for (int64_t i = 1; i < count; ++i) {
    lo = std::min(lo, data[i]);
    hi = std::max(hi, data[i]);
  }
  const double m = gguf_legacy_quant_detail::RoundTripFloat16(lo);
  const double d = gguf_legacy_quant_detail::RoundTripFloat16(
      std::max(hi - lo, 1e-12) / kMaxCode);
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round((data[i] - m) / d);
    code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
    data[i] = code * d + m;
  }
}

}  // namespace gguf_legacy_quant_5bit_detail

inline constexpr char kGgufQ5_0Name[] = "gguf_q5_0";
inline constexpr char kGgufQ5_1Name[] = "gguf_q5_1";

using GgufQ5_0 = GgufLegacyQuantBase<
    gguf_legacy_quant_5bit_detail::QuantizeDequantizeQ5_0Block, kGgufQ5_0Name>;
using GgufQ5_1 = GgufLegacyQuantBase<
    gguf_legacy_quant_5bit_detail::QuantizeDequantizeQ5_1Block, kGgufQ5_1Name>;

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
