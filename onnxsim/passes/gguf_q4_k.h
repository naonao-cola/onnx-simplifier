// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's GGUF "Q4_K" K-quant format -- C++ port of
// onnxsim.gguf_kquant's own apply_gguf_q4_k_quantization. See that
// module's docstring for the full rationale and the exact reconstruction
// formula this port targets (transcribed from, and kept consistent with,
// this repo's own verified onnxsim/ggml_kquant.h decoder's
// DequantizeQ4_KBlock/GetScaleMinK4): a 256-element super-block split
// into 8 sub-blocks of 32 elements, each sub-block getting its own
// asymmetric affine quantizer (``dequant = q * sub_scale + sub_min``)
// whose element code ``q`` is 4 bits ([0, 15]); each sub-block's own
// (scale, min) pair is itself re-quantized to 6-bit codes ([0, 63])
// relative to one shared pair of super-block-level float16 reference
// values (d/dmin) -- ``value = d * sc_j * q - dmin * m_j``. Same overall
// two-field (scale, min) sub-block re-quantization scheme as
// gguf_q2_k.h's own Q2_K port, just with wider sub-blocks (32 vs 16),
// fewer of them per super-block (8 vs 16), a 4-bit element code (vs
// 2-bit) and 6-bit sub-scale/sub-min codes (vs 4-bit).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 256-element-
//                                   super-block Q4_K quantize-dequantize
//                                   round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_q6_k.h/gguf_q2_k.h use. Unlike
// onnxsim.gguf_kquant's own apply_gguf_q4_k_quantization, this port does
// not add an include_conv option -- several other *_cpp ports in this
// repo already establish that a C++ port need not mirror every optional
// knob its Python counterpart has.
//
// Super-blocks are laid out over the weight's own flattened, row-major
// storage, exactly like gguf_q6_k.h's/gguf_q2_k.h's own block layout --
// NOT per (output-channel, K-block). A ragged final super-block (when
// the weight's total element count isn't itself a multiple of 256) is
// quantized using only its own real elements, one ragged sub-block at a
// time; the same max/min-based zero-padding-invariance argument
// gguf_legacy_quant.h's own comment gives applies to each sub-block's own
// min/max here too.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM onnxsim.gguf_kquant: as with
// gguf_q2_k.h, this scheme has no accumulation/iterative-refinement step
// -- every sub-block's own (scale, min) is computed independently from
// that sub-block's own min/max, and every super-block's own shared
// reference (d, dmin) from the max over its own 8 sub-block (scale,
// neg-offset) values, so this port is expected to track the Python
// port's own float64 numpy implementation closely, up to float16
// round-trip and floating-point summation order differences.
// apply_gguf_q4_k_quantization and its _cpp counterpart remain
// independently-correct, non-interchangeable entry points -- this port's
// own tests check structural/algebraic properties and comparable (not
// bit-identical) reconstruction error, matching this repo's established
// contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml_kquant.h"
#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_fp16.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace gguf_q4_k_detail {

constexpr int64_t kSubBlockSize = 32;
constexpr int64_t kSubBlocksPerSuperBlock = 8;
constexpr int64_t kSuperBlockSize =
    kSubBlockSize * kSubBlocksPerSuperBlock;  // 256
constexpr int64_t kMaxCode = 15;          // 4-bit element code range [0, 15]
constexpr int64_t kMaxSubScaleCode = 63;  // 6-bit sub_scale/sub_min code

// Round-trips a float32 value through ggml_half (IEEE754 binary16),
// reproducing the same precision loss real Q4_K's d/dmin fields carry --
// mirrors onnxsim.gguf_kquant's own
// `.astype(np.float16).astype(np.float64)` step. Reuses this repo's own
// already-verified fp16 codec, the same reuse gguf_q6_k.h's/gguf_q2_k.h's
// own RoundTripFloat16 makes.
inline double RoundTripFloat16(double value) {
  return static_cast<double>(::onnxsim::tensor_pool::gguf::Float16BitsToFloat32(
      FloatToFloat16Bits(static_cast<float>(value))));
}

// Quantize-dequantize round trip for one Q4_K super-block of `count` (<=
// kSuperBlockSize) contiguous elements, written back in place. Mirrors
// onnxsim.gguf_kquant's own _quantize_dequantize_superblocks: for each
// (possibly ragged) 32-element sub-block j, ideal_scale_j =
// max(sub_block) - min(sub_block), clamped to >= 1e-12, / 15;
// ideal_neg_offset_j = max(-min(sub_block), 0); d =
// max_over_sub_blocks(ideal_scale) / 63 (round-tripped through float16);
// dmin = max_over_sub_blocks(ideal_neg_offset) / 63 (round-tripped);
// sc_j = round(ideal_scale_j / d) clamped to [0, 63]; m_j =
// round(ideal_neg_offset_j / dmin) clamped to [0, 63]; sub_scale_j =
// d * sc_j; sub_min_j = -(dmin * m_j); q = round(clip((value - sub_min_j)
// / sub_scale_j, 0, 15)); dequant = q * sub_scale_j + sub_min_j.
inline void QuantizeDequantizeQ4KSuperBlock(double* data, int64_t count) {
  const int64_t num_sub_blocks = (count + kSubBlockSize - 1) / kSubBlockSize;
  std::vector<double> ideal_scale(num_sub_blocks);
  std::vector<double> ideal_neg_offset(num_sub_blocks);
  double max_ideal_scale = 0.0;
  double max_ideal_neg_offset = 0.0;
  for (int64_t j = 0; j < num_sub_blocks; ++j) {
    const int64_t start = j * kSubBlockSize;
    const int64_t sub_count = std::min(kSubBlockSize, count - start);
    double lo = data[start];
    double hi = data[start];
    for (int64_t i = 1; i < sub_count; ++i) {
      lo = std::min(lo, data[start + i]);
      hi = std::max(hi, data[start + i]);
    }
    ideal_scale[j] = std::max(hi - lo, 1e-12) / static_cast<double>(kMaxCode);
    ideal_neg_offset[j] = std::max(-lo, 0.0);
    max_ideal_scale = std::max(max_ideal_scale, ideal_scale[j]);
    max_ideal_neg_offset = std::max(max_ideal_neg_offset, ideal_neg_offset[j]);
  }

  const double d = RoundTripFloat16(std::max(max_ideal_scale, 1e-12) /
                                    static_cast<double>(kMaxSubScaleCode));
  const double dmin = RoundTripFloat16(std::max(max_ideal_neg_offset, 1e-12) /
                                       static_cast<double>(kMaxSubScaleCode));

  for (int64_t j = 0; j < num_sub_blocks; ++j) {
    const int64_t start = j * kSubBlockSize;
    const int64_t sub_count = std::min(kSubBlockSize, count - start);
    double sc = std::round(ideal_scale[j] / d);
    sc = std::min(std::max(sc, 0.0), static_cast<double>(kMaxSubScaleCode));
    double m = std::round(ideal_neg_offset[j] / dmin);
    m = std::min(std::max(m, 0.0), static_cast<double>(kMaxSubScaleCode));

    const double sub_scale = d * sc;
    const double sub_min = -(dmin * m);
    const double safe_sub_scale = sub_scale > 0.0 ? sub_scale : 1.0;
    for (int64_t i = 0; i < sub_count; ++i) {
      double code = std::round((data[start + i] - sub_min) / safe_sub_scale);
      code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
      data[start + i] = code * sub_scale + sub_min;
    }
  }
}

}  // namespace gguf_q4_k_detail

// llama.cpp's Q4_K K-quant format -- matches MatMul/vanilla-Gemm the same
// way GgufQ6K/GgufQ2K do, then quantizes the flattened weight
// super-block-by-super-block.
struct GgufQ4K final : public PredicateBasedPass {
  explicit GgufQ4K()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "gguf_q4_k"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    return w_t != nullptr && w_t->elem_type() == TensorProto_DataType_FLOAT &&
           w_t->sizes().size() == 2;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    const auto& sizes = w_t->sizes();
    const int64_t numel = sizes[0] * sizes[1];
    const std::vector<float> data = ReadFloatMatrix(*w_t);

    std::vector<double> out_data(data.begin(), data.end());
    for (int64_t start = 0; start < numel;
         start += gguf_q4_k_detail::kSuperBlockSize) {
      const int64_t count =
          std::min(gguf_q4_k_detail::kSuperBlockSize, numel - start);
      gguf_q4_k_detail::QuantizeDequantizeQ4KSuperBlock(out_data.data() + start,
                                                        count);
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = sizes;
    std::vector<float> out_float(out_data.begin(), out_data.end());
    w_out.floats() = std::move(out_float);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
