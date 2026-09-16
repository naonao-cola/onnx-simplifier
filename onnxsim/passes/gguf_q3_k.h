// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's GGUF "Q3_K" K-quant format -- C++ port of gguf_q3_k.py's own
// apply_gguf_q3_k_quantization. See that module's docstring for the full
// rationale and the exact reconstruction formula this port targets
// (transcribed from, and kept consistent with, this repo's own verified
// onnxsim/ggml_kquant.h decoder's DequantizeQ3_KBlock): a 256-element
// super-block split into 16 sub-blocks of 16 elements, each sub-block
// getting its own 6-bit unsigned scale code sc_j ([0, 63]), made signed
// by a fixed -32 offset, times one shared float16 super-block scale
// d_all, times the format's own asymmetric 3-bit element code q
// ([-4, 3]) -- ``dequant = d_all * (sc_j - 32) * q``. Structurally this
// is gguf_q6_k.h's own shape (a single shared super-block reference, no
// separate min as in Q2_K/Q4_K/Q5_K), just with a coarser, asymmetric
// element code and this encoder's own choice (matching gguf_q6_k.h's
// own encoder) to only ever emit non-negative scale-offset codes
// (sc_j - 32 in [0, 31]).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 256-element-
//                                   super-block Q3_K quantize-dequantize
//                                   round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_q6_k.h/gguf_legacy_quant.h use.
// Unlike gguf_q3_k.py's own apply_gguf_q3_k_quantization, this port does
// not add an include_conv option -- several other *_cpp ports in this
// repo already establish that a C++ port need not mirror every optional
// knob its Python counterpart has.
//
// Super-blocks are laid out over the weight's own flattened, row-major
// storage, exactly like gguf_q6_k.h's/gguf_legacy_quant.h's own block
// layout -- NOT per (output-channel, K-block). A ragged final super-block
// (when the weight's total element count isn't itself a multiple of 256)
// is quantized using only its own real elements, one ragged sub-block at
// a time; the same max-based zero-padding-invariance argument
// gguf_legacy_quant.h's own comment gives applies to each sub-block's own
// max(|.|) here too.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM gguf_q3_k.py: as with gguf_q6_k.h,
// this scheme has no accumulation/iterative-refinement step -- every
// sub-block's own scale is computed independently from that sub-block's
// own max(|.|), and every super-block's own shared reference d_all from
// the max over its own 16 sub-block scales, so this port is expected to
// track the Python port's own float64 numpy implementation closely, up
// to float16 round-trip and floating-point summation order differences.
// apply_gguf_q3_k_quantization and its _cpp counterpart remain
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

namespace gguf_q3_k_detail {

constexpr int64_t kSubBlockSize = 16;
constexpr int64_t kSubBlocksPerSuperBlock = 16;
constexpr int64_t kSuperBlockSize =
    kSubBlockSize * kSubBlocksPerSuperBlock;  // 256
// The real format's own asymmetric 3-bit element code range (2 low bits
// plus one high-mask bit): {-4, ..., 3}, one more level below zero than
// above -- see gguf_q3_k.py's own docstring.
constexpr int64_t kCodeMin = -4;
constexpr int64_t kCodeMax = 3;
constexpr int64_t kScaleMagnitude = -kCodeMin;  // 4
constexpr int64_t kMaxScaleOffsetCode = 31;  // this encoder's sc_j - 32 range

// Round-trips a float32 value through ggml_half (IEEE754 binary16),
// reproducing the same precision loss real Q3_K's d_all field carries --
// mirrors gguf_q3_k.py's own `.astype(np.float16).astype(np.float64)`
// step. Reuses this repo's own already-verified fp16 codec, the same
// reuse gguf_q6_k.h's own RoundTripFloat16 makes.
inline double RoundTripFloat16(double value) {
  return static_cast<double>(::onnxsim::tensor_pool::gguf::Float16BitsToFloat32(
      FloatToFloat16Bits(static_cast<float>(value))));
}

// Quantize-dequantize round trip for one Q3_K super-block of `count` (<=
// kSuperBlockSize) contiguous elements, written back in place. Mirrors
// gguf_q3_k.py's own _quantize_dequantize_superblocks: for each
// (possibly ragged) 16-element sub-block j, ideal_scale_j =
// max(|sub_block|) / 4; d_all = max_over_sub_blocks(ideal_scale) / 31
// (round-tripped through float16); scale_offset_j =
// round(ideal_scale_j / d_all) clamped to [0, 31]; sub_scale_j =
// d_all * scale_offset_j; q = round(clip(value / sub_scale_j, -4, 3));
// dequant = q * sub_scale_j.
inline void QuantizeDequantizeQ3KSuperBlock(double* data, int64_t count) {
  const int64_t num_sub_blocks = (count + kSubBlockSize - 1) / kSubBlockSize;
  std::vector<double> ideal_scale(num_sub_blocks);
  double max_ideal_scale = 0.0;
  for (int64_t j = 0; j < num_sub_blocks; ++j) {
    const int64_t start = j * kSubBlockSize;
    const int64_t sub_count = std::min(kSubBlockSize, count - start);
    double max_abs = 0.0;
    for (int64_t i = 0; i < sub_count; ++i) {
      max_abs = std::max(max_abs, std::fabs(data[start + i]));
    }
    ideal_scale[j] =
        std::max(max_abs, 1e-12) / static_cast<double>(kScaleMagnitude);
    max_ideal_scale = std::max(max_ideal_scale, ideal_scale[j]);
  }

  const double d_all =
      RoundTripFloat16(std::max(max_ideal_scale, 1e-12) /
                       static_cast<double>(kMaxScaleOffsetCode));

  for (int64_t j = 0; j < num_sub_blocks; ++j) {
    const int64_t start = j * kSubBlockSize;
    const int64_t sub_count = std::min(kSubBlockSize, count - start);
    double scale_offset = std::round(ideal_scale[j] / d_all);
    scale_offset = std::min(std::max(scale_offset, 0.0),
                            static_cast<double>(kMaxScaleOffsetCode));
    const double sub_scale = d_all * scale_offset;
    const double safe_sub_scale = sub_scale > 0.0 ? sub_scale : 1.0;
    for (int64_t i = 0; i < sub_count; ++i) {
      double code = std::round(data[start + i] / safe_sub_scale);
      code = std::min(std::max(code, static_cast<double>(kCodeMin)),
                      static_cast<double>(kCodeMax));
      data[start + i] = code * sub_scale;
    }
  }
}

}  // namespace gguf_q3_k_detail

// llama.cpp's Q3_K K-quant format -- matches MatMul/vanilla-Gemm the same
// way GgufQ6K/GgufQ2K do, then quantizes the flattened weight
// super-block-by-super-block.
struct GgufQ3K final : public PredicateBasedPass {
  explicit GgufQ3K()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "gguf_q3_k"; }

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
         start += gguf_q3_k_detail::kSuperBlockSize) {
      const int64_t count =
          std::min(gguf_q3_k_detail::kSuperBlockSize, numel - start);
      gguf_q3_k_detail::QuantizeDequantizeQ3KSuperBlock(out_data.data() + start,
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
