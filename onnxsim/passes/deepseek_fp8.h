// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// DeepSeek-V3-style fine-grained block FP8 weight quantization -- C++ port
// of the *weight* half of deepseek_fp8.py's own apply_deepseek_fp8. See
// that module's docstring for the full rationale (DeepSeek-AI, 2024,
// "DeepSeek-V3 Technical Report", Section 3.3): the weight, oriented
// output-channel-first ([N, K]), is tiled into 128x128 blocks; each block
// gets its own scale, `scale = max(|block|) / 448` (448 is FLOAT8E4M3FN's
// own largest finite magnitude, floored at 1e-12 to avoid a divide by
// zero on an all-zero block), and every element in the block is quantized
// via a **real** FLOAT8E4M3FN round-trip (round-to-nearest, ties-to-even,
// saturating) at `value / scale`, then dequantized back and rescaled by
// `scale`.
//
// SCOPE NARROWING: deepseek_fp8.py's own apply_deepseek_fp8 is a W8A8
// scheme -- it ALSO inserts new Reshape/Abs/ReduceMax/Clip/Div/Cast/Mul
// nodes to block-quantize the *activation* at graph-run time (one scale
// per token per 128-element channel group). This port only reproduces
// the **weight** side (the part that needs no calibration data and no
// new graph nodes at all -- just a replaced initializer, exactly this
// repo's own established weight-only *_cpp port shape). Activation
// quantization is out of scope for this pass entirely, the same kind of
// scope decision gguf_q6_k.h already makes for Conv (documented there as
// "a C++ port need not mirror every optional knob its Python counterpart
// has") -- except here the omitted half is not a knob but the entire
// activation-side rewrite, so the result of this pass is only ever a
// weight-only FP8 round trip, never a full W8A8 model. Given
// deepseek_fp8.py's own scope note that this repo's other FP8 pass
// (quantize_fp8.h/QuantizeFp8Pass) is a whole-tensor unscaled cast, this
// port's own name reflects that it is specifically the *weight-blockwise*
// scheme, not a generic "fp8" alias.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 128x128-block
//                                   FP8 E4M3 quantize-dequantize round
//                                   trip (activation X left untouched)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_q6_k.h/gguf_legacy_quant.h use.
// Blocks tile BOTH axes of the weight's own logical [N, K] (output-
// channel-first) orientation -- Gemm's transB==1 already stores the
// weight that way ([N, K] directly); MatMul's plain [K, N] weight is
// tiled by swapping which axis plays N/K when indexing, without an
// actual in-memory transpose, then written back in the tensor's own
// original storage order. A ragged final block along either axis (when a
// dimension isn't itself a multiple of 128) is quantized using only its
// own real elements -- the same max-based zero-padding-invariance
// argument gguf_legacy_quant.h's own comment gives applies unchanged
// here, since appending zero-valued padding can never change a block's
// own max(|.|) unless the whole block is already all-zero.
//
// NUMERICAL NOTE (unlike this repo's other GGUF-family *_cpp ports):
// every other weight-only *_cpp port in this repo carries an "ACCEPTED,
// PERMANENT DIVERGENCE" note because its Python counterpart's own
// encoder is an honestly-scoped, not-independently-verified heuristic
// (llama.cpp's own real encoder procedure was unavailable to read
// alongside its decoder). This format is different: DeepSeek-V3's own
// block-FP8 scheme has no such ambiguity -- both directions are a real,
// fully-specified FLOAT8E4M3FN cast (round-to-nearest, ties-to-even,
// saturating), reusing this repo's own already-verified encoder
// (quantize_fp8.h's FloatToFloat8Bits, the same bit-exact routine
// QuantizeFp8Pass itself uses for onnxsim's other, whole-tensor FP8
// pass) plus a symmetric decoder implemented here. So this port is
// expected to track deepseek_fp8.py's own quantize_dequantize_block_fp8
// (which itself delegates to ml_dtypes' real FLOAT8E4M3FN cast) far more
// closely than the GGUF family's honestly-scoped heuristics -- verified
// empirically in this port's own cross-check test, not merely asserted.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_fp8.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace deepseek_fp8_detail {

constexpr int64_t kBlockSize = 128;

// Decodes a FLOAT8E4M3FN bit pattern back to float32 -- the missing other
// half of quantize_fp8.h's own FloatToFloat8Bits encoder (which this file
// reuses unchanged for the forward direction). Mirrors FLOAT8E4M3FN's
// layout exactly (1 sign, 4 exponent bits, bias 7, 3 mantissa bits, no
// infinities, a single NaN encoding at exponent=1111/mantissa=111): every
// value this can possibly decode is either exactly representable in
// float32 with no rounding at all (a small integer mantissa multiplied by
// an exact power of two via std::ldexp) or the canonical NaN.
inline float Float8E4M3FNBitsToFloat32(uint8_t bits) {
  const uint32_t sign32 = (static_cast<uint32_t>(bits) & 0x80u) << 24;
  const uint32_t exp = (static_cast<uint32_t>(bits) >> 3) & 0xFu;
  const uint32_t mant = static_cast<uint32_t>(bits) & 0x7u;

  auto with_sign = [sign32](float magnitude) {
    uint32_t bits32;
    std::memcpy(&bits32, &magnitude, sizeof(bits32));
    bits32 |= sign32;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
  };

  if (exp == 0xFu && mant == 0x7u) {
    return with_sign(std::numeric_limits<float>::quiet_NaN());
  }
  if (exp == 0) {
    if (mant == 0) {
      return with_sign(0.0f);
    }
    // Subnormal: value = (mant / 8) * 2^(1 - bias), bias = 7.
    return with_sign(static_cast<float>(mant) / 8.0f * std::ldexp(1.0f, -6));
  }
  // Normalized: value = (1 + mant / 8) * 2^(exp - bias).
  return with_sign((1.0f + static_cast<float>(mant) / 8.0f) *
                   std::ldexp(1.0f, static_cast<int>(exp) - 7));
}

// Quantize-dequantize round trip for one value already divided by its
// own block's scale -- round-to-nearest ties-to-even, saturating at
// FLOAT8E4M3FN's own +-448 range (matches quantize_fp8.h's own
// FloatToFloat8Bits semantics exactly, the same routine QuantizeFp8Pass
// itself uses).
inline double Fp8E4M3RoundTrip(double scaled_value) {
  const uint8_t bits = FloatToFloat8Bits(static_cast<float>(scaled_value),
                                         Float8Format::kE4M3FN);
  return static_cast<double>(Float8E4M3FNBitsToFloat32(bits));
}

// Quantize-dequantize round trip for the weight's own logical [N, K]
// (output-channel-first) view, tiled into 128x128 blocks, written back
// in place into `data` -- `data` holds the tensor's own original
// [rows, cols] row-major storage (rows/cols as declared by the tensor's
// own shape, NOT necessarily [N, K] -- `weight_transposed` says which).
// Mirrors deepseek_fp8.py's own quantize_dequantize_block_fp8, applied to
// `w if weight_transposed else w.T` there.
inline void QuantizeDequantizeBlockFp8(std::vector<float>& data, int64_t rows,
                                       int64_t cols, bool weight_transposed) {
  const int64_t n_extent = weight_transposed ? rows : cols;
  const int64_t k_extent = weight_transposed ? cols : rows;
  // Gemm's transB==1 stores the weight as [N, K] directly (row-major:
  // element (n, k) at n * cols + k, cols == K); MatMul's plain weight is
  // [K, N] (row-major: element (n, k) -- i.e. output channel n, reduction
  // index k -- lives at k * cols + n, cols == N). Either way this indexes
  // the tensor's own real storage directly, without an actual transpose.
  auto at = [&](int64_t n_idx, int64_t k_idx) -> float& {
    return weight_transposed ? data[n_idx * cols + k_idx]
                             : data[k_idx * cols + n_idx];
  };

  for (int64_t n0 = 0; n0 < n_extent; n0 += kBlockSize) {
    const int64_t n_count = std::min(kBlockSize, n_extent - n0);
    for (int64_t k0 = 0; k0 < k_extent; k0 += kBlockSize) {
      const int64_t k_count = std::min(kBlockSize, k_extent - k0);

      double max_abs = 0.0;
      for (int64_t ni = 0; ni < n_count; ++ni) {
        for (int64_t ki = 0; ki < k_count; ++ki) {
          max_abs = std::max(
              max_abs, std::fabs(static_cast<double>(at(n0 + ni, k0 + ki))));
        }
      }
      const double scale =
          std::max(max_abs, 1e-12) / static_cast<double>(kFloat8E4M3FNMax);

      for (int64_t ni = 0; ni < n_count; ++ni) {
        for (int64_t ki = 0; ki < k_count; ++ki) {
          float& v = at(n0 + ni, k0 + ki);
          const double dequant =
              Fp8E4M3RoundTrip(static_cast<double>(v) / scale) * scale;
          v = static_cast<float>(dequant);
        }
      }
    }
  }
}

}  // namespace deepseek_fp8_detail

// DeepSeek-V3's own weight-blockwise FP8 scheme -- matches MatMul/vanilla-
// Gemm the same way every sibling weight-only *_cpp port does, then
// quantizes the weight's own logical [N, K] view block-by-block.
struct DeepSeekFp8 final : public PredicateBasedPass {
  explicit DeepSeekFp8()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "deepseek_fp8"; }

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
    std::vector<float> data = ReadFloatMatrix(*w_t);
    deepseek_fp8_detail::QuantizeDequantizeBlockFp8(data, sizes[0], sizes[1],
                                                    info.weight_transposed);

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = sizes;
    w_out.floats() = std::move(data);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
