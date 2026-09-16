// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// HQQ -- Half-Quadratic Quantization (Badri & Shaji, 2023) -- C++ port of
// hqq.py's own quantize_weight_only_int4_hqq. See that module's docstring
// for the full rationale: an asymmetric (nonzero zero-point) affine INT4
// quantizer whose zero-point is refined per (output-channel, K-block)
// group by a bounded number of Iteratively Reweighted Least Squares
// (IRLS) steps targeting a robust (Lp, p < 2) reconstruction loss instead
// of the ordinary L2 a single min/max-derived fit implicitly optimizes --
// elements with a larger current residual are downweighted before each
// re-solve, so a couple of outliers can no longer force every other
// element in the block toward a handful of coarse levels the way an
// unweighted min/max fit would let them.
//
// Reconstruction, per (output-channel n, K-block j) group of
// hqq_detail::kBlockSize contiguous-in-K elements (transcribed from, and
// kept consistent with, hqq.py's own _irls_affine_quantize_blockwise):
//   scale = max((max(block) - min(block)) / 15, 1e-12)
//   zero_0 = -min(block) / scale                       (initial fit)
//   repeat hqq_detail::kNumIterations times:
//     code_i = round(value_i / scale + zero)
//     d_i = value_i - scale * code_i
//     resid_i = value_i - scale * (code_i - zero)
//     w_i = (|resid_i| + 1e-8) ^ (hqq_detail::kLpNorm - 2)
//     zero = -sum(w_i * d_i) / (scale * sum(w_i) + 1e-8)   (closed-form
//                                                            weighted LS)
//   zero = clip(round(zero), 0, 15)
//   code_i = clip(round(value_i / scale + zero), 0, 15)
//   dequant_i = scale * (code_i - zero)
// Only the scale is fixed up front; the zero-point alone is IRLS-refined,
// matching hqq.py's own note that "most of the benefit comes from the
// zero-point fit."
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, 32-element K-block)
//                                   group replaced by its own HQQ
//                                   quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// hqq_detail::kBlockSize -- the same scope gguf_q6_k.h/gguf_q2_k.h use,
// plus the block-divisibility requirement hqq.py's own encoder already
// imposes (a layer that fails it is left untouched, exactly like the
// Python side's own `continue`).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM hqq.py, two-fold: (1) unlike
// hqq.py's own quantize_weight_only_int4_hqq, which builds a real
// DequantizeLinear(Wq, Ws, Wz, axis=..., block_size=...) node with packed
// UINT4 codes and zero-points (needing opset 21+ for UINT4/blocked
// DequantizeLinear), this port folds the round trip directly into a
// plain replacement float32 initializer instead -- several other *_cpp
// ports in this repo already establish that a C++ port need not mirror
// every optional knob or exact tensor representation its Python
// counterpart has, and this port consequently needs no opset gate at
// all, unlike the Python side. (2) block_size/num_iterations/lp_norm are
// fixed at hqq.py's own defaults (32, 10, 0.7) rather than exposed as
// pass parameters, matching every other data-free *_cpp port's
// no-configurable-knobs convention. Numerically, IRLS here is a
// deterministic fixed-point iteration with no RNG and no cross-block
// interaction, so this port is expected to track the Python port's own
// float64 numpy implementation closely, up to floating-point
// summation-order differences -- apply_gguf_q2_k_quantization-style
// "ACCEPTED, PERMANENT DIVERGENCE" caveat, not a genuine algorithmic one.
// quantize_weight_only_int4_hqq and this port's _cpp counterpart remain
// independently-correct, non-interchangeable entry points.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace hqq_detail {

constexpr int64_t kBlockSize = 32;
constexpr int64_t kNumIterations = 10;
constexpr double kLpNorm = 0.7;
constexpr int64_t kMaxCode = 15;  // unsigned 4-bit code range [0, 15]
constexpr double kEps = 1e-8;

// IRLS-refined asymmetric affine quantize-dequantize round trip for one
// (output-channel, K-block) group of exactly kBlockSize contiguous
// elements, written back in place. Mirrors hqq.py's own
// _irls_affine_quantize_blockwise -- see this header's own top-of-file
// comment for the exact formula.
inline void IrlsAffineQuantizeBlock(double* block, int64_t count) {
  double lo = block[0];
  double hi = block[0];
  for (int64_t i = 1; i < count; ++i) {
    lo = std::min(lo, block[i]);
    hi = std::max(hi, block[i]);
  }
  const double scale =
      std::max((hi - lo) / static_cast<double>(kMaxCode), 1e-12);
  double zero = -lo / scale;

  for (int64_t iter = 0; iter < kNumIterations; ++iter) {
    double weighted_d_sum = 0.0;
    double weight_sum = 0.0;
    for (int64_t i = 0; i < count; ++i) {
      const double code = std::round(block[i] / scale + zero);
      const double d = block[i] - scale * code;
      const double dequant = scale * (code - zero);
      const double resid = block[i] - dequant;
      const double weight = std::pow(std::fabs(resid) + kEps, kLpNorm - 2.0);
      weighted_d_sum += weight * d;
      weight_sum += weight;
    }
    zero = -weighted_d_sum / (scale * weight_sum + kEps);
  }

  zero =
      std::min(std::max(std::round(zero), 0.0), static_cast<double>(kMaxCode));
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round(block[i] / scale + zero);
    code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
    block[i] = scale * (code - zero);
  }
}

}  // namespace hqq_detail

// HQQ -- matches MatMul/vanilla-Gemm the same way GgufQ2K/GgufQ6K do, then
// quantizes each (output-channel, K-block) group of the weight
// independently. Unlike the row-major-flattened block layout every
// sibling GGUF port uses, HQQ's own blocks are grouped along the
// reduction dimension K *per output channel* (hqq.py's own [N, K]
// convention) -- so a Gemm's transB=1 weight (already stored [N, K]) is
// walked contiguously, while a MatMul's weight (stored [K, N]) is walked
// with an N-stride, mirroring hqq.py's own w if weight_transposed else
// w.T normalization exactly, without physically transposing memory.
struct HQQ final : public PredicateBasedPass {
  explicit HQQ()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "hqq"; }

  bool patternMatchPredicate(Node* n) override {
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
    const int64_t k = info.weight_transposed ? sizes[1] : sizes[0];
    return k % hqq_detail::kBlockSize == 0;
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
    const int64_t dim0 = sizes[0];
    const int64_t dim1 = sizes[1];
    const int64_t n_dim = info.weight_transposed ? dim0 : dim1;
    const int64_t k_dim = info.weight_transposed ? dim1 : dim0;
    if (k_dim % hqq_detail::kBlockSize != 0) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);

    // Materialize the logical [N, K] view (hqq.py's own w if
    // weight_transposed else w.T) so each (output-channel, K-block) group
    // is contiguous for the block kernel above, exactly like hqq.py's own
    // reshape(n, num_blocks, block_size) does after its own transpose.
    std::vector<double> w_nk(static_cast<size_t>(n_dim * k_dim));
    for (int64_t n_idx = 0; n_idx < n_dim; ++n_idx) {
      for (int64_t k_idx = 0; k_idx < k_dim; ++k_idx) {
        const int64_t src = info.weight_transposed ? n_idx * k_dim + k_idx
                                                   : k_idx * dim1 + n_idx;
        w_nk[static_cast<size_t>(n_idx * k_dim + k_idx)] = data[src];
      }
    }

    for (int64_t n_idx = 0; n_idx < n_dim; ++n_idx) {
      double* row = w_nk.data() + n_idx * k_dim;
      for (int64_t k_start = 0; k_start < k_dim;
           k_start += hqq_detail::kBlockSize) {
        hqq_detail::IrlsAffineQuantizeBlock(row + k_start,
                                            hqq_detail::kBlockSize);
      }
    }

    std::vector<float> out_float(static_cast<size_t>(dim0 * dim1));
    for (int64_t n_idx = 0; n_idx < n_dim; ++n_idx) {
      for (int64_t k_idx = 0; k_idx < k_dim; ++k_idx) {
        const int64_t dst = info.weight_transposed ? n_idx * k_dim + k_idx
                                                   : k_idx * dim1 + n_idx;
        out_float[static_cast<size_t>(dst)] = static_cast<float>(
            w_nk[static_cast<size_t>(n_idx * k_dim + k_idx)]);
      }
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = sizes;
    w_out.floats() = std::move(out_float);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
