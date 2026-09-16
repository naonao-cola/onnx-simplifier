// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// OliVe -- Outlier-Victim Pair quantization (Guo et al., ISCA 2023,
// "OliVe: Accelerating Large Language Models via Hardware-friendly
// Outlier-Victim Pair Quantization") -- C++ port of olive.py's own
// quantize_weight_only_olive. See that module's own docstring for the
// full rationale and the exact bit-accounting argument (an OVP pair costs
// exactly the same total bit budget as two ordinary group-wide-quantized
// elements would). Reconstruction, per (output channel, block_size=32)
// group along the reduction axis (transcribed from, and kept consistent
// with, olive.py's own _olive_quantize_blockwise):
//   typical_scale = median(|block|)
//   is_outlier[i] = |block[i]| > 4.0 * typical_scale     (outlier_threshold)
//   base_scale    = max(|non-outlier elements|, eps) / 7      (ordinary_qmax,
//                   bits=4; falls back to max(|block|) if every element in
//                   the block is an outlier)
//   outlier_scale = max(|outlier elements|, eps) / 15          (outlier_qmax,
//                   bits+1; falls back to base_scale if the block has no
//                   outlier at all)
// Elements are grouped into non-overlapping adjacent pairs (2i, 2i+1)
// within the block. A pair with *exactly one* outlier member becomes an
// OVP pair: its outlier element quantizes against outlier_scale with a
// wider ([-15, 15]) code, its victim element quantizes against the SAME
// base_scale as ordinary elements but with a narrower ([-3, 3], victim_qmax
// = bits-2) code. A pair with zero or two outlier members is declined --
// both members fall back to ordinary quantization ([-7, 7] against
// base_scale).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, 32-element K-block)
//                                   group replaced by its own OVP-encoded
//                                   quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// olive_detail::kBlockSize -- the same scope gguf_q6_k.h/hqq.h use, plus
// the block-divisibility requirement olive.py's own encoder already
// imposes (a layer that fails it is left untouched, exactly like the
// Python side's own `continue`).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM olive.py, two-fold: (1) unlike
// olive.py's own quantize_weight_only_olive, which builds a real
// <int8 codes>+BaseScale+OutlierScale+OutlierMask graph rewrite
// (DequantizeLinear x2 + Cast + Where + MatMul[+Add], needing opset 21
// for DequantizeLinear's own block_size attribute), this port folds the
// round trip directly into a plain replacement float32 initializer
// instead -- several other *_cpp ports in this repo already establish
// that a C++ port need not mirror every optional knob or exact tensor
// representation its Python counterpart has, and this port consequently
// needs no opset gate at all, unlike the Python side. (2) bits/block_size/
// outlier_threshold are fixed at olive.py's own defaults (4, 32, 4.0)
// rather than exposed as pass parameters, matching every other data-free
// *_cpp port's no-configurable-knobs convention. Numerically, this scheme
// has no accumulation/iterative-refinement step -- every block's own
// (base_scale, outlier_scale) and every pair's own outlier/victim
// assignment are computed independently per block -- so this port is
// expected to track the Python port's own float64 numpy implementation
// closely, up to floating-point summation-order/median-tie-breaking
// differences. quantize_weight_only_olive and this port's _cpp
// counterpart remain independently-correct, non-interchangeable entry
// points.

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

namespace olive_detail {

constexpr int64_t kBits = 4;
constexpr int64_t kBlockSize = 32;
constexpr double kOutlierThreshold = 4.0;
constexpr double kEps = 1e-12;

// 2**(num_bits-1) - 1, mirroring olive.py's own _qmax.
constexpr int64_t QMax(int64_t num_bits) {
  return (int64_t{1} << (num_bits - 1)) - 1;
}

constexpr int64_t kOrdinaryQMax = QMax(kBits);     // 7
constexpr int64_t kOutlierQMax = QMax(kBits + 1);  // 15
constexpr int64_t kVictimQMax = QMax(kBits - 1);   // 3

// OVP-encodes one (output-channel, block) group of exactly kBlockSize
// contiguous elements, written back in place. Mirrors olive.py's own
// _olive_quantize_blockwise -- see this header's own top-of-file comment
// for the exact formula. `count` is always kBlockSize here (an even
// number), matching olive.py's own even-block_size requirement.
inline void OliveQuantizeBlock(double* block, int64_t count) {
  std::vector<double> abs_vals(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    abs_vals[static_cast<size_t>(i)] = std::fabs(block[i]);
  }

  // np.median over an even-length array: the mean of the two middle
  // values of the sorted array.
  std::vector<double> sorted_abs = abs_vals;
  std::sort(sorted_abs.begin(), sorted_abs.end());
  const int64_t mid = count / 2;
  const double typical_scale = count % 2 == 0
                                   ? (sorted_abs[static_cast<size_t>(mid - 1)] +
                                      sorted_abs[static_cast<size_t>(mid)]) /
                                         2.0
                                   : sorted_abs[static_cast<size_t>(mid)];

  std::vector<bool> is_outlier(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    is_outlier[static_cast<size_t>(i)] =
        abs_vals[static_cast<size_t>(i)] > kOutlierThreshold * typical_scale;
  }

  // base_scale: from non-outlier elements only, falling back to every
  // element's own |.| if the whole block is outliers.
  double base_max = 0.0;
  bool any_non_outlier = false;
  for (int64_t i = 0; i < count; ++i) {
    if (!is_outlier[static_cast<size_t>(i)]) {
      any_non_outlier = true;
      base_max = std::max(base_max, abs_vals[static_cast<size_t>(i)]);
    }
  }
  if (!any_non_outlier) {
    for (int64_t i = 0; i < count; ++i) {
      base_max = std::max(base_max, abs_vals[static_cast<size_t>(i)]);
    }
  }
  const double base_scale =
      std::max(base_max, kEps) / static_cast<double>(kOrdinaryQMax);

  // outlier_scale: from outlier elements only, falling back to base_scale
  // if the block has no outlier at all.
  double outlier_max = 0.0;
  bool any_outlier = false;
  for (int64_t i = 0; i < count; ++i) {
    if (is_outlier[static_cast<size_t>(i)]) {
      any_outlier = true;
      outlier_max = std::max(outlier_max, abs_vals[static_cast<size_t>(i)]);
    }
  }
  const double outlier_scale =
      any_outlier
          ? std::max(outlier_max, kEps) / static_cast<double>(kOutlierQMax)
          : base_scale;

  // Adjacent, non-overlapping pairing: (2i, 2i+1). A pair with exactly one
  // outlier member becomes an OVP pair (its outlier gets the wide code
  // against outlier_scale, its neighbor becomes the victim, a narrow code
  // against base_scale); a pair with zero or two outliers is declined --
  // both members fall back to ordinary quantization against base_scale.
  for (int64_t p = 0; p < count / 2; ++p) {
    const int64_t i0 = 2 * p;
    const int64_t i1 = 2 * p + 1;
    const bool ovp_pair = is_outlier[static_cast<size_t>(i0)] !=
                          is_outlier[static_cast<size_t>(i1)];
    for (const int64_t idx : {i0, i1}) {
      double code;
      double scale;
      if (!ovp_pair) {
        code = std::round(block[idx] / base_scale);
        code = std::min(std::max(code, -static_cast<double>(kOrdinaryQMax)),
                        static_cast<double>(kOrdinaryQMax));
        scale = base_scale;
      } else if (is_outlier[static_cast<size_t>(idx)]) {
        code = std::round(block[idx] / outlier_scale);
        code = std::min(std::max(code, -static_cast<double>(kOutlierQMax)),
                        static_cast<double>(kOutlierQMax));
        scale = outlier_scale;
      } else {
        code = std::round(block[idx] / base_scale);
        code = std::min(std::max(code, -static_cast<double>(kVictimQMax)),
                        static_cast<double>(kVictimQMax));
        scale = base_scale;
      }
      block[idx] = code * scale;
    }
  }
}

}  // namespace olive_detail

// OliVe's OVP encoding -- matches MatMul/vanilla-Gemm the same way
// GgufQ6K/HQQ do, then quantizes each (output-channel, K-block) group of
// the weight independently. Unlike the row-major-flattened block layout
// every sibling GGUF port uses, OliVe's own blocks are grouped along the
// reduction dimension K *per output channel* (olive.py's own [N, K]
// convention) -- so a Gemm's transB=1 weight (already stored [N, K]) is
// walked contiguously, while a MatMul's weight (stored [K, N]) is walked
// with an N-stride, mirroring olive.py's own `w_nk = w if
// weight_transposed else w.T` normalization exactly, without physically
// transposing memory.
struct Olive final : public PredicateBasedPass {
  explicit Olive()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "olive"; }

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
    return k % olive_detail::kBlockSize == 0;
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
    if (k_dim % olive_detail::kBlockSize != 0) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);

    // Materialize the logical [N, K] view (olive.py's own w if
    // weight_transposed else w.T) so each (output-channel, K-block) group
    // is contiguous for the block kernel above.
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
           k_start += olive_detail::kBlockSize) {
        olive_detail::OliveQuantizeBlock(row + k_start,
                                         olive_detail::kBlockSize);
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
