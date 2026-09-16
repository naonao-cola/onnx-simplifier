// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// ICQuant (Li, Hanna, Fragouli, Diggavi, 2025, "ICQuant: Index Coding
// enables Low-bit LLM Quantization") -- C++ port of icquant.py's own
// quantize_weight_only_icquant. See that module's own docstring for the
// full rationale: per (output-channel, group_size-element K-block), the
// `num_outliers` largest-magnitude elements are excluded from the
// group's own scale computation and reconstructed at exact full
// precision instead; the remaining elements are quantized to a
// symmetric 7-level-per-side INT4 grid. ICQuant's own distinguishing
// contribution -- a combinatorial-number-system ("combinadic") encoding
// of which positions were chosen, cheaper than a bitmask or an explicit
// index list -- is a pure storage/graph-representation compaction trick
// with no effect on which positions are chosen or what values they get:
// icquant.py's own encoder immediately decodes the rank it just computed
// (`assert decoded == combo`) and bakes the *decoded* explicit index
// list into the graph either way. This port therefore skips the
// combinadic encode/decode round trip entirely -- it has no observable
// effect on the reconstructed float values, only on an intermediate
// representation this port doesn't build in the first place (see below).
//
// Reconstruction, per (output-channel row, K-block) group of
// icquant_detail::kGroupSize contiguous-in-K elements (transcribed from
// icquant.py's own quantize_weight_only_icquant):
//   outliers = the num_outliers largest-|value| positions in the group
//   scale = max(max(|non-outlier values|), 1e-12) / 7
//   code_i = clip(round(value_i / scale), -7, 7)          (non-outliers)
//   dequant_i = code_i * scale                             (non-outliers)
//   dequant_i = value_i                                    (outliers, exact)
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, K-block) group
//                                   replaced by its own ICQuant
//                                   quantize-dequantize round trip,
//                                   outlier positions reconstructed
//                                   exactly
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm
// with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
// constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by icquant_detail::kGroupSize and greater than num_outliers
// -- matching icquant.py's own scope. Unlike icquant.py's own
// quantize_weight_only_icquant, this port does not build the real
// INT4/DequantizeLinear/ScatterND/MatMul/Add graph rewrite (which also
// needs opset 21 for INT4 and DequantizeLinear's own block_size
// attribute) -- following the exact convention this repo's other
// outlier-aware *_cpp ports already establish, this port instead folds
// the reconstructed float32 values directly into a replacement
// initializer (no new graph nodes, no opset gate at all), and hardcodes
// icquant.py's own defaults (group_size=32, num_outliers=1) rather than
// exposing them as parameters -- several other *_cpp ports in this repo
// already establish that a C++ port need not mirror every optional knob
// its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM icquant.py: this scheme has no
// accumulation/iterative-refinement step -- every group's own scale and
// outlier selection are computed independently from that group's own
// values -- so this port is expected to track the Python port's own
// float64 numpy implementation closely, up to floating-point
// summation-order differences and up to how ties are broken among
// equal-magnitude candidates for the last outlier slot (icquant.py's own
// np.argpartition makes no order guarantee among equal keys either, so
// neither side claims a canonical tie-break). quantize_weight_only_icquant
// and this port remain independently-correct, non-interchangeable entry
// points -- this port's own tests check structural/algebraic properties
// and comparable (not bit-identical) reconstruction error, matching this
// repo's established contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace icquant_detail {

constexpr int64_t kGroupSize = 32;
constexpr int64_t kNumOutliers = 1;
constexpr double kMaxCode = 7.0;  // symmetric 4-bit code magnitude side

// Quantize-dequantize round trip for one (output-channel, K-block) group
// of exactly kGroupSize contiguous elements, written back in place.
// Mirrors icquant.py's own per-group logic: the kNumOutliers
// largest-|value| positions are excluded from the scale computation and
// reconstructed exactly; every other position snaps to a symmetric
// 7-level-per-side grid. Ties among equal-magnitude candidates for the
// last outlier slot are broken by ascending index (an arbitrary but
// deterministic choice -- see this header's own top-of-file comment on
// why neither side claims a canonical tie-break).
inline void QuantizeDequantizeICQuantGroup(double* group, int64_t count) {
  std::vector<int64_t> order(static_cast<size_t>(count));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return std::fabs(group[a]) > std::fabs(group[b]);
  });

  std::vector<bool> is_outlier(static_cast<size_t>(count), false);
  const int64_t num_outliers = std::min(kNumOutliers, count);
  for (int64_t i = 0; i < num_outliers; ++i) {
    is_outlier[static_cast<size_t>(order[static_cast<size_t>(i)])] = true;
  }

  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    if (!is_outlier[static_cast<size_t>(i)]) {
      max_abs = std::max(max_abs, std::fabs(group[i]));
    }
  }
  const double scale = std::max(max_abs, 1e-12) / kMaxCode;

  for (int64_t i = 0; i < count; ++i) {
    if (is_outlier[static_cast<size_t>(i)]) {
      continue;  // reconstructed exactly -- leave group[i] untouched
    }
    double code = std::round(group[i] / scale);
    code = std::min(std::max(code, -kMaxCode), kMaxCode);
    group[i] = code * scale;
  }
}

}  // namespace icquant_detail

// ICQuant -- matches MatMul/vanilla-Gemm the same way every sibling
// *_cpp port does, then quantizes each (output-channel, K-block) group
// of the weight independently. Blocks are grouped along the reduction
// dimension K *per output channel* (icquant.py's own [N, K] convention),
// exactly like hqq.h's own layout -- so a Gemm's transB=1 weight
// (already stored [N, K]) is walked contiguously, while a MatMul's
// weight (stored [K, N]) is walked with an N-stride, without physically
// transposing memory.
struct ICQuant final : public PredicateBasedPass {
  explicit ICQuant()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "icquant"; }

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
    return k % icquant_detail::kGroupSize == 0 &&
           icquant_detail::kNumOutliers < icquant_detail::kGroupSize;
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
    if (k_dim % icquant_detail::kGroupSize != 0 ||
        icquant_detail::kNumOutliers >= icquant_detail::kGroupSize) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);

    // Materialize the logical [N, K] view (icquant.py's own w if
    // weight_transposed else w.T) so each (output-channel, K-block)
    // group is contiguous for the block kernel above.
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
           k_start += icquant_detail::kGroupSize) {
        icquant_detail::QuantizeDequantizeICQuantGroup(
            row + k_start, icquant_detail::kGroupSize);
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
