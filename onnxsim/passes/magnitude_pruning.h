// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Magnitude pruning (Han et al., 2015) -- the data-free unstructured
// pruning baseline, C++ port of pruning.py's own apply_magnitude_pruning.
// Zeros the least-magnitude entries of every MatMul/vanilla-Gemm layer's
// constant 2-D float32 weight and every Conv layer's constant 4-D float32
// weight (ordinary, depthwise, and general grouped Conv alike -- see
// pruning.py's own module docstring for why grouping needs no special-
// casing for unstructured pruning), independently per output row/filter so
// a layer with row-dependent weight scale doesn't get some rows pruned to
// nothing and others left untouched.
//
// Scope note: unlike pruning.py's apply_magnitude_pruning, this port does
// not match com.microsoft::Attention's merged QKV weight (a rare contrib-op
// case) and does not offer N:M (semi-structured) pruning -- only the
// sparsity-ratio mode. Both remain available via the pure-Python
// implementation.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- and Conv are handled the same way, reshaped to
// [out_channels, flattened_rest] first):
//   Y = MatMul(X, W)         W constant, [K, N], float32
// After:
//   Wp = <W with each output channel's own lowest-|w| fraction zeroed>
//   Y  = MatMul(X, Wp)
//
// The weight's shape, dtype, and every other node attribute are unchanged;
// only element values within the existing tensor are zeroed (via a new
// initializer of the same shape, not an in-place edit -- matching every
// other onnxsim rewrite's "replace, don't mutate" convention for constants).
// Since the replacement is still a plain constant float32 tensor of the
// same shape, both patternMatchPredicate methods below explicitly check
// whether pruning would still change anything (any would-be-dropped entry
// that isn't already exactly zero) -- otherwise, unlike every other pass in
// this directory (whose replacement is no longer constant, or has changed
// kind/shape), the predicate would keep matching its own already-pruned
// output forever, looping OptimizeFixed's fixed point indefinitely.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Target fraction of each row's entries to zero. A function-local static,
// the same pattern QuantizeFp16KeepIoTypes() (quantize_fp16.h) uses to pass
// a parameter into a pass that OptimizeFixed's pass-name-list interface has
// no other way to carry -- set by PruneMagnitude (pruning_entry.cpp)
// immediately before calling OptimizeFixed.
inline double& MagnitudePruningSparsity() {
  static double sparsity = 0.5;
  return sparsity;
}

// Row-wise magnitude mask over a flat, row-major [rows, cols] `importance`
// matrix: within each row independently, keeps the
// max(1, round(cols * (1 - sparsity))) highest-importance entries and drops
// the rest -- mirrors pruning.py's own _sparsity_mask exactly (same keep
// count, same "never drop every entry of a row" floor), but via
// std::stable_sort per row instead of a vectorized argsort (tie-breaking on
// exactly-equal importances -- common with exact-zero weights -- may
// therefore differ from the Python port's own choice of which zero-tied
// entry to drop; both are equally valid magnitude-pruning outcomes).
inline std::vector<bool> SparsityMaskRowMajor(const std::vector<float>& importance,
                                              int64_t rows, int64_t cols,
                                              double sparsity) {
  std::vector<bool> mask(importance.size(), true);
  // std::nearbyint (current rounding mode is FE_TONEAREST, i.e.
  // round-half-to-even, by default) rather than std::lround
  // (round-half-away-from-zero) to match Python's own round() -- the two
  // differ only on an exact .5 tie, but pruning.py's _sparsity_mask uses
  // Python's round() for its own keep count.
  const int64_t keep = std::max<int64_t>(
      1, static_cast<int64_t>(std::nearbyint(cols * (1.0 - sparsity))));
  if (keep >= cols) {
    return mask;
  }
  const int64_t drop = cols - keep;
  std::vector<int64_t> order(static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = importance.data() + r * cols;
    for (int64_t c = 0; c < cols; ++c) {
      order[static_cast<size_t>(c)] = c;
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      return row[a] < row[b];
    });
    for (int64_t i = 0; i < drop; ++i) {
      mask[static_cast<size_t>(r * cols + order[static_cast<size_t>(i)])] =
          false;
    }
  }
  return mask;
}

// Whether applying SparsityMaskRowMajor's own mask to `data` (row-major
// [rows, cols], matching layout) would zero out at least one currently
// nonzero entry -- see this file's own doc comment for why every predicate
// below must check this before matching.
inline bool MagnitudePruningWouldChange(const std::vector<float>& data,
                                        int64_t rows, int64_t cols,
                                        double sparsity) {
  std::vector<float> importance(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    importance[i] = std::fabs(data[i]);
  }
  const std::vector<bool> mask =
      SparsityMaskRowMajor(importance, rows, cols, sparsity);
  for (size_t i = 0; i < data.size(); ++i) {
    if (!mask[i] && data[i] != 0.0f) {
      return true;
    }
  }
  return false;
}

// Applies `mask` (row-major [rows, cols], SparsityMaskRowMajor's own layout)
// to `data` (the same flat row-major layout), zeroing every masked-out
// entry in place.
inline void ApplyMaskRowMajor(std::vector<float>& data,
                              const std::vector<bool>& mask) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (!mask[i]) {
      data[i] = 0.0f;
    }
  }
}

// ReadWeightNK (quantize_matmul_common.h) reads a 2-D MatMul/Gemm weight
// into a flat row-major [N, K] (output channel first) buffer, used here so
// SparsityMaskRowMajor's per-row grouping lines up with output channels
// regardless of the weight's own on-disk (transposed or not) layout.

struct MagnitudePruningMatMul final : public PredicateBasedPass {
  explicit MagnitudePruningMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "magnitude_pruning_matmul"; }

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
    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t rows = info.weight_transposed ? dim0 : dim1;
    const int64_t cols = info.weight_transposed ? dim1 : dim0;
    const std::vector<float> w_nk = ReadWeightNK(*w_t, info.weight_transposed);
    return MagnitudePruningWouldChange(w_nk, rows, cols,
                                       MagnitudePruningSparsity());
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

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t rows = info.weight_transposed ? dim0 : dim1;
    const int64_t cols = info.weight_transposed ? dim1 : dim0;

    std::vector<float> w_nk = ReadWeightNK(*w_t, info.weight_transposed);
    std::vector<float> importance(w_nk.size());
    for (size_t i = 0; i < w_nk.size(); ++i) {
      importance[i] = std::fabs(w_nk[i]);
    }
    const std::vector<bool> mask = SparsityMaskRowMajor(
        importance, rows, cols, MagnitudePruningSparsity());
    ApplyMaskRowMajor(w_nk, mask);

    Tensor w_pruned;
    w_pruned.elem_type() = TensorProto_DataType_FLOAT;
    w_pruned.sizes() = {dim0, dim1};
    std::vector<float> out_data(static_cast<size_t>(dim0 * dim1));
    for (int64_t i = 0; i < dim0; ++i) {
      for (int64_t j = 0; j < dim1; ++j) {
        out_data[static_cast<size_t>(i * dim1 + j)] =
            info.weight_transposed ? w_nk[static_cast<size_t>(i * cols + j)]
                                   : w_nk[static_cast<size_t>(j * cols + i)];
      }
    }
    w_pruned.floats() = std::move(out_data);

    Value* w_pruned_v = graph.addInitializerAndCreateValue(w_pruned);
    n->replaceInput(1, w_pruned_v);
    return true;
  }
};

struct MagnitudePruningConv final : public PredicateBasedPass {
  explicit MagnitudePruningConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "magnitude_pruning_conv"; }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 4) {
      return false;
    }
    const int64_t rows = w_t->sizes()[0];
    int64_t cols = 1;
    for (size_t i = 1; i < w_t->sizes().size(); ++i) {
      cols *= w_t->sizes()[i];
    }
    const std::vector<float> data = ReadFloatTensorFlat(*w_t);
    return MagnitudePruningWouldChange(data, rows, cols,
                                       MagnitudePruningSparsity());
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 4) {
      return false;
    }

    // [out_channels, in_channels/groups, kH, kW] -> [rows=out_channels,
    // cols=flattened rest], the same reshape pruning.py's own _prune_weight
    // uses for Conv.
    const int64_t rows = w_t->sizes()[0];
    int64_t cols = 1;
    for (size_t i = 1; i < w_t->sizes().size(); ++i) {
      cols *= w_t->sizes()[i];
    }

    std::vector<float> data = ReadFloatTensorFlat(*w_t);
    std::vector<float> importance(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      importance[i] = std::fabs(data[i]);
    }
    const std::vector<bool> mask = SparsityMaskRowMajor(
        importance, rows, cols, MagnitudePruningSparsity());
    ApplyMaskRowMajor(data, mask);

    Tensor w_pruned;
    w_pruned.elem_type() = TensorProto_DataType_FLOAT;
    w_pruned.sizes() = w_t->sizes();
    w_pruned.floats() = std::move(data);

    Value* w_pruned_v = graph.addInitializerAndCreateValue(w_pruned);
    n->replaceInput(1, w_pruned_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
