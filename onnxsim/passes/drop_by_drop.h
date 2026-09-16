// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Drop-by-Drop (Babaoglu, Chen, Khisti, 2026, "Drop-by-Drop: Multi-Bitwidth
// Quantization for LLMs Using Additive Codebooks") -- C++ port of
// drop_by_drop.py's own quantize_weight_only_drop_by_drop. See that
// module's docstring for the full rationale: like aqlm.h's own AQLM
// scheme, each (output-channel, group_dim-element K-block) group is
// reconstructed as the *sum* of `num_codebooks` greedy-residual codebook
// lookups --
//
//   dequant_group = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]
//
// -- but unlike aqlm.h's own *unweighted* Lloyd's-algorithm fit at each
// stage, every stage here (not just the first) is fit via *weighted*
// Lloyd's-algorithm k-means against one fixed, per-group importance
// weight computed once, up front, from the *original* (unquantized)
// weight's own per-group RMS magnitude:
//
//   importance_g = max(rms(group_g), max_g(rms(group_g)) * 1e-6 + 1e-12)
//
// (the same weight is reused unchanged at every one of the `num_codebooks`
// stages -- not recomputed from the residual). A weighted k-means
// centroid update is the importance-weighted mean of a cluster's own
// assigned points (`sum(weight_i * data_i) / sum(weight_i)`), rather
// than the plain mean; nearest-centroid assignment stays ordinary
// (unweighted) squared-distance. This biases every stage's limited
// codebook capacity toward the layer's higher-magnitude (higher-impact)
// groups, which is what makes a short additive-term prefix a *better*
// reconstruction under this scheme, not merely a *valid* one (see
// drop_by_drop.py's own docstring for the full argument).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, group_dim-element
//                                   K-block) group replaced by its own
//                                   Drop-by-Drop additive-codebook
//                                   reconstruction (the full num_codebooks
//                                   -term sum -- see the scope note below)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// kGroupDim -- the same scope aqlm.h/hqq.h use, plus the
// group-divisibility requirement drop_by_drop.py's own encoder already
// imposes (a layer that fails it is left untouched, exactly like the
// Python side's own `continue`).
//
// SCOPE NARROWING, two-fold: (1) unlike drop_by_drop.py's own
// quantize_weight_only_drop_by_drop, this port does not build the
// `num_codebooks` Gather + `num_codebooks - 1` Add (+ Reshape/Transpose)
// "Matryoshka" running-sum graph rewrite -- it folds the *full*
// num_codebooks-term reconstructed values directly into a replacement
// float32 initializer instead, matching every other *_cpp port in this
// repo (e.g. aqlm.h, kmeans_quantization.h). Since the port never
// materializes the named `partial{k}` intermediate sums
// drop_by_drop.py's own graph exposes, this port also does not implement
// anything corresponding to select_drop_by_drop_prefix -- there is no
// graph-level "prefix" artifact here to retarget; a caller wanting a
// shorter-prefix reconstruction would need to reduce kNumCodebooks below
// and rebuild, the same way any other *_cpp port's hardcoded constants
// work. (2) group_dim/num_codebooks/codebook_size/num_iterations are
// fixed at drop_by_drop.py's own defaults (8, 4, 256, 10) rather than
// exposed as pass parameters -- several other *_cpp ports in this repo
// already establish that a C++ port need not mirror every optional knob
// its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM drop_by_drop.py: k-means has no
// closed form, and unlike kmeans_quantization.h's own scalar
// percentile-based initialization, drop_by_drop.py's own initialization
// (mirroring aqlm.py's own _fit_kmeans_codebook, just with a weighted
// centroid update) samples `codebook_size` group rows uniformly at
// random via a seeded `numpy.random.Generator` for every one of the
// `num_codebooks` stages -- reproducing numpy's own PCG64 bitstream in
// C++ was judged not worth it here, the same judgment aqlm.h's own
// divergence note already makes. This port instead reuses aqlm.h's own
// deterministic initialization scheme unchanged: sort each stage's own
// residual groups by their own squared L2 norm and take `codebook_size`
// evenly-spaced rows from that sorted order (padding by repeating the
// last selected row if there are fewer than `codebook_size` groups).
// Both sides then run the *same* weighted-Lloyd's-algorithm loop (same
// fixed importance weights, same iteration budget), so this is a
// genuinely different (though algorithmically identical) fit, not merely
// a floating-point-order divergence -- the same "independently correct,
// non-interchangeable, comparable-not-identical" contract this repo's
// own tests already establish for aqlm.h/kmeans_quantization.h.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace drop_by_drop_detail {

constexpr int64_t kGroupDim = 8;
constexpr int64_t kNumCodebooks = 4;
constexpr int64_t kCodebookSize = 256;
constexpr int64_t kNumIterations = 10;

// Fits `kCodebookSize` `dim`-dimensional centroids to `data` ([num_points,
// dim], row-major) via *weighted* Lloyd's-algorithm k-means -- ordinary
// (unweighted) nearest-centroid assignment each round, but each
// centroid's update is the `weights`-weighted mean of its currently
// assigned points rather than the plain mean. `weights` ([num_points])
// is the same fixed, per-group importance vector at every stage this
// header's own QuantizeDequantizeDropByDropInPlace makes. Mirrors
// drop_by_drop.py's own _fit_weighted_kmeans_codebook -- see this
// header's own top-of-file comment for the one documented divergence
// (deterministic magnitude-sorted initialization instead of a seeded
// random sample, the same choice aqlm.h's own FitKMeansCodebook makes).
// An empty cluster keeps its previous centroid, matching the Python side.
inline void FitWeightedKMeansCodebook(const std::vector<double>& data,
                                      const std::vector<double>& weights,
                                      int64_t num_points, int64_t dim,
                                      std::vector<double>& centroids_out,
                                      std::vector<int64_t>& assignment_out) {
  auto point = [&](int64_t i) { return data.data() + i * dim; };

  // Deterministic initialization: sort points by their own squared L2
  // norm, then take kCodebookSize evenly-spaced rows from that order
  // (padding by repeating the last one if num_points < kCodebookSize) --
  // the same scheme aqlm.h's own FitKMeansCodebook uses.
  std::vector<int64_t> order(static_cast<size_t>(num_points));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::vector<double> norm_sq(static_cast<size_t>(num_points));
  for (int64_t i = 0; i < num_points; ++i) {
    double s = 0.0;
    const double* p = point(i);
    for (int64_t d = 0; d < dim; ++d) {
      s += p[d] * p[d];
    }
    norm_sq[static_cast<size_t>(i)] = s;
  }
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return norm_sq[static_cast<size_t>(a)] < norm_sq[static_cast<size_t>(b)];
  });

  centroids_out.assign(static_cast<size_t>(kCodebookSize * dim), 0.0);
  for (int64_t c = 0; c < kCodebookSize; ++c) {
    const double t =
        kCodebookSize > 1
            ? static_cast<double>(c) / static_cast<double>(kCodebookSize - 1)
            : 0.0;
    int64_t idx = static_cast<int64_t>(
        std::llround(t * static_cast<double>(num_points - 1)));
    idx = std::min(std::max(idx, int64_t{0}), num_points - 1);
    const double* p = point(order[static_cast<size_t>(idx)]);
    std::copy(p, p + dim, centroids_out.begin() + c * dim);
  }

  assignment_out.assign(static_cast<size_t>(num_points), 0);
  std::vector<double> weighted_sum(static_cast<size_t>(kCodebookSize * dim));
  std::vector<double> weight_sum(static_cast<size_t>(kCodebookSize));
  for (int64_t iter = 0; iter < kNumIterations; ++iter) {
    for (int64_t i = 0; i < num_points; ++i) {
      const double* p = point(i);
      int64_t best = 0;
      double best_dist = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < kCodebookSize; ++c) {
        const double* cen = centroids_out.data() + c * dim;
        double dist = 0.0;
        for (int64_t d = 0; d < dim; ++d) {
          const double diff = p[d] - cen[d];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      assignment_out[static_cast<size_t>(i)] = best;
    }

    std::fill(weighted_sum.begin(), weighted_sum.end(), 0.0);
    std::fill(weight_sum.begin(), weight_sum.end(), 0.0);
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t c = assignment_out[static_cast<size_t>(i)];
      const double w = weights[static_cast<size_t>(i)];
      const double* p = point(i);
      double* s = weighted_sum.data() + c * dim;
      for (int64_t d = 0; d < dim; ++d) {
        s[d] += w * p[d];
      }
      weight_sum[static_cast<size_t>(c)] += w;
    }
    for (int64_t c = 0; c < kCodebookSize; ++c) {
      if (weight_sum[static_cast<size_t>(c)] > 0.0) {
        double* cen = centroids_out.data() + c * dim;
        const double* s = weighted_sum.data() + c * dim;
        for (int64_t d = 0; d < dim; ++d) {
          cen[d] = s[d] / weight_sum[static_cast<size_t>(c)];
        }
      }
    }
  }

  // Final assignment against the fully-converged centroids.
  for (int64_t i = 0; i < num_points; ++i) {
    const double* p = point(i);
    int64_t best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int64_t c = 0; c < kCodebookSize; ++c) {
      const double* cen = centroids_out.data() + c * dim;
      double dist = 0.0;
      for (int64_t d = 0; d < dim; ++d) {
        const double diff = p[d] - cen[d];
        dist += diff * diff;
      }
      if (dist < best_dist) {
        best_dist = dist;
        best = c;
      }
    }
    assignment_out[static_cast<size_t>(i)] = best;
  }
}

// Quantize-dequantize round trip for `w_t` (a 2-D float32 constant, laid
// out as [N, K] when `weight_transposed` else [K, N]), written into
// `out_data` (same flat row-major layout/size as `w_t`). Returns false
// (leaving `out_data` unspecified) when K is not evenly divisible by
// kGroupDim, mirroring drop_by_drop.py's own `k % group_dim != 0` skip.
//
// Groups are laid out per (output channel, contiguous kGroupDim block
// along the reduction axis K), exactly matching drop_by_drop.py's own
// `w_nk.reshape(num_groups, group_dim)` grouping. The per-group
// importance weight is computed once, from the *original* (pre-
// quantization) group values, and reused unchanged at every one of the
// kNumCodebooks greedy residual stages: fit a weighted codebook to the
// current residual (raw group values for stage 0), add its own
// reconstruction into the running total, then subtract that
// reconstruction from the residual before the next stage.
inline bool QuantizeDequantizeDropByDropInPlace(const Tensor& w_t,
                                                bool weight_transposed,
                                                std::vector<float>& out_data) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t channel_axis = weight_transposed ? 0 : 1;
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  const int64_t N = channel_axis == 0 ? dim0 : dim1;
  if (K % kGroupDim != 0) {
    return false;
  }
  const int64_t num_blocks = K / kGroupDim;
  const int64_t num_groups = N * num_blocks;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto coord = [&](int64_t c, int64_t k) -> std::pair<int64_t, int64_t> {
    return channel_axis == 0 ? std::make_pair(c, k) : std::make_pair(k, c);
  };
  auto at = [&](int64_t i, int64_t j) -> double {
    return static_cast<double>(data[static_cast<size_t>(i * dim1 + j)]);
  };

  // Gather every group's own kGroupDim values into one flat
  // [num_groups, kGroupDim] buffer, row-major, group g = n * num_blocks +
  // block.
  std::vector<double> residual(static_cast<size_t>(num_groups * kGroupDim));
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int64_t g = n * num_blocks + b;
      for (int64_t d = 0; d < kGroupDim; ++d) {
        const auto [i, j] = coord(n, b * kGroupDim + d);
        residual[static_cast<size_t>(g * kGroupDim + d)] = at(i, j);
      }
    }
  }

  // Fixed, per-group importance from the *original* weight's own
  // magnitude, reused unchanged at every stage -- see this header's own
  // top-of-file comment for why this is what makes a short additive-term
  // prefix better, not just valid.
  std::vector<double> importance(static_cast<size_t>(num_groups));
  double max_importance = 0.0;
  for (int64_t g = 0; g < num_groups; ++g) {
    double sum_sq = 0.0;
    const double* group = residual.data() + g * kGroupDim;
    for (int64_t d = 0; d < kGroupDim; ++d) {
      sum_sq += group[d] * group[d];
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(kGroupDim));
    importance[static_cast<size_t>(g)] = rms;
    max_importance = std::max(max_importance, rms);
  }
  const double importance_floor = max_importance * 1e-6 + 1e-12;
  for (double& imp : importance) {
    imp = std::max(imp, importance_floor);
  }

  std::vector<double> reconstruction(residual.size(), 0.0);
  std::vector<double> centroids;
  std::vector<int64_t> assignment;
  for (int64_t m = 0; m < kNumCodebooks; ++m) {
    FitWeightedKMeansCodebook(residual, importance, num_groups, kGroupDim,
                              centroids, assignment);
    for (int64_t g = 0; g < num_groups; ++g) {
      const double* cen =
          centroids.data() + assignment[static_cast<size_t>(g)] * kGroupDim;
      double* recon = reconstruction.data() + g * kGroupDim;
      double* res = residual.data() + g * kGroupDim;
      for (int64_t d = 0; d < kGroupDim; ++d) {
        recon[d] += cen[d];
        res[d] -= cen[d];
      }
    }
  }

  out_data.assign(data.begin(), data.end());
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int64_t g = n * num_blocks + b;
      for (int64_t d = 0; d < kGroupDim; ++d) {
        const auto [i, j] = coord(n, b * kGroupDim + d);
        out_data[static_cast<size_t>(i * dim1 + j)] = static_cast<float>(
            reconstruction[static_cast<size_t>(g * kGroupDim + d)]);
      }
    }
  }
  return true;
}

}  // namespace drop_by_drop_detail

// Drop-by-Drop -- matches MatMul/vanilla-Gemm the same way AQLM/HQQ do,
// then quantizes each (output-channel, K-group) group of the weight with
// a shared, greedily-fit, importance-weighted additive multi-codebook
// reconstruction.
struct DropByDrop final : public PredicateBasedPass {
  explicit DropByDrop()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "drop_by_drop"; }

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
    return k % drop_by_drop_detail::kGroupDim == 0;
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

    std::vector<float> out_float;
    if (!drop_by_drop_detail::QuantizeDequantizeDropByDropInPlace(
            *w_t, info.weight_transposed, out_float)) {
      return false;
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = w_t->sizes();
    w_out.floats() = std::move(out_float);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
