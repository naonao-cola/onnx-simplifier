// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// AQLM -- Additive Quantization for Language Models (Egiazarian et al.,
// 2024, "Extreme Compression of Large Language Models via Additive
// Quantization") -- C++ port of aqlm.py's own quantize_weight_only_aqlm.
// See that module's docstring for the full rationale: unlike
// kmeans_quantization.h's own single fit-per-layer codebook, AQLM
// represents each small (output-channel, group_dim-element K-block)
// group as the *sum* of `num_codebooks` lookups, one from each of
// `num_codebooks` separate codebooks shared across every group in the
// layer:
//
//   dequant_group = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]
//
// Codebooks are fit via the classical greedy residual k-means strategy
// aqlm.py's own docstring documents: fit codebook 1 with ordinary
// Lloyd's-algorithm k-means to every group's own raw group_dim-dimensional
// values, subtract what it reconstructs to get a residual, fit codebook 2
// to *that residual*, and so on -- not AQLM's own more sophisticated
// joint beam-search code assignment (aqlm.py's own honesty note: this
// greedy residual fitting is the textbook baseline additive/residual
// quantization is built on, not a reproduction of the paper's own
// bespoke calibrated optimizer).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, group_dim-element
//                                   K-block) group replaced by its own
//                                   AQLM additive-codebook reconstruction
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// kGroupDim -- the same scope gguf_q6_k.h/hqq.h use, plus the
// group-divisibility requirement aqlm.py's own encoder already imposes (a
// layer that fails it is left untouched, exactly like the Python side's
// own `continue`).
//
// Unlike aqlm.py's own quantize_weight_only_aqlm, this port does not
// build the `num_codebooks` Gather + `num_codebooks - 1` Add (+ Reshape/
// Transpose) graph rewrite -- it folds the reconstructed values directly
// into a replacement float32 initializer instead, matching every other
// *_cpp port in this repo (e.g. kmeans_quantization.h, gguf_q2_k.h)
// rather than aqlm.py's own choice to keep the per-stage codebooks/codes
// visible in the graph. This port also hardcodes aqlm.py's own defaults
// (group_dim=8, num_codebooks=2, codebook_size=256, num_iterations=10)
// rather than exposing them as parameters -- several other *_cpp ports in
// this repo already establish that a C++ port need not mirror every
// optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM aqlm.py: k-means has no closed
// form, and unlike kmeans_quantization.h's own scalar percentile-based
// initialization (which matches its Python counterpart exactly outside a
// narrow too-few-distinct-values edge case), aqlm.py's own initialization
// samples `codebook_size` group *rows* uniformly at random via a seeded
// `numpy.random.Generator` for every one of the `num_codebooks` stages --
// reproducing numpy's own PCG64 bitstream in C++ was judged not worth it
// here, the same judgment kmeans_quantization.h's own divergence note
// already makes for its own (narrower) random-sampling fallback. This
// port instead initializes each stage's codebook deterministically: sort
// that stage's own residual groups by their own squared L2 norm and take
// `codebook_size` evenly-spaced rows from that sorted order (padding by
// repeating the last selected row if there are fewer than `codebook_size`
// groups) -- a different, but not unreasonable, spread across the
// residual's own magnitude range. Both sides then run the *same* greedy
// residual Lloyd's-algorithm loop to the same fixed iteration budget, so
// this is a genuinely different (though algorithmically identical) fit,
// not merely a floating-point-order divergence -- the same "independently
// correct, non-interchangeable, comparable-not-identical" contract this
// repo's own tests already establish for kmeans_quantization.h. More
// codebook stages can only reduce this port's own reconstruction error
// too (each new stage targets exactly the previous stages' own leftover
// residual), independently of what Python's own random initialization
// happened to pick -- checked directly in this port's own test file.

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

namespace aqlm_detail {

constexpr int64_t kGroupDim = 8;
constexpr int64_t kNumCodebooks = 2;
constexpr int64_t kCodebookSize = 256;
constexpr int64_t kNumIterations = 10;

// Fits `kCodebookSize` `dim`-dimensional centroids to `data` ([num_points,
// dim], row-major) via ordinary (unweighted) Lloyd's-algorithm k-means,
// returning the centroids ([kCodebookSize, dim], row-major) and each
// point's own nearest-centroid assignment. Mirrors aqlm.py's own
// _fit_kmeans_codebook -- see this header's own top-of-file comment for
// the one documented divergence (deterministic magnitude-sorted
// initialization instead of a seeded random sample). An empty cluster
// keeps its previous centroid rather than going undefined, matching the
// Python side.
inline void FitKMeansCodebook(const std::vector<double>& data,
                              int64_t num_points, int64_t dim,
                              std::vector<double>& centroids_out,
                              std::vector<int64_t>& assignment_out) {
  auto point = [&](int64_t i) { return data.data() + i * dim; };

  // Deterministic initialization: sort points by their own squared L2
  // norm, then take kCodebookSize evenly-spaced rows from that order
  // (padding by repeating the last one if num_points < kCodebookSize).
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
  std::vector<double> sum(static_cast<size_t>(kCodebookSize * dim));
  std::vector<int64_t> cluster_count(static_cast<size_t>(kCodebookSize));
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

    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cluster_count.begin(), cluster_count.end(), 0);
    for (int64_t i = 0; i < num_points; ++i) {
      const int64_t c = assignment_out[static_cast<size_t>(i)];
      const double* p = point(i);
      double* s = sum.data() + c * dim;
      for (int64_t d = 0; d < dim; ++d) {
        s[d] += p[d];
      }
      cluster_count[static_cast<size_t>(c)] += 1;
    }
    for (int64_t c = 0; c < kCodebookSize; ++c) {
      if (cluster_count[static_cast<size_t>(c)] > 0) {
        double* cen = centroids_out.data() + c * dim;
        const double* s = sum.data() + c * dim;
        for (int64_t d = 0; d < dim; ++d) {
          cen[d] =
              s[d] / static_cast<double>(cluster_count[static_cast<size_t>(c)]);
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
// kGroupDim, mirroring aqlm.py's own `k % group_dim != 0` skip.
//
// Groups are laid out per (output channel, contiguous kGroupDim block
// along the reduction axis K), exactly matching aqlm.py's own
// `w_nk.reshape(num_groups, group_dim)` grouping. For kNumCodebooks
// greedy residual stages: fit a codebook to the current residual (raw
// group values for stage 0), add its own reconstruction into the running
// total, then subtract that reconstruction from the residual before the
// next stage.
inline bool QuantizeDequantizeAQLMInPlace(const Tensor& w_t,
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

  std::vector<double> reconstruction(residual.size(), 0.0);
  std::vector<double> centroids;
  std::vector<int64_t> assignment;
  for (int64_t m = 0; m < kNumCodebooks; ++m) {
    FitKMeansCodebook(residual, num_groups, kGroupDim, centroids, assignment);
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

}  // namespace aqlm_detail

// AQLM -- matches MatMul/vanilla-Gemm the same way GgufQ6K/HQQ do, then
// quantizes each (output-channel, K-group) group of the weight with a
// shared, greedily-fit additive multi-codebook reconstruction.
struct AQLM final : public PredicateBasedPass {
  explicit AQLM()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "aqlm"; }

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
    return k % aqlm_detail::kGroupDim == 0;
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
    if (!aqlm_detail::QuantizeDequantizeAQLMInPlace(
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
