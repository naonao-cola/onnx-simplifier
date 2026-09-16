// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// LO-BCQ -- Block Clustered Quantization (Elangovan, Sakr, Raghunathan,
// Khailany, 2025, "LO-BCQ: Block Clustered Quantization for 4-bit (W4A4)
// LLM Inference") -- C++ port of the *weight-side* half of lo_bcq.py's
// own quantize_weight_only_lo_bcq. See that module's own docstring for
// the full rationale: unlike kmeans_quantization.h's own single
// whole-tensor codebook, LO-BCQ fits several small (2**bits-entry)
// codebooks -- one per data-driven *cluster of blocks*, not one per
// fixed position -- via an outer alternating loop:
//
//   1. Split the weight's own [N, K] (output-channel-first) view into
//      contiguous block_size-element blocks along K.
//   2. Cluster the blocks themselves by their own [mean, std] summary
//      statistics (NOT their position) into num_clusters groups, via
//      ordinary multi-dimensional Lloyd's k-means.
//   3. Fit one small 1-D Lloyd-max codebook per cluster, from only that
//      cluster's own currently-assigned blocks' values (reusing the same
//      percentile-initialized 1-D k-means kmeans_quantization.h's own
//      QuantizeDequantizeKMeans already established, generalized here to
//      return centroids rather than immediately overwrite the data).
//   4. Alternate for up to outer_iters rounds: re-fit every cluster's own
//      codebook from its current blocks, then re-assign every block to
//      whichever cluster's *current* codebook reconstructs it with the
//      lowest MSE -- stopping early once no block's assignment changes.
//   5. Every element's final code is its own nearest value in its own
//      block's own (final) cluster's codebook.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, block_size-element
//                                   K-block) group replaced by its own
//                                   block-clustered codebook value
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm
// with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
// constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by lo_bcq_detail::kBlockSize -- the same scope
// gguf_q6_k.h/hqq.h/aqlm.h use, plus the block-divisibility requirement
// lo_bcq.py's own encoder already imposes (a layer that fails it is left
// untouched, exactly like the Python side's own `continue`).
//
// Unlike lo_bcq.py's own quantize_weight_only_lo_bcq, this port does not
// build the real Gather(Codebooks, ClusterIds)+Cast+GatherElements+
// Reshape[+Transpose] graph rewrite -- it folds the reconstructed values
// directly into a replacement float32 initializer instead, matching
// every other *_cpp port in this repo (e.g. kmeans_quantization.h,
// aqlm.h) rather than lo_bcq.py's own choice to keep the per-cluster
// codebooks/cluster-ids/codes split visible in the graph. This port also
// hardcodes lo_bcq.py's own defaults (bits=4, block_size=32,
// num_clusters=4, outer_iters=10) rather than exposing them as
// parameters -- several other *_cpp ports in this repo already establish
// that a C++ port need not mirror every optional knob its Python
// counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM lo_bcq.py, two-fold, mirroring the
// exact same two precedents this repo's own kmeans_quantization.h and
// aqlm.h already establish for k-means-based *_cpp ports: (1) every
// *1-D* per-cluster/fallback codebook fit (lo_bcq.py's own _kmeans_1d)
// initializes deterministically at percentile-derived centroids, exactly
// like kmeans_quantization.h's own QuantizeDequantizeKMeans -- outside
// the narrow too-few-distinct-percentiles edge case (padded here by
// repeating the largest centroid found, not lo_bcq.py's own seeded-
// random-sample fallback), this piece is expected to track the Python
// port's own float64 numpy implementation closely. (2) The *block*-
// clustering step (lo_bcq.py's own _kmeans_blocks_by_features, a
// genuinely multi-dimensional k-means over each block's own [mean, std]
// feature vector) initializes from a seeded uniform-random sample of
// blocks in lo_bcq.py; reproducing numpy's own PCG64 bitstream in C++
// was judged not worth it here either (the same judgment aqlm.h's own
// divergence note already makes for its own multi-dimensional k-means
// step) -- this port instead sorts blocks by their own feature vector's
// squared L2 norm and takes num_clusters evenly-spaced ones as initial
// centroids. Both sides then run the *same* alternating block-clustering/
// codebook-refit loop to the same fixed iteration budget, so this is a
// genuinely different (though algorithmically identical) fit for step
// (2), not merely a floating-point-order divergence -- the same
// "independently correct, non-interchangeable, comparable-not-identical"
// contract this repo's own tests already establish for
// kmeans_quantization.h/aqlm.h. quantize_weight_only_lo_bcq and this
// port's _cpp counterpart remain independently-correct, non-
// interchangeable entry points.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

namespace lo_bcq_detail {

constexpr int64_t kBits = 4;
constexpr int64_t kNumCodes = int64_t{1} << kBits;  // 16
constexpr int64_t kBlockSize = 32;
constexpr int64_t kNumClusters = 4;
constexpr int64_t kOuterIters = 10;
constexpr int64_t kInner1DIters = 20;  // lo_bcq.py's own _kmeans_1d iters
constexpr int64_t kBlockKMeansIters =
    20;                         // its own _kmeans_blocks_by_features iters
constexpr double kAtol = 1e-8;  // numpy.allclose's own defaults
constexpr double kRtol = 1e-5;

// numpy.percentile's own default "linear" interpolation method, mirroring
// kmeans_quantization.h's own PercentileSorted exactly.
inline double PercentileSorted(const std::vector<double>& sorted_vals,
                               double p) {
  const int64_t n = static_cast<int64_t>(sorted_vals.size());
  if (n == 1) {
    return sorted_vals[0];
  }
  const double index = (p / 100.0) * static_cast<double>(n - 1);
  const int64_t lower = static_cast<int64_t>(std::floor(index));
  const int64_t upper = static_cast<int64_t>(std::ceil(index));
  if (lower == upper) {
    return sorted_vals[lower];
  }
  const double frac = index - static_cast<double>(lower);
  return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower]);
}

// Fits `k` 1-D centroids to `values` via Lloyd's-algorithm k-means,
// returning them *sorted ascending* (every lo_bcq.py call site sorts its
// own _kmeans_1d output immediately, so this bakes that in). Mirrors
// kmeans_quantization.h's own QuantizeDequantizeKMeans percentile-init/
// Lloyd's-loop/allclose-convergence structure exactly, generalized to
// return centroids rather than immediately reassigning `values` in
// place, since lo_bcq.py's own _fit_lo_bcq discards _kmeans_1d's own
// per-element assignments and only keeps the fitted centroids.
inline std::vector<double> Kmeans1DSortedCentroids(
    const std::vector<double>& values, int64_t k) {
  std::vector<double> sorted_vals = values;
  std::sort(sorted_vals.begin(), sorted_vals.end());

  std::vector<double> centroids;
  centroids.reserve(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    const double p =
        k > 1 ? 100.0 * static_cast<double>(i) / static_cast<double>(k - 1)
              : 0.0;
    const double c = PercentileSorted(sorted_vals, p);
    if (centroids.empty() || c > centroids.back()) {
      centroids.push_back(c);
    }
  }
  while (static_cast<int64_t>(centroids.size()) < k) {
    centroids.push_back(centroids.back());
  }

  const int64_t count = static_cast<int64_t>(values.size());
  std::vector<int64_t> assignments(static_cast<size_t>(count), 0);
  for (int64_t iter = 0; iter < kInner1DIters; ++iter) {
    for (int64_t i = 0; i < count; ++i) {
      int64_t best = 0;
      double best_dist =
          std::fabs(values[static_cast<size_t>(i)] - centroids[0]);
      for (int64_t c = 1; c < k; ++c) {
        const double dist = std::fabs(values[static_cast<size_t>(i)] -
                                      centroids[static_cast<size_t>(c)]);
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      assignments[static_cast<size_t>(i)] = best;
    }

    std::vector<double> sum(static_cast<size_t>(k), 0.0);
    std::vector<int64_t> cluster_count(static_cast<size_t>(k), 0);
    for (int64_t i = 0; i < count; ++i) {
      const int64_t c = assignments[static_cast<size_t>(i)];
      sum[static_cast<size_t>(c)] += values[static_cast<size_t>(i)];
      cluster_count[static_cast<size_t>(c)] += 1;
    }
    std::vector<double> new_centroids(centroids);
    for (int64_t c = 0; c < k; ++c) {
      if (cluster_count[static_cast<size_t>(c)] > 0) {
        new_centroids[static_cast<size_t>(c)] =
            sum[static_cast<size_t>(c)] /
            static_cast<double>(cluster_count[static_cast<size_t>(c)]);
      }
    }

    bool converged = true;
    for (int64_t c = 0; c < k; ++c) {
      if (std::fabs(new_centroids[static_cast<size_t>(c)] -
                    centroids[static_cast<size_t>(c)]) >
          kAtol + kRtol * std::fabs(centroids[static_cast<size_t>(c)])) {
        converged = false;
        break;
      }
    }
    centroids = std::move(new_centroids);
    if (converged) {
      break;
    }
  }

  std::sort(centroids.begin(), centroids.end());
  return centroids;
}

// Clusters `num_blocks` blocks by their own 2-D [mean, std] feature
// vector into `kNumClusters` groups via multi-dimensional Lloyd's
// k-means, returning the per-block cluster assignment. Mirrors
// lo_bcq.py's own _kmeans_blocks_by_features -- see this header's own
// top-of-file comment for the one documented divergence (deterministic,
// feature-norm-sorted initialization instead of a seeded random sample).
// When num_blocks <= kNumClusters, mirrors the Python side's own
// `(arange(num_blocks) % k)` exactly (no k-means at all needed).
inline std::vector<int64_t> ClusterBlocksByFeatures(
    const std::vector<double>& mean, const std::vector<double>& stddev,
    int64_t num_blocks) {
  std::vector<int64_t> cluster_ids(static_cast<size_t>(num_blocks));
  if (num_blocks <= lo_bcq_detail::kNumClusters) {
    for (int64_t i = 0; i < num_blocks; ++i) {
      cluster_ids[static_cast<size_t>(i)] = i % lo_bcq_detail::kNumClusters;
    }
    return cluster_ids;
  }

  // Deterministic initialization: sort blocks by their own feature
  // vector's squared L2 norm, then take kNumClusters evenly-spaced rows
  // from that order (see this header's own top-of-file divergence note).
  std::vector<int64_t> order(static_cast<size_t>(num_blocks));
  std::iota(order.begin(), order.end(), int64_t{0});
  std::vector<double> norm_sq(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) {
    const double m = mean[static_cast<size_t>(i)];
    const double s = stddev[static_cast<size_t>(i)];
    norm_sq[static_cast<size_t>(i)] = m * m + s * s;
  }
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return norm_sq[static_cast<size_t>(a)] < norm_sq[static_cast<size_t>(b)];
  });

  constexpr int64_t k = lo_bcq_detail::kNumClusters;
  std::vector<double> cen_mean(k), cen_std(k);
  for (int64_t c = 0; c < k; ++c) {
    const double t =
        k > 1 ? static_cast<double>(c) / static_cast<double>(k - 1) : 0.0;
    int64_t idx = static_cast<int64_t>(
        std::llround(t * static_cast<double>(num_blocks - 1)));
    idx = std::min(std::max(idx, int64_t{0}), num_blocks - 1);
    const int64_t row = order[static_cast<size_t>(idx)];
    cen_mean[static_cast<size_t>(c)] = mean[static_cast<size_t>(row)];
    cen_std[static_cast<size_t>(c)] = stddev[static_cast<size_t>(row)];
  }

  for (int64_t iter = 0; iter < lo_bcq_detail::kBlockKMeansIters; ++iter) {
    for (int64_t i = 0; i < num_blocks; ++i) {
      int64_t best = 0;
      double best_dist = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < k; ++c) {
        const double dm =
            mean[static_cast<size_t>(i)] - cen_mean[static_cast<size_t>(c)];
        const double ds =
            stddev[static_cast<size_t>(i)] - cen_std[static_cast<size_t>(c)];
        const double dist = dm * dm + ds * ds;
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      cluster_ids[static_cast<size_t>(i)] = best;
    }

    std::vector<double> sum_mean(k, 0.0), sum_std(k, 0.0);
    std::vector<int64_t> cnt(k, 0);
    for (int64_t i = 0; i < num_blocks; ++i) {
      const int64_t c = cluster_ids[static_cast<size_t>(i)];
      sum_mean[static_cast<size_t>(c)] += mean[static_cast<size_t>(i)];
      sum_std[static_cast<size_t>(c)] += stddev[static_cast<size_t>(i)];
      cnt[static_cast<size_t>(c)] += 1;
    }
    std::vector<double> new_cen_mean(cen_mean), new_cen_std(cen_std);
    for (int64_t c = 0; c < k; ++c) {
      if (cnt[static_cast<size_t>(c)] > 0) {
        new_cen_mean[static_cast<size_t>(c)] =
            sum_mean[static_cast<size_t>(c)] /
            static_cast<double>(cnt[static_cast<size_t>(c)]);
        new_cen_std[static_cast<size_t>(c)] =
            sum_std[static_cast<size_t>(c)] /
            static_cast<double>(cnt[static_cast<size_t>(c)]);
      }
    }

    bool converged = true;
    for (int64_t c = 0; c < k; ++c) {
      const bool ok_mean =
          std::fabs(new_cen_mean[static_cast<size_t>(c)] -
                    cen_mean[static_cast<size_t>(c)]) <=
          kAtol + kRtol * std::fabs(cen_mean[static_cast<size_t>(c)]);
      const bool ok_std =
          std::fabs(new_cen_std[static_cast<size_t>(c)] -
                    cen_std[static_cast<size_t>(c)]) <=
          kAtol + kRtol * std::fabs(cen_std[static_cast<size_t>(c)]);
      if (!ok_mean || !ok_std) {
        converged = false;
      }
    }
    cen_mean = std::move(new_cen_mean);
    cen_std = std::move(new_cen_std);
    if (converged) {
      break;
    }
  }

  for (int64_t i = 0; i < num_blocks; ++i) {
    int64_t best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int64_t c = 0; c < k; ++c) {
      const double dm =
          mean[static_cast<size_t>(i)] - cen_mean[static_cast<size_t>(c)];
      const double ds =
          stddev[static_cast<size_t>(i)] - cen_std[static_cast<size_t>(c)];
      const double dist = dm * dm + ds * ds;
      if (dist < best_dist) {
        best_dist = dist;
        best = c;
      }
    }
    cluster_ids[static_cast<size_t>(i)] = best;
  }
  return cluster_ids;
}

// Runs LO-BCQ's own alternating block-clustering/per-cluster-codebook
// loop over `blocks` ([num_blocks, kBlockSize], row-major) and writes
// each element's own final reconstructed value back in place. Mirrors
// lo_bcq.py's own _fit_lo_bcq exactly (see this header's own top-of-file
// comment for the two documented divergences).
inline void FitAndReconstructLoBcq(std::vector<double>& blocks,
                                   int64_t num_blocks) {
  auto block_at = [&](int64_t b, int64_t i) -> double& {
    return blocks[static_cast<size_t>(b * kBlockSize + i)];
  };

  // A single global fallback codebook seeds every cluster (see
  // lo_bcq.py's own comment on why: an unlucky empty cluster still
  // reconstructs reasonably instead of falling back to all-zero).
  std::vector<double> flat(blocks.begin(), blocks.end());
  const std::vector<double> fallback = Kmeans1DSortedCentroids(flat, kNumCodes);
  std::vector<std::vector<double>> codebooks(static_cast<size_t>(kNumClusters),
                                             fallback);

  std::vector<double> mean(static_cast<size_t>(num_blocks));
  std::vector<double> stddev(static_cast<size_t>(num_blocks));
  for (int64_t b = 0; b < num_blocks; ++b) {
    double sum = 0.0;
    for (int64_t i = 0; i < kBlockSize; ++i) {
      sum += block_at(b, i);
    }
    const double m = sum / static_cast<double>(kBlockSize);
    double var_sum = 0.0;
    for (int64_t i = 0; i < kBlockSize; ++i) {
      const double d = block_at(b, i) - m;
      var_sum += d * d;
    }
    mean[static_cast<size_t>(b)] = m;
    stddev[static_cast<size_t>(b)] =
        std::sqrt(var_sum / static_cast<double>(kBlockSize));
  }

  std::vector<int64_t> cluster_ids =
      ClusterBlocksByFeatures(mean, stddev, num_blocks);

  for (int64_t outer = 0; outer < kOuterIters; ++outer) {
    for (int64_t c = 0; c < kNumClusters; ++c) {
      std::vector<double> values;
      for (int64_t b = 0; b < num_blocks; ++b) {
        if (cluster_ids[static_cast<size_t>(b)] == c) {
          for (int64_t i = 0; i < kBlockSize; ++i) {
            values.push_back(block_at(b, i));
          }
        }
      }
      if (values.empty()) {
        continue;
      }
      codebooks[static_cast<size_t>(c)] =
          Kmeans1DSortedCentroids(values, kNumCodes);
    }

    std::vector<int64_t> new_cluster_ids(static_cast<size_t>(num_blocks));
    for (int64_t b = 0; b < num_blocks; ++b) {
      int64_t best_c = 0;
      double best_err = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < kNumClusters; ++c) {
        const auto& cb = codebooks[static_cast<size_t>(c)];
        double sq_err = 0.0;
        for (int64_t i = 0; i < kBlockSize; ++i) {
          const double v = block_at(b, i);
          double best_dist = std::fabs(v - cb[0]);
          for (int64_t j = 1; j < kNumCodes; ++j) {
            const double dist = std::fabs(v - cb[static_cast<size_t>(j)]);
            if (dist < best_dist) {
              best_dist = dist;
            }
          }
          sq_err += best_dist * best_dist;
        }
        sq_err /= static_cast<double>(kBlockSize);
        if (sq_err < best_err) {
          best_err = sq_err;
          best_c = c;
        }
      }
      new_cluster_ids[static_cast<size_t>(b)] = best_c;
    }

    if (new_cluster_ids == cluster_ids) {
      cluster_ids = std::move(new_cluster_ids);
      break;
    }
    cluster_ids = std::move(new_cluster_ids);
  }

  for (int64_t b = 0; b < num_blocks; ++b) {
    const auto& cb =
        codebooks[static_cast<size_t>(cluster_ids[static_cast<size_t>(b)])];
    for (int64_t i = 0; i < kBlockSize; ++i) {
      double& v = block_at(b, i);
      double best_val = cb[0];
      double best_dist = std::fabs(v - cb[0]);
      for (int64_t j = 1; j < kNumCodes; ++j) {
        const double dist = std::fabs(v - cb[static_cast<size_t>(j)]);
        if (dist < best_dist) {
          best_dist = dist;
          best_val = cb[static_cast<size_t>(j)];
        }
      }
      v = best_val;
    }
  }
}

}  // namespace lo_bcq_detail

// LO-BCQ -- matches MatMul/vanilla-Gemm the same way GgufQ6K/HQQ/AQLM do,
// then quantizes the weight's own [N, K] (output-channel-first) view via
// its own alternating block-clustering/per-cluster-codebook loop. Blocks
// are grouped along the reduction dimension K *per output channel*
// (lo_bcq.py's own [N, K] convention) -- so a Gemm's transB=1 weight
// (already stored [N, K]) is walked contiguously, while a MatMul's
// weight (stored [K, N]) is walked with an N-stride, without physically
// transposing memory.
struct LoBcq final : public PredicateBasedPass {
  explicit LoBcq()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "lo_bcq"; }

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
    return k % lo_bcq_detail::kBlockSize == 0;
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
    if (k_dim % lo_bcq_detail::kBlockSize != 0) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);

    // Materialize the logical [N, K] view (lo_bcq.py's own w if
    // weight_transposed else w.T), flattened into row-major
    // [num_blocks, kBlockSize] blocks (num_blocks = n_dim *
    // (k_dim / kBlockSize)), so each block is contiguous for the kernel
    // above.
    std::vector<double> blocks(static_cast<size_t>(n_dim * k_dim));
    for (int64_t n_idx = 0; n_idx < n_dim; ++n_idx) {
      for (int64_t k_idx = 0; k_idx < k_dim; ++k_idx) {
        const int64_t src = info.weight_transposed ? n_idx * k_dim + k_idx
                                                   : k_idx * dim1 + n_idx;
        blocks[static_cast<size_t>(n_idx * k_dim + k_idx)] =
            data[static_cast<size_t>(src)];
      }
    }

    const int64_t num_blocks = n_dim * (k_dim / lo_bcq_detail::kBlockSize);
    lo_bcq_detail::FitAndReconstructLoBcq(blocks, num_blocks);

    std::vector<float> out_float(static_cast<size_t>(dim0 * dim1));
    for (int64_t n_idx = 0; n_idx < n_dim; ++n_idx) {
      for (int64_t k_idx = 0; k_idx < k_dim; ++k_idx) {
        const int64_t dst = info.weight_transposed ? n_idx * k_dim + k_idx
                                                   : k_idx * dim1 + n_idx;
        out_float[static_cast<size_t>(dst)] = static_cast<float>(
            blocks[static_cast<size_t>(n_idx * k_dim + k_idx)]);
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
