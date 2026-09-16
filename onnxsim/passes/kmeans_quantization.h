// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// K-means weight-codebook quantization (Han et al., 2015, "Deep
// Compression", Section 3's own "trained quantization") -- C++ port of
// kmeans_quantization.py's own quantize_weight_only_kmeans. See that
// module's docstring for the full rationale: unlike every fixed-codebook
// scheme in this repo (gguf_legacy_quant.h's Q4_0/Q4_1, iq4_nl.h), this
// codebook is fit *per layer*, directly to that layer's own weight
// values, via an ordinary Lloyd's-algorithm k-means fit -- not a
// data-independent table.
//
// Algorithm, transcribed from kmeans_quantization.py's own _kmeans_1d:
// flatten the weight to `n` scalar values; initialize `k` (= 2**bits)
// centroids at `k` evenly-spaced percentiles of the flattened values
// (numpy's own default "linear" interpolation method), deduplicated;
// then alternate, for a fixed iteration budget, (a) assigning each value
// to its nearest centroid by absolute difference and (b) recomputing
// each centroid as the mean of the values assigned to it, stopping early
// once no centroid moves (matching numpy.allclose's own default
// rtol=1e-5/atol=1e-8 tolerance). Every element is then replaced by its
// own assigned centroid's value.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own layer-fitted
//                                   k-means centroid value
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_q2_k.h/iq4_nl.h use. Unlike
// kmeans_quantization.py's own quantize_weight_only_kmeans, this port
// does not accept a `bits`/`iters`/`seed` configuration (hardcoded to
// this module's own defaults, bits=4/iters=20 -- see kNumCodes/kIters
// below) -- several other *_cpp ports in this repo already establish
// that a C++ port need not mirror every optional knob its Python
// counterpart has.
//
// Unlike kmeans_quantization.py's own Gather(Codebook, Cast(Codes,
// INT64))-based graph rewrite (a real per-layer codebook + per-element
// uint8 codes, both stored as separate initializers), this port folds
// the quantize-dequantize round trip directly into a single replacement
// float32 initializer -- matching every other *_cpp port in this repo
// (e.g. gguf_q2_k.h, iq4_nl.h) rather than kmeans_quantization.py's own
// choice to keep the codebook/codes split visible in the graph. ONNX has
// no tensor type below INT4 for a lot of this repo's other codebook
// schemes, but INT8 (this module's own uint8 code storage) is a real
// representable type -- this port narrows scope anyway, for consistency
// with every sibling data-free *_cpp port here.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM kmeans_quantization.py: k-means has
// no closed form, so this is not merely a floating-point-order divergence
// like every other *_cpp port in this repo -- it is a genuinely different
// (though algorithmically identical) fit whenever the percentile-based
// initialization alone doesn't already yield `k` distinct centroids (a
// narrow edge case: a weight tensor with fewer than `k` distinct
// percentile-derived values, e.g. many exact zeros/duplicates).
// kmeans_quantization.py falls back to `k` uniform-random samples (seeded
// numpy Generator) to pad out the missing centroids in that case; this
// port instead deterministically pads by repeating the largest centroid
// found so far -- reproducing numpy's own PCG64 bitstream in C++ was
// judged not worth it for an edge case neither side's own encoder claims
// is load-bearing. Outside that edge case (the common case for any
// weight tensor with enough distinct values, which is every realistic
// float32 weight), initialization is fully deterministic and identical
// between the two ports, so Lloyd's algorithm itself -- run to the same
// fixed iteration budget -- is expected to converge to the same
// centroids up to ordinary floating-point summation-order noise, the
// same "comparable, not bit-identical" contract this repo's other *_cpp
// ports already establish. quantize_weight_only_kmeans and this port
// remain independently-correct, non-interchangeable entry points.

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

namespace kmeans_quantization_detail {

constexpr int64_t kBits = 4;
constexpr int64_t kNumCodes = int64_t{1} << kBits;  // 16
constexpr int64_t kIters = 20;

// numpy.percentile's own default "linear" interpolation method over an
// already-sorted array: index = (p / 100) * (n - 1), then linearly
// interpolate between the values at floor(index) and ceil(index).
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

// Fits `kNumCodes` centroids to `values` via Lloyd's-algorithm k-means,
// then overwrites `values` in place with each element's own assigned
// centroid. Mirrors kmeans_quantization.py's own _kmeans_1d (see this
// header's own top-of-file comment for the one documented divergence:
// the too-few-distinct-percentiles padding fallback).
inline void QuantizeDequantizeKMeans(double* data, int64_t count) {
  std::vector<double> sorted_vals(data, data + count);
  std::sort(sorted_vals.begin(), sorted_vals.end());

  // Initialize centroids at kNumCodes evenly-spaced percentiles
  // (np.linspace(0, 100, kNumCodes)), deduplicated in ascending order
  // (np.unique).
  std::vector<double> centroids;
  centroids.reserve(kNumCodes);
  for (int64_t i = 0; i < kNumCodes; ++i) {
    const double p = kNumCodes > 1 ? 100.0 * static_cast<double>(i) /
                                         static_cast<double>(kNumCodes - 1)
                                   : 0.0;
    const double c = PercentileSorted(sorted_vals, p);
    if (centroids.empty() || c > centroids.back()) {
      centroids.push_back(c);
    }
  }
  // Documented divergence: pad by repeating the largest centroid found,
  // rather than kmeans_quantization.py's own seeded-random-sample
  // fallback (see top-of-file comment).
  while (static_cast<int64_t>(centroids.size()) < kNumCodes) {
    centroids.push_back(centroids.back());
  }

  std::vector<int64_t> assignments(count, 0);
  for (int64_t iter = 0; iter < kIters; ++iter) {
    for (int64_t i = 0; i < count; ++i) {
      int64_t best = 0;
      double best_dist = std::fabs(data[i] - centroids[0]);
      for (int64_t c = 1; c < kNumCodes; ++c) {
        const double dist = std::fabs(data[i] - centroids[c]);
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      assignments[i] = best;
    }

    std::vector<double> sum(kNumCodes, 0.0);
    std::vector<int64_t> cluster_count(kNumCodes, 0);
    for (int64_t i = 0; i < count; ++i) {
      sum[assignments[i]] += data[i];
      cluster_count[assignments[i]] += 1;
    }
    std::vector<double> new_centroids(centroids);
    for (int64_t c = 0; c < kNumCodes; ++c) {
      if (cluster_count[c] > 0) {
        new_centroids[c] = sum[c] / static_cast<double>(cluster_count[c]);
      }
    }

    // numpy.allclose's own default tolerance: |a - b| <= atol + rtol*|b|.
    bool converged = true;
    for (int64_t c = 0; c < kNumCodes; ++c) {
      const double atol = 1e-8;
      const double rtol = 1e-5;
      if (std::fabs(new_centroids[c] - centroids[c]) >
          atol + rtol * std::fabs(centroids[c])) {
        converged = false;
        break;
      }
    }
    centroids = std::move(new_centroids);
    if (converged) {
      break;
    }
  }

  for (int64_t i = 0; i < count; ++i) {
    int64_t best = 0;
    double best_dist = std::fabs(data[i] - centroids[0]);
    for (int64_t c = 1; c < kNumCodes; ++c) {
      const double dist = std::fabs(data[i] - centroids[c]);
      if (dist < best_dist) {
        best_dist = dist;
        best = c;
      }
    }
    data[i] = centroids[best];
  }
}

}  // namespace kmeans_quantization_detail

// K-means per-layer codebook quantization -- matches MatMul/vanilla-Gemm
// the same way GgufQ2K/IQ4NL do, then fits and applies one k-means
// codebook to the whole flattened weight tensor.
struct KMeansQuant final : public PredicateBasedPass {
  explicit KMeansQuant()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "kmeans_quantization"; }

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
    kmeans_quantization_detail::QuantizeDequantizeKMeans(out_data.data(),
                                                         numel);

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
