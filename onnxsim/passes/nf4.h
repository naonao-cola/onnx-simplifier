// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// bitsandbytes' NF4 (NormalFloat 4-bit) weight-only quantization -- C++
// port of nf4.py's own quantize_weight_only_nf4. See that module's
// docstring for the full rationale (QLoRA, Dettmers et al. 2023): a
// fixed, 16-value, zero-symmetric non-uniform codebook (the quantile
// points of a standard normal distribution, published verbatim in
// bitsandbytes' own source and unchanged across releases), denser near
// zero than a uniform 4-bit grid -- exactly the same motivation as
// iq4_nl.h's own codebook, just this format's own fixed table instead of
// that header's independently-derived Lloyd-Max one.
//
// Unlike iq4_nl.h (which blocks over the weight's own flattened
// row-major storage, ignoring output-channel boundaries), NF4's own
// blocking is *per output channel*: the reduction dimension K is split
// into kBlockSize=64 consecutive elements, each group getting its own
// scale, independently for every output channel N -- exactly
// quantize_matmul_common.h's own TryQuantizeWeightBlockwiseInt4InPlace
// index geometry (reduction axis vs. channel axis, chosen via
// MatMulLikeInfo::weight_transposed), just with NF4's own non-uniform
// codebook snap instead of a uniform INT4 round, and writing the
// dequantized float directly in place instead of packing a q/scale pair.
// Per block: scale = max(|block|) (floored at 1e-12 to avoid a
// divide-by-zero on an all-zero block; the codebook's own largest-
// magnitude entry is already exactly 1.0, so no separate normalization
// by max(|codebook|) is needed), normalized = value / scale, code =
// argmin_i |normalized - codebook[i]|, dequant = codebook[code] * scale.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   element replaced by its own
//                                   64-element (per output channel)
//                                   NF4 quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm
// with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
// constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by kBlockSize -- matching nf4.py's own scope (a
// non-block-divisible layer is left untouched there too, not
// zero-padded, since padding would change which elements share a scale
// group along a *per-channel* axis, unlike gguf_legacy_quant.h's own
// flattened, order-agnostic blocking). Unlike nf4.py's own
// quantize_weight_only_nf4, this port does not build the dequantization
// out of Gather/Reshape/Mul graph nodes -- following the exact
// convention apply_gguf_q4_k_quantization/iq4_nl.h already established
// for a non-uniform-codebook format with no native ONNX representation,
// this port instead folds the reconstructed float32 values directly into
// a replacement initializer (no new graph nodes), and does not expose
// nf4.py's own adjustable block_size (fixed at nf4.py's own default, 64)
// -- several other *_cpp ports in this repo already establish that a C++
// port need not mirror every optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM nf4.py: this scheme has no
// accumulation/iterative-refinement step -- every block's own scale is
// computed independently from that block's own max(|.|) -- so this port
// is expected to track the Python port's own float64 numpy
// implementation closely, up to floating-point summation-order/argmin
// tie-breaking differences. quantize_weight_only_nf4 and this port's own
// _cpp counterpart remain independently-correct, non-interchangeable
// entry points -- this port's own tests check structural/algebraic
// properties and comparable (not bit-identical) reconstruction error,
// matching this repo's established contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace nf4_detail {

constexpr int64_t kBlockSize = 64;
constexpr int64_t kNumLevels = 16;

// bitsandbytes' own NF4 codebook (`bitsandbytes/functional.py`'s
// hardcoded NF4 table), transcribed verbatim from nf4.py's own
// NF4_CODEBOOK -- the 16 quantile points of a standard normal
// distribution, adjusted to be symmetric and include an exact 0.
constexpr double kCodebook[kNumLevels] = {
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
};

// Snaps `normalized` to its nearest of the 16 fixed NF4 codebook levels,
// mirroring nf4.py's own _nearest_codebook_index (a dense 16-way argmin,
// not a sorted-codebook binary search -- also fine here since 16 is
// tiny).
inline double NearestCodebookValue(double normalized) {
  double best_value = kCodebook[0];
  double best_dist = std::fabs(normalized - kCodebook[0]);
  for (int64_t i = 1; i < kNumLevels; ++i) {
    const double dist = std::fabs(normalized - kCodebook[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best_value = kCodebook[i];
    }
  }
  return best_value;
}

// Quantize-dequantize round trip for `w_t` (a 2-D float32 constant, laid
// out as [N, K] when `transposed` else [K, N]) in place, one NF4 code
// per element: for each output channel and each contiguous kBlockSize
// group along the reduction axis K, scale = max(|group|) (floored at
// 1e-12), then every element in the group snaps to
// NearestCodebookValue(value / scale) * scale. Mirrors nf4.py's own
// _quantize_nf4_blockwise exactly (same per-(channel, K-block) grouping,
// same scale formula), just writing the dequantized float directly
// instead of returning separate (code, scale) tensors. Precondition
// (checked by the caller's own patternMatchPredicate): `K` is evenly
// divisible by kBlockSize.
inline void QuantizeDequantizeNF4InPlace(std::vector<double>& data,
                                         int64_t dim0, int64_t dim1,
                                         bool transposed) {
  const int64_t reduction_axis = transposed ? 1 : 0;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  const int64_t num_blocks = K / kBlockSize;
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  auto scale_index = [&](int64_t i, int64_t j) {
    const int64_t si = reduction_axis == 0 ? i / kBlockSize : i;
    const int64_t sj = reduction_axis == 1 ? j / kBlockSize : j;
    return si * scale_dim1 + sj;
  };
  auto at = [&](int64_t i, int64_t j) -> double& { return data[i * dim1 + j]; };

  std::vector<double> scale(static_cast<size_t>(scale_dim0 * scale_dim1), 0.0);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      double& s = scale[static_cast<size_t>(scale_index(i, j))];
      s = std::max(s, std::fabs(at(i, j)));
    }
  }
  for (double& s : scale) {
    s = std::max(s, 1e-12);
  }

  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const double s = scale[static_cast<size_t>(scale_index(i, j))];
      at(i, j) = NearestCodebookValue(at(i, j) / s) * s;
    }
  }
}

}  // namespace nf4_detail

// bitsandbytes' NF4 format -- matches MatMul/vanilla-Gemm the same way
// every other *_cpp port in this repo does, additionally requiring the
// reduction dimension to be evenly divisible by nf4_detail::kBlockSize
// (matching nf4.py's own scope), then quantizes the weight in place, one
// output channel and reduction-axis block at a time.
struct NF4 final : public PredicateBasedPass {
  explicit NF4()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "nf4"; }

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
    const int64_t K = info.weight_transposed ? sizes[1] : sizes[0];
    return K % nf4_detail::kBlockSize == 0;
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
    const int64_t K = info.weight_transposed ? dim1 : dim0;
    if (K % nf4_detail::kBlockSize != 0) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);
    std::vector<double> out_data(data.begin(), data.end());
    nf4_detail::QuantizeDequantizeNF4InPlace(out_data, dim0, dim1,
                                             info.weight_transposed);

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
