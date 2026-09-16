// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// IF4 (Adaptive Block-Scaled Data Types) -- C++ port of
// if4_quantization.py's own quantize_weight_only_if4. See that module's
// docstring for the format's own definition and rationale: per
// (output-channel, 16-element reduction-axis block), try BOTH a plain
// signed 4-bit integer grid ({-8, ..., 7}) and MXFP4's own E2M1 codebook
// ({0, 0.5, 1, 1.5, 2, 3, 4, 6}, signed -- reused here via
// quantize_mxfp4_common.h's own MXFP4Codebook(), the same reuse
// if4_quantization.py's own docstring documents on the Python side), each
// with its own best-fit linear scale (max(|block|) / that codebook's own
// largest magnitude, NOT a power-of-two E8M0 scale the way
// weight_only_quantize_mxfp4_matmul.h's own MXFP4 port picks it -- IF4's
// own encoder uses a plain division, exactly matching
// if4_quantization.py's own `fp4_scale = max_abs / _FP4_MAX_MAGNITUDE`),
// and keep whichever candidate reconstructs that block with lower mean
// squared error.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, 16-element block)
//                                   group replaced by its own IF4
//                                   quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_q6_k.h/iq4_nl.h use. Unlike
// if4_quantization.py's own quantize_weight_only_if4, this port folds the
// round trip directly into a replacement float32 initializer (no new
// Cast/Gather/Reshape/Mul graph nodes, and no combined 32-entry codebook
// initializer either -- there is nothing left to Gather once the winning
// candidate's own reconstructed value is written straight into the
// weight), matching iq4_nl.h's/every K-quant *_cpp port's own established
// convention here ("several other *_cpp ports in this repo already
// establish that a C++ port need not mirror every optional knob its
// Python counterpart has").
//
// Unlike every GGUF *_cpp port in this repo, IF4's own blocks are laid
// out per (output channel, contiguous group of 16 along the *reduction*
// axis) -- not over the weight's own flattened row-major storage --
// exactly matching if4_quantization.py's own `w_nk.reshape(n, num_blocks,
// block_size)` grouping (`w_nk` being the weight re-oriented to
// [output_channel, K] first). This port reads/writes the weight in its
// own native [dim0, dim1] storage layout directly (via an `at(i, j)`
// accessor keyed by `channel_axis`/`reduction_axis`), the same style
// quantize_matmul_common.h's own
// TryQuantizeWeightBlockwiseInt4InPlace/TryQuantizeWeightBlockwiseInt8InPlace
// already use, rather than transposing into an [N, K] buffer the way
// if4_quantization.py's own `w_nk = w if weight_transposed else w.T` does
// -- purely an implementation-style difference, not a numerics one.
//
// A layer whose reduction dimension K is not evenly divisible by 16 is
// left completely untouched -- matching if4_quantization.py's own
// `if k % block_size != 0: continue` (no ragged-last-block handling here,
// unlike the GGUF ports, because if4_quantization.py itself has none to
// match).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM if4_quantization.py: this scheme
// has no accumulation/iterative-refinement step -- every block's own two
// candidate scales, reconstructions and the resulting MSE comparison are
// computed independently per block, so this port is expected to track
// the Python port's own float64 numpy implementation closely, up to
// floating-point summation-order differences in the per-block MSE
// reduction. quantize_weight_only_if4 and this port remain
// independently-correct, non-interchangeable entry points -- this port's
// own tests check structural/algebraic properties and comparable (not
// bit-identical) reconstruction error, matching this repo's established
// contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/quantize_mxfp4_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace if4_quantization_detail {

constexpr int64_t kBlockSize = 16;
constexpr double kFp4MaxMagnitude = 6.0;   // E2M1's own largest magnitude
constexpr double kInt4MaxMagnitude = 7.0;  // symmetric range's "max" side

// Plain signed 4-bit integer grid, two's-complement range -8..7 -- raw
// (unscaled) values, mirroring if4_quantization.py's own _INT4_CODEBOOK.
inline const std::vector<double>& Int4Codebook() {
  static const std::vector<double> kCodebook = [] {
    std::vector<double> c(16);
    for (int i = 0; i < 16; ++i) {
      c[static_cast<size_t>(i)] = static_cast<double>(i - 8);
    }
    return c;
  }();
  return kCodebook;
}

// Nearest codebook *value* (not index) to `normalized`, by linear scan --
// codebooks here are always 16 entries, so this is cheap.
template <typename Codebook>
inline double NearestCodebookValue(double normalized,
                                   const Codebook& codebook) {
  double best = static_cast<double>(codebook[0]);
  double best_diff = std::fabs(normalized - best);
  for (size_t i = 1; i < codebook.size(); ++i) {
    const double candidate = static_cast<double>(codebook[i]);
    const double diff = std::fabs(normalized - candidate);
    if (diff < best_diff) {
      best_diff = diff;
      best = candidate;
    }
  }
  return best;
}

// Quantize-dequantize round trip for `w_t` (a 2-D float32 constant, laid
// out as [N, K] when `weight_transposed` else [K, N]), written into
// `out_data` (same flat row-major layout/size as `w_t`). Returns false
// (leaving `out_data` unspecified) when K is not evenly divisible by
// kBlockSize -- mirrors if4_quantization.py's own `k % block_size != 0`
// skip.
//
// Per (output channel, 16-element reduction-axis block): mirrors
// if4_quantization.py's own _quantize_if4_blockwise -- max_abs =
// max(|block|) clamped to >= 1e-30; fp4_scale = max_abs / 6.0, int4_scale
// = max_abs / 7.0; each element snaps to its own codebook's nearest
// value once divided by that codebook's own scale; whichever candidate
// has the lower per-block mean squared error is kept for every element
// of that block.
inline bool QuantizeDequantizeIF4InPlace(const Tensor& w_t,
                                         bool weight_transposed,
                                         std::vector<float>& out_data) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t channel_axis = weight_transposed ? 0 : 1;
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  const int64_t C = channel_axis == 0 ? dim0 : dim1;
  if (K % kBlockSize != 0) {
    return false;
  }
  const int64_t num_blocks = K / kBlockSize;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) -> double {
    return static_cast<double>(data[static_cast<size_t>(i * dim1 + j)]);
  };
  // Maps a (channel, reduction-axis-position) pair to the underlying
  // (i, j) storage coordinate, regardless of which axis is which.
  auto coord = [&](int64_t c, int64_t k) -> std::pair<int64_t, int64_t> {
    return channel_axis == 0 ? std::make_pair(c, k) : std::make_pair(k, c);
  };

  out_data.assign(data.begin(), data.end());
  const std::vector<float>& fp4_codebook = MXFP4Codebook();
  const std::vector<double>& int4_codebook = Int4Codebook();

  std::vector<double> raw(static_cast<size_t>(kBlockSize));
  std::vector<double> fp4_recon(static_cast<size_t>(kBlockSize));
  std::vector<double> int4_recon(static_cast<size_t>(kBlockSize));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double max_abs = 0.0;
      for (int64_t off = 0; off < kBlockSize; ++off) {
        const auto [i, j] = coord(c, b * kBlockSize + off);
        raw[static_cast<size_t>(off)] = at(i, j);
        max_abs = std::max(max_abs, std::fabs(raw[static_cast<size_t>(off)]));
      }
      max_abs = std::max(max_abs, 1e-30);
      const double fp4_scale = max_abs / kFp4MaxMagnitude;
      const double int4_scale = max_abs / kInt4MaxMagnitude;

      double fp4_mse = 0.0;
      double int4_mse = 0.0;
      for (int64_t off = 0; off < kBlockSize; ++off) {
        const double v = raw[static_cast<size_t>(off)];
        const double fp4_v =
            NearestCodebookValue(v / fp4_scale, fp4_codebook) * fp4_scale;
        const double int4_v =
            NearestCodebookValue(v / int4_scale, int4_codebook) * int4_scale;
        fp4_recon[static_cast<size_t>(off)] = fp4_v;
        int4_recon[static_cast<size_t>(off)] = int4_v;
        fp4_mse += (v - fp4_v) * (v - fp4_v);
        int4_mse += (v - int4_v) * (v - int4_v);
      }
      const bool use_int4 = int4_mse < fp4_mse;
      for (int64_t off = 0; off < kBlockSize; ++off) {
        const auto [i, j] = coord(c, b * kBlockSize + off);
        const double v = use_int4 ? int4_recon[static_cast<size_t>(off)]
                                  : fp4_recon[static_cast<size_t>(off)];
        out_data[static_cast<size_t>(i * dim1 + j)] = static_cast<float>(v);
      }
    }
  }
  return true;
}

}  // namespace if4_quantization_detail

// IF4 -- matches MatMul/vanilla-Gemm the same way GgufQ6K/IQ4NL do, then
// quantizes the weight per (output channel, reduction-axis block).
struct IF4 final : public PredicateBasedPass {
  explicit IF4()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "if4"; }

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
    return K % if4_quantization_detail::kBlockSize == 0;
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
    if (!if4_quantization_detail::QuantizeDequantizeIF4InPlace(
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
