// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// LeptoQuant (Tencent Hunyuan AI Infra Team, 2026, "AngelSlim") -- C++
// port of leptoquant.py's own apply_leptoquant. See that module's
// docstring for the full rationale: it is a drop-in refinement of
// onnxsim.deepseek_fp8's own DeepSeek-V3-style block FP8 weight scheme
// (one FLOAT8E4M3FN scale per 128x128 output-channel x input-feature
// tile, scale = max(|tile|) / 448), except each tile's *clip boundary*
// is grid-searched over an outlier fraction `alpha` instead of always
// being that tile's own true max: for each candidate alpha, clip_value
// is the (1 - alpha) quantile of the tile's own |values| (alpha == 0
// meaning "the tile's true max", i.e. deepseek_fp8's own choice, always
// included as the grid's safety-net candidate), and whichever alpha
// yields the lowest exact reconstruction MSE for that tile wins.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   128x128 output-channel x
//                                   input-feature tile replaced by its
//                                   own LeptoQuant FP8 E4M3 quantize-
//                                   dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm
// with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
// constant 2-D float32 tensor -- exactly leptoquant.py's own scope (no
// Conv, no include_conv option at all on the Python side either, so
// there is no knob being dropped here, unlike several sibling *_cpp
// ports). Tiles are laid out over the weight's own [N, K]
// (output-channel-first) orientation exactly like leptoquant.py's own
// `w_nk = w if weight_transposed else w.T` -- NOT the flattened
// row-major storage every other *_cpp port in this repo uses, because
// LeptoQuant (like the DeepSeek-V3 scheme it refines) tiles along both
// matrix axes independently, not over a flattened 1-D block sequence. A
// ragged edge tile (when N or K isn't itself a multiple of 128) is
// zero-padded to a full 128x128 tile before the grid search runs, exactly
// like leptoquant.py's own zero-pad-then-discard approach -- unlike the
// max-only statistics every flattened-1-D-block *_cpp port here relies
// on, LeptoQuant's own alpha grid search scores each candidate by a
// per-tile MSE average and a quantile, both of which a padding zero
// really does shift, so this port cannot skip the padding the way its
// GGUF-family siblings do.
//
// Reuses passes/quantize_fp8.h's own already-verified FLOAT8E4M3FN
// encode bit-trick (FloatToFloat8Bits, mantissa_bits=3, bias=7,
// max=448, NaN pattern 0x7F -- the real ONNX Cast semantics, matching
// this repo's own established "reuse an already-verified building
// block header" convention, e.g. gguf_legacy_quant.h reusing
// quantize_fp16.h/ggml_kquant.h), but adds its own decode half (bits
// back to float) since no *_cpp port before this one has needed a full
// FP8 round trip evaluated entirely on the host rather than via a
// graph-level Cast node.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM leptoquant.py: as with every
// other data-free *_cpp port in this repo, this scheme's own grid
// search is closed-form/deterministic (no gradient descent or training
// loop), so this port is expected to track the Python port's own
// float64 numpy implementation closely, up to floating-point
// summation-order/quantile-interpolation differences. This port also
// hardcodes leptoquant.py's own defaults (block_size=128, the 5-point
// alpha grid {0, 1e-4, 2e-4, 5e-4, 1e-3}) rather than exposing them as
// parameters -- several other *_cpp ports in this repo already
// establish that a C++ port need not mirror every optional knob its
// Python counterpart has. apply_leptoquant and its _cpp counterpart
// remain independently-correct, non-interchangeable entry points --
// this port's own tests check structural/algebraic properties and
// comparable (not bit-identical) reconstruction error, matching this
// repo's established contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_fp8.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace leptoquant_detail {

constexpr int64_t kBlockSize = 128;
constexpr double kFp8Max =
    448.0;  // FLOAT8E4M3FN's own largest finite magnitude
constexpr double kAlphaGrid[] = {0.0, 1e-4, 2e-4, 5e-4, 1e-3};
constexpr size_t kAlphaGridSize = 5;

// Decodes a FLOAT8E4M3FN bit pattern back to its exact real value, as a
// double. Mirrors the format quantize_fp8.h's own FloatToFloat8Bits
// encodes into: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits;
// the single exponent-all-ones/mantissa-all-ones pattern (0x7F for
// positive, 0xFF for negative) is the format's only NaN encoding, every
// other exponent-all-ones pattern is an ordinary finite value (up to
// the format's own max magnitude 448).
inline double Float8E4M3FNBitsToDouble(uint8_t bits) {
  const uint32_t sign = (bits >> 7) & 0x1u;
  const uint32_t exp = (bits >> 3) & 0xFu;
  const uint32_t mant = bits & 0x7u;
  double magnitude;
  if (exp == 0) {
    magnitude =
        mant == 0 ? 0.0 : std::ldexp(static_cast<double>(mant) / 8.0, 1 - 7);
  } else if (exp == 0xFu && mant == 0x7u) {
    return std::numeric_limits<double>::quiet_NaN();
  } else {
    magnitude = std::ldexp(1.0 + static_cast<double>(mant) / 8.0,
                           static_cast<int>(exp) - 7);
  }
  return sign != 0 ? -magnitude : magnitude;
}

// Full FLOAT8E4M3FN quantize-dequantize round trip for one value, as a
// double -- the exact real ONNX Cast(to=FLOAT8E4M3FN)/Cast(to=FLOAT)
// pair semantics (round-to-nearest ties-to-even, saturating), matching
// onnxsim.deepseek_fp8's own `_fp8_round_trip` (backed by ml_dtypes).
inline double Fp8E4M3RoundTrip(double value) {
  const uint8_t bits =
      FloatToFloat8Bits(static_cast<float>(value), Float8Format::kE4M3FN);
  return Float8E4M3FNBitsToDouble(bits);
}

// numpy.quantile's own default ("linear") interpolation method: `sorted`
// must already be sorted ascending. Mirrors leptoquant.py's own
// `np.quantile(abs_block, 1.0 - alpha)` exactly (not merely
// approximately -- the interpolation rule itself is reproduced, not
// just its rough effect).
inline double LinearQuantile(const std::vector<double>& sorted, double q) {
  const size_t n = sorted.size();
  if (n == 1) {
    return sorted[0];
  }
  const double position = q * static_cast<double>(n - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = std::min(lower + 1, n - 1);
  const double frac = position - static_cast<double>(lower);
  return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
}

// Quantize-dequantize round trip for one (possibly ragged) tile,
// `rows` x `cols` logical elements read via `get`/written via `set`
// (both indexed by a single row-major offset i * cols + j within the
// tile). Mirrors
// leptoquant.py's own per-tile grid search: sorts the tile's own
// |values| once, then for each alpha in kAlphaGrid computes
// clip_value (alpha == 0: the tile's true max via the sorted array's
// own last element; alpha > 0: the (1 - alpha)-quantile), scale =
// max(clip_value, 1e-12) / 448, reconstructs every element through
// Fp8E4M3RoundTrip(value / scale) * scale, and keeps whichever alpha's
// reconstruction has the lowest exact MSE against the tile's own real
// values.
template <typename GetFn, typename SetFn>
void QuantizeDequantizeLeptoquantTile(GetFn get, SetFn set, int64_t rows,
                                      int64_t cols) {
  const int64_t count = rows * cols;
  std::vector<double> values(static_cast<size_t>(count));
  std::vector<double> abs_sorted(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const double v = get(i);
    values[static_cast<size_t>(i)] = v;
    abs_sorted[static_cast<size_t>(i)] = std::fabs(v);
  }
  std::sort(abs_sorted.begin(), abs_sorted.end());

  std::vector<double> best(static_cast<size_t>(count));
  double best_mse = std::numeric_limits<double>::infinity();
  std::vector<double> candidate(static_cast<size_t>(count));
  for (size_t g = 0; g < kAlphaGridSize; ++g) {
    const double alpha = kAlphaGrid[g];
    const double clip_value = alpha == 0.0
                                  ? abs_sorted.back()
                                  : LinearQuantile(abs_sorted, 1.0 - alpha);
    const double scale = std::max(clip_value, 1e-12) / kFp8Max;
    double sq_err = 0.0;
    for (int64_t i = 0; i < count; ++i) {
      const double recon =
          Fp8E4M3RoundTrip(values[static_cast<size_t>(i)] / scale) * scale;
      candidate[static_cast<size_t>(i)] = recon;
      const double diff = values[static_cast<size_t>(i)] - recon;
      sq_err += diff * diff;
    }
    const double mse = sq_err / static_cast<double>(count);
    if (mse < best_mse) {
      best_mse = mse;
      best.swap(candidate);
    }
  }
  for (int64_t i = 0; i < count; ++i) {
    set(i, best[static_cast<size_t>(i)]);
  }
}

}  // namespace leptoquant_detail

// LeptoQuant's outlier-aware block FP8 scheme -- matches MatMul/
// vanilla-Gemm the same way every sibling *_cpp port does, then
// quantizes the weight's own [N, K] (output-channel-first) view
// tile-by-tile.
struct LeptoQuant final : public PredicateBasedPass {
  explicit LeptoQuant()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "leptoquant"; }

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
    const std::vector<float> data = ReadFloatMatrix(*w_t);
    // `data` is row-major over `sizes` (the weight's own on-disk
    // orientation); build an [N, K] (output-channel-first) working copy,
    // exactly like leptoquant.py's own `w_nk = w if weight_transposed
    // else w.T`.
    const int64_t rows = sizes[0];
    const int64_t cols = sizes[1];
    const int64_t num_n = info.weight_transposed ? rows : cols;
    const int64_t num_k = info.weight_transposed ? cols : rows;
    std::vector<double> nk(static_cast<size_t>(num_n) *
                           static_cast<size_t>(num_k));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        const double v = data[static_cast<size_t>(r * cols + c)];
        const int64_t out_n = info.weight_transposed ? r : c;
        const int64_t out_k = info.weight_transposed ? c : r;
        nk[static_cast<size_t>(out_n * num_k + out_k)] = v;
      }
    }

    // A ragged edge tile is always processed as a *full* kBlockSize x
    // kBlockSize buffer, zero-padded past the real (tile_rows, tile_cols)
    // extent, exactly like leptoquant.py's own zero-pad-then-discard
    // approach -- unlike every flattened-1-D-block *_cpp port in this
    // repo, padding here is NOT a no-op: the alpha grid search's own
    // quantile and per-tile MSE (its very selection criterion) are both
    // computed over the *whole* padded tile in Python, so a padding zero
    // measurably shifts both statistics whenever the tile is ragged. Only
    // the real (tile_rows, tile_cols) corner is ever written back.
    constexpr int64_t kBS = leptoquant_detail::kBlockSize;
    for (int64_t n_start = 0; n_start < num_n; n_start += kBS) {
      const int64_t tile_rows = std::min(kBS, num_n - n_start);
      for (int64_t k_start = 0; k_start < num_k; k_start += kBS) {
        const int64_t tile_cols = std::min(kBS, num_k - k_start);
        auto get = [&](int64_t idx) -> double {
          const int64_t i = idx / kBS;
          const int64_t j = idx % kBS;
          if (i >= tile_rows || j >= tile_cols) {
            return 0.0;
          }
          return nk[static_cast<size_t>((n_start + i) * num_k + k_start + j)];
        };
        auto set = [&](int64_t idx, double value) {
          const int64_t i = idx / kBS;
          const int64_t j = idx % kBS;
          if (i >= tile_rows || j >= tile_cols) {
            return;
          }
          nk[static_cast<size_t>((n_start + i) * num_k + k_start + j)] = value;
        };
        leptoquant_detail::QuantizeDequantizeLeptoquantTile(get, set, kBS, kBS);
      }
    }

    // Map the quantized [N, K] view back to the weight's own original
    // [rows, cols] orientation.
    std::vector<float> out_float(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        const int64_t out_n = info.weight_transposed ? r : c;
        const int64_t out_k = info.weight_transposed ? c : r;
        out_float[static_cast<size_t>(r * cols + c)] =
            static_cast<float>(nk[static_cast<size_t>(out_n * num_k + out_k)]);
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
