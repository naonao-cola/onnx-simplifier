// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// QuIP# (Tseng et al., 2024, "QuIP#: Even Better LLM Quantization with
// Hadamard Incoherence and Lattice Codebooks", building on QuIP's own
// incoherence-processing idea, Chee et al., 2023) -- C++ port of
// quip_sharp.py's own apply_quip_sharp. See that module's docstring for
// the full rationale; the short version:
//
//   1. Incoherence processing: conjugating the weight by a pair of random
//      orthogonal matrices (``Wtilde = V @ W @ U``, U along the
//      reduction/K dimension, V along the output/N dimension) makes the
//      transformed weight's entries look like i.i.d. Gaussian noise with
//      overwhelming probability, regardless of the original weight's own
//      structure -- a uniform/lattice quantizer fits that far better than
//      data with a handful of outlier directions, with no calibration.
//   2. E8 lattice vector quantization: rather than rounding each weight
//      element independently, groups of 8 consecutive (post-rotation)
//      elements are jointly quantized to the nearest point in the E8
//      lattice (the densest known sphere packing in 8 dimensions) via the
//      classical fast algorithm of Conway & Sloane, 1982 -- see
//      ClosestPointD8/ClosestPointE8 below, a direct transcription of
//      quip_sharp.py's own _closest_point_d8/_closest_point_e8.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W
//
// SCOPE NARROWING (beyond quip_sharp.py's own MatMulLike/constant-2D-float
// weight/K-divisible-by-8 scope, which this port matches exactly): unlike
// quip_sharp.py's own graph rewrite -- which keeps U, V and the packed
// INT4 lattice codes/scale as explicit initializers and rebuilds the node
// as MatMul(X, U) -> dequantize -> MatMul(_, Wtilde_hat) -> MatMul(_, V)
// [+ bias] -- this port folds the ENTIRE rotate/quantize/rotate-back
// sandwich into a single replacement float32 initializer of the same
// shape as the original weight, leaving the matched node itself untouched
// (matching kmeans_quantization.h/aqlm.h's own fold-to-initializer
// convention, not quip_sharp.py's own choice to keep the rotations
// visible in the graph). This fold is exact, not an approximation: since
// U and V are square and only sandwich the *weight* (X itself is rotated
// by U and then un-rotated by V with no intervening nonlinearity or
// data-dependent step -- every op between them is a plain MatMul), matrix
// multiplication's associativity gives
//   X @ ((X @ U) => quantized-dequantized middle => (_ @ V))
//     == X @ (U @ Wtilde_hat @ V)
// exactly, so a single effective [K, N] (or [N, K], transposed) matrix
// W' = U @ Wtilde_hat @ V reproduces the same linear map with zero new
// graph nodes. This is fundamentally different from quarot.h's own
// situation, which cannot fold this way: QuaRot ALSO quantizes the
// *activation* X itself at runtime (a calibration-free absmax scale
// computed from X's own live values), a genuinely runtime-dependent step
// this offline pass cannot precompute. QuIP# never quantizes X -- only
// the weight -- so nothing here depends on X's own runtime values, and
// the fold is lossless with respect to quip_sharp.py's own graph (up to
// ordinary floating-point rounding-order differences, the same caveat
// every other fold-to-initializer port in this repo already carries).
//
// This port also hardcodes quip_sharp.py's own defaults (seed=0,
// epsilon=1e-8) rather than exposing them as parameters -- several other
// *_cpp ports in this repo already establish that a C++ port need not
// mirror every optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM quip_sharp.py (do not "fix" without
// re-reading this comment in full): the random orthogonal matrices U/V
// are generated exactly as quarot.h's own precedent already establishes
// for this codebase -- passes/random_orthogonal.h's Gram-Schmidt-based
// RandomOrthogonalMatrix (not quip_sharp.py's own QR-with-sign-correction
// construction), and a fresh std::mt19937_64 is reseeded per matched node
// from a hash of a fixed base seed and the node's own unique id (U then V
// drawn from that same per-node generator, in that order -- mirroring
// quip_sharp.py's own within-layer draw order, just not its
// across-layers-in-one-generator sequencing). Cross-language bit parity
// with apply_quip_sharp(seed=N) is explicitly NOT a goal here, for the
// same reasons random_orthogonal.h's and quarot.h's own top-of-file
// comments already give in full for their own identical choice: both
// constructions are independently Haar-uniform, and reproducing numpy's
// own PCG64 bitstream in C++ is not worth it for a rotation whose only
// real requirement is being SOME uniformly random orthogonal matrix. The
// E8 lattice decoding itself (ClosestPointD8/ClosestPointE8) has no such
// divergence -- it is a closed-form, deterministic algorithm transcribed
// directly from quip_sharp.py's own implementation, so it is expected to
// track the Python port's own float64 numpy implementation closely up to
// ordinary floating-point summation-order differences, given the SAME
// input (which, because of the RNG divergence above, this port's U/V/
// group values never are). apply_quip_sharp and this port remain
// independently-correct, non-interchangeable entry points.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/random_orthogonal.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace quip_sharp_detail {

constexpr int64_t kGroupSize = 8;
constexpr double kEpsilon = 1e-8;
// Fixed base seed -- quip_sharp.py's own default `seed=0` parameter is
// hardcoded rather than exposed (see this header's own top-of-file scope
// note); combined with the matched node's own unique id below, this still
// gives a deterministic, reproducible-per-model rotation.
constexpr uint64_t kBaseSeed = 0;

// Nearest point in D8 (integer vectors with an even coordinate sum) to
// `v`, via Conway & Sloane's fast algorithm: round to the nearest
// integers, then if the sum is odd, flip the least-confidently-rounded
// coordinate (largest residual) to the other side. Direct transcription
// of quip_sharp.py's own _closest_point_d8, operating on one 8-element
// group at a time (that module's own batched-over-numpy-array version
// applied elementwise).
inline std::array<double, 8> ClosestPointD8(const std::array<double, 8>& v) {
  std::array<double, 8> f;
  std::array<double, 8> delta;
  for (int i = 0; i < 8; ++i) {
    f[i] = std::round(v[i]);
    delta[i] = v[i] - f[i];
  }
  int64_t sum_f = 0;
  for (int i = 0; i < 8; ++i) {
    sum_f += static_cast<int64_t>(f[i]);
  }
  if ((sum_f % 2) != 0) {
    int idx = 0;
    double best = std::fabs(delta[0]);
    for (int i = 1; i < 8; ++i) {
      if (std::fabs(delta[i]) > best) {
        best = std::fabs(delta[i]);
        idx = i;
      }
    }
    const double adjustment =
        delta[idx] > 0.0 ? 1.0 : (delta[idx] < 0.0 ? -1.0 : 1.0);
    f[idx] += adjustment;
  }
  return f;
}

// Nearest point in the E8 lattice (D8 union its half-integer coset) to
// `v`: the closer of the two candidates ClosestPointD8 gives for `v`
// itself and for `v` shifted onto the coset. Direct transcription of
// quip_sharp.py's own _closest_point_e8.
inline std::array<double, 8> ClosestPointE8(const std::array<double, 8>& v) {
  const std::array<double, 8> g0 = ClosestPointD8(v);
  std::array<double, 8> shifted;
  for (int i = 0; i < 8; ++i) {
    shifted[i] = v[i] - 0.5;
  }
  const std::array<double, 8> g1_raw = ClosestPointD8(shifted);
  std::array<double, 8> g1;
  for (int i = 0; i < 8; ++i) {
    g1[i] = g1_raw[i] + 0.5;
  }
  double d0 = 0.0;
  double d1 = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double diff0 = v[i] - g0[i];
    d0 += diff0 * diff0;
    const double diff1 = v[i] - g1[i];
    d1 += diff1 * diff1;
  }
  return d0 <= d1 ? g0 : g1;
}

}  // namespace quip_sharp_detail

// QuIP# -- matches MatMul/vanilla-Gemm the same way quarot.h/aqlm.h do,
// then folds the rotate/E8-quantize/rotate-back sandwich into a single
// replacement weight initializer (see this header's own top-of-file
// comment for why that fold is exact, not an approximation).
struct QuipSharp final : public PredicateBasedPass {
  explicit QuipSharp()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "quip_sharp"; }

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
    const int64_t K =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    return K % quip_sharp_detail::kGroupSize == 0;
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

    const int64_t N =
        info.weight_transposed ? w_t->sizes()[0] : w_t->sizes()[1];
    const int64_t K =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    if (K % quip_sharp_detail::kGroupSize != 0) {
      return false;
    }
    const int64_t num_blocks = K / quip_sharp_detail::kGroupSize;

    // w_nk: [N, K], output channel first.
    const std::vector<float> w_nk = ReadWeightNK(*w_t, info.weight_transposed);

    // Fresh per-node RNG (see this header's own ACCEPTED, PERMANENT
    // DIVERGENCE note): U then V drawn from the same generator, in that
    // order, mirroring quip_sharp.py's own within-layer draw order.
    std::mt19937_64 rng(quip_sharp_detail::kBaseSeed ^
                        (0x9E3779B97F4A7C15ULL * (n->output()->unique() + 1)));
    const std::vector<float> u = RandomOrthogonalMatrix(K, rng);  // [K, K]
    const std::vector<float> v = RandomOrthogonalMatrix(N, rng);  // [N, N]

    // w_tilde_nk = V @ W_nk @ U -- [N, K], exact (double precision) before
    // quantization.
    std::vector<double> tmp(static_cast<size_t>(N * K), 0.0);  // W_nk @ U
    for (int64_t r = 0; r < N; ++r) {
      const float* w_row = w_nk.data() + r * K;
      for (int64_t c = 0; c < K; ++c) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < K; ++kk) {
          acc += static_cast<double>(w_row[kk]) *
                 static_cast<double>(u[static_cast<size_t>(kk * K + c)]);
        }
        tmp[static_cast<size_t>(r * K + c)] = acc;
      }
    }
    std::vector<double> w_tilde_nk(static_cast<size_t>(N * K),
                                   0.0);  // V @ tmp
    for (int64_t r = 0; r < N; ++r) {
      for (int64_t c = 0; c < K; ++c) {
        double acc = 0.0;
        for (int64_t m = 0; m < N; ++m) {
          acc += static_cast<double>(v[static_cast<size_t>(r * N + m)]) *
                 tmp[static_cast<size_t>(m * K + c)];
        }
        w_tilde_nk[static_cast<size_t>(r * K + c)] = acc;
      }
    }

    // Per-8-element-group quantize/dequantize round trip. The dequant
    // arithmetic itself (codes / 2 * scale) is deliberately done in
    // float32, matching the precision quip_sharp.py's own exported graph
    // actually computes it in (a Cast-to-float32/Div/Mul/Reshape chain);
    // everything else in this pass stays in double for the fold's own
    // numerical stability, the same convention aqlm.h/quarot.h use.
    std::vector<double> recon_nk(static_cast<size_t>(N * K), 0.0);
    for (int64_t r = 0; r < N; ++r) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        std::array<double, 8> group;
        double sum_sq = 0.0;
        for (int64_t d = 0; d < quip_sharp_detail::kGroupSize; ++d) {
          const double value = w_tilde_nk[static_cast<size_t>(
              r * K + b * quip_sharp_detail::kGroupSize + d)];
          group[static_cast<size_t>(d)] = value;
          sum_sq += value * value;
        }
        const double scale =
            std::sqrt(sum_sq /
                      static_cast<double>(quip_sharp_detail::kGroupSize)) +
            quip_sharp_detail::kEpsilon;
        std::array<double, 8> native;
        for (int64_t d = 0; d < quip_sharp_detail::kGroupSize; ++d) {
          native[static_cast<size_t>(d)] =
              group[static_cast<size_t>(d)] / scale;
        }
        const std::array<double, 8> lattice =
            quip_sharp_detail::ClosestPointE8(native);
        const float scale_f32 = static_cast<float>(scale);
        for (int64_t d = 0; d < quip_sharp_detail::kGroupSize; ++d) {
          double code = std::round(lattice[static_cast<size_t>(d)] * 2.0);
          code = std::clamp(code, -7.0, 7.0);
          const float code_f32 = static_cast<float>(code);
          const float recon = code_f32 / 2.0f * scale_f32;
          recon_nk[static_cast<size_t>(r * K +
                                       b * quip_sharp_detail::kGroupSize + d)] =
              static_cast<double>(recon);
        }
      }
    }

    // W' = U @ Wtilde_hat @ V, Wtilde_hat = recon_nk transposed to [K, N]
    // (recon_nk[n][k] read as Wtilde_hat[k][n], a pure index relabeling).
    std::vector<double> tmp2(static_cast<size_t>(K * N),
                             0.0);  // U @ Wtilde_hat
    for (int64_t i = 0; i < K; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        double acc = 0.0;
        for (int64_t k2 = 0; k2 < K; ++k2) {
          acc += static_cast<double>(u[static_cast<size_t>(i * K + k2)]) *
                 recon_nk[static_cast<size_t>(j * K + k2)];
        }
        tmp2[static_cast<size_t>(i * N + j)] = acc;
      }
    }
    std::vector<double> w_eff(static_cast<size_t>(K * N), 0.0);  // tmp2 @ V
    for (int64_t i = 0; i < K; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        double acc = 0.0;
        for (int64_t m = 0; m < N; ++m) {
          acc += tmp2[static_cast<size_t>(i * N + m)] *
                 static_cast<double>(v[static_cast<size_t>(m * N + j)]);
        }
        w_eff[static_cast<size_t>(i * N + j)] = acc;
      }
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    std::vector<float> out_data;
    if (info.weight_transposed) {
      w_out.sizes() = {N, K};
      out_data.assign(static_cast<size_t>(N * K), 0.0f);
      for (int64_t nn = 0; nn < N; ++nn) {
        for (int64_t kk = 0; kk < K; ++kk) {
          out_data[static_cast<size_t>(nn * K + kk)] =
              static_cast<float>(w_eff[static_cast<size_t>(kk * N + nn)]);
        }
      }
    } else {
      w_out.sizes() = {K, N};
      out_data.assign(w_eff.begin(), w_eff.end());
    }
    w_out.floats() = std::move(out_data);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
