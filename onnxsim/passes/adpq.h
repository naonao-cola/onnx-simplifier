// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// AdpQ (Ghaffari et al., 2024, "AdpQ: A Zero-shot Calibration Free Adaptive
// Post Training Quantization Method for LLMs") -- C++ port of adpq.py's
// own quantize_weight_only_adpq. See that module's docstring for the full
// rationale: a calibration-free salient/non-salient weight split, decided
// purely from a weight tensor's own values (no Hessian, no activations),
// borrowing its threshold rule from Adaptive LASSO regression (Zou,
// 2006).
//
// Per (output-channel n, group-of-128-along-K j), transcribed from
// adpq.py's own _adaptive_thresholds/quantize_weight_only_adpq:
//   median   = median(group)
//   mad      = median(|group - median|)
//   sigma    = max(1.4826 * mad, 1e-12)              (robust scale estimate)
//   threshold = lambda_ * sigma ^ (1 - gamma)         (adaptive soft threshold,
//                                                       lambda_=3.0, gamma=0.3)
//   salient_i = |value_i| > threshold                 (per element)
//   scale     = max(max(|value_i| : not salient_i), 1e-12) / 7.0
//                                                       (excludes salient
//                                                        elements from the
//                                                        scale fit, the same
//                                                        "exclude the
//                                                        outliers from the
//                                                        scale" trick
//                                                        spqr.py uses)
//   dequant_i = clip(round(value_i / scale), -7, 7) * scale   (non-salient)
//   dequant_i = value_i                                        (salient --
//                                                       exact, since
//                                                       adpq.py's own
//                                                       ScatterND
//                                                       correction is
//                                                       precisely
//                                                       value_i - dequant_i
//                                                       added back)
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every
//                                   (output-channel, 128-element K-group)
//                                   group replaced by its own AdpQ
//                                   quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// adpq_detail::kGroupSize -- the same scope gguf_q6_k.h/hqq.h use, plus
// the group-divisibility requirement adpq.py's own encoder already
// imposes (a layer that fails it is left untouched, exactly like the
// Python side's own `continue`).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM adpq.py, two-fold: (1) unlike
// adpq.py's own quantize_weight_only_adpq, which builds a real
// DequantizeLinear(codes, scale)+ScatterND(salient correction)+Add graph
// rewrite with packed INT4 codes (needing opset 21+ for INT4/blocked
// DequantizeLinear), this port folds the whole reconstruction directly
// into a plain replacement float32 initializer instead -- several other
// *_cpp ports in this repo already establish that a C++ port need not
// mirror every optional knob or exact tensor representation its Python
// counterpart has, and this port consequently needs no opset gate at
// all, unlike the Python side. (2) group_size/lambda_/gamma are fixed at
// adpq.py's own defaults (128, 3.0, 0.3) rather than exposed as pass
// parameters, matching every other data-free *_cpp port's
// no-configurable-knobs convention. Numerically, this scheme has no
// accumulation/iterative-refinement step -- every group's own threshold
// and scale are computed independently from that group's own median/MAD/
// max(|.|) -- so this port is expected to track the Python port's own
// float64 numpy implementation closely, up to floating-point median-
// tie-breaking and summation-order differences.
// quantize_weight_only_adpq and this port's _cpp counterpart remain
// independently-correct, non-interchangeable entry points.

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

namespace adpq_detail {

constexpr int64_t kGroupSize = 128;
constexpr double kLambda = 3.0;
constexpr double kGamma = 0.3;
constexpr int64_t kMaxCode =
    7;  // symmetric signed 4-bit range's magnitude side

// numpy.median's own even-count convention: the mean of the two middle
// sorted values. `values` is consumed by value (sorted in place) since
// every caller here already owns a disposable copy.
inline double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t count = values.size();
  if (count % 2 == 1) {
    return values[count / 2];
  }
  return 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

// AdpQ quantize-dequantize round trip for one (output-channel, K-group)
// group of exactly kGroupSize contiguous elements, written back in place
// -- salient elements (see this header's own top-of-file formula) are
// left completely untouched, reproducing adpq.py's own ScatterND
// correction's exact-reconstruction effect without needing a separate
// correction overlay. Mirrors adpq.py's own _adaptive_thresholds plus
// quantize_weight_only_adpq's own per-group body.
inline void QuantizeDequantizeAdpqGroup(double* group, int64_t count) {
  const std::vector<double> values(group, group + count);
  const double median = Median(values);
  std::vector<double> abs_dev(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    abs_dev[static_cast<size_t>(i)] = std::fabs(group[i] - median);
  }
  const double mad = Median(std::move(abs_dev));
  const double sigma_hat = std::max(1.4826 * mad, 1e-12);
  const double threshold = kLambda * std::pow(sigma_hat, 1.0 - kGamma);

  double max_abs_non_salient = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    if (std::fabs(group[i]) <= threshold) {
      max_abs_non_salient = std::max(max_abs_non_salient, std::fabs(group[i]));
    }
  }
  const double scale =
      std::max(max_abs_non_salient, 1e-12) / static_cast<double>(kMaxCode);

  for (int64_t i = 0; i < count; ++i) {
    if (std::fabs(group[i]) > threshold) {
      continue;  // salient -- exact reconstruction, left unchanged
    }
    double code = std::round(group[i] / scale);
    code = std::min(std::max(code, -static_cast<double>(kMaxCode)),
                    static_cast<double>(kMaxCode));
    group[i] = code * scale;
  }
}

}  // namespace adpq_detail

// AdpQ -- matches MatMul/vanilla-Gemm the same way GgufQ2K/HQQ do, then
// quantizes each (output-channel, K-group) group of the weight
// independently. Groups are laid out along the reduction dimension K
// *per output channel* (adpq.py's own [N, K] convention) -- so a Gemm's
// transB=1 weight (already stored [N, K]) is walked contiguously, while
// a MatMul's weight (stored [K, N]) is walked with an N-stride,
// mirroring adpq.py's own `w_nk = w if weight_transposed else w.T`
// normalization exactly, without physically transposing memory.
struct ADPQ final : public PredicateBasedPass {
  explicit ADPQ()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "adpq"; }

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
    return k % adpq_detail::kGroupSize == 0;
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
    if (k_dim % adpq_detail::kGroupSize != 0) {
      return false;
    }

    const std::vector<float> data = ReadFloatMatrix(*w_t);

    // Materialize the logical [N, K] view (adpq.py's own w if
    // weight_transposed else w.T) so each (output-channel, K-group) is
    // contiguous for the group kernel above.
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
           k_start += adpq_detail::kGroupSize) {
        adpq_detail::QuantizeDequantizeAdpqGroup(row + k_start,
                                                 adpq_detail::kGroupSize);
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
