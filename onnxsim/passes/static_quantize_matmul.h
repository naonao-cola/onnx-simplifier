// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After (QDQ -- "quantize/dequantize" -- format: the MatMul/Gemm node itself
// is untouched, only its inputs change):
//   Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: CALIBRATED, fixed
//   Xdq = DequantizeLinear(Xq, Xs, Xzp)
//   Wdq = DequantizeLinear(Wq, Ws, axis=<W's output-channel axis>)
//   Y   = MatMul(Xdq, Wdq)
//
// Wq (int8) and Ws (float32, one scale per output channel) are computed once,
// here, from W's static values -- the same per-output-channel symmetric INT8
// quantization dynamic_quantize_matmul.h uses. Xs/Xzp (X's scale and
// zero-point) are *calibrated*: unlike dynamic quantization, which defers X's
// quantization parameters to a DynamicQuantizeLinear computed fresh on every
// run, this pass bakes in a single, fixed (scale, zero_point) pair --
// supplied by the caller as one (min, max) activation range per quantized
// tensor, ordinarily obtained by running the float model over representative
// calibration data and recording each candidate tensor's observed range (see
// QuantizeStatic / the Python `quantize_static` wrapper). X's range is
// asymmetric uint8, widened to include 0 (the standard affine-quantization
// convention, since a zero activation must be exactly representable).
//
// This mirrors ONNX Runtime's "static quantization" in its QDQ format (the
// now-preferred format over the older QOperator/QLinearMatMul one): the graph
// still computes in float32 -- MatMul is untouched -- but a QDQ-aware runtime
// recognizes the bracketing Q/DQ pair and fuses it into a true integer kernel
// at load time. See `passes/dynamic_quantize_matmul.h` for the no-calibration
// alternative.
//
// Only the common, unambiguous shape is handled -- see
// dynamic_quantize_matmul.h's doc comment, which applies identically here --
// plus: opsets older than 13 (DequantizeLinear's per-channel `axis` attribute
// requires opset 13) are left alone, and a MatMul/Gemm is only rewritten when
// its activation input's name has a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeStatic in onnxsim.h).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// The calibrated (min, max) activation range for every tensor name the last
// QuantizeStatic call wants quantized, consulted by StaticQuantizeMatMul.
// QuantizeStatic (onnxsim.cpp) overwrites this immediately before running the
// pass, the same way onnxsim's global `config` feeds ordinary passes --
// see that function's doc comment. A function-local static in an `inline`
// function is one shared instance across every translation unit that
// includes this header (unlike a plain header-defined global), so this is
// safe to call from both this pass (compiled into custom_optimizer_passes.cpp)
// and QuantizeStatic (compiled into onnxsim.cpp).
inline std::unordered_map<std::string, std::pair<float, float>>&
StaticQuantizationCalibrationRanges() {
  static std::unordered_map<std::string, std::pair<float, float>> ranges;
  return ranges;
}

// Computes the (scale, zero_point) of an asymmetric uint8 affine quantization
// covering [min_val, max_val], widened to include 0 so a zero activation
// value is exactly representable (the standard convention -- and the common
// case for a post-ReLU/post-Softmax activation, whose calibrated range is
// often one-sided).
inline void ComputeAsymmetricUint8QuantParams(float min_val, float max_val,
                                              float& scale,
                                              int32_t& zero_point) {
  float lo = std::min(0.0f, min_val);
  float hi = std::max(0.0f, max_val);
  if (hi <= lo) {
    hi = lo + 1.0f;  // Degenerate calibration range (e.g. all-zero data).
  }
  scale = (hi - lo) / 255.0f;
  const float zp = std::round(-lo / scale);
  zero_point = static_cast<int32_t>(std::clamp(zp, 0.0f, 255.0f));
}

struct StaticQuantizeMatMul final : public PredicateBasedPass {
  explicit StaticQuantizeMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "static_quantize_matmul"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 13) {
      return false;  // DequantizeLinear's per-channel `axis` needs opset >= 13.
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (StaticQuantizationCalibrationRanges().count(info.x->uniqueName()) ==
        0) {
      return false;  // No calibrated range for this activation.
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    return w_t != nullptr && w_t->elem_type() == TensorProto_DataType_FLOAT &&
           w_t->sizes().size() == 2;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // Unlike dynamic_quantize_matmul.h, this pass never replaces `n` itself
    // -- only its inputs -- so it never needs to destroy the current node.
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto range_it = ranges.find(info.x->uniqueName());
    if (range_it == ranges.end()) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    float x_scale_f = 1.0f;
    int32_t x_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(range_it->second.first,
                                      range_it->second.second, x_scale_f,
                                      x_zp_i);

    // W's output-channel axis in its own (untransposed) layout: axis 0 when
    // Gemm's transB made W [N, K], else axis 1 ([K, N], MatMul's own layout).
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelInPlace(*w_t, channel_axis, w_q, w_scale);

    Tensor x_scale_t;
    x_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    x_scale_t.floats() = {x_scale_f};
    Tensor x_zp_t;
    x_zp_t.elem_type() = TensorProto_DataType_UINT8;
    x_zp_t.int32s() = {x_zp_i};
    Value* x_scale_v = graph.addInitializerAndCreateValue(x_scale_t);
    Value* x_zp_v = graph.addInitializerAndCreateValue(x_zp_t);

    // Xq = QuantizeLinear(X, Xs, Xzp)
    Node* ql = graph.create(Symbol("QuantizeLinear"), 1);
    ql->addInput(info.x);
    ql->addInput(x_scale_v);
    ql->addInput(x_zp_v);
    ql->insertBefore(n);
    ql->output()->setElemType(TensorProto_DataType_UINT8);

    // Xdq = DequantizeLinear(Xq, Xs, Xzp)
    Node* xdq = graph.create(Symbol("DequantizeLinear"), 1);
    xdq->addInput(ql->output());
    xdq->addInput(x_scale_v);
    xdq->addInput(x_zp_v);
    xdq->insertBefore(n);
    xdq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.x->sizes().size() > 0) {
      xdq->output()->setSizes(info.x->sizes());
    }

    Value* w_q_v = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_v = graph.addInitializerAndCreateValue(w_scale);

    // Wdq = DequantizeLinear(Wq, Ws, axis=channel_axis) -- zero_point
    // omitted (symmetric, i.e. always 0).
    Node* wdq = graph.create(Symbol("DequantizeLinear"), 1);
    wdq->addInput(w_q_v);
    wdq->addInput(w_scale_v);
    wdq->i_(kaxis, channel_axis);
    wdq->insertBefore(n);
    wdq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.w->sizes().size() > 0) {
      wdq->output()->setSizes(info.w->sizes());
    }

    // The MatMul/Gemm node itself is left in place; only its activation and
    // weight inputs change, to their dequantized (QDQ) counterparts.
    n->replaceInput(0, xdq->output());
    n->replaceInput(1, wdq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
