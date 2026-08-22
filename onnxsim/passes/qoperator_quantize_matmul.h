// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes MatMul/Gemm into the "QOperator"
// format -- ``QLinearMatMul``, ONNX's directly-quantized matmul op -- rather
// than static_quantize_matmul.h's QDQ (QuantizeLinear/DequantizeLinear
// wrapping a float MatMul) format. Both are standard ONNX, and both need the
// same calibrated activation range; QOperator format is the older,
// still-standard alternative some runtimes' int8 kernels key off of
// specifically (QDQ is the now-preferred, more composable format when
// several quantized ops chain together -- see that file's doc comment).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C added back in float):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Xq = QuantizeLinear(X, Xs, Xzp)                    -- Xs/Xzp: CALIBRATED
//   Yq = QLinearMatMul(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp)  -- true int8 compute
//   Y  = DequantizeLinear(Yq, Ys, Yzp)                 -- Ys/Yzp: CALIBRATED
//
// Unlike QDQ format, QLinearMatMul computes in int8 directly -- there is no
// float MatMul left in the graph at all -- so it needs a calibrated range for
// the *output* Y too, not just the activation X (QDQ's DequantizeLinear can
// leave Y in float and let whatever consumes it worry about its own range;
// QLinearMatMul cannot, since Y itself is quantized). See
// ListQOperatorQuantizableOutputs in onnxsim.h for how a caller discovers
// which output tensors need calibrating, on top of
// ListQuantizableActivations' input tensors.
//
// Wq/Ws (int8, per-output-channel symmetric) are the same weight
// quantization every other onnxsim quantization pass uses; Wzp is an
// explicit all-zero tensor of the same shape as Ws (QLinearMatMul, unlike
// DequantizeLinear, has no optional-zero-point convention -- all 8 inputs
// are required). X and Y both use a single (scalar) scale/zero-point rather
// than QLinearMatMul's per-row/per-column option, matching
// static_quantize_matmul.h's activation convention.
//
// Only the common, unambiguous shape is handled -- see
// dynamic_quantize_matmul.h's doc comment, which applies identically here --
// plus: opsets older than 10 (QLinearMatMul's own minimum) are left alone,
// and a MatMul/Gemm is only rewritten when both its activation input's name
// and its own output's name have a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeQOperator in
// onnxsim.h).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct QOperatorQuantizeMatMul final : public PredicateBasedPass {
  explicit QOperatorQuantizeMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "qoperator_quantize_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 10) {
      return false;  // QLinearMatMul needs opset >= 10.
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    if (ranges.count(info.x->uniqueName()) == 0 ||
        ranges.count(n->output()->uniqueName()) == 0) {
      return false;  // No calibrated range for the activation and/or output.
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
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto x_range_it = ranges.find(info.x->uniqueName());
    const auto y_range_it = ranges.find(n->output()->uniqueName());
    if (x_range_it == ranges.end() || y_range_it == ranges.end()) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    float x_scale_f = 1.0f;
    int32_t x_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(x_range_it->second.first,
                                      x_range_it->second.second, x_scale_f,
                                      x_zp_i);
    float y_scale_f = 1.0f;
    int32_t y_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(y_range_it->second.first,
                                      y_range_it->second.second, y_scale_f,
                                      y_zp_i);

    // W's output-channel axis in its own (untransposed) layout: axis 0 when
    // Gemm's transB made W [N, K], else axis 1 ([K, N], MatMul's own layout).
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelInPlace(*w_t, channel_axis, w_q, w_scale);
    // QLinearMatMul (unlike DequantizeLinear) has no optional-zero-point
    // convention -- b_zero_point is required, so an explicit all-zero tensor
    // (symmetric quantization) of w_scale's shape is needed.
    Tensor w_zp;
    w_zp.elem_type() = TensorProto_DataType_INT8;
    w_zp.sizes() = w_scale.sizes();
    w_zp.int32s().assign(w_scale.floats().size(), 0);

    Tensor x_scale_t;
    x_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    x_scale_t.floats() = {x_scale_f};
    Tensor x_zp_t;
    x_zp_t.elem_type() = TensorProto_DataType_UINT8;
    x_zp_t.int32s() = {x_zp_i};
    Value* x_scale_v = graph.addInitializerAndCreateValue(x_scale_t);
    Value* x_zp_v = graph.addInitializerAndCreateValue(x_zp_t);

    Tensor y_scale_t;
    y_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    y_scale_t.floats() = {y_scale_f};
    Tensor y_zp_t;
    y_zp_t.elem_type() = TensorProto_DataType_UINT8;
    y_zp_t.int32s() = {y_zp_i};
    Value* y_scale_v = graph.addInitializerAndCreateValue(y_scale_t);
    Value* y_zp_v = graph.addInitializerAndCreateValue(y_zp_t);

    Value* w_q_v = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_v = graph.addInitializerAndCreateValue(w_scale);
    Value* w_zp_v = graph.addInitializerAndCreateValue(w_zp);

    // Xq = QuantizeLinear(X, Xs, Xzp)
    Node* ql = graph.create(Symbol("QuantizeLinear"), 1);
    ql->addInput(info.x);
    ql->addInput(x_scale_v);
    ql->addInput(x_zp_v);
    ql->insertBefore(n);
    ql->output()->setElemType(TensorProto_DataType_UINT8);

    // Yq = QLinearMatMul(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp)
    Node* qlmm = graph.create(Symbol("QLinearMatMul"), 1);
    qlmm->addInput(ql->output());
    qlmm->addInput(x_scale_v);
    qlmm->addInput(x_zp_v);
    qlmm->addInput(w_q_v);
    qlmm->addInput(w_scale_v);
    qlmm->addInput(w_zp_v);
    qlmm->addInput(y_scale_v);
    qlmm->addInput(y_zp_v);
    qlmm->insertBefore(n);
    qlmm->output()->setElemType(TensorProto_DataType_UINT8);

    // Y = DequantizeLinear(Yq, Ys, Yzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qlmm->output());
    dq->addInput(y_scale_v);
    dq->addInput(y_zp_v);
    dq->insertBefore(n);
    dq->output()->setElemType(TensorProto_DataType_FLOAT);

    Value* result = dq->output();
    if (info.bias != nullptr) {
      Node* add = graph.create(kAdd, 1);
      add->addInput(dq->output());
      add->addInput(info.bias);
      add->insertBefore(n);
      add->output()->setElemType(TensorProto_DataType_FLOAT);
      result = add->output();
    }

    if (n->output()->sizes().size() > 0) {
      result->setSizes(n->output()->sizes());
    }

    const bool replacing_success = tryReplacingAllUsesWith(n->output(), result);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
