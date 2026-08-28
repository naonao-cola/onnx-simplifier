// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes Conv into the "QOperator" format
// -- ``QLinearConv``, ONNX's directly-quantized convolution op -- exactly
// mirroring qoperator_quantize_matmul.h's MatMul/Gemm rewrite (see that
// file's doc comment for the QOperator-vs-QDQ tradeoff and why the *output*
// needs a calibrated range too, not just the activation).
//
// Before:
//   Y = Conv(X, W[, B])      W constant, float32, rank >= 3; B optional,
//                            constant, float32, [Cout]
// After:
//   Xq = QuantizeLinear(X, Xs, Xzp)                              -- CALIBRATED
//   Yq = QLinearConv(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp[, Bq])     -- true int8
//   Y  = DequantizeLinear(Yq, Ys, Yzp)                            -- CALIBRATED
//
// Unlike static_quantize_conv.h's QDQ rewrite (which keeps the original Conv
// node and its float bias untouched), QLinearConv takes its optional bias
// (input 8) pre-quantized to INT32, with a *fixed* per-channel scale of
// ``x_scale * w_scale[c]`` and zero_point 0 (see the QLinearConv_ver10_doc
// comment in onnx's own schema) -- there is no float Add left in the graph
// for it to ride on, unlike qoperator_quantize_matmul.h's Gemm bias (MatMul
// has no analogous "quantize the bias ahead of time" input to use). This
// means a Conv whose bias is present but not a constant float32 [Cout]
// tensor cannot be rewritten and is left alone -- see the predicate below.
//
// Wq/Ws (int8, per-output-channel symmetric, axis 0 -- Conv's weight layout
// gives no other choice, see quantize_conv_common.h) are the same weight
// quantization static_quantize_conv.h uses; Wzp is an explicit all-zero
// tensor of the same shape as Ws (QLinearConv, like QLinearMatMul, has no
// optional-zero-point convention -- w_zero_point is required whenever w is
// present). X and Y both use a single (scalar) scale/zero-point, matching
// static_quantize_matmul.h's/qoperator_quantize_matmul.h's activation
// convention (QLinearConv's own per-channel option for x/y is not used).
//
// Conv's kernel_shape/strides/pads/dilations/group/auto_pad attributes are
// copied verbatim onto the new QLinearConv node -- both ops share the same
// attribute set.
//
// Only the common, unambiguous shape is handled: opsets older than 10
// (QLinearConv's own minimum) are left alone, and a Conv is only rewritten
// when both its activation input's name and its own output's name have a
// calibrated range (set via StaticQuantizationCalibrationRanges(), see
// QuantizeQOperator in onnxsim.h).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct QOperatorQuantizeConv final : public PredicateBasedPass {
  explicit QOperatorQuantizeConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "qoperator_quantize_conv"; }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 10) {
      return false;  // QLinearConv needs opset >= 10.
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
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      return false;
    }
    if (info.bias != nullptr) {
      // QLinearConv needs its bias pre-quantized to a static INT32 tensor;
      // only a constant float32 [Cout] bias can be turned into one.
      const Tensor* b_t = FetchConstantTensor(info.bias);
      if (b_t == nullptr || b_t->elem_type() != TensorProto_DataType_FLOAT ||
          b_t->sizes().size() != 1) {
        return false;
      }
    }
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    ConvInfo info;
    if (!MatchConv(n, info)) {
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
        w_t->sizes().size() < 3) {
      return false;
    }
    const Tensor* b_t = nullptr;
    if (info.bias != nullptr) {
      b_t = FetchConstantTensor(info.bias);
      if (b_t == nullptr || b_t->elem_type() != TensorProto_DataType_FLOAT ||
          b_t->sizes().size() != 1) {
        return false;
      }
    }

    float x_scale_f = 1.0f;
    int32_t x_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        x_range_it->second.first, x_range_it->second.second, x_scale_f, x_zp_i);
    float y_scale_f = 1.0f;
    int32_t y_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        y_range_it->second.first, y_range_it->second.second, y_scale_f, y_zp_i);

    Tensor w_q;
    Tensor w_scale;
    QuantizeConvWeightPerOutputChannel(*w_t, w_q, w_scale);
    // QLinearConv (unlike DequantizeLinear) has no optional-zero-point
    // convention -- w_zero_point is required, so an explicit all-zero tensor
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

    // Bq = quantized INT32 bias: per-channel scale = x_scale * w_scale[c],
    // zero_point = 0 (fixed by QLinearConv's own contract, not calibrated).
    Value* bias_v = nullptr;
    if (b_t != nullptr) {
      const std::vector<float> bias_data = ReadFloatTensorFlat(*b_t);
      const auto& w_scale_data = w_scale.floats();
      Tensor bias_q;
      bias_q.elem_type() = TensorProto_DataType_INT32;
      bias_q.sizes() = b_t->sizes();
      bias_q.int32s().resize(bias_data.size());
      for (size_t c = 0; c < bias_data.size(); ++c) {
        const float bias_scale = x_scale_f * w_scale_data[c];
        const float q = std::round(bias_data[c] / bias_scale);
        bias_q.int32s()[c] =
            static_cast<int32_t>(std::clamp(q, -2147483648.0f, 2147483520.0f));
      }
      bias_v = graph.addInitializerAndCreateValue(bias_q);
    }

    // Yq = QLinearConv(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp[, Bq])
    Node* qlconv = graph.create(Symbol("QLinearConv"), 1);
    qlconv->addInput(ql->output());
    qlconv->addInput(x_scale_v);
    qlconv->addInput(x_zp_v);
    qlconv->addInput(w_q_v);
    qlconv->addInput(w_scale_v);
    qlconv->addInput(w_zp_v);
    qlconv->addInput(y_scale_v);
    qlconv->addInput(y_zp_v);
    if (bias_v != nullptr) {
      qlconv->addInput(bias_v);
    }
    qlconv->copyAttributes(*n);
    qlconv->insertBefore(n);
    qlconv->output()->setElemType(TensorProto_DataType_UINT8);

    // Y = DequantizeLinear(Yq, Ys, Yzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qlconv->output());
    dq->addInput(y_scale_v);
    dq->addInput(y_zp_v);
    dq->insertBefore(n);
    dq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (n->output()->sizes().size() > 0) {
      dq->output()->setSizes(n->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(n->output(), dq->output());
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
