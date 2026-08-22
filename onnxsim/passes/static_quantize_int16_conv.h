// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes Conv with a finer **activation**
// quantization step, exactly mirroring static_quantize_int16_matmul.h's
// MatMul/Gemm rewrite -- see that file's doc comment for the "W8A16"
// rationale (weight stays INT8, only the activation widens to UINT16) and
// the calibration-range global/quant-param helper both passes share
// (StaticQuantizationCalibrationRanges / ComputeAsymmetricUint16QuantParams,
// both defined in static_quantize_matmul.h). The only real difference from
// the MatMul/Gemm version is the weight's per-output-channel axis: Conv's
// weight layout ([Cout, Cin/groups, k...]) always puts the output channel on
// axis 0, so unlike Gemm there is no transposed-layout case to pick between
// -- the same relationship static_quantize_conv.h has to
// static_quantize_matmul.h.
//
// Before:
//   Y = Conv(X, W)            W constant, float32, rank >= 3
// After (Conv itself is untouched, only its inputs change):
//   Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: CALIBRATED, fixed,
//   uint16 Xdq = DequantizeLinear(Xq, Xs, Xzp) Wdq = DequantizeLinear(Wq, Ws,
//   axis=0)  -- Wq: int8 Y   = Conv(Xdq, Wdq)
//
// Only Conv's ``X`` and ``W`` inputs are quantized; its optional bias (a
// third input, added back untouched) and its kernel/stride/pad/dilation/
// group/auto_pad attributes are left exactly as they were. Needs opset >= 21
// (UINT16 QuantizeLinear/DequantizeLinear support), unlike
// static_quantize_conv.h's UINT8 scheme, which only needs opset 13.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct StaticQuantizeInt16Conv final : public PredicateBasedPass {
  explicit StaticQuantizeInt16Conv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "static_quantize_int16_conv";
  }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 21) {
      return false;  // UINT16 QuantizeLinear/DequantizeLinear needs opset
                     // >= 21.
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
           w_t->sizes().size() >= 3;
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
    const auto range_it = ranges.find(info.x->uniqueName());
    if (range_it == ranges.end()) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      return false;
    }

    float x_scale_f = 1.0f;
    int32_t x_zp_i = 0;
    ComputeAsymmetricUint16QuantParams(
        range_it->second.first, range_it->second.second, x_scale_f, x_zp_i);

    Tensor w_q;
    Tensor w_scale;
    QuantizeConvWeightPerOutputChannel(*w_t, w_q, w_scale);

    Tensor x_scale_t;
    x_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    x_scale_t.floats() = {x_scale_f};
    Tensor x_zp_t;
    x_zp_t.elem_type() = TensorProto_DataType_UINT16;
    x_zp_t.int32s() = {x_zp_i};
    Value* x_scale_v = graph.addInitializerAndCreateValue(x_scale_t);
    Value* x_zp_v = graph.addInitializerAndCreateValue(x_zp_t);

    // Xq = QuantizeLinear(X, Xs, Xzp)
    Node* ql = graph.create(Symbol("QuantizeLinear"), 1);
    ql->addInput(info.x);
    ql->addInput(x_scale_v);
    ql->addInput(x_zp_v);
    ql->insertBefore(n);
    ql->output()->setElemType(TensorProto_DataType_UINT16);

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

    // Wdq = DequantizeLinear(Wq, Ws, axis=0) -- zero_point omitted
    // (symmetric, i.e. always 0).
    Node* wdq = graph.create(Symbol("DequantizeLinear"), 1);
    wdq->addInput(w_q_v);
    wdq->addInput(w_scale_v);
    wdq->i_(kaxis, 0);
    wdq->insertBefore(n);
    wdq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.w->sizes().size() > 0) {
      wdq->output()->setSizes(info.w->sizes());
    }

    // Conv itself is left in place; only its activation and weight inputs
    // change, to their dequantized (QDQ) counterparts. Its optional bias
    // (input 2) is untouched.
    n->replaceInput(0, xdq->output());
    n->replaceInput(1, wdq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
