// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes MatMul/Gemm with a finer
// **activation** quantization step than static_quantize_matmul.h's: this is
// a "W8A16" scheme -- the weight stays INT8 (the same per-output-channel
// symmetric scheme every onnxsim static/dynamic pass uses), but the
// activation is quantized to UINT16 instead of UINT8, an 8x finer
// calibrated affine step (1/65535 relative vs UINT8's 1/255).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After (QDQ format: the MatMul/Gemm node itself is untouched, only its
// inputs change):
//   Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: CALIBRATED, fixed,
//   uint16 Xdq = DequantizeLinear(Xq, Xs, Xzp) Wdq = DequantizeLinear(Wq, Ws,
//   axis=<W's output-channel axis>)  -- Wq: int8 Y   = MatMul(Xdq, Wdq)
//
// Why the activation gets the finer type and not the weight: this pass's
// only real difference from static_quantize_matmul.h is how much rounding
// error the QuantizeLinear -> DequantizeLinear round trip introduces on the
// activation before it reaches the (still-float32) MatMul -- the weight's
// own INT8 precision is unaffected either way. So this targets exactly the
// activations most sensitive to that round trip (e.g. post-softmax
// attention scores, or any tensor whose calibrated range is unusually wide
// relative to its typical value), without giving up INT8's weight
// compression the way widening the weight too would.
//
// See static_quantize_matmul.h for the calibration-range global
// (StaticQuantizationCalibrationRanges) and ComputeAsymmetricUint16QuantParams
// this pass shares with static_quantize_int16_conv.h.
//
// Only the common, unambiguous shape is handled -- see
// dynamic_quantize_matmul.h's doc comment, which applies identically here --
// plus: opsets older than 21 (UINT16 QuantizeLinear/DequantizeLinear support
// requires opset 21, unlike static_quantize_matmul.h's UINT8 scheme, which
// only needs opset 13) are left alone, and a MatMul/Gemm is only rewritten
// when its activation input's name has a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeStaticInt16 in
// onnxsim.h).

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct StaticQuantizeInt16MatMul final : public PredicateBasedPass {
  explicit StaticQuantizeInt16MatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "static_quantize_int16_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
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
    ComputeAsymmetricUint16QuantParams(
        range_it->second.first, range_it->second.second, x_scale_f, x_zp_i);

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
