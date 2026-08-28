// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Weight-only INT16 quantization: identical in structure to
// weight_only_quantize_matmul.h's INT8 scheme -- per-output-channel
// symmetric quantization of only the constant weight, activation and
// DequantizeLinear-fusion behavior unchanged -- just with a wider,
// 16-bit-integer step (scale = max(|w|) / 32767 per channel, instead of
// INT8's / 127). See that file's doc comment for the general rationale
// (no calibration data, no activation quantization).
//
// The reason this exists as a separate scheme rather than a parameter to
// the INT8 pass: INT16's ~8x finer step (1/32767 relative vs INT8's
// 1/127) matters specifically for channels with a few extreme-outlier
// weights, where INT8's coarse step leaves the channel's *typical*
// (median-magnitude) weight rounding to within one quantization step of
// zero -- effectively lost. onnxsim.estimate_quantization_precision (see
// precision_estimator.py) flags exactly this case (a channel's
// max(|w|)/median(|w|) ratio past 127) and recommends INT16 as one fix;
// this pass is that fix. It trades away half of INT8's ~4x weight
// compression (INT16 is ~2x smaller than float32, not ~4x) for that
// resolution.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=<W's output-channel axis>)
//   Y   = MatMul(X, Wdq)
//
// Wq is INT16 here (INT8 in weight_only_quantize_matmul.h). Only the common,
// unambiguous shape is handled -- see dynamic_quantize_matmul.h's doc
// comment, which applies identically here -- plus: opsets older than 21
// (INT16 is a QuantizeLinear/DequantizeLinear type only from opset 21
// onward) are left alone.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeInt16MatMul final : public PredicateBasedPass {
  explicit WeightOnlyQuantizeInt16MatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_int16_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 21) {
      return false;  // INT16 QuantizeLinear/DequantizeLinear needs opset >= 21.
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    return w_t != nullptr && w_t->elem_type() == TensorProto_DataType_FLOAT &&
           w_t->sizes().size() == 2;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // This pass only ever replaces `n`'s weight input, never `n` itself.
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    // W's output-channel axis in its own (untransposed) layout: axis 0 when
    // Gemm's transB made W [N, K], else axis 1 ([K, N], MatMul's own layout).
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelInPlaceInt16(*w_t, channel_axis, w_q, w_scale);

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

    // The MatMul/Gemm node and its activation input are left untouched; only
    // the weight input changes, to its dequantized counterpart.
    n->replaceInput(1, wdq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
