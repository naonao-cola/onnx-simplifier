// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Weight-only quantization: unlike dynamic_quantize_matmul.h and
// static_quantize_matmul.h, only the constant weight is quantized -- the
// activation is left exactly as it was, with no QuantizeLinear/
// DequantizeLinear pair and no calibration data of any kind. This is the
// scheme real-world weight-heavy models (large embedding/linear layers in
// transformer-style ASR/TTS decoders, for example) actually ship most often:
// activations stay at full precision because they are cheap to compute and
// quantizing them buys little, while the weights dominate model size and
// compress cleanly since they never change at runtime.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=<W's output-channel axis>)
//   Y   = MatMul(X, Wdq)
//
// Wq (int8) and Ws (float32, one scale per output channel) are computed once,
// here, from W's static values -- the same per-output-channel symmetric INT8
// quantization the dynamic and static passes use. X is never touched, so this
// pass needs no calibration step and adds no runtime cost to the activation
// path -- only the serialized/loaded model shrinks (weights compress ~4x)
// and a QDQ-aware runtime can still fuse the DequantizeLinear into the
// consuming op's weight-loading step.
//
// Only the common, unambiguous shape is handled -- see
// dynamic_quantize_matmul.h's doc comment, which applies identically here --
// plus: opsets older than 13 (DequantizeLinear's per-channel `axis` attribute
// requires opset 13) are left alone.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeMatMul final : public PredicateBasedPass {
  explicit WeightOnlyQuantizeMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_matmul";
  }

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
    QuantizeWeightPerChannelInPlace(*w_t, channel_axis, w_q, w_scale);

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
