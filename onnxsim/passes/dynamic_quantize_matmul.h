// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, with its optional bias C added back in
// float after dequantization):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Acc         = MatMulInteger(Xq, Wq, Xzp)          // int32
//   Y           = Cast<float>(Acc) * (Xs * Ws)        // dequantize
//
// Wq (int8) and Ws (float32, one scale per output column of W) are computed
// once, here, from W's static values -- ordinary per-output-channel symmetric
// INT8 quantization, zero_point 0. X, by contrast, is not known until the
// model runs, so it is quantized to uint8 *in the graph* by
// ``DynamicQuantizeLinear``, which computes its own scale/zero-point from
// each run's actual input range. The whole rewrite is therefore a pure,
// data-independent graph transform -- same shape as onnxsim's other passes --
// unlike calibration-based ("static") quantization, which needs sample data
// run through the model first to pick activation ranges.
//
// This mirrors the "dynamic quantization" scheme ONNX Runtime's
// ``quantize_dynamic`` applies to MatMul/Gemm.
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input 1)
// is a constant 2-D float32 tensor and whose activation (input 0) is float32.
// Everything else -- non-constant or non-2-D weights, non-default Gemm
// attributes, non-float32 operands, opsets older than 11 (DynamicQuantizeLinear
// requires opset 11) -- is left alone. So is a node whose reduction depth K
// is large enough that MatMulInteger's int32 accumulator could overflow in
// the worst case (see quantize_matmul_common.h's IsSafeInt32ReductionDepth) --
// such a node is left in float rather than silently producing a
// quantization that can wrap around and corrupt its output.

#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct DynamicQuantizeMatMul final : public PredicateBasedPass {
  explicit DynamicQuantizeMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "dynamic_quantize_matmul"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 11) {
      return false;  // DynamicQuantizeLinear needs opset >= 11.
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    return IsSafeInt32ReductionDepth(k);
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
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    if (!IsSafeInt32ReductionDepth(k)) {
      return false;
    }

    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelKN(*w_t, info.weight_transposed, w_q, w_scale);

    // Xq, Xs, Xzp = DynamicQuantizeLinear(X)
    Node* dql = graph.create(Symbol("DynamicQuantizeLinear"), 3);
    dql->addInput(info.x);
    dql->insertBefore(n);
    Value* x_q = dql->outputs()[0];
    Value* x_scale = dql->outputs()[1];
    Value* x_zp = dql->outputs()[2];
    x_q->setElemType(TensorProto_DataType_UINT8);
    x_scale->setElemType(TensorProto_DataType_FLOAT);
    x_zp->setElemType(TensorProto_DataType_UINT8);

    Value* w_q_value = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_value = graph.addInitializerAndCreateValue(w_scale);

    // Acc = MatMulInteger(Xq, Wq, Xzp)   -- b_zero_point omitted (symmetric,
    // i.e. always 0).
    Node* mmi = graph.create(Symbol("MatMulInteger"), 1);
    mmi->addInput(x_q);
    mmi->addInput(w_q_value);
    mmi->addInput(x_zp);
    mmi->insertBefore(n);
    mmi->output()->setElemType(TensorProto_DataType_INT32);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(mmi->output());
    cast->i_(kto, TensorProto_DataType_FLOAT);
    cast->insertBefore(n);
    cast->output()->setElemType(TensorProto_DataType_FLOAT);

    // combined_scale[j] = Xs * Ws[j], computed at runtime since Xs is not
    // known until the model runs.
    Node* scale_mul = graph.create(kMul, 1);
    scale_mul->addInput(x_scale);
    scale_mul->addInput(w_scale_value);
    scale_mul->insertBefore(n);
    scale_mul->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* dequant = graph.create(kMul, 1);
    dequant->addInput(cast->output());
    dequant->addInput(scale_mul->output());
    dequant->insertBefore(n);
    dequant->output()->setElemType(TensorProto_DataType_FLOAT);

    Value* result = dequant->output();
    if (info.bias != nullptr) {
      Node* add = graph.create(kAdd, 1);
      add->addInput(dequant->output());
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
