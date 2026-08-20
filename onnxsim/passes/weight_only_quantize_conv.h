// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Weight-only quantization for Conv, mirroring
// weight_only_quantize_matmul.h's MatMul/Gemm rewrite -- see that file's doc
// comment for the rationale (no activation quantization, no calibration
// data). The only real difference is the weight's per-output-channel axis:
// Conv's weight layout ([Cout, Cin/groups, k...]) always puts the output
// channel on axis 0, so unlike Gemm there is no transposed-layout case to
// pick between.
//
// Before:
//   Y = Conv(X, W)            W constant, float32, rank >= 3
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=0)
//   Y   = Conv(X, Wdq)
//
// Only Conv's ``W`` input is quantized; its activation, optional bias (a
// third input, left untouched), and kernel/stride/pad/dilation/group/
// auto_pad attributes are unchanged.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeConv final : public PredicateBasedPass {
  explicit WeightOnlyQuantizeConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_conv";
  }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
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
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      return false;
    }

    Tensor w_q;
    Tensor w_scale;
    QuantizeConvWeightPerOutputChannel(*w_t, w_q, w_scale);

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

    // Conv and its activation input are left untouched; only the weight
    // input changes, to its dequantized counterpart. Its optional bias
    // (input 2) is untouched.
    n->replaceInput(1, wdq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
