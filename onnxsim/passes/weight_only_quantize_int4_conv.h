// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Block-wise INT4 weight-only quantization for Conv, mirroring
// weight_only_quantize_int4_matmul.h's rationale (no activation
// quantization, no calibration data, ONNX opset 21's native INT4 type and
// DequantizeLinear's block_size attribute -- not a contrib op). The only
// real difference is Conv's weight shape: unlike MatMul/Gemm, whose weight
// has one clean reduction axis (K) to block along, Conv's weight
// ([Cout, Cin/groups, k...]) has no single such axis -- every one of
// Cin/groups and the kernel spatial dims contributes to one output pixel's
// sum. So instead of blocking one of Conv's own axes directly (which would
// put an independent scale on every kernel spatial position -- little
// compression benefit for, say, a 3x3 kernel), this flattens everything but
// the output channel into one sequence and blocks *that*
// (TryQuantizeConvWeightBlockwiseInt4Flat), then reshapes the dequantized
// result back to Conv's expected weight shape:
//
// Before:
//   Y = Conv(X, W)             W constant, float32, [Cout, Cin/groups, k...]
// After:
//   Wq_flat  = <int4, [Cout, inner]>              inner = Cin/groups * prod(k...)
//   Ws_flat  = <float32, [Cout, inner / kBlockSize]>
//   Wdq_flat = DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=kBlockSize)
//   Wdq      = Reshape(Wdq_flat, [Cout, Cin/groups, k...])
//   Y        = Conv(X, Wdq)
//
// Only Conv's `W` input is quantized; its activation, optional bias (a
// third input, left untouched), and kernel/stride/pad/dilation/group/
// auto_pad attributes are unchanged.
//
// Only the common, unambiguous shape is handled: a Conv whose weight (input
// 1) is a constant float32 tensor, rank >= 3, whose flattened
// Cin/groups * prod(k...) is evenly divisible by kBlockSize, and whose
// activation (input 0) is float32. Everything else -- non-constant weights,
// an `inner` not divisible by kBlockSize, non-float32 operands, an opset
// older than 21 (INT4 tensors and DequantizeLinear's `block_size` both
// require it) -- is left alone.

#pragma once

#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeInt4Conv final : public PredicateBasedPass {
  // Same default as weight_only_quantize_int4_matmul.h -- see that file's
  // doc comment for the rationale.
  static constexpr int64_t kBlockSize = 32;

  explicit WeightOnlyQuantizeInt4Conv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_int4_conv";
  }

  static int64_t InnerSize(const std::vector<int64_t>& sizes) {
    int64_t inner = 1;
    for (size_t i = 1; i < sizes.size(); ++i) {
      inner *= sizes[i];
    }
    return inner;
  }

  bool patternMatchPredicate(Node* n) override {
    ConvInfo info;
    if (!MatchConv(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 21) {
      // INT4 tensors and DequantizeLinear's `block_size` attribute both need
      // opset >= 21.
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
    return InnerSize(w_t->sizes()) % kBlockSize == 0;
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
    if (!TryQuantizeConvWeightBlockwiseInt4Flat(*w_t, kBlockSize, w_q,
                                                w_scale)) {
      return false;
    }

    Value* w_q_v = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_v = graph.addInitializerAndCreateValue(w_scale);

    // Wdq_flat = DequantizeLinear(Wq_flat, Ws_flat, axis=1,
    // block_size=kBlockSize) -- zero_point omitted (symmetric, i.e. always
    // 0).
    Node* wdq = graph.create(Symbol("DequantizeLinear"), 1);
    wdq->addInput(w_q_v);
    wdq->addInput(w_scale_v);
    wdq->i_(kaxis, 1);
    wdq->i_(Symbol("block_size"), kBlockSize);
    wdq->insertBefore(n);
    wdq->output()->setElemType(TensorProto_DataType_FLOAT);
    // Value::setSizes takes vector<Dimension>, not Tensor::sizes()'s
    // vector<int64_t> -- Dimension(int64_t) is an explicit converting
    // constructor, so a range-construction from the int64 shape works.
    wdq->output()->setSizes(
        std::vector<Dimension>(w_q.sizes().begin(), w_q.sizes().end()));

    // Wdq = Reshape(Wdq_flat, W's original [Cout, Cin/groups, k...] shape).
    Tensor shape_t;
    shape_t.elem_type() = TensorProto_DataType_INT64;
    shape_t.sizes().push_back(static_cast<int64_t>(w_t->sizes().size()));
    shape_t.int64s() = w_t->sizes();
    Value* shape_v = graph.addInitializerAndCreateValue(shape_t);

    Node* reshape = graph.create(kReshape, 1);
    reshape->addInput(wdq->output());
    reshape->addInput(shape_v);
    reshape->insertBefore(n);
    reshape->output()->setElemType(TensorProto_DataType_FLOAT);
    reshape->output()->setSizes(
        std::vector<Dimension>(w_t->sizes().begin(), w_t->sizes().end()));

    // Conv and its activation input are left untouched; only the weight
    // input changes, to its dequantized (and reshaped-back) counterpart.
    // Its optional bias (input 2) is untouched.
    n->replaceInput(1, reshape->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
