// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Block-wise INT4 weight-only quantization: like weight_only_quantize_matmul.h,
// only the constant weight is quantized and the activation is left exactly as
// it was -- no calibration data, no runtime quantize/dequantize cost on the
// activation path. The difference is bit width and granularity: INT4 instead
// of INT8, with a separate scale per (block-of-K, output-channel) group
// instead of one scale per output channel, matching the weight-only INT4
// quantization real LLM/ASR deployments increasingly ship (llm.int4/GPTQ/
// AWQ-style block quantization) for roughly half the storage of the INT8
// scheme at a comparable accuracy cost, since finer per-block scales absorb
// most of what a single wider INT8 range would otherwise lose.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N]
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=<K's axis>, block_size=kBlockSize)
//   Y   = MatMul(X, Wdq)
//
// Wq (int4, values in [-7, 7]) and Ws (float32, one scale per (block, output
// channel) pair) are computed once, here, from W's static values -- see
// TryQuantizeWeightBlockwiseInt4InPlace. This uses ONNX opset 21's INT4
// tensor type and DequantizeLinear's `block_size` attribute -- both standard
// ONNX, not a contrib op -- so, like every other onnxsim quantization pass,
// the result loads on any conformant opset-21+ runtime.
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input 1)
// is a constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by kBlockSize, and whose activation (input 0) is float32.
// Everything else -- non-constant or non-2-D weights, a K not divisible by
// kBlockSize, non-default Gemm attributes, non-float32 operands, an opset
// older than 21 (INT4 tensors and DequantizeLinear's `block_size` both
// require it) -- is left alone.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeInt4MatMul final : public PredicateBasedPass {
  // GPTQ/AWQ-style weight-only INT4 quantization commonly blocks in groups
  // of 32 or 128 along the reduction dimension; 32 favors accuracy (more
  // scales, smaller quantization groups) over the scale-tensor overhead a
  // larger block would save, matching this pass's "correctness over squeezed
  // storage" default -- same spirit as the INT8 pass's symmetric-only,
  // zero-point-free scheme.
  static constexpr int64_t kBlockSize = 32;

  explicit WeightOnlyQuantizeInt4MatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_int4_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
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
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    const int64_t K = w_t->sizes()[1 - channel_axis];
    return K % kBlockSize == 0;
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
    // The reduction axis (the one blocked) is the other one.
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    const int64_t reduction_axis = 1 - channel_axis;
    Tensor w_q;
    Tensor w_scale;
    if (!TryQuantizeWeightBlockwiseInt4InPlace(*w_t, channel_axis, kBlockSize,
                                               w_q, w_scale)) {
      return false;
    }

    Value* w_q_v = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_v = graph.addInitializerAndCreateValue(w_scale);

    // Wdq = DequantizeLinear(Wq, Ws, axis=reduction_axis,
    // block_size=kBlockSize) -- zero_point omitted (symmetric, i.e. always
    // 0).
    Node* wdq = graph.create(Symbol("DequantizeLinear"), 1);
    wdq->addInput(w_q_v);
    wdq->addInput(w_scale_v);
    wdq->i_(kaxis, reduction_axis);
    wdq->i_(Symbol("block_size"), kBlockSize);
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
