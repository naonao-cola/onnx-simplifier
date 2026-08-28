// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Block-wise INT8 weight-only quantization: like weight_only_quantize_matmul.h,
// only the constant weight is quantized and the activation is left exactly as
// it was -- no calibration data, no runtime quantize/dequantize cost on the
// activation path. The difference from that pass is granularity, not bit
// width: a separate scale per (block-of-K, output-channel) group instead of
// one scale per output channel, the same block-wise scheme
// weight_only_quantize_int4_matmul.h uses, just at INT8's wider code range.
//
// This sits between quantize_weight_only's INT8 (one scale per channel --
// no block overhead, but a channel with a wide value range under-resolves
// its smaller-magnitude weights) and quantize_weight_only_int4's INT4
// (finer blocks, but only 15 representable codes per block): the same
// storage as the plain INT8 pass (INT8 codes are still 1 byte each; only the
// scale tensor grows, from one float per channel to one float per
// (block, channel) pair), with resolution closer to a per-block scheme.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N]
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=<K's axis>, block_size=kBlockSize)
//   Y   = MatMul(X, Wdq)
//
// Wq (int8, values in [-127, 127]) and Ws (float32, one scale per (block,
// output channel) pair) are computed once, here, from W's static values --
// see TryQuantizeWeightBlockwiseInt8InPlace. This uses ONNX opset 21's
// DequantizeLinear `block_size` attribute (standard ONNX, not a contrib
// op) -- INT8 itself is much older, but the blocked form needs the same
// opset 21 floor as weight_only_quantize_int4_matmul.h's INT4 scheme.
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input 1)
// is a constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by kBlockSize, and whose activation (input 0) is float32.
// Everything else -- non-constant or non-2-D weights, a K not divisible by
// kBlockSize, non-default Gemm attributes, non-float32 operands, an opset
// older than 21 -- is left alone.

#pragma once

#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeInt8BlockMatMul final : public PredicateBasedPass {
  // Same default as weight_only_quantize_int4_matmul.h -- see that file's
  // doc comment for the rationale (32 favors accuracy over the scale-tensor
  // overhead a larger block would save).
  static constexpr int64_t kBlockSize = 32;

  explicit WeightOnlyQuantizeInt8BlockMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_int8_block_matmul";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 21) {
      // DequantizeLinear's `block_size` attribute needs opset >= 21.
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
    if (!TryQuantizeWeightBlockwiseInt8InPlace(*w_t, channel_axis, kBlockSize,
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
