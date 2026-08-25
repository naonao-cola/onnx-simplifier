// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Dynamically quantizes an existing "com.microsoft" `Attention` node (see
// fuse_attention.h -- this pass does not fuse attention itself, it expects
// one to already be present, the same "single, self-contained rewrite"
// division of labor dynamic_quantize_matmul.h and the other dynamic_quantize_*
// passes use) into ONNX Runtime's "com.microsoft" `QAttention` contrib op --
// its quantized counterpart.
//
// Before:
//   Y = Attention(X, Wqkv, Bqkv, num_heads=H, scale=s,
//                 qkv_hidden_sizes=[N,N,N])   -- Wqkv, Bqkv constant float32
// After:
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Y = QAttention(Xq, Wqkv_q, Bqkv, Xs, Wqkv_s, <mask_index skipped>,
//                  Xzp, Wqkv_zp, num_heads=H, scale=s)
//
// Wqkv_q (int8) and Wqkv_s (float32, one scale per output column of Wqkv,
// i.e. per Q/K/V output channel) are computed once, here, from Wqkv's static
// values -- ordinary per-output-channel symmetric INT8 quantization, exactly
// QuantizeWeightPerChannelKN's own scheme (already used by
// dynamic_quantize_matmul.h); Wqkv_zp is an explicit all-zero tensor of the
// same shape (see below for why explicit rather than omitted). X, by
// contrast, is not known until the model runs, so it is quantized to uint8
// *in the graph* by `DynamicQuantizeLinear`, which computes its own
// scale/zero-point from each run's actual input range -- mirroring exactly
// how dynamic_quantize_matmul.h quantizes MatMul's activation. Bqkv is left
// untouched: QAttention's own `bias` input stays float (its schema never
// quantizes it), so it is simply reused as-is.
//
// `mask_index` (QAttention's optional attention-mask input, sitting between
// `weight_scale` and `input_zero_point` in the input list) is skipped via a
// dedicated `kUndefined`-kind node -- the standard ONNX IR representation for
// "this optional input, positioned before others that ARE provided, is
// omitted" (see fuse_gqa.h's own use of the same mechanism for
// `past_key`/`past_value`) -- since fuse_attention.h's own `Attention`
// output never carries a mask either.
//
// `weight_zero_point` is provided *explicitly* as an all-zero tensor rather
// than omitted (even though it is schema-optional and
// QuantizeWeightPerChannelKN's scheme is already zero-point-symmetric):
// fuse_attention.h's own top comment documents a real ONNX Runtime CPU build
// (1.29.0) segfaulting on `Attention` with an omitted optional input it
// otherwise expects, so this mirrors that same "always synthesize rather than
// trust omission of an optional kernel input" caution for its quantized sibling
// op.
//
// QAttention's schema has no `qkv_hidden_sizes`-equivalent attribute -- its
// `weight` shape doc (`[input_hidden_size, 3 * hidden_size]`) assumes Q, K,
// and V all have exactly the *same* hidden size -- unlike `Attention`, which
// fuse_attention.h can (and does, see
// test_fuse_attention_different_v_hidden_size) fire on with a differing V
// hidden size via its own `qkv_hidden_sizes` attribute. This pass declines
// whenever that attribute (fuse_attention.h always emits it explicitly) is not
// exactly `[N, N, N]` for some N, rather than guess how -- or whether --
// QAttention's kernel would even handle an uneven split.
//
// Only an `Attention` node whose `weight` (input 1) is a constant 2-D
// float32 tensor and whose activation (input 0) is float32 is handled.
// Everything else -- a non-constant or non-2-D weight, a non-float32
// operand, an opset older than 11 (`DynamicQuantizeLinear` requires opset
// 11), a `weight` reduction depth (`input_hidden_size`) large enough that
// `QAttention`'s int32 accumulation could overflow in the worst case (see
// quantize_matmul_common.h's `IsSafeInt32ReductionDepth`), or an uneven
// `qkv_hidden_sizes` split -- is left in float rather than silently
// producing a quantization that can misbehave.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

inline bool IsAttentionOp(Node* n) {
  return CheckKind(n, "Attention") && n->has_domain() &&
         n->domain() == "com.microsoft";
}

// Verifies `attn`'s own `qkv_hidden_sizes` attribute (fuse_attention.h
// always emits it) splits evenly -- `[N, N, N]` -- which QAttention's
// schema implicitly requires (see this file's top comment).
inline bool HasEvenQKVSplit(Node* attn) {
  std::vector<int64_t> qkv_hidden_sizes;
  if (!GetValueFromAttr(attn, "qkv_hidden_sizes", qkv_hidden_sizes) ||
      qkv_hidden_sizes.size() != 3) {
    return false;
  }
  return qkv_hidden_sizes[0] == qkv_hidden_sizes[1] &&
         qkv_hidden_sizes[1] == qkv_hidden_sizes[2];
}

struct DynamicQuantizeAttention final : public PredicateBasedPass {
  explicit DynamicQuantizeAttention()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dynamic_quantize_attention";
  }

  bool patternMatchPredicate(Node* n) override {
    if (!IsAttentionOp(n) || n->inputs().size() < 3) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 11) {
      return false;  // DynamicQuantizeLinear needs opset >= 11.
    }
    if (n->input(0)->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(n->input(1));
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    if (!HasEvenQKVSplit(n)) {
      return false;
    }
    return IsSafeInt32ReductionDepth(w_t->sizes()[0]);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (!IsAttentionOp(n) || n->inputs().size() < 3) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 11) {
      return false;
    }
    Value* x = n->input(0);
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    Value* weight = n->input(1);
    Value* bias = n->input(2);
    const Tensor* w_t = FetchConstantTensor(weight);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    if (!HasEvenQKVSplit(n)) {
      return false;
    }
    if (!IsSafeInt32ReductionDepth(w_t->sizes()[0])) {
      return false;
    }

    int64_t num_heads = 0;
    if (!GetValueFromAttr(n, "num_heads", num_heads)) {
      return false;
    }
    double scale = 0.0;
    const bool has_scale = GetValueFromAttr(n, "scale", scale);

    // Wqkv is already stored [input_hidden_size, 3*hidden_size] (K, N) --
    // exactly fuse_attention.h's own ReadAttentionWeightAsKN convention, so
    // no transpose is needed (unlike a generic MatMul weight, which may be
    // stored transposed).
    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelKN(*w_t, /*transposed=*/false, w_q, w_scale);
    const int64_t n_cols = w_scale.sizes()[0];

    Tensor w_zp;
    w_zp.elem_type() = TensorProto_DataType_INT8;
    w_zp.sizes() = {n_cols};
    w_zp.int32s().assign(static_cast<size_t>(n_cols), 0);

    // Xq, Xs, Xzp = DynamicQuantizeLinear(X)
    Node* dql = graph.create(Symbol("DynamicQuantizeLinear"), 3);
    dql->addInput(x);
    dql->insertBefore(n);
    Value* x_q = dql->outputs()[0];
    Value* x_scale = dql->outputs()[1];
    Value* x_zp = dql->outputs()[2];
    x_q->setElemType(TensorProto_DataType_UINT8);
    x_scale->setElemType(TensorProto_DataType_FLOAT);
    x_zp->setElemType(TensorProto_DataType_UINT8);

    Value* w_q_value = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_value = graph.addInitializerAndCreateValue(w_scale);
    Value* w_zp_value = graph.addInitializerAndCreateValue(w_zp);

    // mask_index is skipped (see this file's top comment): a dedicated
    // kUndefined-kind node's output stands in for the omitted middle
    // optional input, which the ONNX encoder serializes back to an empty
    // input name regardless of where in the node list it sits.
    Node* undef = graph.create(kUndefined, 1);
    undef->insertBefore(n);
    undef->output()->setUniqueName("");

    Node* qattn = graph.create(Symbol("QAttention"), 1);
    qattn->addInput(x_q);
    qattn->addInput(w_q_value);
    qattn->addInput(bias);
    qattn->addInput(x_scale);
    qattn->addInput(w_scale_value);
    qattn->addInput(undef->output());  // mask_index (unused)
    qattn->addInput(x_zp);
    qattn->addInput(w_zp_value);
    qattn->insertBefore(n);
    qattn->i_(Symbol("num_heads"), num_heads);
    if (has_scale) {
      qattn->f_(Symbol("scale"), static_cast<float>(scale));
    }
    qattn->setDomain("com.microsoft");
    qattn->output()->copyMetadata(n->output());

    if (!tryReplacingAllUsesWith(n, qattn)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
