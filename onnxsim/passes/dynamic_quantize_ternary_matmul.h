// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Dynamically quantizes MatMul/Gemm nodes whose constant weight is
// *structurally ternary* -- every element of every output column is one of
// {-s, 0, +s} for that column's own scale `s` -- the weight representation
// BitNet b1.58 (https://github.com/microsoft/BitNet) and similar
// ternary-weight models use internally. A generic ONNX export still stores
// such a weight as a dense float32 initializer, so the graph runs on the
// generic float MatMul kernel at 16x the size the weight actually needs.
// This pass detects that structure (see
// ``quantize_matmul_common.h``'s ``TryQuantizeWeightTernaryKN``) and rewrites
// it into exactly the same shape ``dynamic_quantize_matmul.h`` produces for
// an ordinary float weight:
//
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Acc         = MatMulInteger(Xq, Wq, Xzp)          // int32
//   Y           = Cast<float>(Acc) * (Xs * Ws)        // dequantize
//
// The only difference from ``dynamic_quantize_matmul.h`` is *how* `Wq` is
// derived: instead of quantizing the weight's full dynamic range to INT8
// ([-127, 127], lossy), this pass only fires when the weight's values are
// already exactly {-1, 0, 1} once divided by each column's scale, so `Wq` is
// a lossless int8 encoding of the original ternary weight, not a rounded
// approximation. Everything downstream of `Wq` -- activation quantization,
// the integer matmul, dequantization, opset/dtype requirements -- is
// identical to ``dynamic_quantize_matmul.h``; see that file's doc comment for
// the shared parts of the rewrite.
//
// This intentionally targets only standard ONNX operators, the same as every
// other onnxsim quantization pass, rather than a contrib op like
// ``com.microsoft::MatMulNBits`` (which packs ternary codes down to 2 bits
// instead of 8, for a further ~4x weight-storage saving on top of what this
// pass gets from `Wq` alone, but only runs on onnxruntime builds that ship
// it) -- see the roadmap in docs/ternary-quantization.md for that as
// vendor-specific follow-up work, not something this portable pass does.

#pragma once

#include <string>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct DynamicQuantizeTernaryMatMul final : public PredicateBasedPass {
  explicit DynamicQuantizeTernaryMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dynamic_quantize_ternary_matmul";
  }

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
    Tensor q_probe;
    Tensor scale_probe;
    return TryQuantizeWeightTernaryKN(*w_t, info.weight_transposed, q_probe,
                                      scale_probe);
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
    Tensor w_q;
    Tensor w_scale;
    if (!TryQuantizeWeightTernaryKN(*w_t, info.weight_transposed, w_q,
                                    w_scale)) {
      return false;
    }

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
