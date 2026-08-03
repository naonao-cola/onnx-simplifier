// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = MatMul(X, W)          // X rank >= 3, W a 2-D constant [K, N]
//   A = Z + b                 // b broadcastable over the last (N) axis
// After:
//   X2 = Reshape(X, [-1, K])
//   G  = Gemm(X2, W, b)       // alpha=beta=1, transA=transB=0  -> [-1, N]
//   A  = Reshape(G, [d0, ..., d_{r-2}, N])
//
// `fuse_matmul_add_bias_into_gemm` only fuses the 2-D case. Transformer models
// apply linear layers to rank-3 activations `[B, S, K] . [K, N]`, so the MatMul
// is batched and that pass bails. This pass collapses the leading dims to a
// single 2-D Gemm and reshapes back.
//
// NOTE: This is a node-count / graph-shape rewrite (it trades a batched MatMul
// for Reshape + Gemm + Reshape). Batched MatMul and 2-D Gemm are equivalent
// work and modern runtimes execute batched MatMul natively, so this is not
// guaranteed to be faster. It is registered as PassType::Other so it is NOT in
// the default fuse set; invoke it explicitly by name when a Gemm-centric graph
// is wanted.

#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseMatMulAddBiasIntoGemmBatched final : public PredicateBasedPass {
  explicit FuseMatMulAddBiasIntoGemmBatched()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_matmul_add_bias_into_gemm_batched";
  }

  static Value* MakeInt64Constant(Graph& graph, std::vector<int64_t> data) {
    Tensor t;
    t.sizes().push_back(static_cast<int64_t>(data.size()));
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s() = std::move(data);
    return graph.addInitializerAndCreateValue(t);
  }

  // Add is commutative, so the MatMul may be either operand. Exporters differ:
  // HuggingFace linear layers emit ``Add(bias, MatMul(x, W))`` (MatMul second).
  // Return the operand index feeding the fusible MatMul, or -1.
  static int MatMulOperandIndex(Node* node) {
    if (node->kind() != kAdd || node->inputs().size() != 2) {
      return -1;
    }
    for (int i = 0; i < 2; ++i) {
      if (CheckKind(node->input(i), kMatMul) &&
          node->input(i)->uses().size() == 1) {
        return i;
      }
    }
    return -1;
  }

  bool patternMatchPredicate(Node* node) override {
    const int mm_idx = MatMulOperandIndex(node);
    if (mm_idx < 0) {
      return false;
    }
    Value* matmul_out = node->input(mm_idx);
    Node* matmul = matmul_out->node();
    Value* x = matmul->input(0);
    Value* w = matmul->input(1);

    // X: rank >= 3 with a static trailing dim K.
    if (!x->has_sizes()) {
      return false;
    }
    const auto& x_shape = x->sizes();
    if (x_shape.size() < 3 || !x_shape.back().is_int) {
      return false;
    }
    const int64_t k = x_shape.back().dim;

    // W: a 2-D constant [K, N] with static dims, matching K.
    if (!IsConstantTensor(w) || !w->has_sizes()) {
      return false;
    }
    const auto& w_shape = w->sizes();
    if (w_shape.size() != 2 || !w_shape[0].is_int || !w_shape[1].is_int) {
      return false;
    }
    if (w_shape[0].dim != k) {
      return false;
    }
    const int64_t n = w_shape[1].dim;

    // bias: 1-D, broadcastable over the N axis only ([N] or [1]).
    Value* bias = node->input(1 - mm_idx);
    if (!bias->has_sizes()) {
      return false;
    }
    const auto& bias_shape = bias->sizes();
    if (bias_shape.size() != 1 || !bias_shape[0].is_int) {
      return false;
    }
    if (bias_shape[0].dim != n && bias_shape[0].dim != 1) {
      return false;
    }

    // Dynamic leading dims need Shape/Slice, i.e. opset >= 10.
    bool leading_static = true;
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) {
      leading_static &= x_shape[i].is_int;
    }
    if (!leading_static) {
      const int opset = getOpsetVersion(*node->owningGraph());
      if (opset != 0 && opset < 10) {
        return false;
      }
    }
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const int mm_idx = MatMulOperandIndex(n);
    Node* matmul = n->input(mm_idx)->node();
    Value* x = matmul->input(0);
    Value* w = matmul->input(1);
    Value* bias = n->input(1 - mm_idx);

    const auto& x_shape = x->sizes();
    const int64_t rank = static_cast<int64_t>(x_shape.size());
    const int64_t k = x_shape.back().dim;
    const int64_t out_n = w->sizes()[1].dim;

    // X2 = Reshape(X, [-1, K])
    Node* pre = graph.create(kReshape, 1);
    pre->addInput(x);
    pre->addInput(MakeInt64Constant(graph, {-1, k}));

    // G = Gemm(X2, W, bias)
    Node* gemm = graph.create(kGemm, 1);
    gemm->addInput(pre->output());
    gemm->addInput(w);
    gemm->addInput(bias);
    gemm->f_(kalpha, 1.0);
    gemm->f_(kbeta, 1.0);
    gemm->i_(ktransA, 0);
    gemm->i_(ktransB, 0);

    // Reconstruct the output shape [d0, ..., d_{r-2}, N].
    bool leading_static = true;
    std::vector<int64_t> leading_dims;
    for (int64_t i = 0; i + 1 < rank; ++i) {
      leading_static &= x_shape[i].is_int;
      if (x_shape[i].is_int) {
        leading_dims.push_back(x_shape[i].dim);
      }
    }

    // A = Reshape(G, out_shape)
    Node* post = graph.create(kReshape, n->outputs().size());
    post->addInput(gemm->output());

    // Order nodes: pre -> gemm -> [shape ops] -> post -> n.
    pre->insertBefore(n);
    gemm->insertBefore(n);

    if (leading_static) {
      std::vector<int64_t> out_shape = leading_dims;
      out_shape.push_back(out_n);
      post->addInput(MakeInt64Constant(graph, std::move(out_shape)));
    } else {
      // shape(X) -> slice off the last dim -> concat with [N]
      Node* shape = graph.create(Symbol("Shape"), 1);
      shape->addInput(x);
      shape->insertBefore(n);

      Node* slice = graph.create(kSlice, 1);
      slice->addInput(shape->output());
      slice->addInput(MakeInt64Constant(graph, {0}));         // starts
      slice->addInput(MakeInt64Constant(graph, {rank - 1}));  // ends
      slice->addInput(MakeInt64Constant(graph, {0}));         // axes
      slice->insertBefore(n);

      Node* concat = graph.create(kConcat, 1);
      concat->addInput(slice->output());
      concat->addInput(MakeInt64Constant(graph, {out_n}));
      concat->i_(kaxis, 0);
      concat->insertBefore(n);

      post->addInput(concat->output());
    }

    for (int i = 0; i < static_cast<int>(n->outputs().size()); ++i) {
      post->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    post->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, post)) {
      return false;
    }
    // Destroy the Add; the now-dead MatMul is cleaned up by DCE.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
