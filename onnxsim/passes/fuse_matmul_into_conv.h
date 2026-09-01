// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Before (one of):
//   Y = MatMul(X, W)                        // W: constant 2-D [K, N]
//   Z = MatMul(X, W) + B                    // B: constant, broadcastable
//                                            //    over the N axis
//   Z = Gemm(X, W[, B], transA=0, alpha=1[, beta=1])
//                                            // W: constant 2-D, [K, N]
//                                            //    (transB=0) or [N, K]
//                                            //    (transB=1)
// After:
//   X2 = Reshape(X, [-1, K, 1])
//   W2 = Transpose/Unsqueeze(W) -> [N, K, 1]
//   C  = Conv(X2, W2[, B'])                 // a 1-D, 1x1-kernel convolution
//   Z  = Reshape(C, [d0, ..., d_{r-2}, N])
//
// X may have any rank >= 2; only its last (contracted) axis needs a static
// size K. Every leading dim collapses into Conv's batch axis and is restored
// by the trailing Reshape -- the same leading-dims-to-a-single-axis
// scaffolding ``fuse_matmul_add_bias_into_gemm_batched`` uses to turn a
// batched MatMul into a 2-D Gemm, except the target here is a 1x1 Conv, not
// a Gemm.
//
// Rationale: some accelerators (embedded/mobile NPUs and DSPs among them)
// ship a heavily tuned, general-purpose Conv datapath but a much weaker (or
// altogether unaccelerated) generic MatMul/Gemm one. Every "Linear" layer
// (nn.Linear, a Transformer's QKV/output projections and FFN, a CNN's final
// classifier head after global pooling, ...) is mathematically a 1x1
// convolution once its activation is reshaped to put the contracted axis on
// the channel dimension, so rewriting it that way lets such backends keep
// the whole network on their fast Conv engine instead of falling back to a
// slow (or CPU) path for every Linear layer.
//
// This is a graph-shape rewrite, not a node-count reduction, and is a
// *regression* on a MatMul/Gemm-first backend -- so, like
// ``fuse_matmul_add_bias_into_gemm_batched``, it is registered as
// PassType::Other (opt-in only, via ``extra_optimizers``), not part of the
// default fuse set.
//
// Only ``transA=0`` and ``alpha=1``/``beta=1`` Gemm is handled -- the shape a
// plain ``nn.Linear`` export (or onnxsim's own
// ``fuse_matmul_add_bias_into_gemm``/``_batched``) always produces. Any
// other Gemm is left untouched rather than taught to rescale the folded
// weight/bias for an alpha/beta that essentially never occurs in practice.
// A MatMul whose sole use is an Add this same pass could otherwise fuse the
// bias from defers to that Add match instead of racing it, so the bias ends
// up folded into the Conv directly rather than left behind as a trailing
// Add the rewritten (rank >= 3) Conv output can no longer broadcast against
// in the usual per-channel way.

#include <cstdint>
#include <string>
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

struct FuseMatMulIntoConv final : public PredicateBasedPass {
  explicit FuseMatMulIntoConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_matmul_into_conv"; }

  // This pass instance is reused across every round of onnxsim's
  // simplification fixed point, but a batch of reserved names is only valid
  // for the graph state it was reserved against -- other passes/rounds can
  // introduce new names in between. Drop any leftover reservation from a
  // prior runPass() call so nextReservedName() always reserves fresh against
  // the current graph (mirrors fuse_matmul_add_bias_into_gemm_batched's own
  // initializePass, same rationale).
  bool initializePass(Graph&) override {
    reserved_names_.clear();
    reserved_used_ = 0;
    return false;
  }

  static Value* MakeInt64Constant(Graph& graph, std::vector<int64_t> data,
                                  std::string name) {
    Tensor t;
    t.setName(std::move(name));
    t.sizes().push_back(static_cast<int64_t>(data.size()));
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s() = std::move(data);
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  // The pieces of a MatMul/Gemm-shaped Linear layer this pass rewrites.
  struct Match {
    bool ok = false;
    Value* x = nullptr;     // activation, rank >= 2, static last dim K
    Value* w = nullptr;     // constant weight, 2-D
    bool w_is_n_k = false;  // true: w is laid out [N, K] (Gemm transB=1);
                            // false: w is laid out [K, N] (MatMul, or Gemm
                            // transB=0)
    Value* bias = nullptr;  // constant, rank <= 1, may be null
    int64_t k = 0;
    int64_t n = 0;
  };

  static Match MatchNode(Node* node) {
    Match m;
    Node* mm = nullptr;
    Value* bias = nullptr;

    if (node->kind() == kAdd && node->inputs().size() == 2) {
      // Add is commutative, so the MatMul may be either operand. Exporters
      // differ: HuggingFace linear layers emit ``Add(bias, MatMul(x, W))``
      // (MatMul second).
      int mm_idx = -1;
      for (int i = 0; i < 2; ++i) {
        if (CheckKind(node->input(i), kMatMul) &&
            node->input(i)->uses().size() == 1) {
          mm_idx = i;
          break;
        }
      }
      if (mm_idx < 0) {
        return m;
      }
      mm = node->input(mm_idx)->node();
      bias = node->input(1 - mm_idx);
    } else if (node->kind() == kMatMul) {
      // Defer to the Add case above when this MatMul's only use is an Add
      // this pass would rather fuse the bias from directly -- converting the
      // bare MatMul first would leave that Add stranded after a Conv whose
      // rank (>= 3, batch/channel/spatial) it can no longer broadcast a
      // plain [N] bias against the way it could against a 2-D Gemm.
      Value* out = node->output();
      if (out->uses().size() == 1) {
        Node* use = out->uses()[0].user;
        if (use->kind() == kAdd && MatchNode(use).ok) {
          return m;
        }
      }
      mm = node;
    } else if (node->kind() == kGemm) {
      const int64_t trans_a =
          GetValueFromAttrWithDefault(node, ktransA, int64_t(0));
      const double alpha = GetValueFromAttrWithDefault(node, kalpha, 1.0);
      if (trans_a != 0 || alpha != 1.0) {
        return m;
      }
      mm = node;
    } else {
      return m;
    }

    Value* x = mm->input(0);
    Value* w = mm->input(1);
    bool w_is_n_k = false;
    if (mm->kind() == kGemm) {
      const int64_t trans_b =
          GetValueFromAttrWithDefault(mm, ktransB, int64_t(0));
      w_is_n_k = trans_b != 0;
      if (mm->inputs().size() == 3) {
        const double beta = GetValueFromAttrWithDefault(mm, kbeta, 1.0);
        if (beta != 1.0) {
          return m;
        }
        bias = mm->input(2);
      }
    }

    if (!x->has_sizes()) {
      return m;
    }
    const auto& x_shape = x->sizes();
    if (x_shape.size() < 2 || !x_shape.back().is_int) {
      return m;
    }
    const int64_t k = x_shape.back().dim;

    if (!IsConstantTensor(w) || !w->has_sizes()) {
      return m;
    }
    const auto& w_shape = w->sizes();
    if (w_shape.size() != 2 || !w_shape[0].is_int || !w_shape[1].is_int) {
      return m;
    }
    int64_t n_out;
    if (w_is_n_k) {
      if (w_shape[1].dim != k) {
        return m;
      }
      n_out = w_shape[0].dim;
    } else {
      if (w_shape[0].dim != k) {
        return m;
      }
      n_out = w_shape[1].dim;
    }

    if (bias != nullptr) {
      if (!IsConstantTensor(bias) || !bias->has_sizes()) {
        return m;
      }
      const auto& b_shape = bias->sizes();
      if (b_shape.size() > 1) {
        return m;
      }
      if (b_shape.size() == 1) {
        if (!b_shape[0].is_int) {
          return m;
        }
        if (b_shape[0].dim != n_out && b_shape[0].dim != 1) {
          return m;
        }
      }
    }

    // Dynamic leading dims need Shape/Slice, i.e. opset >= 10 (mirrors
    // fuse_matmul_add_bias_into_gemm_batched's own guard).
    bool leading_static = true;
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) {
      leading_static &= x_shape[i].is_int;
    }
    if (!leading_static) {
      const int opset = getOpsetVersion(*node->owningGraph());
      if (opset != 0 && opset < 10) {
        return m;
      }
    }

    m.ok = true;
    m.x = x;
    m.w = w;
    m.w_is_n_k = w_is_n_k;
    m.bias = bias;
    m.k = k;
    m.n = n_out;
    return m;
  }

  bool patternMatchPredicate(Node* n) override { return MatchNode(n).ok; }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const Match match = MatchNode(n);
    if (!match.ok) {
      return false;
    }

    const auto& x_shape = match.x->sizes();
    const int64_t rank = static_cast<int64_t>(x_shape.size());

    // X2 = Reshape(X, [-1, K, 1])
    Node* pre = graph.create(kReshape, 1);
    pre->addInput(match.x);
    pre->addInput(MakeInt64Constant(graph, {-1, match.k, 1},
                                    nextReservedName(graph)));
    pre->insertBefore(n);

    // W2 = [N, K, 1], built from W by Transpose (if laid out [K, N]) then
    // Unsqueeze. Both operate on a constant, so the constant folder
    // materializes W2 as a plain initializer (fuse_mul_into_conv's own
    // weight-folding convention).
    Value* w2 = match.w;
    if (!match.w_is_n_k) {
      Node* transpose = graph.create(kTranspose, 1);
      transpose->addInput(w2);
      transpose->is_(kperm, std::vector<int64_t>{1, 0});
      transpose->insertBefore(n);
      w2 = transpose->output();
    }
    {
      Node* unsqueeze = graph.create(kUnsqueeze, 1);
      unsqueeze->addInput(w2);
      const int opset = getOpsetVersion(graph);
      std::vector<int64_t> axes{2};
      if (opset < 13 && opset != 0) {
        unsqueeze->is_(kaxes, std::move(axes));
      } else {
        unsqueeze->addInput(
            MakeInt64Constant(graph, axes, nextReservedName(graph)));
      }
      unsqueeze->insertBefore(n);
      w2 = unsqueeze->output();
    }

    // Bias, if present, must end up as an exact 1-D [N] Conv bias (Conv's own
    // B input has no broadcasting rules of its own, unlike Add).
    Value* bias = nullptr;
    if (match.bias != nullptr) {
      bias = match.bias;
      const auto& b_shape = bias->sizes();
      const int64_t bias_numel = b_shape.size() == 0 ? 1 : b_shape[0].dim;
      if (b_shape.size() == 0) {
        Node* reshape = graph.create(kReshape, 1);
        reshape->addInput(bias);
        reshape->addInput(
            MakeInt64Constant(graph, {1}, nextReservedName(graph)));
        reshape->insertBefore(n);
        bias = reshape->output();
      }
      if (bias_numel != match.n) {
        Node* expand = graph.create(kExpand, 1);
        expand->addInput(bias);
        expand->addInput(
            MakeInt64Constant(graph, {match.n}, nextReservedName(graph)));
        expand->insertBefore(n);
        bias = expand->output();
      }
    }

    // C = Conv(X2, W2[, bias]) -- kernel_shape/strides/pads/dilations/group
    // are all left at their defaults ([1]/[1]/[0,0]/[1]/1), which Conv's own
    // shape inference derives from W2's shape when unset.
    Node* conv = graph.create(kConv, 1);
    conv->addInput(pre->output());
    conv->addInput(w2);
    if (bias != nullptr) {
      conv->addInput(bias);
    }
    conv->insertBefore(n);

    // Reconstruct the output shape [d0, ..., d_{r-2}, N].
    bool leading_static = true;
    std::vector<int64_t> leading_dims;
    for (int64_t i = 0; i + 1 < rank; ++i) {
      leading_static &= x_shape[i].is_int;
      if (x_shape[i].is_int) {
        leading_dims.push_back(x_shape[i].dim);
      }
    }

    Node* post = graph.create(kReshape, n->outputs().size());
    post->addInput(conv->output());

    if (leading_static) {
      std::vector<int64_t> out_shape = leading_dims;
      out_shape.push_back(match.n);
      post->addInput(MakeInt64Constant(graph, std::move(out_shape),
                                       nextReservedName(graph)));
    } else {
      // shape(X) -> slice off the last dim -> concat with [N]
      Node* shape = graph.create(Symbol("Shape"), 1);
      shape->addInput(match.x);
      shape->insertBefore(n);

      Node* slice = graph.create(kSlice, 1);
      slice->addInput(shape->output());
      slice->addInput(
          MakeInt64Constant(graph, {0}, nextReservedName(graph)));  // starts
      slice->addInput(MakeInt64Constant(graph, {rank - 1},
                                        nextReservedName(graph)));  // ends
      slice->addInput(
          MakeInt64Constant(graph, {0}, nextReservedName(graph)));  // axes
      slice->insertBefore(n);

      Node* concat = graph.create(kConcat, 1);
      concat->addInput(slice->output());
      concat->addInput(
          MakeInt64Constant(graph, {match.n}, nextReservedName(graph)));
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
    // Destroy n (the Add, the bare MatMul, or the Gemm); if n was an Add,
    // the now-dead MatMul underneath it is cleaned up by DCE.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }

 private:
  // Each match mints several fresh initializer names (the pre-reshape's
  // shape, the weight-transform's axes/unsqueeze constant, optionally the
  // bias-expand shape, plus either the post-reshape's shape or -- on the
  // dynamic-leading-dims path -- Slice's starts/ends/axes and Concat's [N]).
  // Same batching fix as fuse_matmul_add_bias_into_gemm_batched (see its own
  // comment on this exact mechanism, and onnxsim issue #651): draw from a
  // batch reserved via Graph::reserveUniqueNames() (one scan per batch)
  // instead of paying a full graph scan per name via
  // Graph::addInitializerAndCreateValue -> getNextUniqueName().
  static constexpr size_t kNameBatchSize = 256;
  std::vector<std::string> reserved_names_;
  size_t reserved_used_ = 0;

  std::string nextReservedName(Graph& graph) {
    if (reserved_used_ >= reserved_names_.size()) {
      reserved_names_ = graph.reserveUniqueNames(kNameBatchSize);
      reserved_used_ = 0;
    }
    return std::move(reserved_names_[reserved_used_++]);
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
