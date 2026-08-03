// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Cancel the reshape scaffolding that
// ``fuse_matmul_add_bias_into_gemm_batched`` leaves around a chain of
// element-wise ops.
//
// That pass rewrites a rank-3 ``MatMul(X,W)+b`` into
//   X2 = Reshape(X, [-1, K])   ;   G = Gemm(X2, W, b)   ;   A = Reshape(G, [..,
//   N])
// so a transformer's linear layers become 2-D Gemms. Consecutive linears are
// separated only by element-wise ops (a GELU/act block, a bias add, ...), so
// the graph ends up as
//   Gemm -> Reshape(to N-D) -> <element-wise chain, all N-D> -> Reshape(to 2-D)
//   -> Gemm
// where the two reshapes are exact inverses (they only merge / split the
// leading dims and keep the last dim). Element-wise ops commute with a
// leading-dim flatten, so the chain can run on the already-2-D Gemm output and
// both reshapes disappear -- the Gemms (and their tuned-kernel dispatch) stay.
//
// Before (K = feature dims collapsed, last dim preserved):
//   g0 = Gemm(...)                      // [P, H]   (P = merged leading dims)
//   u  = Reshape(g0, [d0, .., H])       // back to N-D
//   e  = <element-wise chain on u>      // [d0, .., H]
//   f  = Reshape(e, [-1, H])            // flatten to 2-D
//   g1 = Gemm(f, W, b)
// After:
//   g0 = Gemm(...)                      // [P, H]
//   e' = <same chain, on g0 directly>   // [P, H]
//   g1 = Gemm(e', W, b)
//
// This matches on the flattening ``Reshape`` (``f`` above) and only fires when
// the whole element-wise region between it and the inverse reshapes is *closed*
// -- every value it would flatten is consumed only inside the region -- so no
// out-of-region consumer ever sees a reshaped tensor. Registered ``Nop`` so it
// is part of the default fuse/eliminate set and honours
// ``--skip-optimization``.

#include <functional>
#include <unordered_map>
#include <unordered_set>
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

struct EliminateReshapeAroundElementwise final : public PredicateBasedPass {
  explicit EliminateReshapeAroundElementwise()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "eliminate_reshape_around_elementwise";
  }

  // Ops whose output has exactly the same shape as their (broadcast) inputs and
  // that compute independently per element on the last axis, so merging the
  // leading dims of every operand leaves the result unchanged. Kept to plain
  // element-wise arithmetic/activations; normalizations (which reduce over the
  // last axis) are intentionally excluded.
  static bool IsElementwise(const Node* n) {
    static const std::vector<Symbol> kinds = {
        Symbol("Add"),        Symbol("Sub"),     Symbol("Mul"),
        Symbol("Div"),        Symbol("Pow"),     Symbol("Erf"),
        Symbol("Relu"),       Symbol("Sigmoid"), Symbol("Tanh"),
        Symbol("Sqrt"),       Symbol("Exp"),     Symbol("Log"),
        Symbol("Neg"),        Symbol("Abs"),     Symbol("Softplus"),
        Symbol("Reciprocal"), Symbol("Gelu"),    Symbol("Cast"),
    };
    for (const auto& k : kinds) {
      if (n->kind() == k) {
        return true;
      }
    }
    return false;
  }

  static bool SameDim(const Dimension& a, const Dimension& b) {
    if (a.is_int && b.is_int) {
      return a.dim == b.dim;
    }
    if (!a.is_int && !b.is_int) {
      return !a.is_unknown && !b.is_unknown && a.param == b.param;
    }
    return false;
  }

  static bool SameShape(const std::vector<Dimension>& a,
                        const std::vector<Dimension>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!SameDim(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  // A constant that broadcasts against both the N-D and the flattened 2-D form,
  // i.e. it only broadcasts on the preserved last axis. That is exactly the
  // scalars / length-N (or length-1) vectors linear-layer activations use.
  static bool IsTrailingBroadcastConstant(const Value* v) {
    if (!IsConstantTensor(v)) {
      return false;
    }
    const Tensor* t = FetchConstantTensor(v);
    return t != nullptr && t->sizes().size() <= 1;
  }

  // ``r`` is the inverse of the flatten ``[P, N]``: a Reshape whose data input
  // already has the flattened shape, so bypassing it feeds the 2-D tensor
  // straight into the chain.
  static bool IsInverseReshape(const Node* r,
                               const std::vector<Dimension>& flat_shape) {
    return r->kind() == kReshape && r->inputs().size() >= 1 &&
           r->input(0)->has_sizes() &&
           SameShape(r->input(0)->sizes(), flat_shape);
  }

  bool patternMatchPredicate(Node* node) override {
    // A Reshape that flattens the leading dims of a rank>=3 tensor into a 2-D
    // [P, N] with the last dim preserved.
    if (node->kind() != kReshape || !IsConstantTensor(node, 1)) {
      return false;
    }
    Value* in = node->input(0);
    Value* out = node->output();
    if (!in->has_sizes() || !out->has_sizes()) {
      return false;
    }
    const auto& in_shape = in->sizes();
    const auto& out_shape = out->sizes();
    if (in_shape.size() < 3 || out_shape.size() != 2) {
      return false;
    }
    // Last dim static and preserved.
    return in_shape.back().is_int && out_shape.back().is_int &&
           in_shape.back().dim == out_shape.back().dim;
  }

  bool runTransform(Node* flat, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const std::vector<Dimension> flat_shape = flat->output()->sizes();

    Value* top = flat->input(0);
    // The flattened value must feed only this reshape; otherwise a non-region
    // consumer would still expect the N-D shape.
    if (top->uses().size() != 1) {
      return false;
    }
    Node* root = top->node();
    if (!IsElementwise(root)) {
      return false;
    }

    // Walk the element-wise region backwards from ``root``. Every internal
    // value must be consumed only inside the region; every leaf must be a
    // trailing-broadcast constant (left as-is) or an inverse reshape
    // (bypassed).
    std::unordered_set<Node*> region;
    std::vector<Node*> worklist;
    // Rewires to apply on success: (node, input index) -> the 2-D value.
    std::vector<std::tuple<Node*, size_t, Value*>> rewires;

    region.insert(root);
    worklist.push_back(root);

    while (!worklist.empty()) {
      Node* n = worklist.back();
      worklist.pop_back();

      // Closure: an internal node's output may only be used inside the region.
      // ``root`` is the sole exception (its one use is the flatten reshape).
      if (n != root) {
        for (const auto& u : n->output()->uses()) {
          if (region.count(u.user) == 0) {
            return false;
          }
        }
      }

      for (size_t i = 0; i < n->inputs().size(); ++i) {
        Value* v = n->input(i);
        Node* pv = v->node();
        if (IsInverseReshape(pv, flat_shape)) {
          rewires.emplace_back(n, i, pv->input(0));
        } else if (IsTrailingBroadcastConstant(v)) {
          // leave the operand untouched; it broadcasts in both shapes
        } else if (IsElementwise(pv) && v->has_sizes() &&
                   SameShape(v->sizes(), top->sizes())) {
          // A full-shape element-wise producer: pull it into the region.
          if (region.insert(pv).second) {
            worklist.push_back(pv);
          }
        } else {
          // Anything else (a non-element-wise producer, a large constant, a
          // partially-broadcast operand) cannot be safely flattened.
          return false;
        }
      }
    }

    // Need at least one reshape to actually cancel, else this is a no-op that
    // would loop against the fixed point.
    if (rewires.empty()) {
      return false;
    }

    // Apply: feed each bypassed operand its 2-D source, then drop the flatten
    // (its consumers now read the -- now 2-D -- region output directly). The
    // bypassed inverse reshapes are left for dead-node elimination.
    for (const auto& [n, i, src] : rewires) {
      n->replaceInput(i, src);
    }
    // The region now computes on the flattened (2-D) tensors, so the cached N-D
    // shapes on its outputs are stale. Wipe them so the next shape-inference
    // pass recomputes the 2-D shapes; a leftover 3-D value_info would otherwise
    // trip strict consumers (e.g. onnxruntime's Gemm rank check) even though
    // the graph is numerically correct.
    for (Node* n : region) {
      for (Value* out : n->outputs()) {
        out->wipeSizes();
      }
    }
    flat->output()->replaceAllUsesWith(top);
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
