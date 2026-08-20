// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before (the textbook RMSNorm forward -- e.g. LLaMA/Mistral/Qwen-family
// exports; see test_mnn_llm_export.py's `_RMSNorm`):
//   variance = X.pow(2).mean(-1, keepdim=True)  # ReduceMean(Pow(X,2)|Mul(X,X))
//   Y = weight * (X * rsqrt(variance + eps))    # or weight * (X / sqrt(...))
// After:
//   Y = RMSNormalization<axis, epsilon>(X, weight)
//
// RMSNormalization is a standard ONNX op (opset 23, onnx/defs/nn/defs.cc)
// with a portable function-body fallback, so replacing the hand-unrolled
// decomposition with it is a pure node-count win everywhere, and it
// additionally lets any backend that *does* carry a fused/accelerated
// RMSNorm kernel dispatch straight to it -- the same role the
// Attention/RotaryEmbedding ops play for their own subgraphs. This collapses
// the 6-7 node chain (Pow-or-Mul, ReduceMean, Add, Sqrt, Reciprocal-or-Div,
// Mul, Mul) most exporters emit into a single node; a 32-layer LLaMA-style
// model has ~65 of these (2 per block plus the final norm).
//
// Handles the exporter spellings of `X * rsqrt(v)` seen in practice:
//   Div(X, Sqrt(v))                 -- matches RMSNormalization's own
//                                       reference decomposition exactly
//   Mul(X, Reciprocal(Sqrt(v)))
//   Mul(X, Div(one=1, Sqrt(v)))
// with the weight-Mul and the X-operand of each inner op allowed in either
// order. `v` must be `Add(ReduceMean(square, axes), eps)` with a single
// reduced axis (attribute, or opset >= 18 input) and keepdims=1, where
// `square` is `Mul(X, X)` or `Pow(X, 2)` over the *same* X used above, and
// every intermediate value along the chain is used exactly once.
//
// The whole matched chain (everything strictly between X/weight and the
// outer Mul) is torn down explicitly here rather than left for a later
// dead-code-elimination pass: onnx-optimizer's ``FixedPointPassManager``
// only re-sweeps its full pass list (giving ``eliminate_deadend`` another
// look) when some *other* pass in the list reports
// ``PassEfficiency::Partial`` fixed-point work during the same sweep. On a
// graph where nothing else in the default pass set happens to fire in that
// round (plausible on a small/isolated subgraph), a `Complete`-efficiency
// fuse pass -- this one included -- can otherwise leave its own orphaned
// nodes behind for a full extra `Simplify()` iteration, or indefinitely.
// Node::destroy() requires each destroyed node's outputs to already have
// zero uses, so the chain must come down innermost-consumer first: detach
// the outer Mul's edge into the chain, then destroy in the order
// MatchOuter/MatchDenominator recorded it (mirrors the explicit
// ``bn->removeInput`` + destroy teardown fuse_bn_into_conv.h uses for the
// same reason).
//
// Only fires at opset >= 23 (RMSNormalization's introducing version); pass
// `target_opset_version=23` (or higher) to onnxsim to upgrade first if the
// exported model predates it -- true of essentially all LLM exports today.
//
// Does not match the HuggingFace-style variant that explicitly upcasts to
// fp32 around the reduction (`Cast(X, FLOAT) -> ... -> Cast(_, orig_dtype)`):
// the inner X no longer points at the same Value as the outer multiply's X,
// so the pattern declines rather than misfiring. Left for follow-up.

#include <utility>
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

struct FuseRMSNorm final : public PredicateBasedPass {
  explicit FuseRMSNorm()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_rms_norm"; }

  struct Match {
    Value* x = nullptr;
    Value* scale = nullptr;
    int64_t axis = -1;
    float epsilon = 0.0f;
    // Every node strictly between (X, scale) and the outer Mul, ordered
    // innermost-consumer first (i.e. the node whose output the outer Mul
    // directly used comes first) so destroying them in this order always
    // has a zero-use output to destroy.
    std::vector<Node*> dead_chain;
  };

  static bool FetchScalarAsFloat(Value* v, float& out) {
    if (FetchSoleValueOfTensor(v, out)) {
      return true;
    }
    double d;
    if (FetchSoleValueOfTensor(v, d)) {
      out = static_cast<float>(d);
      return true;
    }
    return false;
  }

  static bool IsConstantTwo(Value* v) {
    float f;
    if (FetchScalarAsFloat(v, f)) {
      return f == 2.0f;
    }
    int64_t i;
    if (FetchSoleValueOfTensor(v, i)) {
      return i == 2;
    }
    return false;
  }

  // Matches `Add(ReduceMean(square, axes), eps)` feeding `sqrt_out`'s Sqrt,
  // where `square` is `Mul(X, X)` or `Pow(X, 2)`. Every intermediate along
  // the chain must be single-use. On success, appends [sqrt, add, mean,
  // square] to `chain` (innermost first).
  static bool MatchDenominator(Value* sqrt_out, Value*& x, int64_t& axis,
                               float& epsilon, std::vector<Node*>& chain) {
    if (!CheckKind(sqrt_out, kSqrt) || sqrt_out->uses().size() != 1) {
      return false;
    }
    Node* sqrt = sqrt_out->node();
    Value* add_out = sqrt->input(0);
    if (!CheckKind(add_out, kAdd) || add_out->uses().size() != 1) {
      return false;
    }
    Node* add = add_out->node();
    if (add->inputs().size() != 2) {
      return false;
    }
    Value* mean_out = nullptr;
    Value* eps_v = nullptr;
    for (int i = 0; i < 2; ++i) {
      if (CheckKind(add->input(i), kReduceMean)) {
        mean_out = add->input(i);
        eps_v = add->input(1 - i);
      }
    }
    if (mean_out == nullptr || mean_out->uses().size() != 1 ||
        !FetchScalarAsFloat(eps_v, epsilon)) {
      return false;
    }

    Node* mean = mean_out->node();
    if (GetValueFromAttrWithDefault(mean, kkeepdims, (int64_t)1) != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(mean, kaxes, (size_t)1, axes) ||
        axes.size() != 1) {
      return false;
    }
    axis = axes[0];

    Value* square_out = mean->input(0);
    if (square_out->uses().size() != 1) {
      return false;
    }
    Node* square = square_out->node();
    bool square_ok = false;
    if (CheckKind(square, kMul) && square->inputs().size() == 2 &&
        square->input(0) == square->input(1)) {
      x = square->input(0);
      square_ok = true;
    } else if (CheckKind(square, kPow) && square->inputs().size() == 2 &&
               IsConstantTwo(square->input(1))) {
      x = square->input(0);
      square_ok = true;
    }
    if (!square_ok) {
      return false;
    }

    chain.push_back(sqrt);
    chain.push_back(add);
    chain.push_back(mean);
    chain.push_back(square);
    return true;
  }

  // Matches the outer `weight * (X * rsqrt(v))` / `weight * (X / sqrt(v))`,
  // trying both operand orders of the scale-Mul and of the two inner
  // commutative ops.
  static bool MatchOuter(Node* n, Match& out) {
    if (!CheckKind(n, kMul) || n->inputs().size() != 2) {
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      Value* norm = n->input(i);
      if (norm->uses().size() != 1) {
        continue;
      }
      Value* scale = n->input(1 - i);
      Node* norm_node = norm->node();

      // Div(X, RMS): the exact spelling of RMSNormalization's own reference
      // decomposition (`Normalized = Div(X, RMS)`).
      if (CheckKind(norm_node, kDiv) && norm_node->inputs().size() == 2) {
        Value* x_direct = norm_node->input(0);
        Value* x_denom;
        int64_t axis;
        float eps;
        std::vector<Node*> chain;
        if (MatchDenominator(norm_node->input(1), x_denom, axis, eps, chain) &&
            x_denom == x_direct) {
          chain.insert(chain.begin(), norm_node);
          out = Match{x_direct, scale, axis, eps, std::move(chain)};
          return true;
        }
      }

      // Mul(X, 1/RMS), 1/RMS = Reciprocal(RMS) or Div(one, RMS).
      if (CheckKind(norm_node, kMul) && norm_node->inputs().size() == 2) {
        for (int j = 0; j < 2; ++j) {
          Value* x_direct = norm_node->input(j);
          Value* inv = norm_node->input(1 - j);
          if (inv->uses().size() != 1) {
            continue;
          }
          Node* inv_node = inv->node();
          Value* sqrt_out = nullptr;
          if (CheckKind(inv_node, "Reciprocal") &&
              inv_node->inputs().size() == 1) {
            sqrt_out = inv_node->input(0);
          } else if (CheckKind(inv_node, kDiv) &&
                     inv_node->inputs().size() == 2) {
            float one;
            if (!FetchScalarAsFloat(inv_node->input(0), one) || one != 1.0f) {
              continue;
            }
            sqrt_out = inv_node->input(1);
          } else {
            continue;
          }
          Value* x_denom;
          int64_t axis;
          float eps;
          std::vector<Node*> chain;
          if (MatchDenominator(sqrt_out, x_denom, axis, eps, chain) &&
              x_denom == x_direct) {
            chain.insert(chain.begin(), inv_node);
            chain.insert(chain.begin(), norm_node);
            out = Match{x_direct, scale, axis, eps, std::move(chain)};
            return true;
          }
        }
      }
    }
    return false;
  }

  bool patternMatchPredicate(Node* n) override {
    if (getOpsetVersion(*n->owningGraph()) < 23) {
      return false;
    }
    Match m;
    return MatchOuter(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!MatchOuter(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    Node* rms = graph.create(Symbol("RMSNormalization"), n->outputs().size());
    rms->addInput(m.x);
    rms->addInput(m.scale);
    rms->i_(kaxis, m.axis);
    rms->f_(kepsilon, m.epsilon);
    for (int i = 0; i < static_cast<int>(n->outputs().size()); ++i) {
      rms->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    rms->insertBefore(n);
    if (!tryReplacingAllUsesWith(n, rms)) {
      return false;
    }

    // `n` itself is destroyed by the driving iterator below, but everything
    // upstream of it is not, so detach n's one remaining edge into the dead
    // chain (dropping its innermost node to zero uses) and tear the rest
    // down explicitly, innermost first.
    Value* chain_out = m.dead_chain.front()->outputs()[0];
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      if (n->input(i) == chain_out) {
        n->removeInput(i);
        break;
      }
    }
    for (Node* dead : m.dead_chain) {
      dead->destroy();
    }

    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
