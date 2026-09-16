// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// I-BERT (Kim, Gholami, Yao, Mahoney, Keutzer, 2021, ICML 2021,
// "I-BERT: Integer-only BERT Quantization") -- C++ port of
// ibert_softmax.py's own apply_ibert_softmax. See that module's docstring
// for the full rationale: this is NOT a weight quantizer (unlike this
// repo's other data-free *_cpp ports) -- it is an activation-function
// rewrite that replaces a plain `Softmax` node's exp-and-normalize
// computation with I-BERT's own integer-arithmetic-friendly substitute
// for the transcendental `exp`.
//
// After the usual numerical-stability max-subtraction (`x <= 0` for every
// element), each element is decomposed as `x = -z*ln2 + p` with
// `z = floor(-x / ln2)` a non-negative integer and `p` the remainder,
// `p in (-ln2, 0]`. Then `exp(x) = exp(p) * 2**(-z)`: `exp(p)` is fit by
// a from-scratch numeric quadratic `A*p^2 + B*p + 1` over the single
// fixed short interval `(-ln2, 0]` (this repo's own numeric min-max
// search minimizing worst-case relative error, A ~= 0.36118,
// B ~= 0.9701 -- see ibert_softmax.py's own docstring for the derivation
// and the honesty note on how these constants were found, not
// transcribed from the paper's own reported ones), and `2**(-z)` is an
// ordinary `Pow` node here (a genuine integer-only accelerator would use
// a bit-shift instead -- see ibert_softmax.py's own scope note, which
// applies unchanged to this port: onnxsim has no lower-than-float32
// arithmetic type to express that distinction).
//
// SCOPE NARROWING (already ibert_softmax.py's own, not an additional one
// this port adds): the final normalization (dividing by the row sum)
// uses a plain `Div` node, not the paper's own integer-only iterative
// reciprocal -- mathematically the same value, just not an integer-only
// op. This module's honest scope is the `exp` piece only, exactly
// matching ibert_softmax.py's own documented scope.
//
// Before:
//   Y = Softmax(X, axis=k)
// After (opset 18+ only -- ReduceMax/ReduceSum's axes-as-input form):
//   RowMax   = ReduceMax(X, [k]), keepdims=1
//   Shifted  = Sub(X, RowMax)                      -- <= 0
//   NegShift = Neg(Shifted)                        -- >= 0
//   Z        = Floor(Div(NegShift, ln2))           -- non-negative integer
//   P        = Add(Shifted, Mul(Z, ln2))           -- in (-ln2, 0]
//   ExpP     = Add(Add(Mul(Mul(P, P), A), Mul(P, B)), 1)   -- poly exp(p)
//   Pow2NegZ = Pow(2.0, Neg(Z))                    -- 2**(-z)
//   ExpX     = Mul(ExpP, Pow2NegZ)                 -- approx exp(x - max)
//   SumExp   = ReduceSum(ExpX, [k]), keepdims=1
//   Y        = Div(ExpX, SumExp)
//
// Only a standalone `Softmax` node with exactly one input is matched (no
// MatMul/weight involvement at all -- this pass looks for the op type
// directly, the same style neg_to_mul.h uses for a single-op-type
// rewrite). A model whose opset is below 18 is left completely
// untouched, matching ibert_softmax.py's own opset gate exactly.
//
// Unlike ibert_softmax.py's own apply_ibert_softmax, this port does not
// accept a skip_names option -- several other *_cpp ports in this repo
// already establish that a C++ port need not mirror every optional knob
// its Python counterpart has. Constants (ln2, the quadratic
// coefficients, 1.0, 2.0) are recreated fresh per matched Softmax node
// rather than shared across the whole model -- the same per-match
// constant-creation convention weight_only_quantize_mxfp4_matmul.h's own
// per-match codebook already establishes (a pass instance is reused
// across future calls via RegisterOrReplace's std::call_once, so it must
// never cache a Value*/Node* pointer into a specific graph across
// matches).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM ibert_softmax.py: none beyond the
// scope narrowing already noted above -- this is a closed-form,
// deterministic polynomial substitution with no accumulation step, so
// this port is expected to track the Python port's own float64/float32
// numpy implementation closely, up to floating-point summation-order
// differences. apply_ibert_softmax and this port remain
// independently-correct, non-interchangeable entry points.

#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace ibert_softmax_detail {

// This repo's own numeric min-max fit of exp(p) ~= A*p^2 + B*p + 1 over
// p in [-ln2, 0] -- transcribed verbatim from ibert_softmax.py's own
// _IBERT_SOFTMAX_QUAD_A/_IBERT_SOFTMAX_QUAD_B (see that module's own
// docstring for the derivation).
constexpr double kQuadA = 0.36118;
constexpr double kQuadB = 0.9701;

}  // namespace ibert_softmax_detail

// I-BERT's own integer-friendly Softmax approximation -- matches any
// standalone Softmax node directly (no weight/MatMul involvement), then
// rebuilds its exp-and-normalize computation out of ordinary opset-18+
// ops (see this header's own top-of-file diagram).
struct IBertSoftmax final : public PredicateBasedPass {
  explicit IBertSoftmax()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "ibert_softmax"; }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != Symbol("Softmax") || n->inputs().size() != 1) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    return opset == 0 || opset >= 18;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (n->kind() != Symbol("Softmax") || n->inputs().size() != 1) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 18) {
      return false;
    }

    Value* x = n->input(0);
    const int64_t axis = n->hasAttribute(kaxis) ? n->i(kaxis) : -1;

    auto make_scalar_f32 = [&](float value) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_FLOAT;
      t.floats() = {value};
      return graph.addInitializerAndCreateValue(t);
    };
    Value* ln2_v = make_scalar_f32(static_cast<float>(std::log(2.0)));
    Value* quad_a_v =
        make_scalar_f32(static_cast<float>(ibert_softmax_detail::kQuadA));
    Value* quad_b_v =
        make_scalar_f32(static_cast<float>(ibert_softmax_detail::kQuadB));
    Value* one_v = make_scalar_f32(1.0f);
    Value* two_v = make_scalar_f32(2.0f);

    Tensor axes_t;
    axes_t.elem_type() = TensorProto_DataType_INT64;
    axes_t.sizes() = {1};
    axes_t.int64s() = {axis};
    Value* axes_v = graph.addInitializerAndCreateValue(axes_t);

    // Same-shape-as-x elementwise op, one input. Propagates only x's own
    // dtype/shape, NOT its name -- Value::copyMetadata (unlike this) also
    // copies the *unique name*, which would give every new intermediate
    // value the graph input's own name ("X"), producing a model onnx
    // itself rejects with "Duplicate definition of name (X)" once more
    // than one such value exists.
    auto unary = [&](NodeKind kind, Value* input) {
      Node* node = graph.create(kind, 1);
      node->addInput(input);
      node->insertBefore(n);
      node->output()->setElemType(x->elemType());
      if (x->has_sizes()) {
        node->output()->setSizes(x->sizes());
      }
      return node->output();
    };
    // Same-shape-as-x elementwise op, two inputs (the second may be a
    // scalar constant that broadcasts).
    auto binary = [&](NodeKind kind, Value* lhs, Value* rhs) {
      Node* node = graph.create(kind, 1);
      node->addInput(lhs);
      node->addInput(rhs);
      node->insertBefore(n);
      node->output()->setElemType(x->elemType());
      if (x->has_sizes()) {
        node->output()->setSizes(x->sizes());
      }
      return node->output();
    };
    auto reduce = [&](NodeKind kind, Value* input) {
      Node* node = graph.create(kind, 1);
      node->addInput(input);
      node->addInput(axes_v);
      node->i_(kkeepdims, 1);
      node->insertBefore(n);
      node->output()->setElemType(x->elemType());
      // keepdims=1 keeps x's own rank; the reduced axis' extent becomes
      // 1, which onnx's own shape inference will fill in correctly on
      // the next inference pass -- not reproduced here since it needs
      // the real Dimension type's symbolic-vs-static distinction.
      return node->output();
    };

    Value* row_max = reduce(kReduceMax, x);
    Value* shifted = binary(kSub, x, row_max);  // x - max <= 0
    Value* neg_shifted = unary(kNeg, shifted);  // -(x - max) >= 0
    Value* z_raw = binary(kDiv, neg_shifted, ln2_v);
    Value* z = unary(Symbol("Floor"), z_raw);
    Value* z_ln2 = binary(kMul, z, ln2_v);
    Value* p = binary(kAdd, shifted, z_ln2);  // p in (-ln2, 0]

    Value* p_sq = binary(kMul, p, p);
    Value* quad_term = binary(kMul, p_sq, quad_a_v);
    Value* lin_term = binary(kMul, p, quad_b_v);
    Value* sum_terms = binary(kAdd, quad_term, lin_term);
    Value* exp_p = binary(kAdd, sum_terms, one_v);  // polynomial exp(p)

    Value* neg_z = unary(kNeg, z);
    Value* pow2_neg_z = binary(kPow, two_v, neg_z);  // 2**(-z)

    Value* exp_x = binary(kMul, exp_p, pow2_neg_z);  // approx exp(x-max)
    Value* sum_exp = reduce(kReduceSum, exp_x);

    Node* result = graph.create(kDiv, 1);
    result->addInput(exp_x);
    result->addInput(sum_exp);
    result->insertBefore(n);
    result->output()->copyMetadata(n->output());

    n->output()->replaceAllUsesWith(result->output());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
