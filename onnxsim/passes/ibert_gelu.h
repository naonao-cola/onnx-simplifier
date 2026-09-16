// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// I-BERT (Kim, Gholami, Yao, Mahoney, Keutzer, 2021, ICML 2021,
// "I-BERT: Integer-only BERT Quantization") -- C++ port of ibert_gelu.py's
// own apply_ibert_gelu, the paper's "i-GELU" piece. See that module's own
// docstring for the full rationale and derivation: GELU is almost
// universally exported as `0.5 * x * (1 + Erf(x / sqrt(2)))`, and `Erf`
// is the one piece of that formula with no polynomial-friendly closed
// form. This pass replaces every standalone `Erf(x)` node with the
// paper's own closed-form second-order polynomial approximation:
//
//   L(x) = sign(x) * (a * (clip(|x|, 0, -b) + b)^2 + c)
//
// with `b = -1.69148` (ibert_gelu.py's own numeric min-max fit of the
// paper's functional form -- see that module's own docstring for how it
// was derived and why it is not claimed to reproduce the paper's own
// reported constant exactly), and `a`/`c` pinned by L's own continuity-
// at-0 and asymptote-at-+-1 constraints (c = 1, a = -1/b^2), not
// independently fit.
//
// Unlike every other *_cpp port in this repo, this is not a weight
// quantizer at all: it is a nonlinear-activation replacement, matching
// no weight, no MatMul/Gemm -- it matches a standalone `Erf` node
// anywhere in the graph and rebuilds its output from ordinary
// Abs/Clip/Add/Mul/Sign ops, the same node-building style (graph.create/
// insertBefore/replaceInput) weight_only_quantize_mxfp4_matmul.h already
// established for a pass that emits a whole new subgraph rather than
// just replacing a weight initializer.
//
// Before:
//   Y = Erf(X)
// After:
//   AbsX    = Abs(X)
//   Clipped = Clip(AbsX, 0.0, -b)
//   Shifted = Add(Clipped, b)
//   Squared = Mul(Shifted, Shifted)
//   Scaled  = Mul(Squared, a)
//   Poly    = Add(Scaled, c)
//   SignX   = Sign(X)
//   Y       = Mul(SignX, Poly)
//
// Only a single-input, single-output `Erf` node is matched -- exactly
// ibert_gelu.py's own scope (`len(node.input) != 1` is skipped there
// too, though every real ONNX Erf already has exactly one input; this
// port keeps the same defensive check for parity). Unlike
// apply_ibert_gelu, this port does not expose a skip_names parameter --
// several other *_cpp ports in this repo already establish that a C++
// port need not mirror every optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM ibert_gelu.py: none beyond the
// usual floating-point evaluation-order differences between this port's
// own scalar constant folding (there is none needed here -- a, b, c are
// compile-time constants) and the Python port's own numpy scalar ops;
// both sides compute the exact same five ONNX ops with the exact same
// three float32 constants, so this port is expected to be numerically
// identical to apply_ibert_gelu up to onnxruntime's own float32
// evaluation of the resulting graph. apply_ibert_gelu and this port
// remain independently-correct, non-interchangeable entry points, not
// aliases.

#pragma once

#include <string>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace ibert_gelu_detail {

// ibert_gelu.py's own numeric min-max fit of L(x)'s one free parameter
// `b` (see this header's own top-of-file comment); `a`/`c` are then
// pinned by L(x)'s own boundary constraints, not independently fit.
constexpr double kB = -1.69148;
constexpr double kA = -1.0 / (kB * kB);
constexpr double kC = 1.0;

}  // namespace ibert_gelu_detail

// I-BERT's own i-GELU polynomial approximation of Erf -- matches any
// standalone, single-input/single-output `Erf` node and rebuilds its
// output from a fixed sequence of Abs/Clip/Add/Mul/Sign ops instead.
struct IBertGelu final : public PredicateBasedPass {
  explicit IBertGelu()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "ibert_gelu"; }

  bool patternMatchPredicate(Node* n) override {
    return n->kind() == Symbol("Erf") && n->inputs().size() == 1 &&
           n->outputs().size() == 1;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (n->kind() != Symbol("Erf") || n->inputs().size() != 1 ||
        n->outputs().size() != 1) {
      return false;
    }

    Value* x = n->input(0);
    auto propagate_from_x = [&](Value* v) {
      v->setElemType(x->elemType());
      if (x->has_sizes()) {
        v->setSizes(x->sizes());
      }
    };
    auto make_scalar_float = [&](double value) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_FLOAT;
      t.sizes() = {};
      t.floats().assign(1, static_cast<float>(value));
      return graph.addInitializerAndCreateValue(t);
    };

    Node* abs_node = graph.create(Symbol("Abs"), 1);
    abs_node->addInput(x);
    abs_node->insertBefore(n);
    propagate_from_x(abs_node->output());

    Value* clip_min_v = make_scalar_float(0.0);
    Value* clip_max_v = make_scalar_float(-ibert_gelu_detail::kB);
    Node* clip_node = graph.create(Symbol("Clip"), 1);
    clip_node->addInput(abs_node->output());
    clip_node->addInput(clip_min_v);
    clip_node->addInput(clip_max_v);
    clip_node->insertBefore(n);
    propagate_from_x(clip_node->output());

    Value* b_v = make_scalar_float(ibert_gelu_detail::kB);
    Node* add_b_node = graph.create(kAdd, 1);
    add_b_node->addInput(clip_node->output());
    add_b_node->addInput(b_v);
    add_b_node->insertBefore(n);
    propagate_from_x(add_b_node->output());

    Node* square_node = graph.create(kMul, 1);
    square_node->addInput(add_b_node->output());
    square_node->addInput(add_b_node->output());
    square_node->insertBefore(n);
    propagate_from_x(square_node->output());

    Value* a_v = make_scalar_float(ibert_gelu_detail::kA);
    Node* scale_node = graph.create(kMul, 1);
    scale_node->addInput(square_node->output());
    scale_node->addInput(a_v);
    scale_node->insertBefore(n);
    propagate_from_x(scale_node->output());

    Value* c_v = make_scalar_float(ibert_gelu_detail::kC);
    Node* add_c_node = graph.create(kAdd, 1);
    add_c_node->addInput(scale_node->output());
    add_c_node->addInput(c_v);
    add_c_node->insertBefore(n);
    propagate_from_x(add_c_node->output());

    Node* sign_node = graph.create(Symbol("Sign"), 1);
    sign_node->addInput(x);
    sign_node->insertBefore(n);
    propagate_from_x(sign_node->output());

    Node* result_node = graph.create(kMul, 1);
    result_node->addInput(sign_node->output());
    result_node->addInput(add_c_node->output());
    result_node->insertBefore(n);
    propagate_from_x(result_node->output());

    const bool replaced =
        tryReplacingAllUsesWith(n->output(), result_node->output());
    if (!replaced) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
