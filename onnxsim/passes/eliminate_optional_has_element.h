// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

// Folds `OptionalHasElement` to a compile-time-constant bool whenever its
// emptiness is already known at graph-simplification time:
//
//  - No input at all (the op's own input is itself optional-arity) always
//    means "false" per the ONNX spec, regardless of anything else.
//  - `OptionalHasElement(Optional(x))` -- an `Optional` node that wraps a
//    value -- is always "true".
//  - `OptionalHasElement(Optional())` -- an `Optional` node built with no
//    input, i.e. an explicitly-empty optional -- is always "false".
//
// `Optional`/`OptionalHasElement`/`OptionalGetElement` are how ONNX
// represents Python's `Optional[Tensor]` (e.g. a `torch.jit.script`-exported
// function argument that may or may not be provided), but most ONNX
// consumers -- this includes compilers such as TVM's Relax ONNX frontend --
// have no support for the Optional type at all. When the optional's
// emptiness is fixed by construction, as here, there is no need for the
// Optional type to survive into the simplified graph.
//
// Only an `Optional` producer is handled (or no input, above); a value whose
// producer is something else entirely -- including a plain tensor/sequence,
// which newer opsets also allow as input here and always reads as "true" --
// is left alone, since the internal graph representation this pass operates
// on does not carry full ONNX type information (optional-vs-plain) to
// distinguish that case from a genuinely unresolved optional-typed value.
struct EliminateOptionalHasElement final : public PredicateBasedPass {
  explicit EliminateOptionalHasElement()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_optional_has_element";
  }

  // -1: no input (always false). 0: Optional() with no input (empty, false).
  // 1: Optional(x) (non-empty, true). -2: not a foldable shape.
  static int Classify(Node* node) {
    if (node->inputs().empty()) {
      return -1;
    }
    if (node->inputs().size() != 1) {
      return -2;
    }
    Node* producer = node->input(0)->node();
    if (producer->kind() != Symbol("Optional")) {
      return -2;
    }
    return producer->inputs().empty() ? 0 : 1;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == Symbol("OptionalHasElement") && Classify(node) != -2;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const int cls = Classify(node);
    ONNX_ASSERT(cls != -2);
    const bool has_element = cls == 1;

    Node* bool_const = graph.create(kConstant, 1);
    Tensor t;
    t.elem_type() = TensorProto_DataType_BOOL;
    t.int32s().push_back(has_element ? 1 : 0);
    bool_const->t_(kvalue, t);
    bool_const->output()->setElemType(TensorProto_DataType_BOOL);
    bool_const->output()->setSizes({});
    bool_const->insertBefore(node);

    node->output()->replaceAllUsesWith(bool_const->output());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
