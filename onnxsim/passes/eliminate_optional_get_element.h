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

// Folds `OptionalGetElement(Optional(x))` -- an `Optional` node that wraps a
// value -- straight to `x`, dropping the Optional indirection entirely. See
// eliminate_optional_has_element.h's doc comment for why this matters
// (Optional is a common `torch.jit.script` export artifact for
// `Optional[Tensor]` arguments, and most ONNX consumers -- compilers such as
// TVM's Relax ONNX frontend included -- have no support for it).
//
// `OptionalGetElement(Optional())` -- unwrapping an explicitly-empty
// optional -- is undefined behavior per the ONNX spec, so it is deliberately
// left unmatched rather than guessed at. Only an `Optional` producer is
// handled; see eliminate_optional_has_element.h for why a plain
// tensor/sequence producer (also allowed by newer opsets, and trivially an
// identity in that case) is left alone too.
struct EliminateOptionalGetElement final : public PredicateBasedPass {
  explicit EliminateOptionalGetElement()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_optional_get_element";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("OptionalGetElement") ||
        node->inputs().size() != 1) {
      return false;
    }
    Node* producer = node->input(0)->node();
    return producer->kind() == Symbol("Optional") &&
           producer->inputs().size() == 1;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* producer = node->input(0)->node();
    node->output()->replaceAllUsesWith(producer->input(0));
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
