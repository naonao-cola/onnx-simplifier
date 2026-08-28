// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <cstdint>
#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

// Folds `SequenceAt(SequenceConstruct(t0, ..., tn), position)` -- with a
// compile-time-constant `position` -- straight to the indexed tensor `ti`,
// dropping the sequence indirection entirely. `SequenceConstruct` is a common
// PyTorch-export artifact for Python lists/tuples of tensors, and most ONNX
// consumers (this includes compilers such as TVM's Relax ONNX frontend) have
// little to no support for the Sequence type; when the index and the
// sequence's construction are both statically known -- the common case, e.g.
// indexing a fixed-size list -- there is no need for a Sequence value to ever
// exist in the simplified graph.
//
// Only a `SequenceConstruct` producer is handled; a sequence built up via
// `SequenceEmpty`/`SequenceInsert`, or produced by `SplitToSequence`, is left
// alone (a reasonable follow-up, but each needs its own index-to-producer
// resolution logic).
struct EliminateSequenceAtConstruct final : public PredicateBasedPass {
  explicit EliminateSequenceAtConstruct()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_sequence_at_construct";
  }

  // SequenceAt's `position` is tensor(int32) or tensor(int64); read it as a
  // plain int64_t either way.
  static bool GetScalarInt64(const Value* v, int64_t& out) {
    const Tensor* t = FetchConstantTensor(v);
    if (t == nullptr) {
      return false;
    }
    if (t->elem_type() == TensorProto_DataType_INT64) {
      const auto data = ParseTensorData<int64_t>(t);
      if (data.empty()) return false;
      out = data[0];
      return true;
    }
    if (t->elem_type() == TensorProto_DataType_INT32) {
      const auto data = ParseTensorData<int32_t>(t);
      if (data.empty()) return false;
      out = static_cast<int64_t>(data[0]);
      return true;
    }
    return false;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("SequenceAt") || node->inputs().size() != 2) {
      return false;
    }
    Node* construct = node->input(0)->node();
    if (construct->kind() != Symbol("SequenceConstruct")) {
      return false;
    }
    int64_t position;
    if (!GetScalarInt64(node->input(1), position)) {
      return false;
    }
    const int64_t n = static_cast<int64_t>(construct->inputs().size());
    if (position < 0) {
      position += n;
    }
    return position >= 0 && position < n;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* construct = node->input(0)->node();
    int64_t position;
    ONNX_ASSERT(GetScalarInt64(node->input(1), position));
    const int64_t n = static_cast<int64_t>(construct->inputs().size());
    if (position < 0) {
      position += n;
    }
    node->output()->replaceAllUsesWith(construct->input(position));
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
