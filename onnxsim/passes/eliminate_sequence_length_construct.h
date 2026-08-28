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

// Folds `SequenceLength(SequenceConstruct(t0, ..., tn))` to the compile-time
// constant `n`. `SequenceConstruct` is a common PyTorch-export artifact for
// Python lists/tuples of tensors; a `for i in range(len(seq)): ...` loop over
// such a list exports as `SequenceLength` feeding a `Loop`'s trip count, so
// folding it to a constant here is what lets
// eliminate_loop_with_const_trip_count unroll that loop in turn -- together
// the two passes remove the Sequence type and the Loop from a graph shape
// most ONNX consumers (this includes compilers such as TVM's Relax ONNX
// frontend) have little to no support for.
//
// Only a `SequenceConstruct` producer is handled; see
// eliminate_sequence_at_construct.h for why `SequenceEmpty`/`SequenceInsert`
// chains and `SplitToSequence` are left alone for now.
struct EliminateSequenceLengthConstruct final : public PredicateBasedPass {
  explicit EliminateSequenceLengthConstruct()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_sequence_length_construct";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == Symbol("SequenceLength") &&
           node->inputs().size() == 1 &&
           node->input(0)->node()->kind() == Symbol("SequenceConstruct");
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const int64_t n =
        static_cast<int64_t>(node->input(0)->node()->inputs().size());

    Node* length_const = graph.create(kConstant, 1);
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s().push_back(n);
    length_const->t_(kvalue, t);
    length_const->output()->setElemType(TensorProto_DataType_INT64);
    length_const->output()->setSizes({});
    length_const->insertBefore(node);

    node->output()->replaceAllUsesWith(length_const->output());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
