// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Y = Reshape(X, shape1)
//   Z = Reshape(Y, shape2)
// After:
//   Z = Reshape(X, shape2)
//
// Only the final target shape determines Z -- X and Y always have the same
// total element count (that's what makes `Y = Reshape(X, shape1)` valid in
// the first place), so a `-1` (infer-from-size) entry in shape2 resolves
// identically whether it is X or Y that feeds the node.
//
// The one part of Reshape's semantics that *does* depend on which tensor
// feeds the node is a literal `0` entry in shape2 under the default
// allowzero=0: it means "copy this dimension from the node's input", i.e.
// from Y's shape today, which would silently become X's shape after fusion
// -- and X and Y can disagree at that position (that's the whole reason
// shape1 exists). So this pass only fires when shape2 either has no such
// copy-from-input `0` entries (nothing for the fusion to disturb) or the
// node opts out of that semantic entirely via allowzero=1.
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseConsecutiveReshapes final : public PredicateBasedPass {
  explicit FuseConsecutiveReshapes()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_reshapes";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kReshape, 0, kReshape);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const bool allowzero =
        n->hasAttribute(Symbol("allowzero")) && n->i(Symbol("allowzero")) == 1;
    if (!allowzero) {
      const Tensor* new_shape_tensor = FetchConstantTensor(n->inputs()[1]);
      if (!new_shape_tensor || new_shape_tensor->elem_type() !=
                                   ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        // Can't rule out a copy-from-input `0` entry -- leave the chain
        // alone.
        return false;
      }
      const auto new_shape = ParseTensorData<int64_t>(new_shape_tensor);
      if (std::any_of(new_shape.begin(), new_shape.end(),
                      [](int64_t d) { return d == 0; })) {
        return false;
      }
    }

    Node* prev = PrevNode(n, 0);
    n->replaceInput(0, prev->input(0));
    if (prev->output()->uses().empty()) {
      prev->destroy();
    }
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
