// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct EliminateNopDropout final : public PredicateBasedPass {
  explicit EliminateNopDropout()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_nop_dropout"; }

  // An optional input is "omitted" when the graph has fewer inputs than the
  // slot, or when the slot is wired to the dummy ``Undefined`` value onnx uses
  // for a skipped optional input.
  static bool IsOmittedInput(const Node* node, size_t i) {
    return i >= node->inputs().size() ||
           node->input(i)->node()->kind() == kUndefined;
  }

  static bool TensorIsAllZero(const Tensor* t) {
    if (t == nullptr) {
      return false;
    }
    switch (t->elem_type()) {
      case TensorProto_DataType_FLOAT: {
        for (float v : ParseTensorData<float>(t)) {
          if (v != 0.0f) return false;
        }
        return true;
      }
      case TensorProto_DataType_DOUBLE: {
        for (double v : ParseTensorData<double>(t)) {
          if (v != 0.0) return false;
        }
        return true;
      }
      case TensorProto_DataType_BOOL: {
        for (bool v : ParseTensorData<bool>(t)) {
          if (v) return false;
        }
        return true;
      }
      default:
        // Unknown/unsupported element type: don't claim it is zero, so we
        // conservatively keep the Dropout.
        return false;
    }
  }

  // True when optional input ``i`` is omitted or a constant tensor of all
  // zeros (i.e. ratio == 0 or training_mode == false).
  static bool IsOmittedOrConstantZero(const Node* node, size_t i) {
    if (IsOmittedInput(node, i)) {
      return true;
    }
    if (!IsConstantTensor(node->input(i))) {
      return false;
    }
    return TensorIsAllZero(FetchConstantTensor(node->input(i)));
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kDropout) {
      return false;
    }
    // The mask (second) output cannot be reconstructed once the node is gone,
    // so bail if anything consumes it. Inference graphs practically never do.
    if (node->outputs().size() > 1 && !node->outputs()[1]->uses().empty()) {
      return false;
    }
    // opset < 12: ``ratio`` is an attribute; the only no-op is ratio == 0. We
    // deliberately do not touch a Dropout with a nonzero ratio attribute.
    if (node->hasAttribute(kratio)) {
      return node->f(kratio) == 0.0;
    }
    // opset >= 12: ``ratio`` and ``training_mode`` became (optional) inputs.
    // Dropout is the identity in inference mode -- when ``training_mode`` is
    // false, which is also its default -- so require input 2 to be omitted or
    // a constant false. Matching the documented no-op, also require ``ratio``
    // (input 1) to be omitted or a constant 0, so a genuine training Dropout
    // (nonzero ratio, kept for training-mode toggling) is never discarded.
    return IsOmittedOrConstantZero(node, 2) &&  // training_mode
           IsOmittedOrConstantZero(node, 1);    // ratio
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // Rewire the data output; the mask output (if any) is guaranteed unused by
    // the predicate above.
    const bool replacing_success =
        tryReplacingAllUsesWith(node->outputs()[0], node->input(0));
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
