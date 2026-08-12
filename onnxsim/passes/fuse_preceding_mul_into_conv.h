// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   X' = X * S
//   Y = Conv(X', W[, B])
// After:
//   Y = Conv(X, W')   with  W' = W * S
//
// S must be a constant that scales the Conv input per input channel: either a
// single element (scalar broadcast) or a tensor whose only non-unit dimension
// is the channel axis of the Conv input (axis 1), e.g. [C_in], [C_in, 1, 1]
// or [1, C_in, 1, 1]. The scale is folded into the convolution weights by
// creating a Mul node on the constant weight, which the constant folder then
// materialises; the leftover multiplicative op in front of the Conv
// disappears.
//
// The bias (if any) is left untouched: scaling every input channel by the
// per-channel factor before the dot product is equivalent to scaling the
// matching weight slice and does not change the additive bias term --
// unlike fuse_mul_into_conv's Conv -> Mul case (which scales the *output*
// channel and must scale the bias too).
//
// A per-channel S is only handled for an ungrouped Conv (group == 1): it
// lines up directly with the weight's axis-1 size ([C_out, C_in / groups,
// k...]) only then. A grouped Conv would need S re-sliced per group, which
// this pass does not attempt; a scalar S is still folded regardless of
// group count since a uniform scale is group-agnostic. Only Conv (not
// ConvTranspose) is handled: ConvTranspose's weight layout puts the input
// channel on axis 0 with a different broadcast shape.

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

struct FusePrecedingMulIntoConv final : public PredicateBasedPass {
  explicit FusePrecedingMulIntoConv()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_preceding_mul_into_conv";
  }

  // Index (0 or 1) of the Mul operand that is a constant scale -- the other
  // operand becomes the Conv's new (unscaled) input -- or -1 if neither
  // operand is constant.
  static int ScaleOperandIndex(Node* mul) {
    for (int i = 0; i < 2; ++i) {
      if (IsConstantTensor(mul->inputs()[i])) {
        return i;
      }
    }
    return -1;
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != kMul || n->inputs().size() != 2) {
      return false;
    }
    // The Mul must feed exactly one consumer -- a Conv, as that Conv's data
    // input (offset 0) -- so rewiring the Conv away from the Mul leaves the
    // Mul dead and safe to erase. `uses()` returns by value, so bind the
    // element by value rather than taking a reference into the temporary
    // vector.
    if (n->output()->uses().size() != 1) {
      return false;
    }
    const Use use = n->output()->uses()[0];
    if (use.user->kind() != kConv || use.offset != 0 ||
        !IsConstantTensor(use.user, 1)) {
      return false;
    }
    return ScaleOperandIndex(n) >= 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const int scale_idx = ScaleOperandIndex(n);
    if (scale_idx < 0 || n->output()->uses().size() != 1) {
      return false;
    }
    Value* scale = n->inputs()[scale_idx];
    Value* activation = n->inputs()[1 - scale_idx];

    const Use use = n->output()->uses()[0];
    Node* conv = use.user;
    if (conv->kind() != kConv || use.offset != 0) {
      return false;
    }
    Value* weight = conv->inputs()[1];

    const Tensor* weight_t = FetchConstantTensor(weight);
    const Tensor* scale_t = FetchConstantTensor(scale);
    if (weight_t == nullptr || scale_t == nullptr) {
      return false;
    }
    // The scale is multiplied against the constant weight, so the element
    // types must agree.
    if (weight_t->elem_type() != scale_t->elem_type()) {
      return false;
    }

    // Conv weight is [C_out, C_in / groups, k...]; the input channel is axis
    // 1.
    const auto& w_sizes = weight_t->sizes();
    if (w_sizes.size() < 2) {
      return false;
    }
    const int64_t Cg = w_sizes[1];  // C_in / groups
    const int64_t w_rank = static_cast<int64_t>(w_sizes.size());

    // Validate that `scale` scales the Conv input per input channel: a
    // scalar (single element) or a tensor whose only non-unit axis equals
    // Cg and, when right-aligned against the Conv input rank (== w_rank),
    // lands on the channel axis (index 1).
    const auto& s_sizes = scale_t->sizes();
    int64_t s_numel = 1;
    for (const int64_t d : s_sizes) {
      s_numel *= d;
    }
    bool per_channel;
    if (s_numel == 1) {
      per_channel = false;  // scalar: broadcasts to every input channel
    } else if (s_numel == Cg) {
      // A per-channel scale only lines up with weight axis 1 when there is a
      // single group; see the file comment above.
      const int64_t group =
          GetValueFromAttrWithDefault(conv, kgroup, int64_t{1});
      if (group != 1) {
        return false;
      }
      const int64_t s_rank = static_cast<int64_t>(s_sizes.size());
      int64_t non_unit_axis = -1;
      for (int64_t i = 0; i < s_rank; ++i) {
        if (s_sizes[i] != 1) {
          if (non_unit_axis != -1) {
            return false;  // more than one non-unit axis
          }
          non_unit_axis = i;
        }
      }
      if (non_unit_axis < 0 || w_rank - s_rank + non_unit_axis != 1) {
        return false;  // the C-sized axis is not the channel axis
      }
      per_channel = true;
    } else {
      return false;
    }

    // --- All checks passed; build the scaled weight. ---

    Value* weight_scale = scale;
    if (per_channel) {
      // Flatten the scale to [C_in].
      Value* scale_1d = scale;
      if (static_cast<int64_t>(s_sizes.size()) != 1) {
        Node* reshape = graph.create(kReshape, 1);
        reshape->addInput(scale);
        Tensor shape_t;
        shape_t.elem_type() = TensorProto_DataType_INT64;
        shape_t.sizes().push_back(1);
        shape_t.int64s().push_back(Cg);
        reshape->addInput(graph.addInitializerAndCreateValue(shape_t));
        reshape->insertBefore(conv);
        scale_1d = reshape->output();
      }
      // Broadcast [C_in] -> [1, C_in, 1, ..., 1] so it multiplies the weight
      // per input channel; weight axis 0 (the output channel) must stay
      // unscaled, unlike fuse_mul_into_conv's per-output-channel broadcast.
      std::vector<int64_t> axes;
      axes.push_back(0);
      for (int64_t i = 2; i < w_rank; ++i) {
        axes.push_back(i);
      }
      Node* unsqueeze = graph.create(kUnsqueeze, 1);
      unsqueeze->addInput(scale_1d);
      const int opset = getOpsetVersion(graph);
      if (opset >= 13 || opset == 0) {
        Tensor axes_t;
        axes_t.elem_type() = TensorProto_DataType_INT64;
        axes_t.sizes().push_back(static_cast<int64_t>(axes.size()));
        axes_t.int64s() = axes;
        unsqueeze->addInput(graph.addInitializerAndCreateValue(axes_t));
      } else {
        unsqueeze->is_(kaxes, std::move(axes));
      }
      unsqueeze->insertBefore(conv);
      weight_scale = unsqueeze->output();
    }

    Node* mul_w = graph.create(kMul, 1);
    mul_w->addInput(weight);
    mul_w->addInput(weight_scale);
    mul_w->insertBefore(conv);
    conv->replaceInput(1, mul_w->output());
    conv->replaceInput(0, activation);

    // The Mul's output is now unused; erase it.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
