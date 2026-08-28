// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

// Rewrites `ArgMax`/`ArgMin` with `select_last_index=1` (added at opset 12)
// into an equivalent computation that never needs the attribute at all:
//
//   ArgMax(x, axis, keepdims, select_last_index=1)
//     == (dim_size_along_axis - 1) - ArgMax(Flip(x, axis), axis, keepdims)
//
// (`ArgMin` the same way.) Flipping the axis turns "last occurrence of the
// extremum" into "first occurrence" -- select_last_index's default, 0 --
// and the index just needs mapping back through the flip afterward. `Flip`
// itself has no dedicated ONNX op; it's the standard `Slice(..., steps=-1)`
// idiom, and `dim_size_along_axis` is read off `Shape(x)` at runtime rather
// than assumed static, so this applies whether or not the axis's extent is
// known at graph-simplification time.
//
// `select_last_index=1` is rarely used (the default, 0, is far more common,
// so this pass fires on very few real graphs), but no runtime-behavior-
// altering attribute is added: some downstream ONNX consumers -- this
// includes compilers such as TVM's Relax ONNX frontend -- don't implement
// `select_last_index` at all, and this rewrite needs nothing beyond `Shape`,
// `Gather`, `Slice`, `Sub`, and the arg-reduce op itself, all of which are
// far more consistently supported.
struct RewriteArgReduceSelectLastIndex final : public PredicateBasedPass {
  explicit RewriteArgReduceSelectLastIndex()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_arg_reduce_select_last_index";
  }

  static bool IsArgReduce(Node* node) {
    return node->kind() == kArgMax || node->kind() == Symbol("ArgMin");
  }

  bool patternMatchPredicate(Node* node) override {
    if (!IsArgReduce(node) || node->inputs().size() != 1) {
      return false;
    }
    if (!node->hasAttribute(kselect_last_index) ||
        node->i(kselect_last_index) == 0) {
      return false;
    }
    // select_last_index was added at opset 12, well above Slice's own
    // opset-10 minimum for the starts/ends/axes/steps-as-inputs form this
    // rewrite relies on, but check explicitly rather than assume.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 12;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const int64_t axis = node->hasAttribute(kaxis) ? node->i(kaxis) : 0;
    Value* data = node->input(0);

    auto make_i64 = [&](int64_t v, bool rank1) -> Value* {
      Node* c = graph.create(kConstant, 1);
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      if (rank1) {
        t.sizes().push_back(1);
      }
      t.int64s().push_back(v);
      c->t_(kvalue, t);
      c->output()->setElemType(TensorProto_DataType_INT64);
      if (rank1) {
        c->output()->setSizes({Dimension{1}});
      } else {
        c->output()->setSizes({});
      }
      c->insertBefore(node);
      return c->output();
    };

    // Flip(data, axis), the Slice(..., steps=-1) idiom: from the last
    // element (starts=-1) back through (and clamped at) the first
    // (ends=INT64_MIN, per the Slice spec's own clamping-to-bounds
    // convention for an out-of-range end with a negative step).
    Node* flip = graph.create(kSlice, 1);
    flip->addInput(data);
    flip->addInput(make_i64(-1, true));
    flip->addInput(make_i64(std::numeric_limits<int64_t>::min(), true));
    flip->addInput(make_i64(axis, true));
    flip->addInput(make_i64(-1, true));
    flip->insertBefore(node);
    flip->output()->setElemType(data->elemType());

    // ArgMax/ArgMin on the flipped data, with select_last_index reset to its
    // default (0): the first occurrence of the extremum in the flipped
    // tensor is the last occurrence in the original.
    Node* arg_first = graph.create(node->kind(), 1);
    arg_first->addInput(flip->output());
    arg_first->copyAttributes(*node);
    arg_first->i_(kselect_last_index, 0);
    arg_first->insertBefore(node);
    arg_first->output()->setElemType(TensorProto_DataType_INT64);

    // dim_size = Gather(Shape(data), axis) -- `axis` may itself be negative;
    // Gather's own negative-index wraparound (by data's rank, the length of
    // Shape(data)) matches ArgMax/ArgMin's own axis convention exactly, so
    // the same literal `axis` is reused unnormalized.
    Node* shape = graph.create(Symbol("Shape"), 1);
    shape->addInput(data);
    shape->insertBefore(node);
    shape->output()->setElemType(TensorProto_DataType_INT64);

    Node* dim_size = graph.create(Symbol("Gather"), 1);
    dim_size->addInput(shape->output());
    dim_size->addInput(make_i64(axis, false));
    dim_size->i_(kaxis, 0);
    dim_size->insertBefore(node);
    dim_size->output()->setElemType(TensorProto_DataType_INT64);

    Node* dim_size_minus_1 = graph.create(kSub, 1);
    dim_size_minus_1->addInput(dim_size->output());
    dim_size_minus_1->addInput(make_i64(1, false));
    dim_size_minus_1->insertBefore(node);
    dim_size_minus_1->output()->setElemType(TensorProto_DataType_INT64);

    // corrected = (dim_size - 1) - arg_first. `dim_size_minus_1` is a
    // scalar; it broadcasts against `arg_first`'s output regardless of
    // keepdims (a size-1 axis when keepdims=1, dropped entirely otherwise).
    Node* corrected = graph.create(kSub, 1);
    corrected->addInput(dim_size_minus_1->output());
    corrected->addInput(arg_first->output());
    corrected->insertBefore(node);
    corrected->output()->setElemType(TensorProto_DataType_INT64);
    if (!node->output()->sizes().empty()) {
      corrected->output()->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), corrected->output());
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
