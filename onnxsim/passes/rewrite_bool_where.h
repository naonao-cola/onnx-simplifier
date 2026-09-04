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
namespace onnxsim_passes {

// Rewrites `Where(cond, x, y)` when its two *data* operands (`x`/`y` -- not
// `cond`, which the ONNX spec always types `bool`) are themselves bool
// tensors, into an equivalent computation that never runs `Where` on `bool`
// data:
//
//   Where(cond, x, y)   -- x, y: bool
//     == Cast<to=BOOL>(Where(cond, Cast<to=INT32>(x), Cast<to=INT32>(y)))
//
// `bool` is a legal type for `Where`'s `T` (data operand) type constraint
// per the ONNX operator spec -- opset 9 and the current opset 16 both list
// it -- but ONNX Runtime's CPU execution provider only registers a `Where`
// kernel for `string`/`float`/`double`/`int32`/`int64`/`uint8`
// (onnxruntime/core/providers/cpu/cpu_execution_provider.cc); `bool` (along
// with float16/bfloat16/int8/int16/uint16/uint32/uint64/complex64/
// complex128) has no CPU kernel at all, so a graph with a bool-operand
// `Where` fails to *load* on ORT's CPU EP with "NOT_IMPLEMENTED ... Could
// not find an implementation for Where". `int32` does have a kernel, and a
// bool<->int32 `Cast` is a trivial, universally-supported 0/1
// reinterpretation, so routing the select through `int32` and casting the
// result back is enough to dodge the gap without changing what the graph
// computes: this is portability, not optimization, exactly like
// RewriteArgReduceSelectLastIndex before it.
//
// A bool-operand `Where` shows up in practice combining two boolean masks
// (e.g. a causal-attention mask built from more than one condition) ahead of
// a later cast to a float bias -- a pattern XLA's own `Select` op has no
// trouble with, so it is invisible in JAX itself and only surfaces once
// lowered to ONNX (observed from jax2onnx's export of Flax/Gemma-style
// attention masking).
struct RewriteBoolWhere final : public PredicateBasedPass {
  explicit RewriteBoolWhere()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "rewrite_bool_where"; }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("Where") || node->inputs().size() != 3) {
      return false;
    }
    Value* x = node->input(1);
    Value* y = node->input(2);
    return x->elemType() == TensorProto_DataType_BOOL &&
           y->elemType() == TensorProto_DataType_BOOL;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Value* cond = node->input(0);
    Value* x = node->input(1);
    Value* y = node->input(2);

    auto cast_to = [&](Value* v, TensorProto_DataType to) -> Value* {
      Node* cast = graph.create(kCast, 1);
      cast->i_(kto, static_cast<int64_t>(to));
      cast->addInput(v);
      cast->insertBefore(node);
      cast->output()->setElemType(to);
      if (!v->sizes().empty()) {
        cast->output()->setSizes(v->sizes());
      }
      return cast->output();
    };

    Value* x_i32 = cast_to(x, TensorProto_DataType_INT32);
    Value* y_i32 = cast_to(y, TensorProto_DataType_INT32);

    Node* where_i32 = graph.create(Symbol("Where"), 1);
    where_i32->addInput(cond);
    where_i32->addInput(x_i32);
    where_i32->addInput(y_i32);
    where_i32->insertBefore(node);
    where_i32->output()->setElemType(TensorProto_DataType_INT32);
    if (!node->output()->sizes().empty()) {
      where_i32->output()->setSizes(node->output()->sizes());
    }

    Node* cast_back = graph.create(kCast, 1);
    cast_back->i_(kto, static_cast<int64_t>(TensorProto_DataType_BOOL));
    cast_back->addInput(where_i32->output());
    cast_back->insertBefore(node);
    cast_back->output()->setElemType(TensorProto_DataType_BOOL);
    if (!node->output()->sizes().empty()) {
      cast_back->output()->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), cast_back->output());
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
