// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites `GatherElements(data, indices, axis)` into plain `Gather` in the
// (narrow but real) case where a full-rank, elementwise `indices` tensor
// turns out not to actually vary along any axis other than `axis` -- i.e.
// it is really a broadcast of a 1-D index vector along that axis, just
// spelled out to full rank. Plain `Gather` is the most broadly and natively
// supported member of the Gather family on backends like TensorRT (more so
// than `GatherElements`, which has narrower/newer support), so this is worth
// firing whenever it legitimately applies.
//
// This is `PassType::Other` (semantics-preserving but only provably
// equivalent for a subset of inputs -- see below -- so it never runs by
// default). Opt in with `extra_optimizers=["rewrite_gatherelements_to_gather"]`
// (Python) or `--enable-optimization rewrite_gatherelements_to_gather` (CLI).
//
// Spec recap (verified against `onnx/defs/tensor/defs.cc`'s
// `GatherElements_ver13` schema doc and `onnx/reference/ops/op_gather_elements
// .py`'s reference implementation):
//   - `data` and `indices` have the *same* rank `r`; for every axis `d !=
//     axis`, `indices.shape[d] == data.shape[d]` -- only `indices.shape[axis]`
//     may differ from `data.shape[axis]`.
//   - `output[i_0,...,i_{r-1}] = data[i_0,...,i_{axis-1}, indices[i_0,...,
//     i_{r-1}], i_{axis+1},...,i_{r-1}]` -- critically, the index used can
//     depend on *every* coordinate of `indices`, not just the `axis`-th one.
//     This is the fundamental difference from plain `Gather`, which applies
//     the *same* index set uniformly across every combination of the other
//     axes (an outer-product gather, not `GatherElements`' elementwise one)
//     -- so the two ops are equivalent only when `indices`, despite having
//     full rank, happens not to vary along any of those other axes.
//   - Per the doc, `indices` values may be negative:
//     `-data_shape[axis] <= indices[...] <= data_shape[axis]-1` -- exactly
//     the same convention `Gather`'s own `indices` input already documents
//     for its own axis, so a representative index vector extracted here can
//     be handed straight to the replacement `Gather` node unchanged, with no
//     separate normalization step (contrast `rewrite_gathernd_to_gather`,
//     where per-column indices *do* need normalizing before being combined
//     by integer strides -- there is no such combination step here).
//
// Proving "doesn't actually vary along any non-`axis` axis" requires
// inspecting `indices`' actual values, not just its shape, so this only ever
// fires when `indices` is a compile-time constant (a `Constant` node or an
// `is_constant_initializer`-marked initializer -- `FetchConstantTensor`
// covers both forms uniformly, the same idiom
// `eliminate_loop_with_const_trip_count.h` uses for its own constant-tensor
// reads). Most real `GatherElements` uses have a genuinely dynamic,
// data-dependent `indices` and this pass simply will not apply to them --
// that is correct and expected, not a shortcoming.
//
// Invariance check: take the slice of `indices` at all-other-axes-coordinate
// 0 (i.e. every axis except `axis` fixed to index 0) as the "representative"
// 1-D vector of length `indices.shape[axis]`; the rewrite applies iff every
// other fixed combination of the non-`axis` coordinates reproduces that same
// vector exactly. Implemented as a single O(size) linear scan over the flat
// tensor buffer using row-major strides (no need to materialize an N-D
// index): for flat position `i`, `axis_coord = (i / stride[axis]) %
// shape[axis]` is `i`'s coordinate along `axis` (this division/modulo
// isolates one dimension's coordinate regardless of any other dimension's
// size -- the standard row-major flat-index decomposition), and the
// representative position sharing that same `axis` coordinate but with
// every *other* coordinate forced to 0 is exactly `axis_coord *
// stride[axis]` (every other dimension contributes 0 to a flat offset when
// its own coordinate is 0). So the whole check is "does every element equal
// the representative element for its own `axis` coordinate" -- an O(size)
// scan matching the cost profile of other constant-tensor-inspecting passes
// already in this codebase. (Take care not to flip which coordinate gets
// zeroed here: zeroing the `axis` coordinate instead -- i.e. comparing `i`
// against `i - axis_coord * stride[axis]` -- would check invariance along
// completely the wrong axis and was an early bug caught by this pass's own
// numeric equivalence-checked tests.)
//
// If the check fails anywhere, this pass declines (leaves `GatherElements`
// untouched) rather than partially rewriting -- very common, since most
// `GatherElements` uses are genuinely elementwise.
//
// Scope (the predicate declines outside this):
//  - `GatherElements` in the default (empty) domain, exactly 2 inputs, 1
//    output.
//  - `data`'s and `indices`' ranks statically known and equal (always true
//    for a valid model per the spec, but checked defensively rather than
//    assumed).
//  - `indices` is a constant int32 or int64 tensor (`FetchConstantTensor`
//    non-null, element type matching `GatherElements`' own `Tind` type
//    constraint).

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct RewriteGatherElementsToGather final : public PredicateBasedPass {
  explicit RewriteGatherElementsToGather()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_gatherelements_to_gather";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("GatherElements")) {
      return false;
    }
    // Leave a same-named op in a non-ai.onnx domain (e.g. a vendor/plugin
    // "GatherElements") alone.
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 1) {
      return false;
    }
    Value* data = node->input(0);
    Value* indices = node->input(1);
    if (!data->has_sizes() || !indices->has_sizes()) {
      return false;
    }
    if (data->sizes().empty() ||
        data->sizes().size() != indices->sizes().size()) {
      return false;
    }
    const Tensor* t = FetchConstantTensor(indices);
    if (t == nullptr) {
      return false;
    }
    return t->elem_type() == TensorProto_DataType_INT32 ||
           t->elem_type() == TensorProto_DataType_INT64;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* data = node->input(0);
    Value* indices = node->input(1);
    const int64_t r = static_cast<int64_t>(data->sizes().size());
    const int64_t axis =
        GetValueFromAttrWithDefault<int64_t>(node, kaxis, int64_t(0));
    const int64_t norm_axis = axis < 0 ? axis + r : axis;
    if (norm_axis < 0 || norm_axis >= r) {
      return false;  // malformed axis; defensive, shouldn't happen for a
                     // valid model.
    }

    const Tensor* t = FetchConstantTensor(indices);
    if (t == nullptr) {
      return false;  // re-checked: the graph could in principle have
                     // changed between predicate and transform.
    }
    const std::vector<int64_t>& shape = t->sizes();
    if (static_cast<int64_t>(shape.size()) != r) {
      return false;  // the constant tensor's own recorded shape disagrees
                     // with the rank the predicate matched on; defensive.
    }

    int64_t total = 1;
    for (int64_t s : shape) {
      total *= s;
    }

    std::vector<int64_t> vals;
    if (t->elem_type() == TensorProto_DataType_INT32) {
      const std::vector<int32_t> vals32 = ParseTensorData<int32_t>(t);
      vals.assign(vals32.begin(), vals32.end());
    } else {
      vals = ParseTensorData<int64_t>(t);
    }
    if (static_cast<int64_t>(vals.size()) != total) {
      return false;  // defensive: parsed element count mismatch.
    }

    // Row-major strides over `shape`.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
      strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] *
                                        shape[static_cast<size_t>(i + 1)];
    }
    const int64_t axis_stride = strides[static_cast<size_t>(norm_axis)];
    const int64_t axis_len = shape[static_cast<size_t>(norm_axis)];

    // Invariance check: every element must equal the representative element
    // sharing its own axis coordinate but with every *other* coordinate
    // forced to 0 -- see this file's header comment for the flat-index
    // arithmetic (and why it is `axis_coord * axis_stride`, not
    // `i - axis_coord * axis_stride`). (When any shape dim, including
    // axis_len itself, is 0, `total` is 0 and this loop simply never runs,
    // so the `% axis_len` below never sees axis_len==0.)
    for (int64_t i = 0; i < total; ++i) {
      const int64_t axis_coord = (i / axis_stride) % axis_len;
      const int64_t rep_index = axis_coord * axis_stride;
      if (vals[static_cast<size_t>(i)] !=
          vals[static_cast<size_t>(rep_index)]) {
        return false;  // genuinely elementwise -- not reducible to Gather.
      }
    }

    std::vector<int64_t> representative(static_cast<size_t>(axis_len));
    for (int64_t kk = 0; kk < axis_len; ++kk) {
      representative[static_cast<size_t>(kk)] =
          vals[static_cast<size_t>(kk * axis_stride)];
    }

    Tensor new_idx_tensor;
    new_idx_tensor.elem_type() = TensorProto_DataType_INT64;
    new_idx_tensor.sizes().push_back(axis_len);
    for (int64_t v : representative) {
      new_idx_tensor.int64s().push_back(v);
    }
    Value* new_idx =
        graph.addInitializerAndCreateValue(std::move(new_idx_tensor));

    // `axis` is carried through unchanged, including if negative -- `Gather`
    // accepts negative axis identically to `GatherElements`.
    Node* gather = graph.create(Symbol("Gather"), 1);
    gather->addInput(data);
    gather->addInput(new_idx);
    gather->i_(kaxis, axis);
    gather->insertBefore(node);
    gather->output()->setElemType(data->elemType());
    if (!node->output()->sizes().empty()) {
      gather->output()->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), gather->output());
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
