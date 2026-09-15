// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites opset-24 `TensorScatter` -- a KV-cache-update op introduced for
// LLM attention (`past_cache`, `update`, optional `write_indices`, `axis`,
// `mode`) -- into `ScatterElements`, a long-supported op with broad backend
// coverage (unlike `TensorScatter`, which is brand new and unlikely to have
// runtime kernels yet). This is `PassType::Other` (a graph-shape rewrite,
// not a fusion opportunity onnxsim looks for by default), so it never runs
// unless requested. Opt in with
// `extra_optimizers=["rewrite_tensor_scatter_to_scatter_elements"]` (Python)
// or `--enable-optimization rewrite_tensor_scatter_to_scatter_elements`
// (CLI).
//
// Spec recap (verified against `onnx/defs/tensor/defs.cc`'s
// `TensorScatter_ver24` schema doc/shape-inference function and
// `onnx/reference/ops/op_tensor_scatter.py`'s reference implementation):
//   - `past_cache` and `update` have the same rank `r`; every axis other
//     than `axis` (normalized: `axis += r` if negative) has
//     `past_cache.shape[i] == update.shape[i]`, in particular axis 0 (the
//     batch axis) -- `axis` itself must not be 0.
//   - `write_indices` (optional, `tensor(int64)`, shape `(batch_size,)`,
//     defaults to all zero when absent) gives, per batch item, the starting
//     offset into `past_cache`'s `axis` dimension to write `update`'s
//     `axis`-dimension slices at.
//   - Per the reference pseudocode:
//       for prefix_idx in np.ndindex(past_cache.shape[:axis]):
//           batch_idx = prefix_idx[0]
//           for sequence_idx in range(sequence_length):
//               cache_sequence_idx = write_indices[batch_idx] + sequence_idx
//               if mode == "circular":
//                   cache_sequence_idx %= max_sequence_length
//               present_cache[(*prefix_idx, cache_sequence_idx)] = \
//                   update[(*prefix_idx, sequence_idx)]
//     Note `cache_idx`/`update_idx` are only `len(prefix_idx)+1 == axis+1`
//     entries long -- shorter than `past_cache`'s/`update`'s full rank `r` --
//     so NumPy's basic indexing assigns the *entire trailing
//     `(axis+1:)`-shaped slice at once. Also note every coordinate before
//     `axis` (not just the batch axis) is copied through unchanged
//     (`cache_idx[j] == prefix_idx[j] == update_idx[j]` for `j < axis`) --
//     `write_indices` only ever perturbs the single `axis` coordinate, and
//     only as a function of `prefix_idx[0]` (the batch coordinate) and
//     `sequence_idx` (`update`'s own `axis`-coordinate).
//
// That "every coordinate but one is an identity copy, and the one exception
// is driven by a same-shaped index tensor" is *exactly* `ScatterElements`'
// own contract (per `onnx/reference/ops/op_scatter_elements.py`):
//   output = np.copy(data)
//   for idx in np.ndindex(indices.shape):
//       out_idx = idx[:axis] + (indices[idx],) + idx[axis+1:]
//       output[out_idx] = updates[idx]
// with `indices`/`updates` sharing `update`'s own shape (not
// `data`'s/`past_cache`'s) -- so no `ScatterND`-style explicit per-element
// coordinate tuple needs constructing; only one same-shaped-as-`update`
// index tensor along `axis`, computed elementwise as
// `write_indices[batch_coord] + axis_coord` (`axis_coord` being that
// position's own coordinate along `axis`, i.e. `sequence_idx`), optionally
// wrapped modulo `max_sequence_length` for `mode="circular"`.
// `ScatterElements` has never had a reduction concept other than plain
// overwrite before opset 16 added the optional `reduction` attribute
// (default `"none"`, i.e. overwrite) -- exactly what's wanted here, so the
// generated node never sets `reduction` at all and works on every opset
// `ScatterElements` itself supports (11+, trivially satisfied since a valid
// graph containing opset-24 `TensorScatter` is already opset >= 24).
//
// Construction, per original `TensorScatter` node (`R` = static rank of
// `update`/`past_cache`, `axis_norm` = `axis` normalized to `[1, R-1]`):
//   1. `seq_len = Gather(Shape(update), [axis_norm])` (scalar) -- `update`'s
//      own size along `axis`, i.e. `sequence_length`.
//   2. `axis_iota = Range(0, seq_len, 1)` -> shape `(seq_len,)`, values
//      `[0, 1, ..., seq_len-1]` (`sequence_idx` for every position).
//   3. Reshape `axis_iota` to rank `R` with `seq_len` at position
//      `axis_norm` and `1` everywhere else (`Reshape`'s `-1` sentinel at
//      that one position, since `seq_len` may be dynamic).
//   4. If `write_indices` is present: reshape it (cast to int64 first if
//      somehow not already, though the schema fixes its type to
//      `tensor(int64)`) to rank `R` with its own (possibly dynamic) length
//      at position 0 and `1` elsewhere, then `Add` to step 3's tensor --
//      ONNX elementwise broadcasting naturally combines shape
//      `(batch,1,...,1)` and `(1,...,seq_len,...,1)` into
//      `(batch,1,...,seq_len,...,1)`, matching "only perturbed by the batch
//      coordinate and the axis coordinate, identity elsewhere". If absent,
//      `write_indices` defaults to all zero, so this step is skipped
//      entirely and step 3's tensor is used directly (adding a zero tensor
//      would be a no-op).
//   5. If `mode == "circular"`: `Mod` step 4's result by
//      `max_seq = Gather(Shape(past_cache), [axis_norm])` (scalar) --
//      matches the reference's `% max_sequence_length`; `Mod`'s default
//      `fmod=0` produces the same divisor-sign remainder convention NumPy's
//      `%` uses, matching the reference implementation exactly (relevant
//      only if `write_indices` ever carries a negative offset).
//   6. `final_indices = Expand(step 4-or-5's result, Shape(update))` --
//      broadcasts the (still partly size-1) tensor out to `update`'s exact
//      shape, since `ScatterElements` requires `indices` and `updates` to
//      have identical shapes (not just broadcast-compatible ones).
//   7. `ScatterElements(past_cache, final_indices, update, axis=axis)` --
//      `axis` is carried through completely unchanged (including if
//      negative): `ScatterElements` normalizes a negative `axis` the exact
//      same way (`axis += rank`) `TensorScatter` itself does, so there is no
//      need to pre-normalize the attribute value actually written onto the
//      new node (only the local `axis_norm` used to pick reshape positions
//      needs normalizing).
//
// Scope (the predicate declines outside this):
//  - `TensorScatter` in the default (empty) domain, 2 or 3 inputs
//    (`write_indices` optional), 1 output.
//  - `past_cache->sizes()` and `update->sizes()` both known and of equal
//    rank `R >= 2` (`axis != 0` requires at least one axis before it and one
//    at-or-after it).
//  - `axis` (int attribute, default -2) normalizes into `[1, R-1]`.
//  - `mode` (string attribute, default `"linear"`), when present, must be
//    exactly `"linear"` or `"circular"` -- anything else is malformed for
//    this op and the predicate declines rather than guessing.
//  - Static rank is required only to build the reshape/broadcast shape
//    literals (which axis position gets `-1`); no input dimension's actual
//    size need be statically known -- `seq_len`, `max_seq`, and
//    `write_indices`'s own length are all read at runtime via `Shape`/
//    `Gather`, following this codebase's established "no static dims
//    required beyond rank" idiom (e.g. `rewrite_bev_pool_to_scatter.h`).

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Small node-construction helper bound to one `TensorScatter` rewrite:
// every node it creates is inserted immediately before `anchor` (the
// `TensorScatter` node itself).
struct TensorScatterToScatterElementsBuilder {
  Graph& graph;
  Node* anchor;

  Value* ConstI64Vec(const std::vector<int64_t>& v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(static_cast<int64_t>(v.size()));
    for (int64_t x : v) {
      t.int64s().push_back(x);
    }
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  // Rank-0 (scalar) int64 constant -- e.g. `Range`'s `start`/`delta`
  // arguments, which its schema requires to be scalars, not rank-1 vectors.
  Value* ConstI64Scalar(int64_t v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s().push_back(v);
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  Value* UnOp(Symbol op, Value* a, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* BinOp(Symbol op, Value* a, Value* b, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->addInput(b);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* Add(Value* a, Value* b) { return BinOp(kAdd, a, b, a->elemType()); }
  Value* Mod(Value* a, Value* b) { return BinOp(Symbol("Mod"), a, b, a->elemType()); }

  Value* CastTo(Value* a, int32_t to) {
    Node* n = graph.create(kCast, 1);
    n->addInput(a);
    n->i_(kto, static_cast<int64_t>(to));
    n->insertBefore(anchor);
    n->output()->setElemType(to);
    return n->output();
  }

  Value* CastToI64IfNeeded(Value* a) {
    if (a->elemType() == TensorProto_DataType_INT64) {
      return a;
    }
    return CastTo(a, TensorProto_DataType_INT64);
  }

  Value* Shape(Value* a) { return UnOp(Symbol("Shape"), a, TensorProto_DataType_INT64); }

  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // Scalar (rank-0) Gather: drops `axis` entirely, per ONNX Gather's own
  // "rank-0 indices" behavior.
  Value* GatherScalar(Value* data, int64_t idx, int64_t axis) {
    return Gather(data, ConstI64Scalar(idx), axis);
  }

  Value* Reshape(Value* data, Value* shape) {
    Node* n = graph.create(kReshape, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Expand(Value* data, Value* shape) {
    Node* n = graph.create(kExpand, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // `start`/`limit`/`delta` must each be rank-0 (scalar) int64 tensors, per
  // ONNX `Range`'s own schema.
  Value* Range(Value* start, Value* limit, Value* delta) {
    Node* n = graph.create(Symbol("Range"), 1);
    n->addInput(start);
    n->addInput(limit);
    n->addInput(delta);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  Value* ScatterElements(Value* data, Value* indices, Value* updates,
                        int64_t axis) {
    Node* n = graph.create(Symbol("ScatterElements"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->addInput(updates);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }
};

struct RewriteTensorScatterToScatterElements final : public PredicateBasedPass {
  explicit RewriteTensorScatterToScatterElements()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_tensor_scatter_to_scatter_elements";
  }

  // Normalizes `axis` (a `TensorScatter`-style, possibly negative, attribute
  // value) against rank `r`, and checks it lands in `[1, r-1]` (`axis != 0`
  // is one of `TensorScatter`'s own hard requirements -- see this file's
  // header comment). Shared by the predicate and the transform so the two
  // never disagree.
  static bool NormalizeAxis(int64_t axis, int64_t r, int64_t& axis_norm) {
    int64_t a = axis % r;
    if (a < 0) {
      a += r;
    }
    if (a <= 0 || a >= r) {
      return false;
    }
    axis_norm = a;
    return true;
  }

  // True iff `write_indices` (the 3rd input) is actually supplied -- a 3rd
  // input slot can still mean "omitted", represented by an edge from a
  // dedicated `kUndefined`-kind node (ONNX's own convention for a trailing
  // optional input an exporter chose to spell out explicitly as absent).
  static bool HasWriteIndices(Node* node) {
    return node->inputs().size() == 3 &&
           node->input(2)->node()->kind() != kUndefined;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("TensorScatter")) {
      return false;
    }
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    const size_t num_inputs = node->inputs().size();
    if ((num_inputs != 2 && num_inputs != 3) ||
        node->outputs().size() != 1) {
      return false;
    }

    Value* past_cache = node->input(0);
    Value* update = node->input(1);
    if (!past_cache->has_sizes() || !update->has_sizes()) {
      return false;
    }
    const int64_t r = static_cast<int64_t>(past_cache->sizes().size());
    if (r < 2 || static_cast<int64_t>(update->sizes().size()) != r) {
      return false;
    }

    // A 3rd input slot doesn't necessarily mean `write_indices` is actually
    // supplied -- ONNX's own convention for a trailing optional input the
    // exporter chose to spell out explicitly as absent is an edge from a
    // dedicated `kUndefined`-kind node (see e.g. `eliminate_nop_dropout.h`,
    // `magnitude_pruning.h`). That case is handled identically to the
    // 2-input case below (see `HasWriteIndices`), so nothing further to
    // check here beyond a defensive rank check when it is genuinely present.
    if (HasWriteIndices(node)) {
      Value* write_indices = node->input(2);
      if (write_indices->has_sizes() && write_indices->sizes().size() != 1) {
        return false;
      }
    }

    std::string mode;
    if (GetValueFromAttr(node, Symbol("mode"), mode)) {
      if (mode != "linear" && mode != "circular") {
        return false;
      }
    }

    const int64_t axis =
        GetValueFromAttrWithDefault<int64_t>(node, kaxis, int64_t(-2));
    int64_t axis_norm = 0;
    return NormalizeAxis(axis, r, axis_norm);
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* past_cache = node->input(0);
    Value* update = node->input(1);
    Value* write_indices =
        HasWriteIndices(node) ? node->input(2) : nullptr;

    const int64_t r = static_cast<int64_t>(past_cache->sizes().size());
    const int64_t axis =
        GetValueFromAttrWithDefault<int64_t>(node, kaxis, int64_t(-2));
    int64_t axis_norm = 0;
    if (!NormalizeAxis(axis, r, axis_norm)) {
      return false;  // re-checked: the graph could in principle have
                     // changed between predicate and transform.
    }
    const std::string mode =
        GetValueFromAttrWithDefault<std::string>(node, Symbol("mode"),
                                                 std::string("linear"));

    TensorScatterToScatterElementsBuilder b{graph, node};

    // 1-2. axis_iota = Range(0, seq_len, 1) -> (seq_len,), sequence_idx for
    // every position.
    Value* seq_len = b.GatherScalar(b.Shape(update), axis_norm, 0);
    Value* axis_iota =
        b.Range(b.ConstI64Scalar(0), seq_len, b.ConstI64Scalar(1));

    // 3. Reshape to rank r with seq_len at axis_norm, 1 elsewhere.
    std::vector<int64_t> iota_reshape(static_cast<size_t>(r), 1);
    iota_reshape[static_cast<size_t>(axis_norm)] = -1;
    Value* indices = b.Reshape(axis_iota, b.ConstI64Vec(iota_reshape));

    // 4. Fold in write_indices (defaults to all-zero, i.e. a no-op, when
    // absent).
    if (write_indices != nullptr) {
      Value* wi = b.CastToI64IfNeeded(write_indices);
      std::vector<int64_t> wi_reshape(static_cast<size_t>(r), 1);
      wi_reshape[0] = -1;
      Value* wi_reshaped = b.Reshape(wi, b.ConstI64Vec(wi_reshape));
      indices = b.Add(wi_reshaped, indices);
    }

    // 5. Wrap-around for circular mode.
    if (mode == "circular") {
      Value* max_seq = b.GatherScalar(b.Shape(past_cache), axis_norm, 0);
      indices = b.Mod(indices, max_seq);
    }

    // 6. Broadcast out to update's exact shape -- ScatterElements requires
    // indices/updates to have identical shapes.
    Value* final_indices = b.Expand(indices, b.Shape(update));

    // 7. The overwrite-scatter itself; `axis` (possibly negative) carried
    // through unchanged.
    Value* result = b.ScatterElements(past_cache, final_indices, update, axis);

    if (!node->output()->sizes().empty()) {
      result->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), result);
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
