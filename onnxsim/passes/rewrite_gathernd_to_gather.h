// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites `GatherND(data, indices, batch_dims=b)` into a single plain
// `Gather` whenever the `k` axes of `data` it jointly indexes have
// statically-known sizes. This is the natural follow-up to
// `rewrite_gridsample_to_gather` (see that file): plain `Gather` is the most
// broadly and natively supported member of the Gather family on backends
// like TensorRT, better supported than `GatherND`/`GatherElements`, which
// have narrower/newer support. In particular this reduces the
// `GatherND(..., batch_dims=1)` nodes that pass emits for 2-D `GridSample`
// down to plain `Gather` as well, when `H`/`W` (the two axes it jointly
// indexes) are statically known.
//
// This is `PassType::Other` (a node-count-neutral graph-shape rewrite in the
// general case, not a size/op-count reduction) and never runs by default.
// Opt in with `extra_optimizers=["rewrite_gathernd_to_gather"]` (Python) or
// `--enable-optimization rewrite_gathernd_to_gather` (CLI).
//
// Spec recap (verified against `onnx/defs/tensor/defs.cc`'s `GatherND_ver13`
// schema doc and `onnx/reference/ops/op_gathernd.py`'s `_gather_nd_impl`, the
// actual reference semantics -- not just the doc prose):
//   - `data` has rank `r >= 1`, `indices` has rank `q >= 1` with
//     `indices.shape[-1] == k` (`1 <= k <= r - b`), `batch_dims = b`.
//   - `indices`' first `b` dims must equal `data`'s first `b` dims (the
//     shared "batch" dims); indexing starts at `data`'s axis `b`.
//   - Per the doc's own bullet 5, `indices` values may be *negative*:
//     `-data_shape[i] <= indices[...,i] <= data_shape[i]-1`, i.e. ordinary
//     Python-style negative indexing along each jointly-indexed axis -- the
//     reference implementation's `reshaped_data[(batch_dim, *gather_index)]`
//     is plain numpy advanced indexing, which resolves negative components
//     exactly this way. This rewrite normalizes each such component to
//     non-negative *before* combining it into a flat sub-index (see below);
//     skipping that step would be a correctness bug for any negative-index
//     input, not just an edge case -- so it is not optional here.
//   - Output shape = `indices.shape[:-1] + data.shape[b+k:]`.
//
// Derivation (the general trick -- not merely the trivial `b=0,k=1` case,
// though that case is worth sanity-checking the general formula against
// first, see below):
//
// Let `data`'s shape be `(B_0,...,B_{b-1}, D_0,...,D_{k-1}, T_0,...)` (the
// leading `b` "batch" axes, the `k` axes actually being jointly indexed, and
// whatever trailing axes are left over) and `indices`' shape be
// `(B_0,...,B_{b-1}, E_0,...,E_{m-1}, k)` (the same `b` batch axes, `m`
// "extra" axes -- `m = q - b - 1` -- and the trailing index-depth axis).
// Only the `D_0,...,D_{k-1}` sizes need to be statically known for this
// rewrite to fire; the batch sizes `B_i` and trailing sizes `T_i` may stay
// fully dynamic/symbolic, since they never need to appear as compile-time
// integer literals below -- only as runtime `Shape(data)` reads.
//
//  1. Flatten `data`'s leading `b+k` axes into one via
//     `Reshape(data, Concat([-1], Shape(data)[b+k:]))`. Row-major reshape
//     semantics mean the flattened axis has size `B*D` (`B = prod(B_i)`,
//     `D = prod(D_i)`, `D` a compile-time integer literal, `B` inferred by
//     the `-1` since it may be dynamic) and position `(bidx*D + didx, ...)`
//     of the result equals `data[unflatten_b(bidx)..., unflatten_d(didx)...,
//     ...]` for the matching row-major unflattenings of `bidx` and `didx`.
//  2. For column `j` of `indices`' last axis (extracted via a scalar
//     `Gather(indices, j, axis=-1)`, which conveniently drops that axis
//     outright -- no separate `Squeeze` needed), normalize negative values
//     (`idx_j < 0 ? idx_j + D_j : idx_j`, `D_j` a compile-time constant) and
//     combine all `k` columns via the same row-major strides `Reshape`
//     itself uses: `combined = sum_j idx_j_norm * mult_j`, `mult_j =
//     D_{j+1}*D_{j+2}*...*D_{k-1}` (`mult_{k-1} = 1`), so that `combined`
//     equals exactly the `didx` from step 1 for that same set of column
//     values. `combined` has shape `indices.shape[:-1]` (rank `b+m`).
//  3. `b == 0`: `B == 1` and `bidx` is always `0`, so `combined` alone is
//     already the flat index into axis 0 of the step-1 reshape -- skip the
//     rest of this derivation and its offset machinery entirely (this is
//     also exactly the well-known trivial case: `b=0,k=1` degenerates to
//     `combined = Gather(indices, 0, axis=-1)` with no `mult` multiply
//     needed since `mult_0 = 1`, i.e. plain `Gather(data,
//     Squeeze(indices,-1), axis=0)` -- matches `GatherND`'s own doc Example
//     2 exactly when hand-checked).
//  4. `b > 0`: each batch position `(i_0,...,i_{b-1})` needs its own
//     `bidx*D` offset added to `combined` before it is a valid flat index
//     into the *whole* `B*D`-sized axis. `bidx` is exactly the row-major
//     flat index that reshaping `data`'s leading `b` axes down to one axis
//     of size `B` would produce at that position -- which is precisely what
//     `Reshape(Range(0, B, 1), Shape(data)[:b])` computes (a `0..B-1` count
//     up, reshaped back into the batch axes' own shape; `B` itself read at
//     runtime as `prod(Shape(data)[:b])`, computed here via repeated scalar
//     `Gather`+`Mul` off `Shape(data)` rather than `ReduceProd` -- `ReduceProd`
//     moved its `axes` operand from attribute to input at opset 18, and this
//     rewrite otherwise only needs ops stable since `GatherND`'s own
//     opset-11 floor, so this sidesteps that version skew entirely). Multiply
//     by the integer literal `D` to get `batch_offset`.
//  5. `batch_offset` has rank `b`; `combined` has rank `b+m`. Rather than
//     `Unsqueeze` `batch_offset` at `m` trailing axes (the natural-looking
//     move, but `Unsqueeze`'s axes-as-tensor-input form is opset >= 13
//     only), `Reshape` it to `Shape(data)[:b] ++ [1]*m` (built via `Concat`
//     with a compile-time-constant length-`m` vector of `1`s) -- an
//     equivalent trailing-broadcast-dims shape that plain `Reshape` can
//     produce at any opset. ONNX/numpy broadcasting then aligns the two
//     equal-rank (`b+m`) operands position-wise, so `batch_offset`'s value at
//     `(i_0,...,i_{b-1},...)` broadcasts unchanged across all `m` trailing
//     positions. `flat_indices = combined + batch_offset` (skipped when
//     `m == 0`: ranks already match, no reshape needed).
//  6. `output = Gather(flattened_data, flat_indices, axis=0)`. This produces
//     exactly `indices.shape[:-1] + data.shape[b+k:]` -- `Gather`'s own
//     output-shape rule (`data.shape[:axis] + indices.shape +
//     data.shape[axis+1:]`, `axis=0`) applied to the step-1 reshape's shape
//     `(B*D,) + data.shape[b+k:]` gives `indices.shape[:-1] +
//     data.shape[b+k:]`, matching `GatherND`'s own output shape exactly.
//
// Hand/numerically verified (see the derivation-sanity script run during
// development, replicated in this pass's tests) against: the `b=0,k=1`
// trivial case; `GatherND`'s own doc Examples 1-3; `b=1,k=1`; `b=1,k=2`
// (mirroring `rewrite_gridsample_to_gather`'s `(N,H,W,C)`,
// `batch_dims=1` output exactly); `b=2,k=2`; the `m=0` case (`indices` has
// no extra dims beyond the batch dims and the trailing index-depth axis);
// `b+k==r` (no trailing axes at all); and negative-index columns in several
// of the above.
//
// Scope (the predicate declines outside this):
//  - `GatherND` in the default (empty) domain, exactly 2 inputs, 1 output.
//  - `indices`' rank `q` and last-dim size `k` (`k >= 1`) statically known.
//  - `data`'s rank `r` statically known, `0 <= batch_dims`, `batch_dims + k
//    <= r`, `q >= batch_dims + 1`, and axes `[batch_dims, batch_dims+k)` of
//    `data` (the `k` axes actually flattened together) have statically-known
//    sizes. `data`'s other axes -- the leading `batch_dims` batch axes and
//    any trailing axes -- may be fully dynamic/symbolic: they are only ever
//    read off `Shape(data)` at runtime, never assumed static.
//  - `GatherND` with a `batch_dims` attribute did not exist before opset 11
//    (it was introduced at that version), so this pass's opset floor is
//    exactly that -- a defensive check, not load-bearing (the node could not
//    otherwise exist), matching this codebase's other opset-gated rewrites.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Small node-construction helper bound to one `GatherND` rewrite: every node
// it creates is inserted immediately before `anchor` (the `GatherND` node
// itself); scalar/length-1 int64 constants are cached since the same
// literal (0, 1, a repeated `mult`/`D` value, ...) is often needed more than
// once within one rewrite.
struct GatherNDToGatherBuilder {
  Graph& graph;
  Node* anchor;

  std::unordered_map<int64_t, Value*> i64_scalar_cache;
  std::unordered_map<int64_t, Value*> i64_vec1_cache;

  Value* ConstI64Scalar(int64_t v) {
    auto it = i64_scalar_cache.find(v);
    if (it != i64_scalar_cache.end()) {
      return it->second;
    }
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.int64s().push_back(v);
    Value* val = graph.addInitializerAndCreateValue(std::move(t));
    i64_scalar_cache.emplace(v, val);
    return val;
  }

  // Rank-1, single-element int64 tensor -- e.g. the `starts`/`ends`/`axes`
  // inputs `Slice` wants, or a `[-1]` `Reshape` target.
  Value* ConstI64Vec1(int64_t v) {
    auto it = i64_vec1_cache.find(v);
    if (it != i64_vec1_cache.end()) {
      return it->second;
    }
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(1);
    t.int64s().push_back(v);
    Value* val = graph.addInitializerAndCreateValue(std::move(t));
    i64_vec1_cache.emplace(v, val);
    return val;
  }

  // Rank-1 int64 tensor of `count` repeated `1`s -- pads a shape vector with
  // trailing size-1 axes via `Reshape`'s target shape, standing in for
  // `Unsqueeze`'s tensor-axes form (opset >= 13 only) so this pass keeps
  // working down to `GatherND`'s own opset-11 floor. Not cached: `count`
  // (`= m`, the number of `indices`' "extra" dims) is rarely reused across
  // rewrites in one graph.
  Value* ConstI64Ones(int64_t count) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(count);
    for (int64_t i = 0; i < count; ++i) {
      t.int64s().push_back(1);
    }
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  Value* BinOp(Symbol op, Value* a, Value* b, int32_t elem_type) {
    Node* n = graph.create(op, 1);
    n->addInput(a);
    n->addInput(b);
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  Value* Add(Value* a, Value* b) {
    return BinOp(kAdd, a, b, TensorProto_DataType_INT64);
  }
  Value* Mul(Value* a, Value* b) {
    return BinOp(kMul, a, b, TensorProto_DataType_INT64);
  }
  Value* Less(Value* a, Value* b) {
    return BinOp(Symbol("Less"), a, b, TensorProto_DataType_BOOL);
  }

  Value* Where(Value* cond, Value* a, Value* b) {
    Node* n = graph.create(Symbol("Where"), 1);
    n->addInput(cond);
    n->addInput(a);
    n->addInput(b);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  Value* Shape(Value* a) {
    Node* n = graph.create(Symbol("Shape"), 1);
    n->addInput(a);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  // 1-D int64 slice of a `Shape(...)` vector, axis 0, half-open [start,end).
  // The tensor-inputs form of `Slice`, stable since opset 10 -- well below
  // `GatherND`'s own opset-11 floor.
  Value* SliceShape(Value* shape_vec, int64_t start, int64_t end) {
    Node* n = graph.create(kSlice, 1);
    n->addInput(shape_vec);
    n->addInput(ConstI64Vec1(start));
    n->addInput(ConstI64Vec1(end));
    n->addInput(ConstI64Vec1(0));
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  Value* Range(Value* start, Value* limit, Value* delta) {
    Node* n = graph.create(Symbol("Range"), 1);
    n->addInput(start);
    n->addInput(limit);
    n->addInput(delta);
    n->insertBefore(anchor);
    n->output()->setElemType(TensorProto_DataType_INT64);
    return n->output();
  }

  Value* Reshape(Value* data, Value* shape) {
    Node* n = graph.create(kReshape, 1);
    n->addInput(data);
    n->addInput(shape);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  Value* Concat(int64_t axis, const std::vector<Value*>& inputs) {
    Node* n = graph.create(kConcat, 1);
    for (Value* v : inputs) {
      n->addInput(v);
    }
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(inputs[0]->elemType());
    return n->output();
  }

  // Scalar (rank-0) index `Gather` that drops `axis` from the output --
  // used both to read one element off a runtime shape vector and to pull
  // one column off `indices`' last axis.
  Value* GatherScalar(Value* data, Value* index, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(index);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // General (non-scalar-index) `Gather` -- this pass's final replacement
  // node.
  Value* Gather(Value* data, Value* indices, int64_t axis) {
    Node* n = graph.create(Symbol("Gather"), 1);
    n->addInput(data);
    n->addInput(indices);
    n->i_(kaxis, axis);
    n->insertBefore(anchor);
    n->output()->setElemType(data->elemType());
    return n->output();
  }

  // `idx` if non-negative, else `idx + dim` -- `GatherND` (like `Gather` and
  // `GatherElements`) allows negative per-axis indices; each column must be
  // normalized to non-negative before it is combined into a single flat
  // sub-index by integer strides in `runTransform`, since that stride
  // arithmetic is only valid for non-negative components. `dim` is a
  // compile-time-known axis size (this pass declines otherwise), so the
  // shift is a plain scalar constant, not a runtime shape read.
  Value* NormalizeIndex(Value* idx, int64_t dim) {
    Value* is_neg = Less(idx, ConstI64Scalar(0));
    Value* shifted = Add(idx, ConstI64Scalar(dim));
    return Where(is_neg, shifted, idx);
  }
};

struct RewriteGatherNDToGather final : public PredicateBasedPass {
  explicit RewriteGatherNDToGather()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_gathernd_to_gather";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("GatherND")) {
      return false;
    }
    // Leave a same-named op in a non-ai.onnx domain (e.g. a vendor/plugin
    // "GatherND") alone.
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 1) {
      return false;
    }
    Value* data = node->input(0);
    Value* indices = node->input(1);

    if (!indices->has_sizes() || indices->sizes().empty()) {
      return false;
    }
    const Dimension& k_dim = indices->sizes().back();
    if (!k_dim.is_int || k_dim.dim < 1) {
      return false;
    }
    const int64_t k = k_dim.dim;
    const int64_t q = static_cast<int64_t>(indices->sizes().size());

    if (!data->has_sizes()) {
      return false;
    }
    const int64_t r = static_cast<int64_t>(data->sizes().size());

    const int64_t b = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("batch_dims"), int64_t(0));
    // Per the spec: 0 <= b < min(q,r), 1 <= k <= r-b. b>=0 always holds for a
    // valid model (batch_dims has no legal negative value) but is checked
    // defensively rather than assumed.
    if (b < 0 || b + k > r || q < b + 1) {
      return false;
    }

    // Only the k axes actually being flattened together need known sizes --
    // data's leading b batch axes and any trailing axes may stay dynamic.
    for (int64_t j = 0; j < k; ++j) {
      if (!data->sizes()[b + j].is_int) {
        return false;
      }
    }

    // GatherND's batch_dims attribute (and GatherND itself) did not exist
    // before opset 11; defensive, not load-bearing.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 11;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* data = node->input(0);
    Value* indices = node->input(1);
    const int64_t b = GetValueFromAttrWithDefault<int64_t>(
        node, Symbol("batch_dims"), int64_t(0));
    const int64_t r = static_cast<int64_t>(data->sizes().size());
    const int64_t q = static_cast<int64_t>(indices->sizes().size());
    const int64_t k = indices->sizes().back().dim;
    const int64_t m = q - b - 1;

    std::vector<int64_t> dims(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      dims[static_cast<size_t>(j)] = data->sizes()[b + j].dim;
    }
    // mult_j = D_{j+1} * ... * D_{k-1}, mult_{k-1} = 1 -- the row-major
    // stride for column j within the flattened (B*D)-sized axis.
    std::vector<int64_t> mults(static_cast<size_t>(k), 1);
    for (int64_t j = k - 2; j >= 0; --j) {
      mults[static_cast<size_t>(j)] =
          mults[static_cast<size_t>(j + 1)] * dims[static_cast<size_t>(j + 1)];
    }
    int64_t D = 1;
    for (int64_t j = 0; j < k; ++j) {
      D *= dims[static_cast<size_t>(j)];
    }

    GatherNDToGatherBuilder gb{graph, node};

    // Shape(data), computed at most once and shared by whichever of the two
    // uses below (or both) actually need it.
    Value* shape_x = nullptr;
    auto ShapeX = [&]() -> Value* {
      if (shape_x == nullptr) {
        shape_x = gb.Shape(data);
      }
      return shape_x;
    };

    // Step 1: flatten data's leading b+k axes into one.
    Value* reshape_target;
    if (b + k == r) {
      // No trailing axes at all -- fully flatten, no Shape(data) needed for
      // this step.
      reshape_target = gb.ConstI64Vec1(-1);
    } else {
      Value* tail_shape = gb.SliceShape(ShapeX(), b + k, r);
      reshape_target = gb.Concat(0, {gb.ConstI64Vec1(-1), tail_shape});
    }
    Value* flat_data = gb.Reshape(data, reshape_target);

    // Step 2: per-column extraction, negative-index normalization, and
    // stride-weighted combination.
    Value* combined = nullptr;
    for (int64_t j = 0; j < k; ++j) {
      Value* idx_j = gb.GatherScalar(indices, gb.ConstI64Scalar(j), -1);
      Value* idx_j_norm =
          gb.NormalizeIndex(idx_j, dims[static_cast<size_t>(j)]);
      Value* term =
          mults[static_cast<size_t>(j)] == 1
              ? idx_j_norm
              : gb.Mul(idx_j_norm,
                       gb.ConstI64Scalar(mults[static_cast<size_t>(j)]));
      combined = combined == nullptr ? term : gb.Add(combined, term);
    }

    // Steps 3-5: the per-batch-row offset, skipped entirely when b == 0.
    Value* flat_indices = combined;
    if (b > 0) {
      Value* shape_head = gb.SliceShape(ShapeX(), 0, b);
      Value* B = nullptr;
      for (int64_t i = 0; i < b; ++i) {
        Value* di = gb.GatherScalar(ShapeX(), gb.ConstI64Scalar(i), 0);
        B = B == nullptr ? di : gb.Mul(B, di);
      }
      Value* range = gb.Range(gb.ConstI64Scalar(0), B, gb.ConstI64Scalar(1));
      Value* flat_batch_index = gb.Reshape(range, shape_head);
      Value* batch_offset = gb.Mul(flat_batch_index, gb.ConstI64Scalar(D));
      if (m > 0) {
        Value* target_shape = gb.Concat(0, {shape_head, gb.ConstI64Ones(m)});
        batch_offset = gb.Reshape(batch_offset, target_shape);
      }
      flat_indices = gb.Add(combined, batch_offset);
    }

    // Step 6.
    Value* output = gb.Gather(flat_data, flat_indices, 0);
    if (!node->output()->sizes().empty()) {
      output->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), output);
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
