// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites `ScatterElements(data, indices, updates, axis=0)` into a row-wise
// `ScatterND(data, row_indices, updates)` when `indices`, although full rank,
// provably does not vary along any axis other than 0 -- i.e. every element
// of row `k` of `updates` is written to row `indices[k, 0, ..., 0]` of the
// output. That is exactly what torchvision's `MultiScaleRoIAlign` exports
// for Mask R-CNN / Faster R-CNN's per-FPN-level RoIAlign merge:
//
//   ScatterElements(data, Expand(Reshape(idx, [-1,1,1,1]), Shape(updates)),
//                   updates, axis=0)
//
// ONNX Runtime's CPU `ScatterElements` does one indexed element write per
// element of `updates` (N*C*H*W of them, driven by an equally large int64
// `indices` tensor that first has to be materialized by the `Expand`);
// `ScatterND` with `[N, 1]` indices does N contiguous row copies. On the
// real `MaskRCNN-12-qdq` remainder graph that merge was the single most
// expensive node group on a phone CPU (see
// `scripts/android/htp_exploration/rest_htp_findings.md`).
//
// This is `PassType::Other` (opt-in, like every other `rewrite_*` pass in
// onnxsim): enable with
// `extra_optimizers=["rewrite_scatterelements_to_scatternd"]` (Python) or
// `--enable-optimization rewrite_scatterelements_to_scatternd` (CLI).
//
// Spec recap (`onnx/defs/tensor/defs.cc`, `ScatterElements_ver18` /
// `ScatterND_ver18`, and the reference implementations):
//   ScatterElements, axis=0, rank r:
//     output[indices[i0,i1,...], i1, ...] = updates[i0, i1, ...]
//   ScatterND with `indices` of shape [N, 1] (so `updates` must have shape
//   [N] + data.shape[1:]):
//     output[indices[i0, 0], :, ...] = updates[i0, :, ...]
//   These coincide for every (i0, i1, ...) iff `indices[i0, i1, ...] ==
//   indices[i0, 0, ..., 0]` for every i1, ... (the invariance this pass
//   proves) *and* `updates.shape[1:] == data.shape[1:]` (ScatterElements
//   allows `updates`/`indices` to be smaller than `data` along the non-axis
//   dims -- a sub-block scatter ScatterND cannot express -- so that equality
//   is checked statically too, never assumed). `data`'s trailing dims come
//   from shape inference where it knows them; in the torchvision export
//   `data` is `ConstantOfShape(Concat(<row count>, [C], [H], [W]))` and
//   shape inference leaves all four dims unknown, so for a
//   `ConstantOfShape` producer the constant entries of its shape vector are
//   used as well (see `DataDims`).
//   Both ops accept negative index values with the same convention
//   (counting back from the end of axis 0), so index values pass through
//   unchanged. Both forbid duplicate indices under `reduction="none"`
//   ("indices should not have duplicate entries"), and a duplicate row index
//   in the ScatterND form is exactly a duplicate element index in the
//   ScatterElements form, so the two ops are defined on the same set of
//   valid inputs and agree on all of them.
//
// Deliberately NOT rewritten:
//   - `reduction` other than "none". The values agree mathematically, but
//     ScatterND's spec leaves the order in which duplicate indices are
//     combined unspecified, so for float add/mul the result is not
//     guaranteed bit-identical to ScatterElements'; not worth the risk.
//   - `axis` != 0 (after normalization). ScatterND only indexes leading
//     dims; supporting another axis would need Transposes around it, which
//     costs about what this rewrite saves.
//   - `indices` produced by `Tile` (repeats indices rather than broadcasting
//     them) or by any other op not covered below.
//
// Two ways the invariance is proven:
//   1. `indices` is `Expand(X, shape)` where X's shape, left-padded with 1s
//      to rank r as `Expand` broadcasts it, is a statically known 1 on every
//      axis except 0. Then every element of `indices` in row k is X[k] (or
//      X[0] if X's axis-0 size is 1 and broadcast), so the row index vector
//      is `Reshape(X, [-1, 1])`, `Expand`ed to [N, 1] (N = updates.shape[0])
//      unless X's axis-0 size is statically equal to N. That Expand of an
//      N-element vector is always valid: the original Expand already
//      required X's axis-0 size to be 1 or N. X is cast to int64 if it is
//      int32 (ScatterND only takes int64 indices).
//   2. `indices` is a constant (`Constant` / constant initializer): every
//      element is compared against its row's representative
//      `indices[k, 0, ..., 0]` in one linear scan (same row-major flat-index
//      scheme as `rewrite_gatherelements_to_gather`), and the representative
//      vector becomes a new [N, 1] int64 initializer.
//
// The original `Expand` / constant is left in place; if nothing else uses
// it, dead-node elimination removes it.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct RewriteScatterElementsToScatterND final : public PredicateBasedPass {
  explicit RewriteScatterElementsToScatterND()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_scatterelements_to_scatternd";
  }

  static bool SameDim(const Dimension& a, const Dimension& b) {
    if (a.is_int && b.is_int) {
      return a.dim == b.dim;
    }
    return !a.is_int && !b.is_int && !a.is_unknown && !b.is_unknown &&
           !a.param.empty() && a.param == b.param;
  }

  static bool IsOne(const Dimension& d) { return d.is_int && d.dim == 1; }

  // Rank of the scatter, from whichever of indices/updates/data carries a
  // shape (all three have the same rank in a valid model). -1 if unknown.
  static int64_t ScatterRank(Node* node) {
    for (size_t i = 0; i < 3; ++i) {
      Value* v = node->input(i);
      if (v->has_sizes()) {
        return static_cast<int64_t>(v->sizes().size());
      }
    }
    return -1;
  }

  // Statically known entries of a rank-1 int64 shape vector of length r:
  // either a constant, or a `Concat(axis=0)` of pieces that are each
  // constant or of statically known length (their entries then stay
  // unknown). Returns false if even the vector's layout can't be pinned.
  static bool ShapeVectorDims(Value* shape_vec, int64_t r,
                              std::vector<Dimension>& out) {
    out.clear();
    auto append_const = [&out](const Tensor* t) {
      if (t->elem_type() != TensorProto_DataType_INT64 ||
          t->sizes().size() > 1) {
        return false;
      }
      for (int64_t v : ParseTensorData<int64_t>(t)) {
        out.emplace_back(v);
      }
      return true;
    };
    if (const Tensor* t = FetchConstantTensor(shape_vec)) {
      return append_const(t) && static_cast<int64_t>(out.size()) == r;
    }
    Node* cat = shape_vec->node();
    if (cat == nullptr || cat->kind() != kConcat ||
        (cat->has_domain() && !cat->domain().empty())) {
      return false;
    }
    const int64_t axis = GetValueFromAttrWithDefault<int64_t>(cat, kaxis, 0);
    if (axis != 0 && axis != -1) {
      return false;
    }
    for (Value* piece : cat->inputs()) {
      if (const Tensor* t = FetchConstantTensor(piece)) {
        if (!append_const(t)) {
          return false;
        }
        continue;
      }
      if (!piece->has_sizes() || piece->sizes().size() != 1 ||
          !piece->sizes()[0].is_int) {
        return false;
      }
      for (int64_t k = 0; k < piece->sizes()[0].dim; ++k) {
        out.emplace_back();  // unknown entry, known position
      }
    }
    return static_cast<int64_t>(out.size()) == r;
  }

  // data's dims, taking shape inference's sizes where static and, for dims
  // it left unknown, the static entries of the shape vector if `data` is a
  // `ConstantOfShape` (the pattern torchvision exports for the RoIAlign
  // merge buffer, whose trailing dims shape inference does not recover).
  // A scatter's output has exactly its `data` input's shape (both specs),
  // and torchvision merges FPN levels by scattering each level into the
  // previous level's result, so the buffer's shape is also followed back
  // through a chain of ScatterElements / ScatterND.
  static std::vector<Dimension> DataDims(Value* data, int64_t r,
                                         int depth = 0) {
    std::vector<Dimension> dims(static_cast<size_t>(r));
    if (data->has_sizes() && static_cast<int64_t>(data->sizes().size()) == r) {
      dims = data->sizes();
    }
    auto fill_unknown = [&dims, r](const std::vector<Dimension>& from) {
      for (int64_t d = 0; d < r; ++d) {
        auto& dd = dims[static_cast<size_t>(d)];
        if (!dd.is_int && from[static_cast<size_t>(d)].is_int) {
          dd = from[static_cast<size_t>(d)];
        }
      }
    };
    Node* p = data->node();
    if (p == nullptr || (p->has_domain() && !p->domain().empty())) {
      return dims;
    }
    if (p->kind() == Symbol("ConstantOfShape") && p->inputs().size() == 1) {
      std::vector<Dimension> from_shape;
      if (ShapeVectorDims(p->input(0), r, from_shape)) {
        fill_unknown(from_shape);
      }
    } else if ((p->kind() == Symbol("ScatterElements") ||
                p->kind() == Symbol("ScatterND")) &&
               p->inputs().size() == 3 && depth < 64) {
      fill_unknown(DataDims(p->input(0), r, depth + 1));
    }
    return dims;
  }

  // updates.shape[1:] == data.shape[1:], statically.
  static bool TrailingDimsMatch(Value* data, Value* updates, int64_t r) {
    if (!updates->has_sizes() ||
        static_cast<int64_t>(updates->sizes().size()) != r) {
      return false;
    }
    const std::vector<Dimension> ds = DataDims(data, r);
    const auto& us = updates->sizes();
    for (int64_t d = 1; d < r; ++d) {
      if (!SameDim(ds[static_cast<size_t>(d)], us[static_cast<size_t>(d)])) {
        return false;
      }
    }
    return true;
  }

  // For the Expand form: X's shape, aligned to rank r, is 1 on axes 1..r-1.
  static bool ExpandSourceIsRowBroadcast(Value* x, int64_t r) {
    if (!x->has_sizes()) {
      return false;
    }
    const auto& xs = x->sizes();
    const int64_t rx = static_cast<int64_t>(xs.size());
    if (rx > r) {
      return false;
    }
    const int64_t pad = r - rx;  // X's dim j aligns with output axis j + pad
    for (int64_t j = 0; j < rx; ++j) {
      if (j + pad == 0) {
        continue;  // axis 0 itself: any size
      }
      if (!IsOne(xs[static_cast<size_t>(j)])) {
        return false;
      }
    }
    return true;
  }

  static Node* ExpandProducer(Value* indices) {
    Node* p = indices->node();
    if (p == nullptr || p->kind() != Symbol("Expand")) {
      return nullptr;
    }
    if (p->has_domain() && !p->domain().empty()) {
      return nullptr;
    }
    if (p->inputs().size() != 2) {
      return nullptr;
    }
    return p;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("ScatterElements")) {
      return false;
    }
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    if (node->inputs().size() != 3 || node->outputs().size() != 1) {
      return false;
    }
    if (getOpsetVersion(*node->owningGraph()) < 11) {
      return false;
    }
    if (node->hasAttribute(Symbol("reduction")) &&
        node->s(Symbol("reduction")) != "none") {
      return false;
    }
    const int64_t r = ScatterRank(node);
    if (r < 1) {
      return false;
    }
    int64_t axis = GetValueFromAttrWithDefault<int64_t>(node, kaxis, 0);
    if (axis < 0) {
      axis += r;
    }
    if (axis != 0) {
      return false;
    }
    if (!TrailingDimsMatch(node->input(0), node->input(2), r)) {
      return false;
    }
    Value* indices = node->input(1);
    if (Node* e = ExpandProducer(indices)) {
      Value* x = e->input(0);
      const int32_t t = x->elemType();
      if (t != TensorProto_DataType_INT64 && t != TensorProto_DataType_INT32) {
        return false;
      }
      return ExpandSourceIsRowBroadcast(x, r);
    }
    const Tensor* c = FetchConstantTensor(indices);
    return c != nullptr && (c->elem_type() == TensorProto_DataType_INT32 ||
                            c->elem_type() == TensorProto_DataType_INT64);
  }

  // [N, 1] int64 row indices from the constant form, or nullptr if the
  // constant genuinely varies along a non-zero axis.
  static Value* RowIndicesFromConstant(const Tensor* t, int64_t r,
                                       Graph& graph) {
    const std::vector<int64_t>& shape = t->sizes();
    if (static_cast<int64_t>(shape.size()) != r) {
      return nullptr;
    }
    int64_t total = 1;
    for (int64_t s : shape) {
      total *= s;
    }
    std::vector<int64_t> vals;
    if (t->elem_type() == TensorProto_DataType_INT32) {
      const std::vector<int32_t> v32 = ParseTensorData<int32_t>(t);
      vals.assign(v32.begin(), v32.end());
    } else {
      vals = ParseTensorData<int64_t>(t);
    }
    if (total == 0 || static_cast<int64_t>(vals.size()) != total) {
      return nullptr;  // empty scatter (nothing to gain) or malformed
    }
    const int64_t n = shape[0];
    const int64_t row = total / n;  // elements per axis-0 row
    // Element i lies in row i / row; its representative is that row's first
    // element (every non-axis-0 coordinate zero), at flat offset
    // (i / row) * row.
    for (int64_t i = 0; i < total; ++i) {
      if (vals[static_cast<size_t>(i)] !=
          vals[static_cast<size_t>((i / row) * row)]) {
        return nullptr;
      }
    }
    Tensor out;
    out.elem_type() = TensorProto_DataType_INT64;
    out.sizes() = {n, 1};
    for (int64_t k = 0; k < n; ++k) {
      out.int64s().push_back(vals[static_cast<size_t>(k * row)]);
    }
    return graph.addInitializerAndCreateValue(std::move(out));
  }

  static Value* ConstI64(Graph& graph, std::vector<int64_t> v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(static_cast<int64_t>(v.size()));
    for (int64_t x : v) {
      t.int64s().push_back(x);
    }
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  static Value* Emit(Graph& graph, Node* anchor, Symbol kind,
                     std::vector<Value*> inputs, int32_t elem_type) {
    Node* n = graph.create(kind, 1);
    for (Value* v : inputs) {
      n->addInput(v);
    }
    n->insertBefore(anchor);
    n->output()->setElemType(elem_type);
    return n->output();
  }

  // [N, 1] int64 row indices from the Expand form.
  static Value* RowIndicesFromExpand(Node* expand, Node* anchor, int64_t r,
                                     Graph& graph) {
    Value* x = expand->input(0);
    Value* updates = anchor->input(2);
    Value* idx = x;
    if (x->elemType() == TensorProto_DataType_INT32) {
      Node* cast = graph.create(kCast, 1);
      cast->addInput(x);
      cast->i_(kto, static_cast<int64_t>(TensorProto_DataType_INT64));
      cast->insertBefore(anchor);
      cast->output()->setElemType(TensorProto_DataType_INT64);
      idx = cast->output();
    }
    idx = Emit(graph, anchor, kReshape, {idx, ConstI64(graph, {-1, 1})},
               TensorProto_DataType_INT64);

    // Skip the broadcast to [N, 1] only when X's axis-0 size is provably N.
    const auto& xs = x->sizes();
    const bool x_has_axis0 = static_cast<int64_t>(xs.size()) == r;
    const bool rows_known_equal = x_has_axis0 && updates->has_sizes() &&
                                  SameDim(xs[0], updates->sizes()[0]);
    if (!rows_known_equal) {
      Value* shp = Emit(graph, anchor, Symbol("Shape"), {updates},
                        TensorProto_DataType_INT64);
      Value* n = Emit(graph, anchor, kSlice,
                      {shp, ConstI64(graph, {0}), ConstI64(graph, {1}),
                       ConstI64(graph, {0})},
                      TensorProto_DataType_INT64);
      Node* cat = graph.create(kConcat, 1);
      cat->addInput(n);
      cat->addInput(ConstI64(graph, {1}));
      cat->i_(kaxis, 0);
      cat->insertBefore(anchor);
      cat->output()->setElemType(TensorProto_DataType_INT64);
      idx = Emit(graph, anchor, Symbol("Expand"), {idx, cat->output()},
                 TensorProto_DataType_INT64);
    }
    return idx;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const int64_t r = ScatterRank(node);
    Value* indices = node->input(1);

    Value* row_idx = nullptr;
    if (Node* e = ExpandProducer(indices)) {
      row_idx = RowIndicesFromExpand(e, node, r, graph);
    } else if (const Tensor* c = FetchConstantTensor(indices)) {
      row_idx = RowIndicesFromConstant(c, r, graph);
    }
    if (row_idx == nullptr) {
      return false;
    }

    Node* snd = graph.create(Symbol("ScatterND"), 1);
    snd->addInput(node->input(0));
    snd->addInput(row_idx);
    snd->addInput(node->input(2));
    snd->insertBefore(node);
    snd->output()->setElemType(node->output()->elemType());
    if (node->output()->has_sizes()) {
      snd->output()->setSizes(node->output()->sizes());
    }
    if (!tryReplacingAllUsesWith(node->output(), snd->output())) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
