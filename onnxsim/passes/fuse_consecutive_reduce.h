// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Y = ReduceSum(X, axes=A, keepdims=ka)
//   Z = ReduceSum(Y, axes=B, keepdims=kb)
// After:
//   Z = ReduceSum(X, axes=A ∪ B', keepdims=kb)
//
// where B' is B remapped back into X's index space (through the axes A
// already dropped, when ka == 0). This holds for every reduction op whose
// two-step application over disjoint axis groups equals one application over
// their union: Sum, Mean, Max, Min, Prod, L1, L2 and LogSumExp all reduce to
// "combine with the same binary op over the union of elements" once the two
// passes are unfolded algebraically (e.g. LogSumExp(B)(LogSumExp(A)(x)) =
// log(sum_B(exp(log(sum_A(exp(x)))))) = log(sum_B(sum_A(exp(x)))) =
// LogSumExp(A∪B)(x)), so grouping the axes differently changes nothing.
// ReduceLogSum and ReduceSumSquare do *not* have this property -- e.g.
// sum(x)^2 summed again is not sum(x^2) -- so they are intentionally
// excluded.
//
// The two reductions may additionally be separated by a Reshape that only
// inserts/removes the size-1 dimensions keepdims leaves behind -- the
// "reduce(keepdims=False) -> reshape back to the keepdims=True shape" idiom
// some exporters emit before a second reduction -- which this pass verifies
// against the static shapes involved and folds away as part of the same
// rewrite.

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseConsecutiveReduce final : public PredicateBasedPass {
  explicit FuseConsecutiveReduce()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_consecutive_reduce";
  }

  // Reduction ops whose two-step application over disjoint axis groups
  // equals a single application over the union of those axes. See the file
  // comment for why ReduceLogSum/ReduceSumSquare are excluded.
  static bool IsFusableReduceKind(NodeKind k) {
    static const std::unordered_set<NodeKind> kKinds{
        kReduceSum, kReduceMean, kReduceMax,      kReduceMin,
        kReduceProd, kReduceL1,  kReduceL2,       kReduceLogSumExp};
    return kKinds.count(k) != 0;
  }

  // Whether this opset carries a Reduce* node's axes as its second input
  // rather than an attribute: ReduceSum moved at opset 13, the rest at
  // opset 18. Used only as a fallback when neither operand being fused shows
  // the representation directly (see its call site).
  static bool AxesIsInputAtOpset(NodeKind k, int opset) {
    if (opset == 0) {
      return true;  // unknown opset: assume latest
    }
    return k == kReduceSum ? opset >= 13 : opset >= 18;
  }

  static int64_t GetKeepdims(Node* n) {
    int64_t keepdims = 1;
    GetValueFromAttr(n, kkeepdims, keepdims);
    return keepdims;
  }

  // Resolves a Reduce* node's actual reduction axes, normalized to
  // non-negative when `have_rank` is true, handling every way ONNX lets them
  // be spelled: an attribute, a constant input, omitted entirely (meaning
  // "all axes"), or an explicit empty list under noop_with_empty_axes
  // (meaning "no axes" -- the node is an identity, flagged via
  // `is_identity`). Returns false when the axes can't be determined (a
  // non-constant axes input, or negative axes with no rank to resolve them
  // against).
  static bool ResolveAxes(Node* node, bool have_rank, int64_t rank,
                          std::vector<int64_t>& axes, bool& is_identity) {
    is_identity = false;
    std::vector<int64_t> raw;
    bool omitted = false;
    if (node->hasAttribute(kaxes)) {
      raw = node->is(kaxes);
    } else if (node->inputs().size() > 1) {
      if (!GetValueFromInput(node, 1, raw)) {
        return false;  // non-constant axes: can't reason about them
      }
    } else {
      omitted = true;
    }

    if (omitted || raw.empty()) {
      if (!omitted) {
        int64_t noop = 0;
        GetValueFromAttr(node, "noop_with_empty_axes", noop);
        if (noop != 0) {
          axes.clear();
          is_identity = true;
          return true;
        }
      }
      if (!have_rank) {
        return false;
      }
      axes.resize(static_cast<size_t>(rank));
      std::iota(axes.begin(), axes.end(), int64_t{0});
      return true;
    }

    axes = raw;
    if (have_rank) {
      for (auto& a : axes) {
        a = AddYIfNegative(a, rank);
      }
    } else {
      for (int64_t a : axes) {
        if (a < 0) {
          return false;
        }
      }
    }
    return true;
  }

  // The shape a Reduce* over `in_shape` with (normalized, sorted, deduped)
  // `axes` produces, for a given keepdims setting.
  static std::vector<Dimension> ReducedShape(
      const std::vector<Dimension>& in_shape,
      const std::vector<int64_t>& axes, bool keepdims) {
    std::vector<Dimension> out;
    out.reserve(in_shape.size());
    size_t ai = 0;
    for (size_t i = 0; i < in_shape.size(); ++i) {
      if (ai < axes.size() && axes[ai] == static_cast<int64_t>(i)) {
        ++ai;
        if (keepdims) {
          out.push_back(Dimension(int64_t{1}));
        }
      } else {
        out.push_back(in_shape[i]);
      }
    }
    return out;
  }

  static bool SameDim(const Dimension& a, const Dimension& b) {
    if (a.is_int && b.is_int) {
      return a.dim == b.dim;
    }
    if (!a.is_int && !b.is_int) {
      return !a.is_unknown && !b.is_unknown && a.param == b.param;
    }
    return false;
  }

  static bool SameShape(const std::vector<Dimension>& a,
                        const std::vector<Dimension>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!SameDim(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  // Finds the `idx`-th (0-based, ascending) value not present in the sorted
  // `removed` list -- i.e. inverts "drop these positions" to map an index
  // into the post-drop space back into the original one. Needs no upper
  // bound on the index space, so it works without knowing the original rank
  // as long as `removed` and `idx` are themselves non-negative.
  static int64_t MapThroughRemovedAxes(const std::vector<int64_t>& removed,
                                       int64_t idx) {
    int64_t orig = 0, count = 0;
    size_t ri = 0;
    while (true) {
      while (ri < removed.size() && removed[ri] == orig) {
        ++ri;
        ++orig;
      }
      if (count == idx) {
        return orig;
      }
      ++count;
      ++orig;
    }
  }

  bool patternMatchPredicate(Node* node) override {
    if (!IsFusableReduceKind(node->kind())) {
      return false;
    }
    Node* prev = node->input(0)->node();
    if (prev->kind() == node->kind()) {
      return prev->output()->uses().size() == 1;
    }
    if (prev->kind() == kReshape) {
      if (prev->output()->uses().size() != 1) {
        return false;
      }
      Node* prev2 = prev->input(0)->node();
      return prev2->kind() == node->kind() &&
             prev2->output()->uses().size() == 1;
    }
    return false;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* reshape = nullptr;
    Node* a = n->input(0)->node();
    if (a->kind() == kReshape) {
      reshape = a;
      a = reshape->input(0)->node();
    }
    // patternMatchPredicate only lets same-kind reduces (directly, or
    // through a single-use Reshape) reach here; re-check defensively.
    if (a->kind() != n->kind()) {
      return false;
    }

    Value* x = a->input(0);
    const bool have_x_rank = x->has_sizes();
    const int64_t x_rank =
        have_x_rank ? static_cast<int64_t>(x->sizes().size()) : 0;

    std::vector<int64_t> a_axes;
    bool a_identity = false;
    if (!ResolveAxes(a, have_x_rank, x_rank, a_axes, a_identity) ||
        a_identity) {
      return false;
    }
    std::sort(a_axes.begin(), a_axes.end());
    a_axes.erase(std::unique(a_axes.begin(), a_axes.end()), a_axes.end());

    const int64_t ka = GetKeepdims(a);

    // The keepdims value `n` effectively sees `a`'s output through: `ka`
    // directly, or its opposite when the Reshape in between is exactly the
    // keepdims-toggling reshape the "reduce(keepdims=False) -> reshape back"
    // idiom produces (verified against the actual static shapes).
    int64_t effective_ka = ka;
    if (reshape != nullptr) {
      if (!have_x_rank || !reshape->output()->has_sizes()) {
        return false;
      }
      effective_ka = ka != 0 ? 0 : 1;
      const std::vector<Dimension> toggled =
          ReducedShape(x->sizes(), a_axes, effective_ka != 0);
      if (!SameShape(toggled, reshape->output()->sizes())) {
        return false;
      }
    }

    const bool have_mid_rank = have_x_rank;
    const int64_t mid_rank =
        have_x_rank ? (effective_ka != 0
                          ? x_rank
                          : x_rank - static_cast<int64_t>(a_axes.size()))
                    : 0;
    std::vector<int64_t> n_axes;
    bool n_identity = false;
    if (!ResolveAxes(n, have_mid_rank, mid_rank, n_axes, n_identity) ||
        n_identity) {
      return false;
    }
    std::sort(n_axes.begin(), n_axes.end());
    n_axes.erase(std::unique(n_axes.begin(), n_axes.end()), n_axes.end());

    // Map n's axes (relative to a's output) back into x's index space.
    std::vector<int64_t> mapped_n_axes;
    mapped_n_axes.reserve(n_axes.size());
    if (effective_ka != 0) {
      // Same rank as x: indices carry over unchanged.
      mapped_n_axes = n_axes;
    } else {
      // a's output dropped a_axes entirely; invert that.
      for (int64_t idx : n_axes) {
        mapped_n_axes.push_back(MapThroughRemovedAxes(a_axes, idx));
      }
    }

    std::vector<int64_t> fused_axes = a_axes;
    fused_axes.insert(fused_axes.end(), mapped_n_axes.begin(),
                      mapped_n_axes.end());
    std::sort(fused_axes.begin(), fused_axes.end());
    fused_axes.erase(std::unique(fused_axes.begin(), fused_axes.end()),
                     fused_axes.end());

    const int64_t kb = GetKeepdims(n);

    Node* fused = graph.create(n->kind(), n->outputs().size());
    fused->addInput(x);

    bool axes_is_input;
    if (n->hasAttribute(kaxes)) {
      axes_is_input = false;
    } else if (n->inputs().size() > 1) {
      axes_is_input = true;
    } else {
      axes_is_input = AxesIsInputAtOpset(n->kind(), getOpsetVersion(graph));
    }
    if (axes_is_input) {
      Tensor axes_t;
      axes_t.sizes().push_back(static_cast<int64_t>(fused_axes.size()));
      axes_t.elem_type() = TensorProto_DataType_INT64;
      axes_t.int64s().assign(fused_axes.begin(), fused_axes.end());
      fused->addInput(graph.addInitializerAndCreateValue(axes_t));
    } else {
      fused->is_(kaxes, std::move(fused_axes));
    }
    fused->i_(kkeepdims, kb);

    for (size_t i = 0; i < n->outputs().size(); ++i) {
      fused->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    fused->insertBefore(n);
    if (!tryReplacingAllUsesWith(n, fused)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
