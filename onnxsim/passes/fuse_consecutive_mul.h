// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Y = Mul(X, C1)
//   Z = Mul(Y, C2)
// After:
//   Z = Mul(X, C1 * C2)
//
// where C1 and C2 are constants (initializers or Constant nodes) and the inner
// Mul (producing Y) is used only by the outer Mul. C1 * C2 is materialised as a
// new constant using numpy-style broadcasting, so the rewrite is numerically
// equivalent: at every output position the value is X * C1 * C2 regardless of
// how the two constant scales are grouped.
//
// This shows up in models that export a composed constant scale as two
// back-to-back Mul nodes, e.g. PoolFormer's LayerScale (a per-channel (C,1,1)
// scale multiplied by a scalar factor).

#include <functional>
#include <numeric>
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

struct FuseConsecutiveMul final : public PredicateBasedPass {
  explicit FuseConsecutiveMul()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_consecutive_mul"; }

  // Returns the index (0 or 1) of the single constant operand of a 2-input
  // node, or -1 when the node does not have exactly one constant operand.
  static int SoleConstantInput(const Node* n) {
    if (n->inputs().size() != 2) {
      return -1;
    }
    const bool c0 = IsConstantTensor(n->input(0));
    const bool c1 = IsConstantTensor(n->input(1));
    if (c0 && !c1) {
      return 0;
    }
    if (c1 && !c0) {
      return 1;
    }
    return -1;
  }

  // numpy-style broadcast of two shapes. Returns false when incompatible.
  static bool BroadcastShapes(const std::vector<int64_t>& a,
                              const std::vector<int64_t>& b,
                              std::vector<int64_t>& out) {
    const size_t ra = a.size();
    const size_t rb = b.size();
    const size_t r = std::max(ra, rb);
    out.assign(r, 1);
    for (size_t i = 0; i < r; ++i) {
      const int64_t da = i < r - ra ? 1 : a[i - (r - ra)];
      const int64_t db = i < r - rb ? 1 : b[i - (r - rb)];
      if (da == db || db == 1) {
        out[i] = da == 1 ? db : da;
      } else if (da == 1) {
        out[i] = db;
      } else {
        return false;
      }
    }
    return true;
  }

  // Elementwise multiply of two flat buffers with numpy broadcasting, given the
  // (right-aligned) input shapes and the pre-computed output shape.
  template <typename T>
  static std::vector<T> BroadcastMul(const std::vector<T>& va,
                                     const std::vector<int64_t>& sa,
                                     const std::vector<T>& vb,
                                     const std::vector<int64_t>& sb,
                                     const std::vector<int64_t>& so) {
    const size_t r = so.size();
    const int64_t total = std::accumulate(so.begin(), so.end(), (int64_t)1,
                                          std::multiplies<int64_t>{});
    // Strides into each input buffer, right-aligned to the output rank; a
    // broadcast (size-1) dimension gets stride 0.
    auto build_strides = [r](const std::vector<int64_t>& s) {
      std::vector<int64_t> full(r, 1);
      for (size_t i = 0; i < s.size(); ++i) {
        full[r - s.size() + i] = s[i];
      }
      std::vector<int64_t> stride(r, 0);
      int64_t acc = 1;
      for (size_t i = r; i-- > 0;) {
        stride[i] = full[i] == 1 ? 0 : acc;
        acc *= full[i];
      }
      return stride;
    };
    const std::vector<int64_t> stra = build_strides(sa);
    const std::vector<int64_t> strb = build_strides(sb);

    std::vector<T> out(total);
    std::vector<int64_t> idx(r, 0);
    for (int64_t lin = 0; lin < total; ++lin) {
      int64_t ia = 0, ib = 0;
      for (size_t d = 0; d < r; ++d) {
        ia += idx[d] * stra[d];
        ib += idx[d] * strb[d];
      }
      out[lin] = va[ia] * vb[ib];
      for (size_t d = r; d-- > 0;) {
        if (++idx[d] < so[d]) {
          break;
        }
        idx[d] = 0;
      }
    }
    return out;
  }

  // Builds C1 * C2 into `out`. Only FLOAT/DOUBLE are handled; other dtypes are
  // left to constant folding and this pass conservatively declines them.
  static bool MultiplyConstants(const Tensor& c1, const Tensor& c2,
                                Tensor& out) {
    if (c1.elem_type() != c2.elem_type()) {
      return false;
    }
    std::vector<int64_t> so;
    if (!BroadcastShapes(c1.sizes(), c2.sizes(), so)) {
      return false;
    }
    out.elem_type() = c1.elem_type();
    out.sizes() = so;
    switch (c1.elem_type()) {
      case TensorProto_DataType_FLOAT:
        out.floats() =
            BroadcastMul(ParseTensorData<float>(&c1), c1.sizes(),
                         ParseTensorData<float>(&c2), c2.sizes(), so);
        return true;
      case TensorProto_DataType_DOUBLE:
        out.doubles() =
            BroadcastMul(ParseTensorData<double>(&c1), c1.sizes(),
                         ParseTensorData<double>(&c2), c2.sizes(), so);
        return true;
      default:
        return false;
    }
  }

  bool patternMatchPredicate(Node* node) override {
    if (!CheckKind(node, kMul)) {
      return false;
    }
    const int outer_c = SoleConstantInput(node);
    if (outer_c < 0) {
      return false;
    }
    const Value* other = node->input(1 - outer_c);
    if (!CheckKind(other, kMul)) {
      return false;
    }
    // inner Mul must feed only this outer Mul
    if (other->uses().size() != 1) {
      return false;
    }
    return SoleConstantInput(other->node()) >= 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const int outer_c = SoleConstantInput(n);
    Value* c2v = n->input(outer_c);
    Node* inner = n->input(1 - outer_c)->node();
    const int inner_c = SoleConstantInput(inner);
    if (inner_c < 0) {
      return false;
    }
    Value* c1v = inner->input(inner_c);
    Value* xv = inner->input(1 - inner_c);

    const Tensor* c1 = FetchConstantTensor(c1v);
    const Tensor* c2 = FetchConstantTensor(c2v);
    if (c1 == nullptr || c2 == nullptr) {
      return false;
    }

    Tensor fused_const;
    if (!MultiplyConstants(*c1, *c2, fused_const)) {
      return false;
    }
    Value* cv = graph.addInitializerAndCreateValue(fused_const);

    Node* fused = graph.create(kMul, n->outputs().size());
    fused->addInput(xv);
    fused->addInput(cv);
    for (int i = 0; i < static_cast<int>(n->outputs().size()); ++i) {
      fused->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    fused->insertBefore(n);
    if (!tryReplacingAllUsesWith(n, fused)) {
      return false;
    }
    // Destroy the outer Mul; DCE removes the now-dead inner Mul and constants.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
