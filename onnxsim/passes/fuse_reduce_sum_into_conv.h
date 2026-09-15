// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Before:
//   Y = ReduceSum(X, axes=[2, 3, ..., rank-1], keepdims=k)  // X: [N, C, d2, ..., d_{rank-1}]
// After:
//   C0 = Conv(X, W)  // W: ones([C, 1, d2, ..., d_{rank-1}]), group=C,
//                     // kernel_shape=[d2, ..., d_{rank-1}], strides=1,
//                     // pads=0, dilations=1 -> output [N, C, 1, ..., 1]
//   Y  = k != 0 ? C0 : Squeeze(C0, axes=[2, ..., rank-1])
//
// Only a *full* spatial reduction qualifies -- axes must be exactly every
// axis from 2 to rank-1 (N and C, axes 0 and 1, are always left alone). A
// depthwise (group=C) Conv whose kernel covers the entire remaining spatial
// extent computes precisely that: with stride 1 and no padding, each output
// channel co reads only input channel co (group=C keeps channels from ever
// mixing) and its single output position per spatial axis is the sum of
// every element in that axis's full extent -- i.e. exactly
// ReduceSum(axes=[2..rank-1]) computed per channel. This is the same trick
// GlobalAveragePool-as-Conv relies on, specialized to Sum instead of Mean (no
// division) and requiring the reduced axes to be static so the ones weight
// and kernel_shape can be materialized.
//
// Rationale: mirrors fuse_matmul_into_conv's own rationale -- some
// accelerators (embedded/mobile NPUs and DSPs among them) ship a heavily
// tuned, general-purpose Conv datapath but a much weaker (or altogether
// unaccelerated) generic Reduce one. Every full-spatial-extent ReduceSum
// (e.g. a CNN's classifier head doing global sum pooling before a final
// Linear layer) is mathematically a depthwise convolution against an
// all-ones kernel, so rewriting it that way lets such backends keep the
// whole network on their fast Conv engine instead of falling back to a slow
// (or CPU) path for the reduction.
//
// This is a graph-shape rewrite, not a node-count reduction -- it also
// materializes a size C * prod(kernel_shape) all-ones weight tensor, which
// can be large for a big spatial extent -- and is a *regression* on a
// Reduce-first backend, so, like fuse_matmul_into_conv, it is registered as
// PassType::Other (opt-in only, via ``extra_optimizers``), not part of the
// default fuse set.
//
// Only ReduceSum is handled (not ReduceMean, which would additionally need
// to divide by the reduced extent -- either a rescaled 1/N weight, losing the
// exactness of an all-ones kernel, or a trailing Div/Mul this pass does not
// insert). Dtype is restricted to float32, matching Conv's typical
// accelerated path.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseReduceSumIntoConv final : public PredicateBasedPass {
  explicit FuseReduceSumIntoConv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_reduce_sum_into_conv";
  }

  // The pieces of a qualifying ReduceSum this pass rewrites.
  struct Match {
    bool ok = false;
    std::vector<int64_t> kernel_shape;  // size == rank - 2, all > 0
    int64_t channels = 0;               // X's axis-1 size, static
    int64_t keepdims = 1;
  };

  static Match MatchNode(Node* node) {
    Match m;
    if (node->kind() != kReduceSum) {
      return m;
    }

    Value* x = node->input(0);
    if (!x->has_sizes()) {
      return m;
    }
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return m;
    }
    const auto& x_sizes = x->sizes();
    const int64_t rank = static_cast<int64_t>(x_sizes.size());
    // Need at least one spatial axis (2) beyond N (0) and C (1).
    if (rank < 3) {
      return m;
    }

    // Resolve the reduction axes: an attribute (opset < 13), a constant
    // input (opset >= 13), or omitted entirely (meaning "all axes"). A
    // non-constant axes input can't be reasoned about statically.
    std::vector<int64_t> axes;
    if (node->hasAttribute(kaxes)) {
      axes = node->is(kaxes);
    } else if (node->inputs().size() > 1) {
      if (!GetValueFromInput(node, 1, axes)) {
        return m;
      }
    } else {
      axes.resize(static_cast<size_t>(rank));
      std::iota(axes.begin(), axes.end(), int64_t{0});
    }
    for (auto& a : axes) {
      a = AddYIfNegative(a, rank);
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

    // Must be exactly the full spatial range [2, rank) -- N and C are never
    // touched, matching what a depthwise Conv can express (see file
    // comment). This also rejects an empty axes list (whether from an
    // explicitly-empty axes input or a noop_with_empty_axes identity),
    // since rank >= 3 makes that range non-empty.
    if (static_cast<int64_t>(axes.size()) != rank - 2) {
      return m;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(axes.size()); ++i) {
      if (axes[static_cast<size_t>(i)] != i + 2) {
        return m;
      }
    }

    // Channel count (axis 1) and every spatial dim must be statically known
    // to materialize the depthwise ones-weight and Conv's kernel_shape.
    if (!x_sizes[1].is_int || x_sizes[1].dim <= 0) {
      return m;
    }
    std::vector<int64_t> kernel_shape;
    kernel_shape.reserve(static_cast<size_t>(rank - 2));
    for (int64_t i = 2; i < rank; ++i) {
      const Dimension& d = x_sizes[static_cast<size_t>(i)];
      if (!d.is_int || d.dim <= 0) {
        return m;
      }
      kernel_shape.push_back(d.dim);
    }

    int64_t keepdims = 1;
    GetValueFromAttr(node, kkeepdims, keepdims);

    m.ok = true;
    m.kernel_shape = std::move(kernel_shape);
    m.channels = x_sizes[1].dim;
    m.keepdims = keepdims;
    return m;
  }

  bool patternMatchPredicate(Node* node) override {
    return MatchNode(node).ok;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const Match match = MatchNode(n);
    if (!match.ok) {
      return false;
    }

    Value* x = n->input(0);
    const int64_t rank = static_cast<int64_t>(x->sizes().size());
    const int64_t num_spatial = rank - 2;
    const int64_t channels = match.channels;

    int64_t kernel_numel = 1;
    for (int64_t k : match.kernel_shape) {
      kernel_numel *= k;
    }

    // W = ones([C, 1, d2, ..., d_{rank-1}]) -- depthwise (group=C) so the
    // Conv sums each channel's full spatial extent independently, never
    // mixing channels, exactly matching ReduceSum's per-channel semantics.
    Tensor w_t;
    w_t.elem_type() = TensorProto_DataType_FLOAT;
    w_t.sizes() = {channels, 1};
    w_t.sizes().insert(w_t.sizes().end(), match.kernel_shape.begin(),
                       match.kernel_shape.end());
    std::vector<float> ones(static_cast<size_t>(channels * kernel_numel),
                            1.0f);
    w_t.set_raw_data(WriteRawDataLittleEndian(ones));
    // No name set: addInitializerAndCreateValue(Tensor&&) auto-assigns a
    // fresh, guaranteed-unique one when empty.
    Value* w = graph.addInitializerAndCreateValue(std::move(w_t));

    Node* conv = graph.create(kConv, 1);
    conv->addInput(x);
    conv->addInput(w);
    conv->is_(kkernel_shape, match.kernel_shape);
    conv->is_(kstrides,
              std::vector<int64_t>(static_cast<size_t>(num_spatial), 1));
    conv->is_(kpads,
              std::vector<int64_t>(static_cast<size_t>(num_spatial) * 2, 0));
    conv->is_(kdilations,
              std::vector<int64_t>(static_cast<size_t>(num_spatial), 1));
    conv->i_(kgroup, channels);
    conv->insertBefore(n);
    conv->output()->setElemType(x->elemType());

    Value* result;
    if (match.keepdims != 0) {
      // Conv's output is already exactly ReduceSum(keepdims=1)'s shape
      // ([N, C, 1, ..., 1]): kernel == full remaining extent, stride 1, no
      // padding, so every spatial output dim collapses to 1.
      conv->output()->copyMetadata(n->output());
      result = conv->output();
    } else {
      std::vector<Dimension> conv_sizes = {x->sizes()[0], Dimension(channels)};
      for (int64_t i = 0; i < num_spatial; ++i) {
        conv_sizes.push_back(Dimension(int64_t{1}));
      }
      conv->output()->setSizes(conv_sizes);

      std::vector<int64_t> squeeze_axes(static_cast<size_t>(num_spatial));
      std::iota(squeeze_axes.begin(), squeeze_axes.end(), int64_t{2});

      Node* squeeze = graph.create(kSqueeze, 1);
      squeeze->addInput(conv->output());
      const int opset = getOpsetVersion(graph);
      if (opset < 13 && opset != 0) {
        squeeze->is_(kaxes, squeeze_axes);
      } else {
        Tensor axes_t;
        axes_t.elem_type() = TensorProto_DataType_INT64;
        axes_t.sizes().push_back(static_cast<int64_t>(squeeze_axes.size()));
        axes_t.int64s().assign(squeeze_axes.begin(), squeeze_axes.end());
        squeeze->addInput(graph.addInitializerAndCreateValue(std::move(axes_t)));
      }
      squeeze->insertBefore(n);
      squeeze->output()->copyMetadata(n->output());
      result = squeeze->output();
    }

    if (!tryReplacingAllUsesWith(n, result->node())) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
