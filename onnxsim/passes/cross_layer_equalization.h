// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Cross-Layer Equalization (CLE) -- the data-free weight-equalization
// technique from "Data-Free Quantization Through Weight Equalization and
// Bias Correction" (Nagel et al., 2019) and shipped as part of Qualcomm's
// AIMET toolkit. It is a *preprocessing* step for quantization, not a
// quantization scheme itself: run it before a quantize_* pass (this file has
// no `_quantize_` in its own pass name and never introduces a
// Quantize/DequantizeLinear node), to make the surrounding per-tensor or
// per-channel quantization that follows more accurate.
//
// The problem it addresses: two adjacent Conv layers can have very different
// per-channel weight magnitudes (e.g. one channel with a 100x larger range
// than its neighbors), which forces a shared quantization scale to either
// waste most of its resolution on the small channels or clip the large one.
// CLE removes that imbalance losslessly, for a fixed pair of layers
// Conv1 -> [activation] -> Conv2 where the activation (if any) is positive-
// homogeneous of degree 1 -- f(a*x) = a*f(x) for every a > 0, true of Relu,
// PRelu, and LeakyRelu (and trivially true of "no activation at all") --
// by picking, per shared channel c, a scale
//
//   S[c] = sqrt(r1[c] / r2[c])
//
// where r1[c] = max(|Conv1's output-channel-c weights|) and
// r2[c] = max(|Conv2's input-channel-c weights|), then rewriting
//
//   W1'[c, ...] = W1[c, ...] / S[c];  b1'[c] = b1[c] / S[c]
//   W2'[:, c, ...] = W2[:, c, ...] * S[c]
//
// Because the activation between them commutes with a positive scale,
// Conv1'(x) = Conv1(x) / S elementwise per channel, the activation passes
// that /S through unchanged, and Conv2' undoes it exactly (* S) -- so the
// composed function Conv1 -> activation -> Conv2 is *exactly* unchanged
// (same as BatchNorm folding, not an approximation), while r1'[c] == r2'[c]
// for every c: the two layers' per-channel ranges are now identical, which
// is the most balanced a fixed pair can be made.
//
// Scope of this implementation (documented limitations, not correctness
// bugs -- each just declines the match rather than mishandling it):
//   - Conv only (no ConvTranspose, no Gemm/MatMul-based fully-connected
//     equalization -- AIMET supports FC layers too, via the same math on a
//     2-D weight; not implemented here).
//   - `group` must be 1 on both convs -- a grouped/depthwise conv's input
//     and output channels don't correspond 1:1, which this pass's channel
//     bookkeeping assumes.
//   - FLOAT32 weights/bias only (mirrors fuse_bn_into_conv.h's own
//     "currently works for only DOUBLE, FLOAT32" scope, minus DOUBLE).
//   - No "high-bias absorption" (AIMET's optional follow-up step that moves
//     part of Conv1's post-activation bias forward into a following
//     BatchNorm's bias when equalization alone leaves Conv1's activations
//     with an unusually large bias). Plain BN folding (fuse_bn_into_conv)
//     upstream of this pass covers the common case fine on its own.
//
// Iterates to a network-wide fixed point for free: onnxsim's
// FixedPointPassManager reruns every registered pass, including this one,
// until no pass reports a change (see custom_optimizer_passes.cpp), so
// equalizing (Conv1, Conv2) and then, in the same fixed-point sweep,
// (Conv2, Conv3) naturally propagates balance across a whole chain of
// layers -- exactly what the paper's own multi-round equalization does --
// without this file needing its own outer iteration loop.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct CrossLayerEqualization final : public PredicateBasedPass {
  explicit CrossLayerEqualization()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "cross_layer_equalization";
  }

  // A matched, fully-validated Conv1 -> [activation] -> Conv2 chain, ready
  // for RescaleAndApply() below. `activation` is nullptr when Conv1 feeds
  // Conv2 directly (also valid: the identity function is positive-
  // homogeneous too).
  struct Match {
    Node* conv1 = nullptr;
    Node* conv2 = nullptr;
    Node* activation = nullptr;
    const Tensor* w1 = nullptr;
    const Tensor* w2 = nullptr;
    const Tensor* b1 = nullptr;  // nullptr if conv1 has no bias input
  };

  static bool IsPassThroughActivation(Node* n) {
    return CheckKind(n, "Relu") || CheckKind(n, "PRelu") ||
           CheckKind(n, "LeakyRelu");
  }

  // Conv weight layout is [Cout, Cin/group, k...]; only group == 1 gives a
  // clean 1:1 channel correspondence with the adjacent conv, which is what
  // this pass's per-channel bookkeeping assumes throughout.
  static bool HasSingleGroup(Node* conv) {
    return GetValueFromAttrWithDefault(conv, "group", (int64_t)1) == 1;
  }

  // Validates `conv`'s weight (and bias, if present) are constant FLOAT32
  // tensors of the right rank, filling `w_out`/`b_out`. Shared by both
  // Conv1 and Conv2's checks below -- they need the identical validation,
  // just applied to different nodes.
  static bool ValidateConvWeights(Node* conv, const Tensor*& w_out,
                                  const Tensor*& b_out) {
    const size_t num_inputs = conv->inputs().size();
    if (conv->kind() != kConv || (num_inputs != 2 && num_inputs != 3)) {
      return false;
    }
    if (!HasSingleGroup(conv) || !IsConstantTensor(conv, 1)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(conv->inputs()[1]);
    if (w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 2) {
      return false;
    }
    const Tensor* b_t = nullptr;
    if (num_inputs == 3) {
      if (!IsConstantTensor(conv, 2)) {
        return false;
      }
      b_t = FetchConstantTensor(conv->inputs()[2]);
      if (b_t->elem_type() != TensorProto_DataType_FLOAT) {
        return false;
      }
    }
    w_out = w_t;
    b_out = b_t;
    return true;
  }

  // `conv2` is the node PredicateBasedPass's graph walk hands us; CLE always
  // matches on the *second* conv of a pair and looks backward, the same
  // shape as fuse_bn_into_conv.h matching on BatchNormalization and walking
  // back to its producing Conv.
  static bool TryMatch(Node* conv2, Match& m) {
    const Tensor* w2_t = nullptr;
    const Tensor* unused_b2 = nullptr;
    if (!ValidateConvWeights(conv2, w2_t, unused_b2)) {
      return false;
    }

    // conv2's activation input must have no other consumer -- otherwise
    // rescaling what feeds conv2 would change that other consumer's input
    // too, which is not this pass's to alter.
    Value* feed = conv2->inputs()[0];
    if (feed->uses().size() != 1) {
      return false;
    }
    Node* producer = feed->node();
    Node* activation = nullptr;
    Node* conv1 = producer;
    if (producer->kind() != kConv) {
      if (!IsPassThroughActivation(producer) || producer->inputs().empty()) {
        return false;
      }
      Value* act_in = producer->inputs()[0];
      // Same reasoning as above, one hop further back: conv1's output must
      // feed only this activation.
      if (act_in->uses().size() != 1) {
        return false;
      }
      activation = producer;
      conv1 = act_in->node();
    }

    const Tensor* w1_t = nullptr;
    const Tensor* b1_t = nullptr;
    if (!ValidateConvWeights(conv1, w1_t, b1_t)) {
      return false;
    }

    // Channel correspondence: conv1's output channels (w1's axis 0) are
    // conv2's input channels (w2's axis 1, since group == 1 on both).
    // Already implied by the graph being a valid ONNX model, but checked
    // directly rather than assumed.
    if (w1_t->sizes()[0] != w2_t->sizes()[1]) {
      return false;
    }

    m.conv1 = conv1;
    m.conv2 = conv2;
    m.activation = activation;
    m.w1 = w1_t;
    m.w2 = w2_t;
    m.b1 = b1_t;
    return true;
  }

  // Computes the per-channel scale and, if any channel needs a meaningful
  // rescale, rewrites conv1's weight/bias and conv2's weight in place.
  // Returns false (no graph change) when the pair is already balanced --
  // this is what lets the fixed-point driver converge instead of looping
  // forever re-applying a no-op: a single application already makes
  // r1'[c] == r2'[c] exactly for a fixed pair (see this file's own top
  // comment), so a channel only needs rescaling again after a *different*
  // pass application changed one of its neighbors.
  static bool RescaleAndApply(const Match& m, Graph& graph) {
    const auto& s1 = m.w1->sizes();  // [C, Cin1, k...]
    const auto& s2 = m.w2->sizes();  // [Cout2, C, k...]
    const int64_t C = s1[0];
    int64_t inner1 = 1;
    for (size_t i = 1; i < s1.size(); ++i) {
      inner1 *= s1[i];
    }
    const int64_t outer2 = s2[0];
    int64_t inner2 = 1;
    for (size_t i = 2; i < s2.size(); ++i) {
      inner2 *= s2[i];
    }

    std::vector<float> w1 = ReadFloatTensorFlat(*m.w1);
    std::vector<float> w2 = ReadFloatTensorFlat(*m.w2);
    std::vector<float> b1 =
        m.b1 != nullptr ? ReadFloatTensorFlat(*m.b1) : std::vector<float>();

    // r1[c] = max(|W1[c, ...]|) -- conv1's own per-output-channel range.
    std::vector<float> r1(static_cast<size_t>(C), 0.0f);
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t j = 0; j < inner1; ++j) {
        r1[static_cast<size_t>(c)] =
            std::max(r1[static_cast<size_t>(c)],
                     std::fabs(w1[static_cast<size_t>(c * inner1 + j)]));
      }
    }
    // r2[c] = max(|W2[:, c, ...]|) -- conv2's per-*input*-channel range.
    std::vector<float> r2(static_cast<size_t>(C), 0.0f);
    for (int64_t o = 0; o < outer2; ++o) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t j = 0; j < inner2; ++j) {
          const int64_t idx = o * C * inner2 + c * inner2 + j;
          r2[static_cast<size_t>(c)] =
              std::max(r2[static_cast<size_t>(c)],
                       std::fabs(w2[static_cast<size_t>(idx)]));
        }
      }
    }

    // A channel with either range at 0 (a dead/all-zero weight slice) is
    // left unscaled (S=1) -- there is no meaningful ratio to balance, and
    // dividing by 0 would poison the whole channel with inf/NaN.
    std::vector<float> s(static_cast<size_t>(C), 1.0f);
    bool any_change = false;
    constexpr float kConvergedTol = 1e-3f;
    for (int64_t c = 0; c < C; ++c) {
      const float r1c = r1[static_cast<size_t>(c)];
      const float r2c = r2[static_cast<size_t>(c)];
      if (r1c > 0.0f && r2c > 0.0f) {
        const float sc = std::sqrt(r1c / r2c);
        s[static_cast<size_t>(c)] = sc;
        if (std::fabs(sc - 1.0f) > kConvergedTol) {
          any_change = true;
        }
      }
    }
    if (!any_change) {
      return false;
    }

    for (int64_t c = 0; c < C; ++c) {
      const float sc = s[static_cast<size_t>(c)];
      for (int64_t j = 0; j < inner1; ++j) {
        w1[static_cast<size_t>(c * inner1 + j)] /= sc;
      }
      if (m.b1 != nullptr) {
        b1[static_cast<size_t>(c)] /= sc;
      }
    }
    for (int64_t o = 0; o < outer2; ++o) {
      for (int64_t c = 0; c < C; ++c) {
        const float sc = s[static_cast<size_t>(c)];
        for (int64_t j = 0; j < inner2; ++j) {
          const int64_t idx = o * C * inner2 + c * inner2 + j;
          w2[static_cast<size_t>(idx)] *= sc;
        }
      }
    }

    Tensor w1_new;
    w1_new.elem_type() = TensorProto_DataType_FLOAT;
    w1_new.sizes() = s1;
    w1_new.floats() = std::move(w1);
    Value* w1_value = graph.addInitializerAndCreateValue(w1_new);
    Value* old_w1_value = m.conv1->inputs()[1];
    m.conv1->replaceInput(1, w1_value);
    if (old_w1_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_w1_value);
    }

    if (m.b1 != nullptr) {
      Tensor b1_new;
      b1_new.elem_type() = TensorProto_DataType_FLOAT;
      b1_new.sizes() = m.b1->sizes();
      b1_new.floats() = std::move(b1);
      Value* b1_value = graph.addInitializerAndCreateValue(b1_new);
      Value* old_b1_value = m.conv1->inputs()[2];
      m.conv1->replaceInput(2, b1_value);
      if (old_b1_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_b1_value);
      }
    }

    Tensor w2_new;
    w2_new.elem_type() = TensorProto_DataType_FLOAT;
    w2_new.sizes() = s2;
    w2_new.floats() = std::move(w2);
    Value* w2_value = graph.addInitializerAndCreateValue(w2_new);
    Value* old_w2_value = m.conv2->inputs()[1];
    m.conv2->replaceInput(1, w2_value);
    if (old_w2_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_w2_value);
    }

    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    Match m;
    return TryMatch(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // Neither conv is ever destroyed -- only their weight/bias *inputs*
    // change, same as fuse_mul_into_conv.h's own DestroyZero convention for
    // an in-place rewrite that removes no node.
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!TryMatch(n, m)) {
      return false;
    }
    return RescaleAndApply(m, graph);
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
