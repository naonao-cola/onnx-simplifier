// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// QuaRot (Ashkboos et al., 2024, "QuaRot: Outlier-Free 4-Bit Inference in
// Rotated LLMs") -- C++ port of quarot.py's own apply_quarot. See that
// module's docstring for the full rationale (rotating the residual stream
// by a random orthogonal matrix removes activation outliers the same way
// it already removes weight outliers, letting *both* MatMul operands drop
// to INT4 with plain round-to-nearest and no calibration data at all).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C carried through unchanged):
//   Y = MatMul(X, W) [+ bias]                 W constant, [K, N], float32
// After:
//   U: initializer, float32 [K, K]            -- the random rotation
//   Xrot = MatMul(X, U)                       -- runtime activation rotation
//   Xq   = round_to_nearest_int4_per_token(Xrot)   -- data-free, no calibration
//   Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
//                                               -- INT4 codes, [K, N]
//   Y = MatMul(Xq, Wtilde_hat) [+ bias]
//
// Unlike every weight-only scheme in this directory, this pass replaces the
// *entire* matched node (not just its weight input): the activation path
// changes too. Only the common, unambiguous shape is handled: a MatMul, or
// a Gemm with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
// constant 2-D float32 tensor whose reduction dimension K is divisible by
// kBlockSize, on an opset >= 21 model (INT4 tensors and DequantizeLinear's
// block_size attribute both need it). Everything else is left alone.
//
// Scope note: this reuses a fresh Haar-random rotation per matched layer
// (random_orthogonal.h), matching quarot.py's/spinquant.py's own scope --
// not the real QuaRot's single rotation fused across an entire decoder
// stack (see quarot.py's own docstring for why: fusing one global rotation
// needs a model-level residual-stream graph walk this port does not
// attempt).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/random_orthogonal.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Seed for the per-layer random rotations. A function-local static, the
// same pattern MagnitudePruningSparsity() (magnitude_pruning.h) uses to
// pass a parameter into a pass OptimizeFixed's pass-name-list interface has
// no other way to carry -- set by ApplyQuarot (quantize_entry.cpp)
// immediately before calling OptimizeFixed. A fresh std::mt19937_64,
// reseeded from this value plus the matched node's own unique id, is used
// per match so results are deterministic and reproducible for a given
// model and seed, without needing a single stateful RNG threaded across
// every match in graph node order the way quarot.py's own
// numpy.random.Generator sequencing does (cross-language RNG bit-parity
// with the Python port isn't a goal -- only that this port's own rotation
// is deterministic and genuinely random-orthogonal).
inline uint64_t& QuarotSeed() {
  static uint64_t seed = 0;
  return seed;
}

struct Quarot final : public PredicateBasedPass {
  static constexpr int64_t kBlockSize = 32;
  static constexpr float kEpsilon = 1e-12f;

  explicit Quarot()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "quarot"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 21) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    const int64_t K = w_t->sizes()[1 - channel_axis];
    return K % kBlockSize == 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    const int64_t N =
        info.weight_transposed ? w_t->sizes()[0] : w_t->sizes()[1];
    const int64_t K =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    if (K % kBlockSize != 0) {
      return false;
    }
    const int64_t num_blocks = K / kBlockSize;

    // w_nk: [N, K], output channel first.
    const std::vector<float> w_nk = ReadWeightNK(*w_t, info.weight_transposed);

    std::mt19937_64 rng(QuarotSeed() ^
                        (0x9E3779B97F4A7C15ULL * (n->output()->unique() + 1)));
    const std::vector<float> u = RandomOrthogonalMatrix(K, rng);  // [K, K]

    // w_tilde_nk = w_nk @ u  -- [N, K], exact before quantization.
    std::vector<double> w_tilde_nk(static_cast<size_t>(N * K), 0.0);
    for (int64_t r = 0; r < N; ++r) {
      for (int64_t c = 0; c < K; ++c) {
        double acc = 0.0;
        const float* w_row = w_nk.data() + r * K;
        for (int64_t kk = 0; kk < K; ++kk) {
          acc += static_cast<double>(w_row[kk]) *
                 static_cast<double>(u[static_cast<size_t>(kk * K + c)]);
        }
        w_tilde_nk[static_cast<size_t>(r * K + c)] = acc;
      }
    }

    // Block-wise INT4 quantization of w_tilde, directly into [K, N] layout
    // (codes_kn/scale_kn) so the final MatMul(Xq, Wdq) needs no extra
    // Transpose node -- mirrors TryQuantizeWeightBlockwiseInt4InPlace's
    // scale rule (max(|.|) / 7 per (block-of-K, output-channel) group) but
    // written directly transposed, since transposing already-nibble-packed
    // INT4 data afterward would need unpacking/repacking anyway.
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * N), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
      for (int64_t blk = 0; blk < num_blocks; ++blk) {
        float max_abs = 0.0f;
        for (int64_t j = 0; j < kBlockSize; ++j) {
          const int64_t c = blk * kBlockSize + j;
          max_abs = std::max(max_abs,
                             static_cast<float>(std::fabs(
                                 w_tilde_nk[static_cast<size_t>(r * K + c)])));
        }
        scale_kn[static_cast<size_t>(blk * N + r)] =
            max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
      }
    }
    std::vector<int8_t> codes_kn(static_cast<size_t>(K * N));
    for (int64_t r = 0; r < N; ++r) {
      for (int64_t c = 0; c < K; ++c) {
        const int64_t blk = c / kBlockSize;
        const float s = scale_kn[static_cast<size_t>(blk * N + r)];
        const float q = std::round(
            static_cast<float>(w_tilde_nk[static_cast<size_t>(r * K + c)]) / s);
        codes_kn[static_cast<size_t>(c * N + r)] =
            static_cast<int8_t>(std::clamp(q, -7.0f, 7.0f));
      }
    }
    // Pack two int4 codes per byte, low nibble first (K * N is always even:
    // K is a multiple of kBlockSize=32) -- see
    // TryQuantizeWeightBlockwiseInt4InPlace's identical packing for why this
    // bypasses the typed int32_data field.
    const int64_t numel = K * N;
    std::string packed(static_cast<size_t>((numel + 1) / 2), '\0');
    for (int64_t i = 0; i < numel; ++i) {
      const uint8_t nibble =
          static_cast<uint8_t>(codes_kn[static_cast<size_t>(i)]) & 0x0F;
      uint8_t& byte =
          reinterpret_cast<uint8_t&>(packed[static_cast<size_t>(i / 2)]);
      if (i % 2 == 0) {
        byte = nibble;
      } else {
        byte = static_cast<uint8_t>(byte | (nibble << 4));
      }
    }

    Tensor codes_t;
    codes_t.elem_type() = TensorProto_DataType_INT4;
    codes_t.sizes() = {K, N};
    codes_t.set_raw_data(std::move(packed));
    Value* codes_v = graph.addInitializerAndCreateValue(codes_t);

    Tensor scale_t;
    scale_t.elem_type() = TensorProto_DataType_FLOAT;
    scale_t.sizes() = {num_blocks, N};
    scale_t.floats() = scale_kn;
    Value* scale_v = graph.addInitializerAndCreateValue(scale_t);

    Tensor u_t;
    u_t.elem_type() = TensorProto_DataType_FLOAT;
    u_t.sizes() = {K, K};
    u_t.floats() = u;
    Value* u_v = graph.addInitializerAndCreateValue(u_t);

    Tensor eps_t;
    eps_t.elem_type() = TensorProto_DataType_FLOAT;
    eps_t.floats() = {kEpsilon};
    Value* eps_v = graph.addInitializerAndCreateValue(eps_t);

    Tensor seven_t;
    seven_t.elem_type() = TensorProto_DataType_FLOAT;
    seven_t.floats() = {7.0f};
    Value* seven_v = graph.addInitializerAndCreateValue(seven_t);

    Tensor clip_min_t;
    clip_min_t.elem_type() = TensorProto_DataType_FLOAT;
    clip_min_t.floats() = {-7.0f};
    Value* clip_min_v = graph.addInitializerAndCreateValue(clip_min_t);

    Tensor clip_max_t;
    clip_max_t.elem_type() = TensorProto_DataType_FLOAT;
    clip_max_t.floats() = {7.0f};
    Value* clip_max_v = graph.addInitializerAndCreateValue(clip_max_t);

    Tensor axes_t;
    axes_t.elem_type() = TensorProto_DataType_INT64;
    axes_t.sizes() = {1};
    axes_t.int64s() = {-1};
    Value* axes_v = graph.addInitializerAndCreateValue(axes_t);

    auto make_node = [&](Symbol kind, const std::vector<Value*>& inputs) {
      Node* node = graph.create(kind, 1);
      for (Value* v : inputs) {
        node->addInput(v);
      }
      node->insertBefore(n);
      node->output()->setElemType(TensorProto_DataType_FLOAT);
      return node;
    };

    // Xrot = MatMul(X, U)
    Node* x_rot = make_node(kMatMul, {info.x, u_v});

    // Data-free, per-token round-to-nearest INT4 activation quantization --
    // simulated via an immediate dequantize (kept in float32) rather than a
    // true packed INT4 tensor, since X isn't constant.
    Node* x_abs = make_node(Symbol("Abs"), {x_rot->output()});
    Node* x_max = graph.create(kReduceMax, 1);
    x_max->addInput(x_abs->output());
    x_max->addInput(axes_v);
    x_max->i_(kkeepdims, 1);
    x_max->insertBefore(n);
    x_max->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* x_safe_max = graph.create(Symbol("Clip"), 1);
    x_safe_max->addInput(x_max->output());
    x_safe_max->addInput(eps_v);
    x_safe_max->insertBefore(n);
    x_safe_max->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* x_scale = make_node(kDiv, {x_safe_max->output(), seven_v});
    Node* x_scaled = make_node(kDiv, {x_rot->output(), x_scale->output()});
    Node* x_rounded = make_node(Symbol("Round"), {x_scaled->output()});

    Node* x_clipped = graph.create(Symbol("Clip"), 1);
    x_clipped->addInput(x_rounded->output());
    x_clipped->addInput(clip_min_v);
    x_clipped->addInput(clip_max_v);
    x_clipped->insertBefore(n);
    x_clipped->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* x_dequant = make_node(kMul, {x_clipped->output(), x_scale->output()});

    Node* w_dequant = graph.create(Symbol("DequantizeLinear"), 1);
    w_dequant->addInput(codes_v);
    w_dequant->addInput(scale_v);
    w_dequant->i_(kaxis, 0);
    w_dequant->i_(Symbol("block_size"), kBlockSize);
    w_dequant->insertBefore(n);
    w_dequant->output()->setElemType(TensorProto_DataType_FLOAT);
    w_dequant->output()->setSizes({Dimension(K), Dimension(N)});

    Node* core = make_node(kMatMul, {x_dequant->output(), w_dequant->output()});

    Value* final_output;
    if (info.bias != nullptr) {
      Node* bias_add = graph.create(kAdd, 1);
      bias_add->addInput(core->output());
      bias_add->addInput(info.bias);
      bias_add->insertBefore(n);
      bias_add->output()->setElemType(TensorProto_DataType_FLOAT);
      final_output = bias_add->output();
    } else {
      final_output = core->output();
    }
    if (n->output()->sizes().size() > 0) {
      final_output->setSizes(n->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(n->output(), final_output);
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
