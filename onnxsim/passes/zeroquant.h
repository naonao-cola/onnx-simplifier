// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// ZeroQuant (Yao, Aminabadi, Zhang, Wu, Li, He, 2022, "ZeroQuant: Efficient
// and Affordable Post-Training Quantization for Large-Scale Transformers")
// -- C++ port of zeroquant.py's own apply_zeroquant. See that module's
// docstring for the full rationale: this repo already has group-wise INT8
// weight quantization (quantize_weight_only_int8_block) and per-token
// dynamic INT8 activation quantization (quarot.h's own scheme) in
// isolation; ZeroQuant's real contribution is *pairing* them and feeding
// the result into a genuine ``int8 x int8`` MatMulInteger -- unlike every
// other per-token-dynamic-INT8 use in this repo, which immediately
// dequantizes back to float32 (simulating precision loss, not executing
// integer arithmetic).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After (conceptually -- see zeroquant.py's own docstring, "Why grouped
// MatMulInteger", for exactly why this needs one MatMulInteger per K-group
// rather than a single call):
//   Xq = round_to_nearest_int8_per_token(X)     -- computed at runtime,
//                                                   scale = max(|x|)/127
//   Wq, Ws = per_group_symmetric_int8(W)        -- computed once, here
//   Acc = sum over K-groups g of MatMulInteger(Xq[:, group g], Wq[group g])
//   Y = Acc * Xscale * Ws [+ bias]               -- dequantize
//
// Unlike every weight-only fold-to-initializer port in this repo, this
// pass cannot fold into a single replacement weight: the activation's own
// per-token scale is a genuine runtime value (computed fresh from each
// token's own row), the same reason quarot.h/quip_sharp.h's own top-of-file
// comments already give for why QuaRot can't fold either (quip_sharp.h
// explicitly contrasts itself against quarot's situation: QuIP# never
// quantizes the activation, only the weight, so it CAN fold; ZeroQuant, like
// QuaRot, quantizes X too, so it can't). This pass therefore builds real new
// graph nodes, mirroring zeroquant.py's own Shape/Gather/Concat/Reshape
// flatten-to-2-D prelude (so the per-token reduction always sees a plain
// [M, K] matrix regardless of X's own rank, e.g. [batch, seq, K]) and its
// own group-wise Split/MatMulInteger/Cast/Mul/Sum construction, node for
// node -- the two exceptions below are pure graph-size simplifications with
// no numeric effect, not scope narrowing:
//   - zeroquant.py's own graph creates two separately-named int64 constants
//     that both hold the value [-1] ("zq_minus_one" for the Concat's own
//     -1 element, "zq_last_axis" for Gather's index / ReduceMax's axes /
//     Slice's end); this port creates that value once and reuses the same
//     Value* everywhere, the same per-match fresh-constant convention every
//     other *_cpp port in this repo already uses instead of mirroring the
//     Python source's own naming.
//   - Only the common, unambiguous shape is handled: a MatMul, or a Gemm
//     with transA=0, alpha=1 and beta=1, whose weight (input 1) is a
//     constant 2-D float32 tensor whose reduction dimension K is divisible
//     by the configured block size (ZeroQuantBlockSize(), 32 by default,
//     and no larger than the group-level int32-accumulator-overflow bound
//     below), on an opset >= 18 model (ReduceMax's axes-as-input form and
//     Split's num_outputs attribute both need it) -- exactly zeroquant.py's
//     own gate. Everything else is left alone.
//
// This is a closed-form, deterministic quantization scheme (no RNG, no
// fitting algorithm), so -- unlike this repo's k-means-family ports
// (kmeans_quantization.h, aqlm.h, quip_sharp.h) -- this port is expected to
// track zeroquant.py's own float64 numpy implementation closely, up to
// ordinary floating-point summation-order differences; no ACCEPTED,
// PERMANENT DIVERGENCE applies here.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace zeroquant_detail {

// int32 accumulation over a single K-group can't overflow until
// block_size * 255 (int8 activation's own -127..127 range, cast to its
// widest possible magnitude) * 127 (int8 weight range) exceeds INT32_MAX --
// mirrors zeroquant.py's own _MAX_SAFE_GROUP_SIZE, and
// quantize_matmul_common.h's IsSafeInt32ReductionDepth, applied per group
// instead of over the whole (ungrouped) reduction depth.
constexpr int64_t kMaxSafeGroupSize =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / (255 * 127);

// Symmetric INT8 quantization of a canonical [K, N] weight (`w_kn`), one
// scale per (block_size-wide K-group, output column) -- the same
// granularity quantize_weight_only_int8_block's own C++ port
// (TryQuantizeWeightBlockwiseInt8InPlace, quantize_matmul_common.h) uses.
// NOT reused directly here: that helper's own "in place" convention keeps
// whatever transposed-or-not layout the source tensor already has, while
// this pass always needs the canonical [K, N] (reduction-axis-first) layout
// to slice out contiguous per-group rows for its own grouped MatMulInteger
// construction below; and its own zero-scale fallback (a fixed 1.0) differs
// from zeroquant.py's own epsilon floor -- though, since a fully-zero
// group's codes are 0 regardless of which fallback scale later multiplies
// them, this makes no difference to any reconstructed value. Mirrors
// zeroquant.py's own _quantize_weight_groupwise_int8 exactly otherwise.
inline void QuantizeWeightGroupwiseInt8KN(const std::vector<double>& w_kn,
                                          int64_t k, int64_t n,
                                          int64_t block_size, double epsilon,
                                          std::vector<int8_t>& wq_out,
                                          std::vector<float>& scale_out) {
  const int64_t num_groups = k / block_size;
  wq_out.assign(static_cast<size_t>(k * n), 0);
  scale_out.assign(static_cast<size_t>(num_groups * n), 0.0f);
  for (int64_t g = 0; g < num_groups; ++g) {
    for (int64_t col = 0; col < n; ++col) {
      double max_abs = 0.0;
      for (int64_t i = 0; i < block_size; ++i) {
        const int64_t row = g * block_size + i;
        max_abs = std::max(max_abs,
                           std::fabs(w_kn[static_cast<size_t>(row * n + col)]));
      }
      const double scale = std::max(max_abs / 127.0, epsilon);
      scale_out[static_cast<size_t>(g * n + col)] = static_cast<float>(scale);
      for (int64_t i = 0; i < block_size; ++i) {
        const int64_t row = g * block_size + i;
        const double q =
            std::round(w_kn[static_cast<size_t>(row * n + col)] / scale);
        wq_out[static_cast<size_t>(row * n + col)] =
            static_cast<int8_t>(std::clamp(q, -127.0, 127.0));
      }
    }
  }
}

}  // namespace zeroquant_detail

// Weight quantization group size along K (elements per group sharing one
// weight scale) -- a function-local static, the same pattern
// QuarotBlockSize() (quarot.h) uses to pass a parameter into a pass
// OptimizeFixed's pass-name-list interface has no other way to carry. Set
// by ApplyZeroQuant (quantize_entry.cpp) immediately before calling
// OptimizeFixed. Mirrors zeroquant.py's own ``block_size`` parameter.
inline int64_t& ZeroQuantBlockSize() {
  static int64_t block_size = 32;
  return block_size;
}

// Floor applied to a weight group's own max-abs value, and (at graph-run
// time) a token's own quantization range, before using it as a scale --
// read/written the same way ZeroQuantBlockSize() is. Mirrors zeroquant.py's
// own ``epsilon`` parameter.
inline float& ZeroQuantEpsilon() {
  static float epsilon = 1e-12f;
  return epsilon;
}

struct ZeroQuant final : public PredicateBasedPass {
  explicit ZeroQuant()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "zeroquant"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 18) {
      return false;
    }
    const int64_t block_size = ZeroQuantBlockSize();
    if (block_size <= 0 || block_size > zeroquant_detail::kMaxSafeGroupSize) {
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
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    return k % block_size == 0;
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

    const int64_t block_size = ZeroQuantBlockSize();
    const double epsilon = static_cast<double>(ZeroQuantEpsilon());
    if (block_size <= 0 || block_size > zeroquant_detail::kMaxSafeGroupSize) {
      return false;
    }

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t k = info.weight_transposed ? dim1 : dim0;
    const int64_t out_n = info.weight_transposed ? dim0 : dim1;
    if (k % block_size != 0) {
      return false;
    }
    const int64_t num_groups = k / block_size;

    // Canonicalize the weight to [K, N] (zeroquant.py's own
    // ``w_kn = w.T if weight_transposed else w``), so each group's rows are
    // contiguous below.
    const std::vector<float> data = ReadFloatMatrix(*w_t);
    std::vector<double> w_kn(static_cast<size_t>(k * out_n));
    if (info.weight_transposed) {
      for (int64_t nn = 0; nn < out_n; ++nn) {
        for (int64_t kk = 0; kk < k; ++kk) {
          w_kn[static_cast<size_t>(kk * out_n + nn)] =
              static_cast<double>(data[static_cast<size_t>(nn * k + kk)]);
        }
      }
    } else {
      for (int64_t i = 0; i < k * out_n; ++i) {
        w_kn[static_cast<size_t>(i)] =
            static_cast<double>(data[static_cast<size_t>(i)]);
      }
    }

    std::vector<int8_t> wq_kn;
    std::vector<float> scale_gn;
    zeroquant_detail::QuantizeWeightGroupwiseInt8KN(w_kn, k, out_n, block_size,
                                                    epsilon, wq_kn, scale_gn);

    auto make_const_i64 = [&](std::vector<int64_t> vals) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = {static_cast<int64_t>(vals.size())};
      t.int64s() = std::move(vals);
      return graph.addInitializerAndCreateValue(t);
    };
    auto make_const_f32_scalar = [&](float v) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_FLOAT;
      t.floats() = {v};
      return graph.addInitializerAndCreateValue(t);
    };

    // Merges zeroquant.py's own separately-named-but-identically-valued
    // "zq_minus_one"/"zq_last_axis" constants (both [-1]) into one Value --
    // see this header's own top-of-file comment.
    Value* neg1_v = make_const_i64({-1});
    Value* zero_1d_v = make_const_i64({0});
    Value* i127_v = make_const_f32_scalar(127.0f);
    Value* i127_min_v = make_const_f32_scalar(-127.0f);
    Value* eps_v = make_const_f32_scalar(ZeroQuantEpsilon());

    auto before_n = [&](Node* node) { node->insertBefore(n); };

    // --- Flatten X to 2-D [M, K] (M = every leading dim collapsed) --
    // mirrors zeroquant.py's own Shape/Gather/Concat/Reshape prelude, so
    // the per-token quantization below always sees a plain matrix
    // regardless of X's own rank.
    Node* shape_node = graph.create(Symbol("Shape"), 1);
    shape_node->addInput(info.x);
    before_n(shape_node);
    Value* orig_shape = shape_node->output();
    orig_shape->setElemType(TensorProto_DataType_INT64);

    Node* k_dim_node = graph.create(Symbol("Gather"), 1);
    k_dim_node->addInput(orig_shape);
    k_dim_node->addInput(neg1_v);
    k_dim_node->i_(kaxis, 0);
    before_n(k_dim_node);
    Value* k_dim = k_dim_node->output();
    k_dim->setElemType(TensorProto_DataType_INT64);
    k_dim->setSizes({Dimension(1)});

    Node* flat_shape_node = graph.create(Symbol("Concat"), 1);
    flat_shape_node->addInput(neg1_v);
    flat_shape_node->addInput(k_dim);
    flat_shape_node->i_(kaxis, 0);
    before_n(flat_shape_node);
    Value* flat_shape = flat_shape_node->output();
    flat_shape->setElemType(TensorProto_DataType_INT64);
    flat_shape->setSizes({Dimension(2)});

    Node* x2d_node = graph.create(Symbol("Reshape"), 1);
    x2d_node->addInput(info.x);
    x2d_node->addInput(flat_shape);
    before_n(x2d_node);
    Value* x2d = x2d_node->output();
    x2d->setElemType(TensorProto_DataType_FLOAT);

    // --- Per-token (per-row) dynamic INT8 activation quantization,
    // symmetric (scale = max(|x|) / 127, zero point fixed at 0 -- see
    // zeroquant.py's own docstring, point 2, for why symmetric rather than
    // DynamicQuantizeLinear's asymmetric convention). No calibration data:
    // each token's own scale is computed fresh here, at graph-run time.
    Node* x_abs_node = graph.create(Symbol("Abs"), 1);
    x_abs_node->addInput(x2d);
    before_n(x_abs_node);
    Value* x_abs = x_abs_node->output();
    x_abs->setElemType(TensorProto_DataType_FLOAT);

    Node* x_max_node = graph.create(kReduceMax, 1);
    x_max_node->addInput(x_abs);
    x_max_node->addInput(neg1_v);
    x_max_node->i_(kkeepdims, 1);
    before_n(x_max_node);
    Value* x_max_abs = x_max_node->output();
    x_max_abs->setElemType(TensorProto_DataType_FLOAT);

    Node* x_safe_node = graph.create(Symbol("Max"), 1);
    x_safe_node->addInput(x_max_abs);
    x_safe_node->addInput(eps_v);
    before_n(x_safe_node);
    Value* x_safe_max_abs = x_safe_node->output();
    x_safe_max_abs->setElemType(TensorProto_DataType_FLOAT);

    Node* x_scale_node = graph.create(kDiv, 1);
    x_scale_node->addInput(x_safe_max_abs);
    x_scale_node->addInput(i127_v);
    before_n(x_scale_node);
    Value* x_scale = x_scale_node->output();
    x_scale->setElemType(TensorProto_DataType_FLOAT);

    Node* x_scaled_node = graph.create(kDiv, 1);
    x_scaled_node->addInput(x2d);
    x_scaled_node->addInput(x_scale);
    before_n(x_scaled_node);
    Value* x_scaled = x_scaled_node->output();
    x_scaled->setElemType(TensorProto_DataType_FLOAT);

    Node* x_round_node = graph.create(Symbol("Round"), 1);
    x_round_node->addInput(x_scaled);
    before_n(x_round_node);
    Value* x_rounded = x_round_node->output();
    x_rounded->setElemType(TensorProto_DataType_FLOAT);

    Node* x_clip_node = graph.create(Symbol("Clip"), 1);
    x_clip_node->addInput(x_rounded);
    x_clip_node->addInput(i127_min_v);
    x_clip_node->addInput(i127_v);
    before_n(x_clip_node);
    Value* x_clipped = x_clip_node->output();
    x_clipped->setElemType(TensorProto_DataType_FLOAT);

    Node* xq_cast_node = graph.create(kCast, 1);
    xq_cast_node->addInput(x_clipped);
    xq_cast_node->i_(kto, TensorProto_DataType_INT8);
    before_n(xq_cast_node);
    Value* xq_2d = xq_cast_node->output();
    xq_2d->setElemType(TensorProto_DataType_INT8);

    // --- Group-wise INT8 weight (from W's static values) and one real
    // MatMulInteger per K-group -- see this header's own top-of-file
    // comment ("Why grouped MatMulInteger") for why a single call can't
    // express a weight scale that varies partway through K.
    Node* split_node =
        graph.create(Symbol("Split"), static_cast<size_t>(num_groups));
    split_node->addInput(xq_2d);
    split_node->i_(kaxis, 1);
    split_node->i_(Symbol("num_outputs"), num_groups);
    before_n(split_node);
    for (int64_t g = 0; g < num_groups; ++g) {
      split_node->outputs()[static_cast<size_t>(g)]->setElemType(
          TensorProto_DataType_INT8);
    }

    std::vector<Value*> group_terms;
    group_terms.reserve(static_cast<size_t>(num_groups));
    for (int64_t g = 0; g < num_groups; ++g) {
      Tensor wq_g_t;
      wq_g_t.elem_type() = TensorProto_DataType_INT8;
      wq_g_t.sizes() = {block_size, out_n};
      std::vector<int8_t> wq_g(static_cast<size_t>(block_size * out_n));
      for (int64_t i = 0; i < block_size * out_n; ++i) {
        wq_g[static_cast<size_t>(i)] =
            wq_kn[static_cast<size_t>(g * block_size * out_n + i)];
      }
      wq_g_t.set_raw_data(WriteRawDataLittleEndian(wq_g));
      Value* wq_g_v = graph.addInitializerAndCreateValue(wq_g_t);

      Tensor ws_g_t;
      ws_g_t.elem_type() = TensorProto_DataType_FLOAT;
      ws_g_t.sizes() = {out_n};
      ws_g_t.floats() = std::vector<float>(scale_gn.begin() + g * out_n,
                                           scale_gn.begin() + (g + 1) * out_n);
      Value* ws_g_v = graph.addInitializerAndCreateValue(ws_g_t);

      // a_zero_point/b_zero_point both omitted (default 0): both operands
      // are quantized symmetrically, and ONNX Runtime's own MatMulInteger
      // CPU kernel rejects a genuine per-row zero point anyway -- see
      // zeroquant.py's own docstring, point 2.
      Node* mmi_node = graph.create(Symbol("MatMulInteger"), 1);
      mmi_node->addInput(split_node->outputs()[static_cast<size_t>(g)]);
      mmi_node->addInput(wq_g_v);
      before_n(mmi_node);
      Value* acc_g = mmi_node->output();
      acc_g->setElemType(TensorProto_DataType_INT32);

      Node* acc_cast_node = graph.create(kCast, 1);
      acc_cast_node->addInput(acc_g);
      acc_cast_node->i_(kto, TensorProto_DataType_FLOAT);
      before_n(acc_cast_node);
      Value* acc_g_f = acc_cast_node->output();
      acc_g_f->setElemType(TensorProto_DataType_FLOAT);

      Node* scaled_node = graph.create(kMul, 1);
      scaled_node->addInput(acc_g_f);
      scaled_node->addInput(ws_g_v);
      before_n(scaled_node);
      Value* scaled_g = scaled_node->output();
      scaled_g->setElemType(TensorProto_DataType_FLOAT);

      group_terms.push_back(scaled_g);
    }

    Value* unscaled_sum;
    if (num_groups == 1) {
      unscaled_sum = group_terms[0];
    } else {
      Node* sum_node = graph.create(Symbol("Sum"), 1);
      for (Value* v : group_terms) {
        sum_node->addInput(v);
      }
      before_n(sum_node);
      unscaled_sum = sum_node->output();
      unscaled_sum->setElemType(TensorProto_DataType_FLOAT);
    }

    Node* y_unbiased_node = graph.create(kMul, 1);
    y_unbiased_node->addInput(unscaled_sum);
    y_unbiased_node->addInput(x_scale);
    before_n(y_unbiased_node);
    Value* y2d_unbiased = y_unbiased_node->output();
    y2d_unbiased->setElemType(TensorProto_DataType_FLOAT);

    Value* y2d;
    if (info.bias != nullptr) {
      Node* add_node = graph.create(kAdd, 1);
      add_node->addInput(y2d_unbiased);
      add_node->addInput(info.bias);
      before_n(add_node);
      y2d = add_node->output();
      y2d->setElemType(TensorProto_DataType_FLOAT);
    } else {
      y2d = y2d_unbiased;
    }

    // --- Restore the original leading (batch/sequence) dims, with the last
    // dim now out_n instead of K.
    Tensor n_const_t;
    n_const_t.elem_type() = TensorProto_DataType_INT64;
    n_const_t.sizes() = {1};
    n_const_t.int64s() = {out_n};
    Value* n_const_v = graph.addInitializerAndCreateValue(n_const_t);

    Node* lead_shape_node = graph.create(Symbol("Slice"), 1);
    lead_shape_node->addInput(orig_shape);
    lead_shape_node->addInput(zero_1d_v);
    lead_shape_node->addInput(neg1_v);
    before_n(lead_shape_node);
    Value* lead_shape = lead_shape_node->output();
    lead_shape->setElemType(TensorProto_DataType_INT64);

    Node* out_shape_node = graph.create(Symbol("Concat"), 1);
    out_shape_node->addInput(lead_shape);
    out_shape_node->addInput(n_const_v);
    out_shape_node->i_(kaxis, 0);
    before_n(out_shape_node);
    Value* out_shape = out_shape_node->output();
    out_shape->setElemType(TensorProto_DataType_INT64);

    Node* final_node = graph.create(Symbol("Reshape"), 1);
    final_node->addInput(y2d);
    final_node->addInput(out_shape);
    before_n(final_node);
    Value* final_output = final_node->output();
    final_output->setElemType(TensorProto_DataType_FLOAT);
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
