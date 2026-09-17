// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// QServe's QoQ quantization (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang,
// Gan, Han, MLSys 2025, "QServe: W4A8KV4 Quantization and System Co-design
// for Efficient LLM Serving") -- C++ port of qoq.py's own
// quantize_weight_only_qoq (the module's *primary* contribution; that
// module's own apply_smooth_attention is a separate, calibration-driven
// KV-cache-side technique, out of scope here). See qoq.py's own module
// docstring for the full rationale: unlike quantize_weight_only_int4's own
// single-stage rounding (float weight straight to one INT4 grid, one scale
// per block), QoQ quantizes in TWO stages -- first the whole float weight,
// per output channel, to a *protective* INT8 grid (clipped below the full
// [-127, 127] range, leaving headroom for the second stage's own rounding
// error), then, within that already-INT8-quantized tensor, each
// block_size-element group of the reduction dimension is quantized again
// down to INT4. The key numerical difference from
// weight_only_quantize_int4_matmul.h: the INT4 code is derived by rounding
// an ALREADY-INT8-quantized value, not the original float weight, and every
// reconstructed value passes through the INT8 grid on the way back to
// float -- even though the two per-stage scales are folded into one
// combined per-(channel, group) scale, so the graph itself only ever needs
// to emit a single DequantizeLinear, exactly
// weight_only_quantize_int4_matmul.h's own graph shape (INT4 codes plus a
// block-wise scale), differing only in how those codes and that scale were
// computed:
//
//   Stage 1 (per output channel, protective INT8):
//     s1 = max(|W_row|) / int8_clip_max
//     code8 = clip(round(W_row / s1), -int8_clip_max, int8_clip_max)
//
//   Stage 2 (per (channel, block-of-K) group, INT8 grid -> INT4):
//     s2 = max(|code8_group|) / 7
//     code4 = clip(round(code8_group / s2), -7, 7)
//
//   Reconstruction (two-stage, folded into one scale for the graph):
//     W_hat = code4 * s2 * s1
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N]
// After:
//   Wdq = DequantizeLinear(Wq, Ws, axis=<K's axis>, block_size=kBlockSize)
//   Y   = MatMul(X, Wdq)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input
// 1) is a constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by kBlockSize, and whose activation (input 0) is float32. An
// opset older than 21 (INT4 tensors and DequantizeLinear's block_size
// attribute both need it) is left untouched, matching
// quantize_weight_only_qoq's own opset gate exactly.
//
// This port hardcodes qoq.py's own default int8_clip_max (119) and
// block_size (32) rather than exposing them as parameters -- several other
// *_cpp ports in this repo already establish that a C++ port need not
// mirror every optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE: none -- this is a closed-form,
// deterministic two-stage quantization scheme with no RNG and no fitting
// algorithm, so this port is expected to track qoq.py's own float64 numpy
// implementation closely, up to ordinary floating-point summation-order
// differences (there are none here beyond independent per-element/per-group
// max reductions, which this port computes in the same left-to-right order
// numpy's own axis-reduction would for these small block sizes).
// quantize_weight_only_qoq and this port's _cpp counterpart remain
// independently-correct, non-interchangeable entry points.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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

namespace qoq_detail {

constexpr int64_t kBlockSize = 32;
constexpr int64_t kInt8ClipMax = 119;  // qoq.py's own default, in (0, 127].
constexpr double kEps = 1e-12;         // qoq.py's own _EPS.

// Round-half-to-even (banker's rounding), matching numpy's own `np.round`
// -- qoq.py's own two-stage quantizer uses np.round for BOTH stages, unlike
// std::round's half-away-from-zero (which silently disagrees with the
// Python reference whenever a quotient lands exactly, or near-exactly due
// to floating-point representation, on a .5 boundary).
inline double RoundHalfToEven(double v) {
  const double f = std::floor(v);
  const double d = v - f;
  if (d < 0.5) {
    return f;
  }
  if (d > 0.5) {
    return f + 1.0;
  }
  const double half = f / 2.0;
  return (half == std::floor(half)) ? f : f + 1.0;
}

// QoQ's own progressive (INT8-then-INT4) two-stage quantization of `w_t`
// (a 2-D float32 constant) *in its own layout* (no transpose): `channel_axis`
// (0 or 1) is the output-channel axis, so the reduction axis (K) is the
// other one, split into `K / kBlockSize` consecutive blocks. Direct
// transcription of qoq.py's own quantize_weight_only_qoq (see this header's
// own top-of-file comment for the two-stage formula). Returns false when
// `K` is not evenly divisible by kBlockSize -- the common, unambiguous case
// only, mirroring TryQuantizeWeightBlockwiseInt4InPlace's own identical
// scope decision (quantize_matmul_common.h).
inline bool TryQuantizeWeightQoqInPlace(const Tensor& w_t, int64_t channel_axis,
                                        Tensor& q_out, Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t k = reduction_axis == 0 ? dim0 : dim1;
  if (kBlockSize <= 0 || k % kBlockSize != 0) {
    return false;
  }
  const int64_t n = channel_axis == 0 ? dim0 : dim1;
  const int64_t num_blocks = k / kBlockSize;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  // w_nk(i, j): element (output channel i, reduction index j) of the
  // logical [N, K] view, regardless of w_t's own on-disk layout -- mirrors
  // `w_nk = w if weight_transposed else w.T` exactly (channel_axis == 0
  // means w_t is already [N, K] == [dim0, dim1]; channel_axis == 1 means
  // w_t is [K, N] == [dim0, dim1], so w_nk(i, j) reads element (j, i)).
  auto w_nk = [&](int64_t i, int64_t j) {
    return channel_axis == 0 ? data[static_cast<size_t>(i * dim1 + j)]
                             : data[static_cast<size_t>(j * dim1 + i)];
  };

  // Stage 1: FP16/float -> INT8, per output channel, protective clip range.
  std::vector<double> s1(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    double channel_absmax = 0.0;
    for (int64_t j = 0; j < k; ++j) {
      channel_absmax =
          std::max(channel_absmax, std::fabs(static_cast<double>(w_nk(i, j))));
    }
    s1[static_cast<size_t>(i)] = std::max(channel_absmax, qoq_detail::kEps) /
                                 static_cast<double>(kInt8ClipMax);
  }
  std::vector<double> code8(static_cast<size_t>(n * k));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      const double q = RoundHalfToEven(static_cast<double>(w_nk(i, j)) /
                                       s1[static_cast<size_t>(i)]);
      code8[static_cast<size_t>(i * k + j)] =
          std::clamp(q, -static_cast<double>(kInt8ClipMax),
                     static_cast<double>(kInt8ClipMax));
    }
  }

  // Stage 2: that INT8 grid -> INT4, per (channel, block-of-K) group --
  // rounds the already-quantized code8, not the original float weight.
  std::vector<double> s2(static_cast<size_t>(n * num_blocks), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double group_absmax = 0.0;
      for (int64_t jj = 0; jj < kBlockSize; ++jj) {
        const int64_t j = b * kBlockSize + jj;
        group_absmax = std::max(
            group_absmax, std::fabs(code8[static_cast<size_t>(i * k + j)]));
      }
      s2[static_cast<size_t>(i * num_blocks + b)] =
          std::max(group_absmax, qoq_detail::kEps) / 7.0;
    }
  }
  std::vector<int8_t> code4(static_cast<size_t>(n * k));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      const int64_t b = j / kBlockSize;
      const double scale = s2[static_cast<size_t>(i * num_blocks + b)];
      const double q =
          RoundHalfToEven(code8[static_cast<size_t>(i * k + j)] / scale);
      code4[static_cast<size_t>(i * k + j)] =
          static_cast<int8_t>(std::clamp(q, -7.0, 7.0));
    }
  }

  // Final reconstruction folds both stages into one combined scale --
  // code4 * s2 * s1 -- so the graph only needs a single DequantizeLinear,
  // even though the codes were derived via the two-stage round-trip
  // through the INT8 grid above. combined_scale(i, b) = s1[i] * s2[i][b].
  std::vector<float> combined_scale(static_cast<size_t>(n * num_blocks));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      combined_scale[static_cast<size_t>(i * num_blocks + b)] =
          static_cast<float>(s1[static_cast<size_t>(i)] *
                             s2[static_cast<size_t>(i * num_blocks + b)]);
    }
  }

  // Write codes/scale back into w_t's OWN [dim0, dim1] storage layout --
  // mirrors `codes_orig = code4 if weight_transposed else code4.T` /
  // `scale_orig = combined_scale if weight_transposed else
  // combined_scale.T` exactly (channel_axis == 0 <=> weight_transposed).
  std::vector<int8_t> codes_orig(static_cast<size_t>(dim0 * dim1));
  std::vector<float> scale_orig(
      static_cast<size_t>((reduction_axis == 0 ? num_blocks : dim0) *
                          (reduction_axis == 1 ? num_blocks : dim1)));
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      const int64_t oi = channel_axis == 0 ? i : j;
      const int64_t oj = channel_axis == 0 ? j : i;
      codes_orig[static_cast<size_t>(oi * dim1 + oj)] =
          code4[static_cast<size_t>(i * k + j)];
    }
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int64_t osi = reduction_axis == 0 ? b : i;
      const int64_t osj = reduction_axis == 1 ? b : i;
      scale_orig[static_cast<size_t>(osi * scale_dim1 + osj)] =
          combined_scale[static_cast<size_t>(i * num_blocks + b)];
    }
  }

  // Nibble-packed INT4, low-nibble-first -- same convention
  // TryQuantizeWeightBlockwiseInt4InPlace uses (quantize_matmul_common.h):
  // byte[p] = (codes_orig[2p] & 0xF) | ((codes_orig[2p+1] & 0xF) << 4).
  // kBlockSize is even, so K % kBlockSize == 0 makes dim0 * dim1 always
  // even here, same reasoning that function's own comment documents.
  const int64_t numel = dim0 * dim1;
  std::string packed(static_cast<size_t>((numel + 1) / 2), '\0');
  for (int64_t p = 0; p < numel; ++p) {
    const uint8_t nibble =
        static_cast<uint8_t>(codes_orig[static_cast<size_t>(p)]) & 0x0F;
    uint8_t& byte =
        reinterpret_cast<uint8_t&>(packed[static_cast<size_t>(p / 2)]);
    if (p % 2 == 0) {
      byte = nibble;
    } else {
      byte = static_cast<uint8_t>(byte | (nibble << 4));
    }
  }

  q_out.elem_type() = TensorProto_DataType_INT4;
  q_out.sizes() = {dim0, dim1};
  q_out.set_raw_data(std::move(packed));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {scale_dim0, scale_dim1};
  scale_out.floats() = std::move(scale_orig);
  return true;
}

}  // namespace qoq_detail

struct Qoq final : public PredicateBasedPass {
  explicit Qoq()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "qoq"; }

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
    const int64_t k = w_t->sizes()[1 - channel_axis];
    return k % qoq_detail::kBlockSize == 0;
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

    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    const int64_t reduction_axis = 1 - channel_axis;
    Tensor w_q;
    Tensor w_scale;
    if (!qoq_detail::TryQuantizeWeightQoqInPlace(*w_t, channel_axis, w_q,
                                                 w_scale)) {
      return false;
    }

    Value* w_q_v = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_v = graph.addInitializerAndCreateValue(w_scale);

    Node* wdq = graph.create(Symbol("DequantizeLinear"), 1);
    wdq->addInput(w_q_v);
    wdq->addInput(w_scale_v);
    wdq->i_(kaxis, reduction_axis);
    wdq->i_(Symbol("block_size"), qoq_detail::kBlockSize);
    wdq->insertBefore(n);
    wdq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.w->sizes().size() > 0) {
      wdq->output()->setSizes(info.w->sizes());
    }

    n->replaceInput(1, wdq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
