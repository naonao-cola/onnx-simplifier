// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Weight-only quantization to ONNX Runtime's "com.microsoft::MatMulNBits"
// contrib op -- a *vendor-specific* counterpart to
// weight_only_quantize_int4_matmul.h's own portable, standard-ONNX INT4
// scheme (opset 21's native INT4 tensor type + DequantizeLinear's
// `block_size` attribute). Where that pass produces a result any conformant
// opset-21+ runtime can load, this one produces the single fused op ORT's
// own GenAI/quantization tooling (Olive, onnxruntime.quantization's
// matmul_4bits_quantizer, ...) emits for LLM/ASR weight compression --
// smaller and faster on ORT specifically, at the cost of needing ORT (or
// another runtime that implements this contrib op) to run at all. Neither
// scheme supersedes the other; pick this one when the deployment target is
// known to be ONNX Runtime and the smaller single-op footprint matters,
// weight_only_quantize_int4_matmul.h's otherwise.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C passed through as
// MatMulNBits' own optional bias input):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N] (or [N, K]
//                            for a transB=1 Gemm)
// After:
//   Y = com.microsoft::MatMulNBits(X, Wq, Ws, K=K, N=N, bits=4,
//                                  block_size=kBlockSize)
//
// Wq (packed uint8, shape [N, k_blocks, blob_size]) and Ws (float32, shape
// [N, k_blocks]) are computed once, here, from W's static values -- see
// QuantizeWeightForMatMulNBits below. Unlike
// weight_only_quantize_int4_matmul.h, K need not be evenly divisible by
// kBlockSize: MatMulNBits' own `k_blocks = ceil(K / block_size)` already
// accounts for a ragged last (shorter) block, so this quantizes it exactly
// like every other block instead of declining the whole match.
//
// zero_points (input 3) is omitted -- MatMulNBits' own documented default,
// 2^(bits-1) = 8 for 4-bit, is exactly the symmetric-around-8 code this
// pass quantizes to (`code = round(w / scale) + 8`, clamped to [0, 15]), so
// there is nothing an explicit zero_points tensor would add.
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input
// 1) is a constant 2-D float32 tensor, and whose activation (input 0) is
// float32. Everything else -- non-constant or non-2-D weights, non-default
// Gemm attributes, non-float32 operands -- is left alone.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Quantizes `w_t` (a 2-D float32 constant, laid out as [N, K] when
// `transposed` else [K, N]) into MatMulNBits' own 4-bit block format:
// `b_out` (uint8, shape [N, k_blocks, blob_size]) packs two codes per byte,
// low nibble first, matching the op's documented packing; `scale_out`
// (float32, shape [N, k_blocks]) holds one scale per (output channel,
// block). `block_size` must be a power of 2 and >= 16, matching the op's
// own documented constraint; returns false (nothing written) otherwise.
//
// Each block's `block_size` values are quantized symmetrically around the
// op's own default zero point (2^(bits-1) = 8 for 4-bit): `code =
// round(w / scale) + 8`, clamped to [0, 15]. A block's tail past K (when K
// is not evenly divisible by block_size, i.e. the last block is short)
// packs the fixed zero-point code 8 -- dequantizing to exactly 0 -- since
// MatMulNBits itself never reads those positions (K is a real attribute
// the kernel bounds its own reduction by), but every byte still needs a
// defined value to keep the initializer's contents well-defined.
inline bool QuantizeWeightForMatMulNBits(const Tensor& w_t, bool transposed,
                                         int64_t block_size, Tensor& b_out,
                                         Tensor& scale_out) {
  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return false;
  }
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t K = transposed ? dim1 : dim0;
  const int64_t N = transposed ? dim0 : dim1;
  const int64_t k_blocks = (K + block_size - 1) / block_size;
  const int64_t blob_size = block_size / 2;  // bits == 4, block_size even

  const std::vector<float> data = ReadFloatMatrix(w_t);
  // element (k, n) of the logical [K, N] weight, regardless of storage
  // layout.
  auto at = [&](int64_t k, int64_t n) {
    return transposed ? data[n * K + k] : data[k * N + n];
  };

  std::vector<float> scale(static_cast<size_t>(N * k_blocks), 0.0f);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      const int64_t k0 = kb * block_size;
      const int64_t k1 = std::min(K, k0 + block_size);
      float m = 0.0f;
      for (int64_t k = k0; k < k1; ++k) {
        m = std::max(m, std::fabs(at(k, n)));
      }
      scale[static_cast<size_t>(n * k_blocks + kb)] =
          m > 0.0f ? m / 7.0f : 1.0f;
    }
  }

  auto code_at = [&](int64_t k, int64_t n, float s) -> uint8_t {
    if (k >= K) {
      return 8;  // Past the real weight: fixed zero-point code (dequantizes
                 // to 0), never read by the kernel (bounded by K).
    }
    const float q = std::round(at(k, n) / s);
    return static_cast<uint8_t>(std::clamp(q, -8.0f, 7.0f) + 8.0f);
  };

  std::string packed(static_cast<size_t>(N * k_blocks * blob_size), '\0');
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      const float s = scale[static_cast<size_t>(n * k_blocks + kb)];
      const int64_t k0 = kb * block_size;
      for (int64_t j = 0; j < blob_size; ++j) {
        const uint8_t lo = code_at(k0 + 2 * j, n, s);
        const uint8_t hi = code_at(k0 + 2 * j + 1, n, s);
        const int64_t byte_idx = (n * k_blocks + kb) * blob_size + j;
        packed[static_cast<size_t>(byte_idx)] =
            static_cast<char>((lo & 0x0F) | ((hi & 0x0F) << 4));
      }
    }
  }

  b_out.elem_type() = TensorProto_DataType_UINT8;
  b_out.sizes() = {N, k_blocks, blob_size};
  b_out.set_raw_data(std::move(packed));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {N, k_blocks};
  scale_out.floats() = std::move(scale);
  return true;
}

struct WeightOnlyQuantizeMatMulNBits final : public PredicateBasedPass {
  // Matches weight_only_quantize_int4_matmul.h's own default: 32 favors
  // accuracy (more scales, smaller quantization groups) over the
  // scale-tensor overhead a larger block would save.
  static constexpr int64_t kBlockSize = 32;

  explicit WeightOnlyQuantizeMatMulNBits()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_matmul_nbits";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    return w_t != nullptr && w_t->elem_type() == TensorProto_DataType_FLOAT &&
           w_t->sizes().size() == 2;
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

    const auto& sizes = w_t->sizes();
    const int64_t K = info.weight_transposed ? sizes[1] : sizes[0];
    const int64_t N = info.weight_transposed ? sizes[0] : sizes[1];

    Tensor b_q;
    Tensor b_scale;
    if (!QuantizeWeightForMatMulNBits(*w_t, info.weight_transposed, kBlockSize,
                                      b_q, b_scale)) {
      return false;
    }

    Value* b_q_v = graph.addInitializerAndCreateValue(b_q);
    Value* b_scale_v = graph.addInitializerAndCreateValue(b_scale);

    Node* nbits = graph.create(Symbol("MatMulNBits"), 1);
    nbits->addInput(info.x);
    nbits->addInput(b_q_v);
    nbits->addInput(b_scale_v);
    if (info.bias != nullptr) {
      // MatMulNBits' bias is input index 5 -- zero_points (3) and g_idx (4)
      // are both skipped via an Undefined placeholder, the standard ONNX
      // IR mechanism for a skipped *middle* optional input (see
      // dynamic_quantize_attention.h's own use of the same mechanism for
      // QAttention's mask_index).
      Node* undef_zp = graph.create(kUndefined, 1);
      undef_zp->insertBefore(n);
      undef_zp->output()->setUniqueName("");
      Node* undef_gidx = graph.create(kUndefined, 1);
      undef_gidx->insertBefore(n);
      undef_gidx->output()->setUniqueName("");
      nbits->addInput(undef_zp->output());
      nbits->addInput(undef_gidx->output());
      nbits->addInput(info.bias);
    }
    nbits->i_(Symbol("K"), K);
    nbits->i_(Symbol("N"), N);
    nbits->i_(Symbol("bits"), static_cast<int64_t>(4));
    nbits->i_(Symbol("block_size"), kBlockSize);
    nbits->insertBefore(n);
    nbits->setDomain("com.microsoft");
    nbits->output()->setElemType(TensorProto_DataType_FLOAT);
    nbits->output()->copyMetadata(n->output());

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    if (!tryReplacingAllUsesWith(n, nbits)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
