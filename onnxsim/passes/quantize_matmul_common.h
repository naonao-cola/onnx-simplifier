// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Node-matching and weight-quantization helpers shared by
// dynamic_quantize_matmul.h and static_quantize_matmul.h: both passes target
// the same MatMul / "vanilla" Gemm shape and quantize the constant weight to
// INT8 per output channel from its static values; they differ only in how
// the *activation* is quantized (a runtime-computed DynamicQuantizeLinear vs.
// a calibrated, fixed QuantizeLinear/DequantizeLinear pair) and, downstream
// of that, in the rest of the graph they build.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// The pieces of a MatMul/vanilla-Gemm node these passes care about.
struct MatMulLikeInfo {
  Value* x = nullptr;     // activation (not required to be constant)
  Value* w = nullptr;     // weight; must be a constant 2-D float32 tensor
  Value* bias = nullptr;  // Gemm's optional C input; nullptr for MatMul
  bool weight_transposed = false;  // Gemm transB == 1: W stored as [N, K]
};

// Recognizes a MatMul, or a Gemm with transA=0 and (when it has a bias)
// beta=1 and always alpha=1, filling `info`. Attributes are read with ONNX's
// documented defaults when absent.
inline bool MatchMatMulLike(Node* n, MatMulLikeInfo& info) {
  if (n->kind() == kMatMul) {
    if (n->inputs().size() != 2) {
      return false;
    }
    info.x = n->inputs()[0];
    info.w = n->inputs()[1];
    return true;
  }
  if (n->kind() == kGemm) {
    const size_t num_inputs = n->inputs().size();
    if (num_inputs != 2 && num_inputs != 3) {
      return false;
    }
    const int64_t transA = GetValueFromAttrWithDefault(n, ktransA, int64_t(0));
    const int64_t transB = GetValueFromAttrWithDefault(n, ktransB, int64_t(0));
    const double alpha = GetValueFromAttrWithDefault(n, kalpha, 1.0);
    const double beta = GetValueFromAttrWithDefault(n, kbeta, 1.0);
    if (transA != 0 || alpha != 1.0) {
      return false;
    }
    if (num_inputs == 3) {
      if (beta != 1.0) {
        return false;
      }
      info.bias = n->inputs()[2];
    }
    info.x = n->inputs()[0];
    info.w = n->inputs()[1];
    info.weight_transposed = transB != 0;
    return true;
  }
  return false;
}

// Reads `w_t` (a 2-D float32 constant) into a flat row-major, host-byte-order
// float buffer, regardless of whether it is stored as raw bytes (which are
// little-endian on the wire regardless of host -- see endian_read.h) or a
// typed float array (already host-order, decoded by protobuf itself).
inline std::vector<float> ReadFloatMatrix(const Tensor& w_t) {
  const auto& sizes = w_t.sizes();
  const int64_t numel = sizes[0] * sizes[1];
  if (w_t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(w_t.data<float>(), numel);
  }
  return w_t.floats();
}

// Quantizes `w_t` (a 2-D float32 constant, laid out as [N, K] when
// `transposed` else [K, N]) per output channel: `q_out` holds the weight in
// the non-transposed [K, N] layout MatMulInteger's B operand needs, and
// `scale_out`[j] is the symmetric INT8 scale for output channel j
// (max(|w[:, j]|) / 127, or 1.0 for an all-zero channel so no scale is 0).
// Used by the dynamic-quantization pass, whose replacement MatMulInteger
// always consumes B in [K, N] layout.
inline void QuantizeWeightPerChannelKN(const Tensor& w_t, bool transposed,
                                       Tensor& q_out, Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t K = transposed ? dim1 : dim0;
  const int64_t N = transposed ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  // element (k, n) of the logical [K, N] weight, regardless of storage
  // layout.
  auto at = [&](int64_t k, int64_t n) {
    return transposed ? data[n * K + k] : data[k * N + n];
  };

  std::vector<float> scale(static_cast<size_t>(N), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      scale[n] = std::max(scale[n], std::fabs(at(k, n)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {K, N};
  q_out.int32s().resize(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      const float q = std::round(at(k, n) / scale[n]);
      q_out.int32s()[k * N + n] =
          static_cast<int32_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {N};
  scale_out.floats() = std::move(scale);
}

// Quantizes `w_t` (a 2-D float32 constant) per output channel *in its own
// layout* (no transpose): `channel_axis` (0 or 1) is the axis of `w_t` that
// indexes the output channel. `q_out` has the SAME shape as `w_t`, and
// `scale_out`[j] is the symmetric INT8 scale for channel j. Used by the
// static-quantization pass, whose DequantizeLinear replaces the weight input
// in place -- the surrounding MatMul/Gemm node (and therefore the layout it
// expects of its weight input) is left untouched, unlike the dynamic pass's
// MatMulInteger replacement.
inline void QuantizeWeightPerChannelInPlace(const Tensor& w_t,
                                            int64_t channel_axis, Tensor& q_out,
                                            Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t C = channel_axis == 0 ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  auto channel_of = [&](int64_t i, int64_t j) {
    return channel_axis == 0 ? i : j;
  };

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      scale[c] = std::max(scale[c], std::fabs(at(i, j)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {dim0, dim1};
  q_out.int32s().resize(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      const float q = std::round(at(i, j) / scale[c]);
      q_out.int32s()[i * dim1 + j] =
          static_cast<int32_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C};
  scale_out.floats() = std::move(scale);
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
