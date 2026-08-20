// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Node-matching and weight-quantization helpers for Conv, mirroring
// quantize_matmul_common.h's role for MatMul/Gemm: both quantize a constant
// weight to INT8 per output channel from its static values. Conv's weight
// layout ([Cout, Cin/groups, k...]) always puts the output channel on axis 0
// (unlike MatMul/Gemm, whose weight can be transposed), so there is no
// transposed-layout case to handle here.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// The pieces of a Conv node these passes care about.
struct ConvInfo {
  Value* x = nullptr;     // activation (not required to be constant)
  Value* w = nullptr;     // weight; must be a constant float32 tensor, rank
                          // >= 3 ([Cout, Cin/groups, k...])
  Value* bias = nullptr;  // Conv's optional B input; nullptr if absent
};

// Recognizes a Conv node, filling `info`. Kernel/stride/pad/dilation/group/
// auto_pad attributes are untouched by the caller -- Conv itself is never
// replaced, so they need no special handling.
inline bool MatchConv(Node* n, ConvInfo& info) {
  if (n->kind() != kConv) {
    return false;
  }
  const size_t num_inputs = n->inputs().size();
  if (num_inputs != 2 && num_inputs != 3) {
    return false;
  }
  info.x = n->input(0);
  info.w = n->input(1);
  if (num_inputs == 3) {
    info.bias = n->input(2);
  }
  return true;
}

// Reads `t` (a float32 constant of any rank) into a flat row-major,
// host-byte-order buffer, regardless of whether it is stored as raw bytes
// (which are little-endian on the wire regardless of host -- see
// endian_read.h) or a typed float array (already host-order, decoded by
// protobuf itself).
inline std::vector<float> ReadFloatTensorFlat(const Tensor& t) {
  int64_t numel = 1;
  for (const auto& s : t.sizes()) {
    numel *= s;
  }
  if (t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(t.data<float>(), numel);
  }
  return t.floats();
}

// Quantizes `w_t` (a constant float32 Conv weight, [Cout, Cin/groups, k...])
// per output channel (axis 0 -- Conv's layout gives no other choice):
// `q_out` has the same shape as `w_t`, and `scale_out`[c] is the symmetric
// INT8 scale for output channel c (max(|w[c, ...]|) / 127, or 1.0 for an
// all-zero channel so no scale is 0).
inline void QuantizeConvWeightPerOutputChannel(const Tensor& w_t, Tensor& q_out,
                                               Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t C = sizes[0];
  int64_t inner = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    inner *= sizes[i];
  }
  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      scale[c] = std::max(scale[c], std::fabs(data[c * inner + j]));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = sizes;
  q_out.int32s().resize(static_cast<size_t>(C * inner));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      const float q = std::round(data[c * inner + j] / scale[c]);
      q_out.int32s()[c * inner + j] =
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
