// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Shared OCP Microscaling MXFP4 (E2M1 elements + power-of-two E8M0 block
// scale) codebook and block-wise weight quantization helper, used by
// weight_only_quantize_mxfp4_matmul.h. Ports mx_quantization.py's own
// MXFP4_CODEBOOK / _quantize_mxfp4_blockwise -- see that module's docstring
// for the format's own definition (OCP Microscaling Formats (MX)
// Specification v1.0) and why a Gather-a-codebook idiom (rather than a
// native ONNX tensor type, which does not exist for MX formats) is used to
// represent it.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "onnx/common/ir.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"  // ReadFloatMatrix

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// E2M1's own 16 bit patterns, evaluated per the format definition (bias 1,
// subnormal exponent field 0): magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6},
// signed. Fixed by the OCP MX spec -- not fit to any data.
inline const std::vector<float>& MXFP4Codebook() {
  static const std::vector<float> kCodebook = {
      -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f,
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
  };
  return kCodebook;
}

// The largest magnitude E2M1 can represent (6.0 == 1.5 * 2^2) -- used to pick
// each block's own power-of-two shared scale so the block's own
// largest-magnitude element lands within E2M1's representable range.
constexpr float kMXFP4MaxMagnitude = 6.0f;

// OCP MX spec's own canonical block size for every MX format.
constexpr int64_t kMXBlockSize = 32;

inline uint8_t NearestMXFP4Code(float normalized) {
  const auto& codebook = MXFP4Codebook();
  size_t best = 0;
  float best_diff = std::fabs(normalized - codebook[0]);
  for (size_t i = 1; i < codebook.size(); ++i) {
    const float diff = std::fabs(normalized - codebook[i]);
    if (diff < best_diff) {
      best_diff = diff;
      best = i;
    }
  }
  return static_cast<uint8_t>(best);
}

// Block-wise MXFP4 quantization of `w_t` (a 2-D float32 constant) *in its own
// layout* (no transpose), mirroring
// TryQuantizeWeightBlockwiseInt4InPlace's channel_axis/block_size/shape
// conventions (quantize_matmul_common.h) exactly, but for MXFP4: `q_out`
// holds one UINT8 codebook index (0..15) per element (SAME shape as `w_t`)
// instead of a packed INT4 nibble, and `scale_out` holds one power-of-two
// float32 scale per (block, channel) group -- the OCP MX spec's E8M0 *value*
// (not its raw exponent-byte encoding, since ONNX has no E8M0 tensor type to
// store that encoding in).
//
// Returns false (nothing written) when `K` (the reduction axis' size) is not
// evenly divisible by `block_size`.
inline bool TryQuantizeWeightBlockwiseMXFP4InPlace(const Tensor& w_t,
                                                    int64_t channel_axis,
                                                    int64_t block_size,
                                                    Tensor& q_out,
                                                    Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  if (block_size <= 0 || K % block_size != 0) {
    return false;
  }
  const int64_t num_blocks = K / block_size;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  auto scale_index = [&](int64_t i, int64_t j) {
    const int64_t si = reduction_axis == 0 ? i / block_size : i;
    const int64_t sj = reduction_axis == 1 ? j / block_size : j;
    return si * scale_dim1 + sj;
  };

  std::vector<float> max_abs(static_cast<size_t>(scale_dim0 * scale_dim1),
                             0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      float& m = max_abs[static_cast<size_t>(scale_index(i, j))];
      m = std::max(m, std::fabs(at(i, j)));
    }
  }
  std::vector<float> scale(max_abs.size());
  for (size_t idx = 0; idx < max_abs.size(); ++idx) {
    const float m = std::max(max_abs[idx], 1e-30f);
    // The smallest power-of-two scale that keeps the block's own largest
    // magnitude within E2M1's representable range (max 6.0): ceil(), not
    // floor(log2)-2, so max_abs / scale is always <= 6.0 -- see
    // mx_quantization.py's own _quantize_mxfp4_blockwise for why the
    // floor-based alternative would silently clip.
    const float shared_exponent = std::ceil(std::log2(m / kMXFP4MaxMagnitude));
    scale[idx] = std::exp2(shared_exponent);
  }

  q_out.elem_type() = TensorProto_DataType_UINT8;
  q_out.sizes() = {dim0, dim1};
  std::vector<uint8_t> codes(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const float s = scale[static_cast<size_t>(scale_index(i, j))];
      const float normalized = at(i, j) / s;
      codes[static_cast<size_t>(i * dim1 + j)] = NearestMXFP4Code(normalized);
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {scale_dim0, scale_dim1};
  scale_out.floats() = std::move(scale);
  return true;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
