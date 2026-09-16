// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// NVIDIA's NVFP4 weight-only quantization -- C++ port of
// nvfp4_quantization.py's own quantize_weight_only_nvfp4. See that
// module's docstring for the format's full rationale (NVIDIA,
// "Pretraining Large Language Models with NVFP4",
// https://arxiv.org/abs/2509.25149, 2025). NVFP4 shares OCP MXFP4's exact
// 4-bit E2M1 element codebook (reused here from
// passes/quantize_mxfp4_common.h's own MXFP4Codebook()/NearestMXFP4Code()
// -- the same 16-value table, the same nearest-code search) but replaces
// MXFP4's power-of-two-only E8M0 block scale with a two-level scale:
//
//   global_scale       = amax(tensor) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
//   raw_block_scale_i  = amax(block_i) / (global_scale * FLOAT4_E2M1_MAX)
//   block_scale_i      = round_to_nearest_e4m3(raw_block_scale_i)
//   dequant(code, i)   = E2M1_CODEBOOK[code] * block_scale_i * global_scale
//
// Block size 16 (not MXFP4's 32 -- NVFP4_BLOCK_SIZE in the Python
// module). Like weight_only_quantize_mxfp4_matmul.h, this pass stores
// `block_scale_i * global_scale` -- already E4M3-rounded, then combined
// with the per-tensor scale -- as one plain float32 value per (output
// channel, block) group, rather than the on-disk E4M3 byte plus a
// separate FP32 scalar; only the two-field on-disk *bit* layout isn't
// reproduced, the same simplification nvfp4_quantization.py's own
// docstring documents for its own graph-based encoding.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N]
// After (identical graph shape to weight_only_quantize_mxfp4_matmul.h):
//   Codes = Cast(Wq, INT64)                     -- Wq: UINT8 codebook indices
//   Gathered = Gather(Codebook, Codes, axis=0)  -- Codebook: the 16-value E2M1
//   table Blocked  = Reshape(Gathered, <blocked shape>) ScaleB   = Reshape(Ws,
//   <matching blocked shape, block dim singleton>) Scaled   = Mul(Blocked,
//   ScaleB) Wdq      = Reshape(Scaled, W's original shape) Y = MatMul(X, Wdq)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight
// (input 1) is a constant 2-D float32 tensor whose reduction dimension K
// is evenly divisible by kBlockSize. Everything else is left alone.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM nvfp4_quantization.py: none beyond
// the on-disk bit-layout simplification already noted above -- this
// scheme has no accumulation/iterative-refinement step at all (every
// block's own scale is computed independently from that block's own
// max(|.|), and the tensor's own global scale from the overall
// max(|.|)), so this port is expected to track the Python port's own
// float64 numpy implementation closely, up to floating-point summation
// order differences. quantize_weight_only_nvfp4 and this port remain
// independently-correct, non-interchangeable entry points -- this port's
// own tests check structural/algebraic properties and comparable (not
// necessarily bit-identical) reconstruction, matching this repo's
// established contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"
#include "passes/quantize_mxfp4_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace nvfp4_quantization_detail {

// NVFP4's own reference block size: groups of 16, half OCP MX's canonical
// 32 -- see nvfp4_quantization.py's own module docstring.
constexpr int64_t kNVFP4BlockSize = 16;

// E2M1's own largest representable magnitude (same value as
// quantize_mxfp4_common.h's kMXFP4MaxMagnitude, re-exported under
// NVFP4's own naming for readers coming from the NVFP4 literature).
constexpr double kFloat4E2M1Max = kMXFP4MaxMagnitude;

// E4M3 (1 sign, 4 exponent, 3 mantissa bits, the "e4m3fn" variant with no
// infinities and a single NaN pattern) largest representable magnitude.
constexpr double kFloat8E4M3Max = 448.0;

// Every nonnegative magnitude the OCP E4M3 ("e4m3fn") format can
// represent, sorted ascending -- mirrors nvfp4_quantization.py's own
// _e4m3_positive_grid(): 1 sign bit (dropped -- magnitudes only), 4
// exponent bits (bias 7), 3 mantissa bits. Exponent field 0 is subnormal;
// fields 1..15 are normal, except the single reserved NaN pattern
// (exponent 15, mantissa 0b111). 127 distinct magnitudes, 0.0 to 448.0.
// Computed in double throughout (matching nvfp4_quantization.py's own
// float64 numpy implementation) -- a single-precision grid/rounding here
// previously let the effective scale drift measurably past 448.0 once
// divided back out in a caller's own float64 check.
inline const std::vector<double>& E4M3PositiveGrid() {
  static const std::vector<double> kGrid = [] {
    std::vector<double> magnitudes;
    magnitudes.reserve(127);
    for (int exponent_field = 0; exponent_field < 16; ++exponent_field) {
      for (int mantissa = 0; mantissa < 8; ++mantissa) {
        if (exponent_field == 15 && mantissa == 7) {
          continue;  // reserved NaN pattern (S.1111.111)
        }
        double magnitude;
        if (exponent_field == 0) {
          magnitude = (mantissa / 8.0) * std::exp2(1.0 - 7.0);
        } else {
          magnitude = (1.0 + mantissa / 8.0) *
                      std::exp2(static_cast<double>(exponent_field - 7));
        }
        magnitudes.push_back(magnitude);
      }
    }
    std::sort(magnitudes.begin(), magnitudes.end());
    magnitudes.erase(std::unique(magnitudes.begin(), magnitudes.end()),
                     magnitudes.end());
    return magnitudes;
  }();
  return kGrid;
}

// Rounds a nonnegative magnitude to the nearest value E4M3 can represent,
// clamping at kFloat8E4M3Max first -- mirrors nvfp4_quantization.py's own
// _round_to_e4m3(), including its tie-break (prefers the lower neighbor
// on an exact tie, matching `(hi - clamped) < (clamped - lo) ? hi : lo`).
inline double RoundToE4M3(double magnitude) {
  const auto& grid = E4M3PositiveGrid();
  const double clamped = std::min(std::max(magnitude, 0.0), kFloat8E4M3Max);
  auto it = std::lower_bound(grid.begin(), grid.end(), clamped);
  size_t hi_idx = static_cast<size_t>(it - grid.begin());
  if (hi_idx >= grid.size()) {
    hi_idx = grid.size() - 1;
  }
  const size_t lo_idx = hi_idx > 0 ? hi_idx - 1 : 0;
  const double lo = grid[lo_idx];
  const double hi = grid[hi_idx];
  return (hi - clamped) < (clamped - lo) ? hi : lo;
}

// Block-wise NVFP4 quantization of `w_t` (a 2-D float32 constant) *in its
// own layout* (no transpose), mirroring
// TryQuantizeWeightBlockwiseMXFP4InPlace's channel_axis/block_size/shape
// conventions (quantize_mxfp4_common.h) exactly, but with NVFP4's own
// two-level (global_scale, per-block E4M3-rounded scale) rule instead of
// MXFP4's single power-of-two scale. `q_out` holds one UINT8 codebook
// index (0..15) per element (SAME shape as `w_t`); `scale_out` holds one
// *effective* scale (block_scale * global_scale, already E4M3-rounded)
// per (block, channel) group.
//
// Returns false (nothing written) when `K` (the reduction axis' size) is
// not evenly divisible by `block_size`.
inline bool TryQuantizeWeightBlockwiseNVFP4InPlace(const Tensor& w_t,
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

  const std::vector<float> raw_data = ReadFloatMatrix(w_t);
  const std::vector<double> data(raw_data.begin(), raw_data.end());
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  auto scale_index = [&](int64_t i, int64_t j) {
    const int64_t si = reduction_axis == 0 ? i / block_size : i;
    const int64_t sj = reduction_axis == 1 ? j / block_size : j;
    return si * scale_dim1 + sj;
  };

  double tensor_amax = 0.0;
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      tensor_amax = std::max(tensor_amax, std::fabs(at(i, j)));
    }
  }
  tensor_amax = std::max(tensor_amax, 1e-30);
  const double global_scale = tensor_amax / (kFloat4E2M1Max * kFloat8E4M3Max);

  std::vector<double> block_amax(static_cast<size_t>(scale_dim0 * scale_dim1),
                                 0.0);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      double& m = block_amax[static_cast<size_t>(scale_index(i, j))];
      m = std::max(m, std::fabs(at(i, j)));
    }
  }

  std::vector<double> effective_scale(block_amax.size());
  for (size_t idx = 0; idx < block_amax.size(); ++idx) {
    const double bamax = std::max(block_amax[idx], 1e-30);
    const double raw_block_scale = bamax / (global_scale * kFloat4E2M1Max);
    const double block_scale = RoundToE4M3(raw_block_scale);
    effective_scale[idx] = block_scale * global_scale;
  }

  q_out.elem_type() = TensorProto_DataType_UINT8;
  q_out.sizes() = {dim0, dim1};
  std::vector<uint8_t> codes(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const double s = effective_scale[static_cast<size_t>(scale_index(i, j))];
      const double normalized = at(i, j) / s;
      codes[static_cast<size_t>(i * dim1 + j)] =
          NearestMXFP4Code(static_cast<float>(normalized));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {scale_dim0, scale_dim1};
  std::vector<float> effective_scale_f32(effective_scale.begin(),
                                         effective_scale.end());
  scale_out.floats() = std::move(effective_scale_f32);
  return true;
}

}  // namespace nvfp4_quantization_detail

// NVIDIA's NVFP4 weight-only quantization -- matches MatMul/vanilla-Gemm
// the same way WeightOnlyQuantizeMXFP4MatMul does, then rebuilds the
// dequantization out of ordinary opset-11+ ops (Cast/Gather/Reshape/Mul),
// identical graph shape to that sibling pass.
struct NVFP4Quantization final : public PredicateBasedPass {
  static constexpr int64_t kBlockSize =
      nvfp4_quantization_detail::kNVFP4BlockSize;

  explicit NVFP4Quantization()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "nvfp4_quantization"; }

  bool patternMatchPredicate(Node* n) override {
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

    const int64_t channel_axis = info.weight_transposed ? 0 : 1;
    const int64_t reduction_axis = 1 - channel_axis;
    Tensor w_q;
    Tensor w_scale;
    if (!nvfp4_quantization_detail::TryQuantizeWeightBlockwiseNVFP4InPlace(
            *w_t, channel_axis, kBlockSize, w_q, w_scale)) {
      return false;
    }

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t K = reduction_axis == 0 ? dim0 : dim1;
    const int64_t num_blocks = K / kBlockSize;

    // A fresh codebook initializer per matched layer -- see
    // weight_only_quantize_mxfp4_matmul.h's own comment on why (must not
    // cache a Value*/Node* pointer into a specific graph across matches).
    Tensor codebook_t;
    codebook_t.elem_type() = TensorProto_DataType_FLOAT;
    codebook_t.sizes() = {static_cast<int64_t>(MXFP4Codebook().size())};
    codebook_t.floats() = MXFP4Codebook();
    Value* codebook_v = graph.addInitializerAndCreateValue(codebook_t);

    Value* codes_v = graph.addInitializerAndCreateValue(w_q);
    Value* scale_v = graph.addInitializerAndCreateValue(w_scale);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(codes_v);
    cast->i_(kto, TensorProto_DataType_INT64);
    cast->insertBefore(n);
    cast->output()->setElemType(TensorProto_DataType_INT64);
    cast->output()->setSizes(codes_v->sizes());

    Node* gather = graph.create(Symbol("Gather"), 1);
    gather->addInput(codebook_v);
    gather->addInput(cast->output());
    gather->i_(kaxis, 0);
    gather->insertBefore(n);
    gather->output()->setElemType(TensorProto_DataType_FLOAT);
    gather->output()->setSizes(codes_v->sizes());

    std::vector<int64_t> blocked_shape;
    std::vector<int64_t> scale_shape;
    if (reduction_axis == 1) {
      blocked_shape = {dim0, num_blocks, kBlockSize};
      scale_shape = {dim0, num_blocks, 1};
    } else {
      blocked_shape = {num_blocks, kBlockSize, dim1};
      scale_shape = {num_blocks, 1, dim1};
    }

    auto make_shape_initializer = [&](const std::vector<int64_t>& shape) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = {static_cast<int64_t>(shape.size())};
      t.int64s() = shape;
      return graph.addInitializerAndCreateValue(t);
    };

    Node* reshape1 = graph.create(kReshape, 1);
    reshape1->addInput(gather->output());
    reshape1->addInput(make_shape_initializer(blocked_shape));
    reshape1->insertBefore(n);
    reshape1->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* reshape2 = graph.create(kReshape, 1);
    reshape2->addInput(scale_v);
    reshape2->addInput(make_shape_initializer(scale_shape));
    reshape2->insertBefore(n);
    reshape2->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* mul = graph.create(kMul, 1);
    mul->addInput(reshape1->output());
    mul->addInput(reshape2->output());
    mul->insertBefore(n);
    mul->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* reshape3 = graph.create(kReshape, 1);
    reshape3->addInput(mul->output());
    reshape3->addInput(make_shape_initializer({dim0, dim1}));
    reshape3->insertBefore(n);
    reshape3->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.w->sizes().size() > 0) {
      reshape3->output()->setSizes(info.w->sizes());
    }

    n->replaceInput(1, reshape3->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
