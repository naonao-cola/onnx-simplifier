// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Double quantization (Dettmers et al., 2023, "QLoRA: Efficient Finetuning
// of Quantized LLMs", Section 3.2) -- see double_quantization.py's own
// docstring for the full rationale. Unlike every other pass in this
// directory, this one has no "live weight" to quantize: it operates purely
// on an *already-quantized* graph, as a second pass over every
// DequantizeLinear node's own scale tensor.
//
// Every block-wise/per-channel scheme in onnxsim (weight_only_quantize_*,
// static_quantize_*, ...) stores one float32 scale per quantization block --
// e.g. a 32-element INT4 block needs 16 bytes of codes plus 4 bytes of
// scale, a 25% overhead. This pass quantizes the scale tensor itself, one
// more level down: for every DequantizeLinear node whose scale input (input
// 1) is a constant float32 tensor with at least kMinElements values (a
// per-block or per-channel scale -- a single scalar per-tensor scale isn't
// worth a second quantizer's own overhead), the scale is quantized to UINT8
// with a single per-tensor meta-scale and reconstructed via a second, nested
// DequantizeLinear feeding into the original node's own scale input:
//
//   Before:
//     Whatever = DequantizeLinear(Codes, Scale, ...)  -- Scale: constant
//   After:
//     ScaleCodes: initializer, uint8, same shape as Scale
//     MetaScale: initializer, float32 scalar, max(|Scale|) / 255
//     ScaleHat = DequantizeLinear(ScaleCodes, MetaScale)
//     Whatever = DequantizeLinear(Codes, ScaleHat, ...)  -- attrs unchanged
//
// This is technique-agnostic (it only looks at the DequantizeLinear node's
// own scale input, not who produced it), so it composes with every
// block-wise/per-channel onnxsim scheme unchanged. It runs to a fixed point
// naturally: the rewritten outer node's scale input (ScaleHat) is no longer
// a constant initializer, so it never matches a second time, and the inner
// DequantizeLinear's own scale (MetaScale, a single scalar) is always well
// under kMinElements. The original float32 scale initializer is left in the
// graph, unreferenced, like every other onnxsim rewrite's now-dead
// initializer.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_conv_common.h"  // ReadFloatTensorFlat

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// ReadFloatTensorFlat (quantize_conv_common.h) reads every element of a
// tensor of any rank into a flat host-order vector<float> -- used here since
// a DequantizeLinear scale tensor may be scalar, 1-D (per-channel), or 2-D
// (per-block), unlike ReadFloatMatrix (quantize_matmul_common.h), which
// targets only the always-2-D weight tensors its own callers match.

inline bool MatchDoubleQuantizationCandidate(Node* n, int64_t min_elements,
                                             const Tensor** scale_t_out) {
  if (n->kind() != Symbol("DequantizeLinear") || n->inputs().size() < 2) {
    return false;
  }
  const Tensor* scale_t = FetchConstantTensor(n->inputs()[1]);
  if (scale_t == nullptr ||
      scale_t->elem_type() != TensorProto_DataType_FLOAT) {
    return false;
  }
  int64_t numel = 1;
  for (const int64_t d : scale_t->sizes()) {
    numel *= d;
  }
  if (numel < min_elements) {
    return false;
  }
  *scale_t_out = scale_t;
  return true;
}

struct DoubleQuantization final : public PredicateBasedPass {
  // A per-tensor scalar scale (numel == 1) is never worth a second
  // quantizer's own overhead (a meta-scale initializer plus a new node); 64
  // matches double_quantization.py's own default.
  static constexpr int64_t kMinElements = 64;

  explicit DoubleQuantization()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "double_quantization"; }

  bool patternMatchPredicate(Node* n) override {
    const Tensor* scale_t = nullptr;
    return MatchDoubleQuantizationCandidate(n, kMinElements, &scale_t);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const Tensor* scale_t = nullptr;
    if (!MatchDoubleQuantizationCandidate(n, kMinElements, &scale_t)) {
      return false;
    }
    Value* scale_v = n->inputs()[1];

    const std::vector<float> scale_data = ReadFloatTensorFlat(*scale_t);
    double max_abs = 0.0;
    for (const float v : scale_data) {
      max_abs = std::max(max_abs, static_cast<double>(std::fabs(v)));
    }
    // scale_data holds ordinary absmax-derived quantization scales, always
    // non-negative, so a plain unsigned 0..255 range needs no zero-point
    // offset.
    const float meta_scale =
        static_cast<float>(std::max(max_abs, 1e-12) / 255.0);

    std::vector<uint8_t> codes(scale_data.size());
    for (size_t i = 0; i < scale_data.size(); ++i) {
      const float q = std::round(scale_data[i] / meta_scale);
      codes[i] = static_cast<uint8_t>(std::clamp(q, 0.0f, 255.0f));
    }

    Tensor codes_t;
    codes_t.elem_type() = TensorProto_DataType_UINT8;
    codes_t.sizes() = scale_t->sizes();
    codes_t.set_raw_data(WriteRawDataLittleEndian(codes));
    Value* codes_v = graph.addInitializerAndCreateValue(codes_t);

    Tensor meta_scale_t;
    meta_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    meta_scale_t.floats() = {meta_scale};
    Value* meta_scale_v = graph.addInitializerAndCreateValue(meta_scale_t);

    // ScaleHat = DequantizeLinear(ScaleCodes, MetaScale) -- zero_point
    // omitted (symmetric-unsigned, i.e. always 0).
    Node* inner_dq = graph.create(Symbol("DequantizeLinear"), 1);
    inner_dq->addInput(codes_v);
    inner_dq->addInput(meta_scale_v);
    inner_dq->insertBefore(n);
    inner_dq->output()->setElemType(TensorProto_DataType_FLOAT);
    inner_dq->output()->setSizes(scale_v->sizes());

    // `n` (the original DequantizeLinear) is left otherwise untouched; only
    // its scale input changes, to its dequantized-from-UINT8 reconstruction.
    n->replaceInput(1, inner_dq->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
