// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// OCP Microscaling MXFP4 weight-only quantization: like
// weight_only_quantize_int4_matmul.h, only the constant weight is quantized
// and the activation is left exactly as it was -- no calibration data, no
// runtime quantize/dequantize cost on the activation path. Unlike INT4 (an
// ordinary affine DequantizeLinear), MXFP4's scale is constrained to a pure
// power of two and its 4-bit codes follow a fixed, non-uniform (E2M1
// floating-point) codebook -- ONNX has no native MX tensor type, so this
// pass builds the dequantization out of ordinary opset-11+ ops instead of a
// single DequantizeLinear. See quantize_mxfp4_common.h and
// mx_quantization.py's own docstring for the format's full definition.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W)         W constant, 2-D, float32, [K, N]
// After:
//   Codes = Cast(Wq, INT64)                     -- Wq: UINT8 codebook indices
//   Gathered = Gather(Codebook, Codes, axis=0)  -- Codebook: the 16-value E2M1
//   table Blocked  = Reshape(Gathered, <blocked shape>) ScaleB   = Reshape(Ws,
//   <matching blocked shape, block dim singleton>) Scaled   = Mul(Blocked,
//   ScaleB) Wdq      = Reshape(Scaled, W's original shape) Y = MatMul(X, Wdq)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input 1)
// is a constant 2-D float32 tensor whose reduction dimension K is evenly
// divisible by kBlockSize, and whose activation (input 0) is float32.
// Everything else is left alone.

#pragma once

#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"
#include "passes/quantize_mxfp4_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct WeightOnlyQuantizeMXFP4MatMul final : public PredicateBasedPass {
  // The OCP MX spec's own canonical block size for every MX format.
  static constexpr int64_t kBlockSize = kMXBlockSize;

  explicit WeightOnlyQuantizeMXFP4MatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "weight_only_quantize_mxfp4_matmul";
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
    if (!TryQuantizeWeightBlockwiseMXFP4InPlace(*w_t, channel_axis, kBlockSize,
                                                w_q, w_scale)) {
      return false;
    }

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    const int64_t K = reduction_axis == 0 ? dim0 : dim1;
    const int64_t num_blocks = K / kBlockSize;

    // A fresh codebook initializer per matched layer, matching this
    // codebase's existing convention of not sharing constant initializers
    // across separate PredicateBasedPass matches (e.g. fuse_bn_into_conv.h's
    // per-match epsilon constant) -- 16 float32 values is a negligible
    // overhead, and a pass instance is reused (via RegisterOrReplace's
    // std::call_once) across every future Quantize* call in the process, so
    // it must not cache any Value*/Node* pointer into a specific graph
    // across matches.
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

    // Reshape the gathered (elementwise) values and the per-block scale into
    // matching 3-D shapes that expose the block dimension explicitly, so a
    // plain Mul can broadcast each block's single scale across its own
    // block_size elements -- gathered's flat [dim0, dim1] shape has no such
    // dimension of its own to broadcast against scale's smaller
    // [scale_dim0, scale_dim1] shape directly.
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

    // The MatMul/Gemm node and its activation input are left untouched; only
    // the weight input changes, to its dequantized counterpart.
    n->replaceInput(1, reshape3->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
