// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, with its optional bias C added back in
// float after dequantization):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Acc         = MatMulInteger(Xq, Wq, Xzp)          // int32
//   Y           = Cast<float>(Acc) * (Xs * Ws)        // dequantize
//
// Wq (int8) and Ws (float32, one scale per output column of W) are computed
// once, here, from W's static values -- ordinary per-output-channel symmetric
// INT8 quantization, zero_point 0. X, by contrast, is not known until the
// model runs, so it is quantized to uint8 *in the graph* by
// ``DynamicQuantizeLinear``, which computes its own scale/zero-point from
// each run's actual input range. The whole rewrite is therefore a pure,
// data-independent graph transform -- same shape as onnxsim's other passes --
// unlike calibration-based ("static") quantization, which needs sample data
// run through the model first to pick activation ranges.
//
// This mirrors the "dynamic quantization" scheme ONNX Runtime's
// ``quantize_dynamic`` applies to MatMul/Gemm.
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1 (transB may be 0 or 1), whose weight (input 1)
// is a constant 2-D float32 tensor and whose activation (input 0) is float32.
// Everything else -- non-constant or non-2-D weights, non-default Gemm
// attributes, non-float32 operands, opsets older than 11 (DynamicQuantizeLinear
// requires opset 11) -- is left alone.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct DynamicQuantizeMatMul final : public PredicateBasedPass {
  explicit DynamicQuantizeMatMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "dynamic_quantize_matmul"; }

  // The pieces of a MatMul/vanilla-Gemm node this pass cares about.
  struct MatMulLikeInfo {
    Value* x = nullptr;     // activation (not required to be constant)
    Value* w = nullptr;     // weight; must be a constant 2-D float32 tensor
    Value* bias = nullptr;  // Gemm's optional C input; nullptr for MatMul
    bool weight_transposed = false;  // Gemm transB == 1: W stored as [N, K]
  };

  // Recognizes a MatMul, or a Gemm with transA=0 and (when it has a bias)
  // beta=1 and always alpha=1, filling `info`. Attributes are read with
  // ONNX's documented defaults when absent.
  static bool MatchMatMulLike(Node* n, MatMulLikeInfo& info) {
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
      const int64_t transA =
          GetValueFromAttrWithDefault(n, ktransA, int64_t(0));
      const int64_t transB =
          GetValueFromAttrWithDefault(n, ktransB, int64_t(0));
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

  // Quantizes `w_t` (a 2-D float32 constant, laid out as [N, K] when
  // `transposed` else [K, N]) per output channel: `q_out` holds the weight in
  // the non-transposed [K, N] layout MatMulInteger's B operand needs, and
  // `scale_out`[j] is the symmetric INT8 scale for output channel j
  // (max(|w[:, j]|) / 127, or 1.0 for an all-zero channel so no scale is 0).
  static void QuantizeWeightPerChannel(const Tensor& w_t, bool transposed,
                                       Tensor& q_out, Tensor& scale_out) {
    const auto& sizes = w_t.sizes();
    const int64_t dim0 = sizes[0];
    const int64_t dim1 = sizes[1];
    const int64_t K = transposed ? dim1 : dim0;
    const int64_t N = transposed ? dim0 : dim1;

    std::vector<float> data;
    if (w_t.is_raw_data()) {
      const float* raw = w_t.data<float>();
      data.assign(raw, raw + static_cast<size_t>(dim0 * dim1));
    } else {
      data = w_t.floats();
    }
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

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 11) {
      return false;  // DynamicQuantizeLinear needs opset >= 11.
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

    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannel(*w_t, info.weight_transposed, w_q, w_scale);

    // Xq, Xs, Xzp = DynamicQuantizeLinear(X)
    Node* dql = graph.create(Symbol("DynamicQuantizeLinear"), 3);
    dql->addInput(info.x);
    dql->insertBefore(n);
    Value* x_q = dql->outputs()[0];
    Value* x_scale = dql->outputs()[1];
    Value* x_zp = dql->outputs()[2];
    x_q->setElemType(TensorProto_DataType_UINT8);
    x_scale->setElemType(TensorProto_DataType_FLOAT);
    x_zp->setElemType(TensorProto_DataType_UINT8);

    Value* w_q_value = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_value = graph.addInitializerAndCreateValue(w_scale);

    // Acc = MatMulInteger(Xq, Wq, Xzp)   -- b_zero_point omitted (symmetric,
    // i.e. always 0).
    Node* mmi = graph.create(Symbol("MatMulInteger"), 1);
    mmi->addInput(x_q);
    mmi->addInput(w_q_value);
    mmi->addInput(x_zp);
    mmi->insertBefore(n);
    mmi->output()->setElemType(TensorProto_DataType_INT32);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(mmi->output());
    cast->i_(kto, TensorProto_DataType_FLOAT);
    cast->insertBefore(n);
    cast->output()->setElemType(TensorProto_DataType_FLOAT);

    // combined_scale[j] = Xs * Ws[j], computed at runtime since Xs is not
    // known until the model runs.
    Node* scale_mul = graph.create(kMul, 1);
    scale_mul->addInput(x_scale);
    scale_mul->addInput(w_scale_value);
    scale_mul->insertBefore(n);
    scale_mul->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* dequant = graph.create(kMul, 1);
    dequant->addInput(cast->output());
    dequant->addInput(scale_mul->output());
    dequant->insertBefore(n);
    dequant->output()->setElemType(TensorProto_DataType_FLOAT);

    Value* result = dequant->output();
    if (info.bias != nullptr) {
      Node* add = graph.create(kAdd, 1);
      add->addInput(dequant->output());
      add->addInput(info.bias);
      add->insertBefore(n);
      add->output()->setElemType(TensorProto_DataType_FLOAT);
      result = add->output();
    }

    if (n->output()->sizes().size() > 0) {
      result->setSizes(n->output()->sizes());
    }

    const bool replacing_success = tryReplacingAllUsesWith(n->output(), result);
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
