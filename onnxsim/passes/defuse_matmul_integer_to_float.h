// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// The exact inverse of dynamic_quantize_matmul.h's rewrite: folds a
// DynamicQuantizeLinear + MatMulInteger + Cast<float> + Mul + Mul (+ Add)
// dequantization chain back into a single plain MatMul (+ Add for the
// optional bias), by dequantizing the constant INT8 weight to float once,
// here, at pass-transform time.
//
// Before (the exact shape dynamic_quantize_matmul.h emits):
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Acc         = MatMulInteger(Xq, Wq, Xzp)          // int32, Wq int8 [K,N]
//   Y           = Cast<float>(Acc) * (Xs * Ws)        // Ws float32 [N]
//   [Y = Y + Bias]                                    // optional
// After:
//   Y = MatMul(X, Dequant(Wq, Ws))    // Dequant(Wq, Ws)[k,n] = Wq[k,n]*Ws[n]
//   [Y = Y + Bias]
//
// This is a pure, data-independent graph transform: W is already a
// compile-time constant, so its dequantized float value can be computed once
// here rather than left as a runtime DequantizeLinear node.
//
// It exists for consumers of onnxsim's output (e.g. compiler frontends) that
// do not support MatMulInteger / DynamicQuantizeLinear and would rather see
// the original plain-float MatMul than fail to import the model at all --
// accepting the loss of the quantized model's actual runtime benefit
// (smaller weights, integer compute) in exchange for portability. Because
// that tradeoff is a deliberate, model-changing choice only the caller can
// make, this pass is PassType::Other and is never in onnxsim's default pass
// set: it only runs when explicitly requested, via
// `extra_optimizers=["defuse_matmul_integer_to_float"]` (Python) or
// `--enable-optimization defuse_matmul_integer_to_float` (CLI).
//
// Only the exact shape dynamic_quantize_matmul.h produces is recognized: a
// MatMulInteger with exactly 3 inputs (no explicit b_zero_point -- symmetric,
// implicit 0), whose quantized-activation and zero-point inputs come from
// the same DynamicQuantizeLinear node as the scale multiplicand's
// activation-scale operand, and whose weight is a constant 2-D INT8 tensor
// paired with a constant 1-D float32 per-column scale of matching length.
// Anything else -- statically-quantized (QuantizeLinear/DequantizeLinear)
// matmuls, an explicit w_zero_point, ONNX Runtime's fused
// "MatMulIntegerToFloat" contrib op (see
// dynamic_quantize_matmul_integer_to_float.h), a scale computed some other
// way -- is left alone.

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

struct MatMulIntegerDequantInfo {
  Value* x = nullptr;               // original float32 activation
  const Tensor* w_q = nullptr;      // constant INT8 [K, N]
  const Tensor* w_scale = nullptr;  // constant FLOAT [N]
};

// Recognizes `dequant_mul` as the "Y = Cast<float>(Acc) * (Xs * Ws)" node of
// dynamic_quantize_matmul.h's output pattern, filling `info`. See the file
// comment above for the exact shape matched.
inline bool MatchMatMulIntegerDequant(Node* dequant_mul,
                                      MatMulIntegerDequantInfo& info) {
  if (dequant_mul->kind() != kMul || dequant_mul->inputs().size() != 2) {
    return false;
  }
  const Symbol matmul_integer = Symbol("MatMulInteger");
  const Symbol dynamic_quantize_linear = Symbol("DynamicQuantizeLinear");
  for (size_t i = 0; i < 2; ++i) {
    Value* cast_out = dequant_mul->inputs()[i];
    Value* scale_out = dequant_mul->inputs()[1 - i];
    Node* cast = cast_out->node();
    if (cast->kind() != kCast || cast->inputs().size() != 1 ||
        cast->output() != cast_out) {
      continue;
    }
    if (GetValueFromAttrWithDefault(cast, kto, int64_t(0)) !=
        TensorProto_DataType_FLOAT) {
      continue;
    }

    Node* mmi = cast->inputs()[0]->node();
    if (mmi->kind() != matmul_integer || mmi->inputs().size() != 3 ||
        mmi->output() != cast->inputs()[0]) {
      continue;
    }
    Value* x_q = mmi->inputs()[0];
    Value* w_q_val = mmi->inputs()[1];
    Value* x_zp = mmi->inputs()[2];

    Node* dql = x_q->node();
    if (dql->kind() != dynamic_quantize_linear || dql->inputs().size() != 1 ||
        dql->outputs().size() != 3 || dql->outputs()[0] != x_q ||
        dql->outputs()[2] != x_zp) {
      continue;
    }
    Value* x_scale = dql->outputs()[1];

    Node* scale_mul = scale_out->node();
    if (scale_mul->kind() != kMul || scale_mul->inputs().size() != 2 ||
        scale_mul->output() != scale_out) {
      continue;
    }
    Value* w_scale_val = nullptr;
    bool found_x_scale = false;
    for (Value* v : scale_mul->inputs()) {
      if (v == x_scale) {
        found_x_scale = true;
      } else {
        w_scale_val = v;
      }
    }
    if (!found_x_scale || w_scale_val == nullptr) {
      continue;
    }

    const Tensor* w_q_t = FetchConstantTensor(w_q_val);
    if (w_q_t == nullptr || w_q_t->elem_type() != TensorProto_DataType_INT8 ||
        w_q_t->sizes().size() != 2) {
      continue;
    }
    const Tensor* w_scale_t = FetchConstantTensor(w_scale_val);
    if (w_scale_t == nullptr ||
        w_scale_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_scale_t->sizes().size() != 1 ||
        w_scale_t->sizes()[0] != w_q_t->sizes()[1]) {
      continue;
    }

    info.x = dql->inputs()[0];
    info.w_q = w_q_t;
    info.w_scale = w_scale_t;
    return true;
  }
  return false;
}

// Dequantizes `w_q` (INT8, [K, N]) with `w_scale` ([N]) into a FLOAT [K, N]
// Tensor: out[k, n] = w_q[k, n] * w_scale[n].
inline Tensor DequantizeWeightKN(const Tensor& w_q, const Tensor& w_scale) {
  const int64_t K = w_q.sizes()[0];
  const int64_t N = w_q.sizes()[1];
  const std::vector<int8_t> codes = ReadInt8Matrix(w_q);

  // w_scale is 1-D ([N]); ReadFloatMatrix (quantize_matmul_common.h) requires
  // rank 2, so its raw/typed dispatch is inlined here instead of reused.
  std::vector<float> scale_flat;
  if (w_scale.is_raw_data()) {
    scale_flat = ReadRawDataHostOrder<float>(w_scale.data<float>(), N);
  } else {
    scale_flat = w_scale.floats();
  }

  std::vector<float> dequant(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      dequant[static_cast<size_t>(k * N + n)] =
          static_cast<float>(codes[static_cast<size_t>(k * N + n)]) *
          scale_flat[static_cast<size_t>(n)];
    }
  }

  Tensor out;
  out.elem_type() = TensorProto_DataType_FLOAT;
  out.sizes() = {K, N};
  out.set_raw_data(WriteRawDataLittleEndian(dequant));
  return out;
}

struct DefuseMatMulIntegerToFloat final : public PredicateBasedPass {
  explicit DefuseMatMulIntegerToFloat()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "defuse_matmul_integer_to_float";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulIntegerDequantInfo info;
    if (n->kind() == kMul) {
      return MatchMatMulIntegerDequant(n, info);
    }
    if (n->kind() == kAdd && n->inputs().size() == 2) {
      // v->node() can be a multi-output pseudo-node for a graph input or
      // initializer (e.g. the bias itself), so MatchMatMulIntegerDequant's
      // own kind() == kMul check -- not an unconditional output() call,
      // which asserts exactly one output -- must run first.
      for (Value* v : n->inputs()) {
        if (MatchMatMulIntegerDequant(v->node(), info)) {
          return true;
        }
      }
    }
    return false;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    MatMulIntegerDequantInfo info;
    Value* bias = nullptr;
    if (n->kind() == kMul) {
      if (!MatchMatMulIntegerDequant(n, info)) {
        return false;
      }
    } else if (n->kind() == kAdd && n->inputs().size() == 2) {
      bool found = false;
      for (Value* v : n->inputs()) {
        if (MatchMatMulIntegerDequant(v->node(), info)) {
          bias = (v == n->inputs()[0]) ? n->inputs()[1] : n->inputs()[0];
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    } else {
      return false;
    }

    Tensor w_dequant = DequantizeWeightKN(*info.w_q, *info.w_scale);
    Value* w_dequant_value = graph.addInitializerAndCreateValue(w_dequant);

    Node* matmul = graph.create(kMatMul, 1);
    matmul->addInput(info.x);
    matmul->addInput(w_dequant_value);
    matmul->insertBefore(n);
    matmul->output()->setElemType(TensorProto_DataType_FLOAT);

    Value* result = matmul->output();
    if (bias != nullptr) {
      Node* add = graph.create(kAdd, 1);
      add->addInput(matmul->output());
      add->addInput(bias);
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
