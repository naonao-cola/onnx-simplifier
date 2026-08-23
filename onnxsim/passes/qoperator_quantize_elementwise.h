// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes elementwise Add/Mul between two
// runtime (non-constant) float32 tensors into ONNX Runtime's "com.microsoft"
// contrib ops QLinearAdd/QLinearMul -- the elementwise analogue of
// qoperator_quantize_matmul.h's QLinearMatMul rewrite. Unlike every other
// quantization pass in onnxsim, QLinearAdd/QLinearMul are NOT standard ONNX:
// standard ONNX has no quantized elementwise-binary op at all -- only
// QLinearMatMul/QLinearConv, both constrained to a weight-times-activation
// shape (see contrib_schemas.cpp's doc comment on why QLinearAdd/QLinearMul
// exist as contrib ops in the first place). The resulting model needs a
// com.microsoft-aware runtime (ONNX Runtime itself, or another runtime that
// imports the same contrib schemas) to execute, not just any ONNX Runtime.
//
// Before (illustrated for Add; Mul is identical but for the op/QLinear* name):
//   Z = Add(A, B)             A, B: both runtime float32 tensors
// After:
//   Aq = QuantizeLinear(A, As, Azp)                     -- As/Azp: CALIBRATED
//   Bq = QuantizeLinear(B, Bs, Bzp)                     -- Bs/Bzp: CALIBRATED
//   Zq = QLinearAdd(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp)  -- true int8 compute
//   Z  = DequantizeLinear(Zq, Zs, Zzp)                  -- Zs/Zzp: CALIBRATED
//
// Unlike qoperator_quantize_matmul.h (one calibrated activation, one
// statically-quantized-from-its-own-values constant weight), QLinearAdd/
// QLinearMul have no "weight" role at all -- the schema treats both operands
// identically -- so BOTH A and B need a calibrated range, on top of the
// output Z's (QOperator format computes directly in int8, so its output must
// be quantized too, same reason QLinearMatMul's Y needs one -- see that
// file's doc comment). Only a node whose *both* inputs are non-constant is
// matched: a constant operand (e.g. a per-channel bias/embedding added
// elementwise) is better quantized from its own static values than
// force-fed through the runtime calibration harness as if it varied at
// inference time, so this pass leaves that case alone -- the matching
// pattern is deliberately narrower than QLinearAdd/QLinearMul's own schema,
// which allows either operand to be anything.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// Only Add/Mul with exactly 2 inputs, both float32, neither a constant
// value, are matched; a node is only rewritten when both its inputs' names
// and its own output's name have a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeQOperatorElementwise in
// onnxsim.h).

#pragma once

#include <cstdint>
#include <string>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct QOperatorQuantizeElementwise final : public PredicateBasedPass {
  explicit QOperatorQuantizeElementwise()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "qoperator_quantize_elementwise";
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != kAdd && n->kind() != kMul) {
      return false;
    }
    if (n->inputs().size() != 2) {
      return false;
    }
    Value* a = n->inputs()[0];
    Value* b = n->inputs()[1];
    if (a->elemType() != TensorProto_DataType_FLOAT ||
        b->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (FetchConstantTensor(a) != nullptr ||
        FetchConstantTensor(b) != nullptr) {
      return false;  // Constant operand: quantize from its own values instead.
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    return ranges.count(a->uniqueName()) != 0 &&
           ranges.count(b->uniqueName()) != 0 &&
           ranges.count(n->output()->uniqueName()) != 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (n->kind() != kAdd && n->kind() != kMul) {
      return false;
    }
    if (n->inputs().size() != 2) {
      return false;
    }
    Value* a = n->inputs()[0];
    Value* b = n->inputs()[1];
    if (a->elemType() != TensorProto_DataType_FLOAT ||
        b->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (FetchConstantTensor(a) != nullptr ||
        FetchConstantTensor(b) != nullptr) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto a_range_it = ranges.find(a->uniqueName());
    const auto b_range_it = ranges.find(b->uniqueName());
    const auto z_range_it = ranges.find(n->output()->uniqueName());
    if (a_range_it == ranges.end() || b_range_it == ranges.end() ||
        z_range_it == ranges.end()) {
      return false;
    }

    float a_scale_f = 1.0f;
    int32_t a_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        a_range_it->second.first, a_range_it->second.second, a_scale_f, a_zp_i);
    float b_scale_f = 1.0f;
    int32_t b_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        b_range_it->second.first, b_range_it->second.second, b_scale_f, b_zp_i);
    float z_scale_f = 1.0f;
    int32_t z_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        z_range_it->second.first, z_range_it->second.second, z_scale_f, z_zp_i);

    Tensor a_scale_t;
    a_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    a_scale_t.floats() = {a_scale_f};
    Tensor a_zp_t;
    a_zp_t.elem_type() = TensorProto_DataType_UINT8;
    a_zp_t.int32s() = {a_zp_i};
    Value* a_scale_v = graph.addInitializerAndCreateValue(a_scale_t);
    Value* a_zp_v = graph.addInitializerAndCreateValue(a_zp_t);

    Tensor b_scale_t;
    b_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    b_scale_t.floats() = {b_scale_f};
    Tensor b_zp_t;
    b_zp_t.elem_type() = TensorProto_DataType_UINT8;
    b_zp_t.int32s() = {b_zp_i};
    Value* b_scale_v = graph.addInitializerAndCreateValue(b_scale_t);
    Value* b_zp_v = graph.addInitializerAndCreateValue(b_zp_t);

    Tensor z_scale_t;
    z_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    z_scale_t.floats() = {z_scale_f};
    Tensor z_zp_t;
    z_zp_t.elem_type() = TensorProto_DataType_UINT8;
    z_zp_t.int32s() = {z_zp_i};
    Value* z_scale_v = graph.addInitializerAndCreateValue(z_scale_t);
    Value* z_zp_v = graph.addInitializerAndCreateValue(z_zp_t);

    // Aq = QuantizeLinear(A, As, Azp)
    Node* aq = graph.create(Symbol("QuantizeLinear"), 1);
    aq->addInput(a);
    aq->addInput(a_scale_v);
    aq->addInput(a_zp_v);
    aq->insertBefore(n);
    aq->output()->setElemType(TensorProto_DataType_UINT8);

    // Bq = QuantizeLinear(B, Bs, Bzp)
    Node* bq = graph.create(Symbol("QuantizeLinear"), 1);
    bq->addInput(b);
    bq->addInput(b_scale_v);
    bq->addInput(b_zp_v);
    bq->insertBefore(n);
    bq->output()->setElemType(TensorProto_DataType_UINT8);

    // Zq = QLinear{Add,Mul}(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp), a
    // "com.microsoft" contrib op -- see QLinearBinaryShapeInference in
    // contrib_schemas.cpp for its output-type/broadcast convention, which
    // matches this input order (A at index 0, B at index 3).
    const bool is_add = n->kind() == kAdd;
    Node* qlop = graph.create(Symbol(is_add ? "QLinearAdd" : "QLinearMul"), 1);
    qlop->addInput(aq->output());
    qlop->addInput(a_scale_v);
    qlop->addInput(a_zp_v);
    qlop->addInput(bq->output());
    qlop->addInput(b_scale_v);
    qlop->addInput(b_zp_v);
    qlop->addInput(z_scale_v);
    qlop->addInput(z_zp_v);
    qlop->setDomain("com.microsoft");
    qlop->insertBefore(n);
    qlop->output()->setElemType(TensorProto_DataType_UINT8);

    // Z = DequantizeLinear(Zq, Zs, Zzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qlop->output());
    dq->addInput(z_scale_v);
    dq->addInput(z_zp_v);
    dq->insertBefore(n);
    dq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (n->output()->sizes().size() > 0) {
      dq->output()->setSizes(n->output()->sizes());
    }

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

    const bool replacing_success =
        tryReplacingAllUsesWith(n->output(), dq->output());
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
