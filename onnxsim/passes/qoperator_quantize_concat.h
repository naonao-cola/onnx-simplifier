// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes a `Concat` node whose inputs are
// all non-constant float32 tensors into ONNX Runtime's "com.microsoft"
// contrib op QLinearConcat -- the variadic analogue of
// qoperator_quantize_elementwise.h's QLinearAdd/QLinearMul rewrite (see that
// file's doc comment for why these are contrib, not standard, ONNX ops).
//
// Before (illustrated for 2 inputs; QLinearConcat is variadic, so this
// generalizes to any input count):
//   Z = Concat(A, B, axis=ax)   -- A, B: both runtime float32 tensors
// After:
//   Aq = QuantizeLinear(A, As, Azp)   -- As/Azp: CALIBRATED
//   Bq = QuantizeLinear(B, Bs, Bzp)   -- Bs/Bzp: CALIBRATED
//   Zq = QLinearConcat(Zs, Zzp, Aq, As, Azp, Bq, Bs, Bzp, axis=ax)
//        -- true int8 compute
//   Z  = DequantizeLinear(Zq, Zs, Zzp)   -- Zs/Zzp: CALIBRATED
//
// Unlike QLinearAdd/QLinearMul (a fixed, shared output scale/zero-point
// derived independently), QLinearConcat's schema takes `Y_scale`/
// `Y_zero_point` as its *first two* inputs -- the same values this pass
// reuses for the trailing DequantizeLinear, so the output is calibrated once
// and that single (scale, zero_point) pair does double duty. Every operand
// still needs its own calibrated range too, same as QLinearAdd/QLinearMul's
// A and B (see that file's doc comment on why -- QLinearConcat has no
// "weight" role either).
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// Only a Concat whose every input is a non-constant float32 tensor is
// matched: a constant operand is better quantized from its own static
// values than force-fed through the runtime calibration harness as if it
// varied at inference time (same reasoning qoperator_quantize_elementwise.h
// applies to Add/Mul). A node is only rewritten when every input's name and
// its own output's name have a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeQOperatorConcat in
// onnxsim.h).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct QOperatorQuantizeConcat final : public PredicateBasedPass {
  explicit QOperatorQuantizeConcat()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "qoperator_quantize_concat";
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != kConcat) {
      return false;
    }
    if (n->inputs().empty()) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    if (ranges.count(n->output()->uniqueName()) == 0) {
      return false;
    }
    for (Value* in : n->inputs()) {
      if (in->elemType() != TensorProto_DataType_FLOAT) {
        return false;
      }
      if (FetchConstantTensor(in) != nullptr) {
        return false;  // Constant operand: quantize from its own values
                       // instead.
      }
      if (ranges.count(in->uniqueName()) == 0) {
        return false;
      }
    }
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (n->kind() != kConcat) {
      return false;
    }
    if (n->inputs().empty()) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto z_range_it = ranges.find(n->output()->uniqueName());
    if (z_range_it == ranges.end()) {
      return false;
    }
    std::vector<std::pair<float, float>> in_ranges;
    in_ranges.reserve(n->inputs().size());
    for (Value* in : n->inputs()) {
      if (in->elemType() != TensorProto_DataType_FLOAT) {
        return false;
      }
      if (FetchConstantTensor(in) != nullptr) {
        return false;
      }
      const auto in_range_it = ranges.find(in->uniqueName());
      if (in_range_it == ranges.end()) {
        return false;
      }
      in_ranges.push_back(in_range_it->second);
    }
    const int64_t axis = GetValueFromAttrWithDefault(n, kaxis, int64_t(0));

    float z_scale_f = 1.0f;
    int32_t z_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        z_range_it->second.first, z_range_it->second.second, z_scale_f, z_zp_i);
    Tensor z_scale_t;
    z_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    z_scale_t.floats() = {z_scale_f};
    Tensor z_zp_t;
    z_zp_t.elem_type() = TensorProto_DataType_UINT8;
    z_zp_t.int32s() = {z_zp_i};
    Value* z_scale_v = graph.addInitializerAndCreateValue(z_scale_t);
    Value* z_zp_v = graph.addInitializerAndCreateValue(z_zp_t);

    // Yq = QLinearConcat(Ys, Yzp, (Xiq, Xis, Xizp)+, axis=axis), a
    // "com.microsoft" contrib op.
    Node* qlop = graph.create(Symbol("QLinearConcat"), 1);
    qlop->addInput(z_scale_v);
    qlop->addInput(z_zp_v);
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* in = n->inputs()[i];
      float in_scale_f = 1.0f;
      int32_t in_zp_i = 0;
      ComputeAsymmetricUint8QuantParams(in_ranges[i].first, in_ranges[i].second,
                                        in_scale_f, in_zp_i);
      Tensor in_scale_t;
      in_scale_t.elem_type() = TensorProto_DataType_FLOAT;
      in_scale_t.floats() = {in_scale_f};
      Tensor in_zp_t;
      in_zp_t.elem_type() = TensorProto_DataType_UINT8;
      in_zp_t.int32s() = {in_zp_i};
      Value* in_scale_v = graph.addInitializerAndCreateValue(in_scale_t);
      Value* in_zp_v = graph.addInitializerAndCreateValue(in_zp_t);

      // Xiq = QuantizeLinear(Xi, Xis, Xizp)
      Node* xq = graph.create(Symbol("QuantizeLinear"), 1);
      xq->addInput(in);
      xq->addInput(in_scale_v);
      xq->addInput(in_zp_v);
      xq->insertBefore(n);
      xq->output()->setElemType(TensorProto_DataType_UINT8);

      qlop->addInput(xq->output());
      qlop->addInput(in_scale_v);
      qlop->addInput(in_zp_v);
    }
    qlop->i_(kaxis, axis);
    qlop->setDomain("com.microsoft");
    qlop->insertBefore(n);
    qlop->output()->setElemType(TensorProto_DataType_UINT8);

    // Y = DequantizeLinear(Yq, Ys, Yzp)
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
