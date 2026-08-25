// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Converts a model's float32 weights and (by default) internal activations
// to bfloat16 -- the same kind of whole-graph, calibration-free "quantization"
// as quantize_fp16.h, just to a different narrow floating-point format. Like
// float16, bfloat16 is still an IEEE-754-style floating-point format (not an
// integer scheme), so there is no scale, no zero-point, and no calibration
// data of any kind. Unlike float16 (5 exponent bits / 10 mantissa bits),
// bfloat16 keeps float32's full 8 exponent bits and narrows only the mantissa
// to 7 bits -- so a bfloat16 value's bit pattern is simply the top 16 bits of
// its float32 counterpart, rounded, and there is no subnormal handling or
// clamping/overflow concern: bfloat16's exponent range exactly matches
// float32's, so no finite float32 value can overflow it.
//
// See quantize_fp16.h's doc comment for the shared whole-graph-transform
// design this file mirrors exactly (constant conversion via
// FetchConstantTensor, keep_io_types boundary-Cast insertion/removal, no
// per-op bfloat16-support checking, top-level-graph-only scope, optional-
// input-with-default-initializer skip).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Whether QuantizeBf16Pass should keep the graph's own external input/output
// types at float32 (inserting boundary Cast nodes) rather than redeclaring
// them bfloat16 directly. Same function-local-static parameter-passing
// pattern as QuantizeFp16KeepIoTypes() -- set by QuantizeBf16 in onnxsim.cpp
// immediately before calling OptimizeFixed.
inline bool& QuantizeBf16KeepIoTypes() {
  static bool keep_io_types = true;
  return keep_io_types;
}

// Converts `value` to the bit pattern of the nearest representable bfloat16
// value, rounding to nearest (ties away from zero -- see
// FloatToFloat16Bits's comment in quantize_fp16.h for why this tie-breaking
// rule is an acceptable, standard alternative to hardware's usual
// ties-to-even for this use case). bfloat16 shares float32's 8-bit exponent,
// so this is just a truncation of the top 16 bits of the float32 bit pattern
// with rounding based on the discarded low 16 bits -- no subnormal handling
// and no clamping is needed, since no finite float32 exponent can be out of
// bfloat16's range.
inline uint16_t FloatToBFloat16Bits(float value) {
  if (std::isnan(value)) {
    return 0x7FC0u;  // A canonical quiet NaN; any input NaN payload collapses
                     // to it.
  }

  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));

  // Round-to-nearest, ties away from zero: add half an ULP at bfloat16's
  // precision (bit 15, the top bit of the 16 bits being discarded) before
  // truncating. This can carry all the way up through the mantissa into the
  // exponent (rounding up to the next power of two) or, in the rare case of
  // the largest finite float32 values, into float32's own infinity bit
  // pattern -- both are the mathematically correct rounding result, so no
  // special-casing is needed the way quantize_fp16.h's clamp is (bfloat16's
  // exponent range exactly matches float32's, so this can only ever land on
  // a legitimate bfloat16 value, including +-Inf itself, never overflow it).
  const uint32_t rounded = bits + 0x00008000u;
  return static_cast<uint16_t>(rounded >> 16);
}

// Converts a constant float32 tensor of any rank/shape to bfloat16, keeping
// the same shape. Like float16, bfloat16 has no dedicated typed field in
// TensorProto -- see ConvertFloatTensorToFp16's comment in quantize_fp16.h
// for why this packs the bit patterns into raw_data (via
// WriteRawDataLittleEndian, endian_read.h) rather than the far less compact
// typed `int32_data` field ONNX's wire format also allows for it.
inline Tensor ConvertFloatTensorToBf16(const Tensor& t) {
  const std::vector<float> data = ReadFloatTensorFlat(t);
  Tensor out;
  out.elem_type() = TensorProto_DataType_BFLOAT16;
  out.sizes() = t.sizes();
  std::vector<uint16_t> bits(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    bits[i] = FloatToBFloat16Bits(data[i]);
  }
  out.set_raw_data(WriteRawDataLittleEndian(bits));
  return out;
}

struct QuantizeBf16Pass final : public FullGraphBasedPass {
  explicit QuantizeBf16Pass()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "quantize_bf16"; }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    const bool keep_io_types = QuantizeBf16KeepIoTypes();

    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());
    std::unordered_set<std::string> graph_input_names;
    for (Value* v : graph.inputs()) {
      graph_input_names.insert(v->uniqueName());
    }

    // 1. Convert every constant float32 tensor (a true initializer, or a
    // Constant node's embedded value -- FetchConstantTensor covers both
    // uniformly) to bfloat16, once per unique value, skipping any that is
    // also a graph input (see this file's doc comment).
    std::unordered_set<std::string> seen;
    std::vector<Value*> candidates;
    for (Node* n : graph.nodes()) {
      for (Value* in : n->inputs()) {
        if (!seen.insert(in->uniqueName()).second) {
          continue;
        }
        if (graph_input_names.count(in->uniqueName()) > 0) {
          continue;
        }
        const Tensor* t = FetchConstantTensor(in);
        if (t != nullptr && t->elem_type() == TensorProto_DataType_FLOAT) {
          candidates.push_back(in);
        }
      }
    }
    for (Value* old_v : candidates) {
      const Tensor* t = FetchConstantTensor(old_v);
      if (t == nullptr) {
        continue;  // Defensive: shouldn't happen, nothing else touches these.
      }
      Tensor bf16_t = ConvertFloatTensorToBf16(*t);
      Value* new_v = graph.addInitializerAndCreateValue(bf16_t);
      tryReplacingAllUsesWith(old_v, new_v);
    }

    // 1.5. Clear stale float32 elemType/shape metadata on every existing
    // node's output (except graph outputs, which step 3 below still needs
    // to see as FLOAT to decide whether to convert/cast them, and sets
    // explicitly itself either way) -- see quantize_fp16.h's own step 1.5
    // for the full rationale: this pass never re-runs shape inference, so a
    // stale float32 declaration left over from an earlier pass (e.g.
    // ``simplify()``) would otherwise round-trip into the exported model's
    // value_info as a wrong type for a tensor now actually bfloat16, which
    // ONNX Runtime's own load-time type-checking rejects outright. A
    // cleared (omitted) value_info entry is always safe -- inferred fresh
    // by any conformant consumer -- unlike a wrong one.
    std::unordered_set<std::string> graph_output_names;
    for (Value* v : graph.outputs()) {
      graph_output_names.insert(v->uniqueName());
    }
    for (Node* n : graph.nodes()) {
      for (Value* out : n->outputs()) {
        if (out->elemType() == TensorProto_DataType_FLOAT &&
            graph_output_names.count(out->uniqueName()) == 0) {
          out->setElemType(TensorProto_DataType_UNDEFINED);
          out->wipeSizes();
        }
      }
    }

    // 2. Graph inputs.
    for (Value* value : graph.inputs()) {
      if (initializer_names.count(value->uniqueName()) > 0 ||
          value->elemType() != TensorProto_DataType_FLOAT) {
        continue;
      }
      if (!keep_io_types) {
        value->setElemType(TensorProto_DataType_BFLOAT16);
        continue;
      }
      // Cast(to=BFLOAT16) right after the input; redirect every existing
      // consumer to the cast's output, leaving the input's own declared
      // type at float32. Snapshot uses() before creating the cast so the
      // cast itself is never among the uses redirected.
      auto use_list = value->uses();
      Node* cast = graph.create(kCast, 1);
      cast = graph.appendNode(cast);
      cast->i_(kto, static_cast<int64_t>(TensorProto_DataType_BFLOAT16));
      cast->addInput(value);
      cast->output()->setUniqueName(graph.getNextUniqueName());
      cast->output()->setElemType(TensorProto_DataType_BFLOAT16);
      if (value->sizes().size() > 0) {
        cast->output()->setSizes(value->sizes());
      }
      for (auto& use : use_list) {
        if (!cast->isBefore(use.user)) {
          cast->moveBefore(use.user);
        }
        use.user->replaceInput(use.offset, cast->output());
      }
    }

    // 3. Graph outputs. Snapshot the list first: redirecting an output slot
    // below mutates graph.outputs() (it is literally return_node()'s own
    // input list), which would invalidate a live iteration over it.
    std::vector<Value*> orig_outputs(graph.outputs().begin(),
                                     graph.outputs().end());
    for (Value* out_v : orig_outputs) {
      if (out_v->elemType() != TensorProto_DataType_FLOAT) {
        continue;
      }
      if (!keep_io_types) {
        out_v->setElemType(TensorProto_DataType_BFLOAT16);
        continue;
      }
      // Rename the true producer internally (its real new type is
      // bfloat16), freeing up the original external name; Cast(to=FLOAT)
      // back to it. Unlike the input case, do NOT redirect out_v's other
      // uses (if any -- e.g. this value also feeds another node internally,
      // not just being a graph output): those should keep consuming out_v
      // directly, now correctly bfloat16, same as everything else in the
      // converted graph. Only the specific graph-output slot(s) that
      // referenced out_v need redirecting to the cast.
      const std::string original_name = out_v->uniqueName();
      out_v->setElemType(TensorProto_DataType_BFLOAT16);
      out_v->setUniqueName(graph.getNextUniqueName());

      Node* cast = graph.create(kCast, 1);
      cast = graph.appendNode(cast);
      cast->i_(kto, static_cast<int64_t>(TensorProto_DataType_FLOAT));
      cast->addInput(out_v);
      cast->output()->setUniqueName(original_name);
      cast->output()->setElemType(TensorProto_DataType_FLOAT);
      if (out_v->sizes().size() > 0) {
        cast->output()->setSizes(out_v->sizes());
      }

      for (size_t i = 0; i < graph.outputs().size(); ++i) {
        if (graph.outputs()[i]->uniqueName() == out_v->uniqueName()) {
          graph.return_node()->replaceInput(i, cast->output());
        }
      }
    }

    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
