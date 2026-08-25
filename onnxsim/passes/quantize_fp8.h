// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Converts a model's float32 weights and (by default) internal activations
// to an 8-bit floating-point format -- the same kind of whole-graph,
// calibration-free "quantization" as quantize_fp16.h/quantize_bf16.h, just
// to a much narrower floating-point format, selectable at conversion time
// between the two formats introduced in onnx 1.15.0 that are most widely
// implemented by real hardware/runtimes (see
// third_party/onnx-optimizer/third_party/onnx/docs/docsgen/source/technical/float8.md):
//
//   - E4M3FN (Float8Format::kE4M3FN): 1 sign bit, 4 exponent bits, 3
//     mantissa bits, exponent bias 7, max finite magnitude 448. No
//     infinities -- the exponent-all-ones pattern is reserved entirely for
//     NaN (a single NaN encoding per sign). Typically used for weights.
//   - E5M2 (Float8Format::kE5M2): 1 sign bit, 5 exponent bits, 2 mantissa
//     bits, exponent bias 15, max finite magnitude 57344. Has a real
//     infinity encoding (exponent all-ones, mantissa 00), with the other
//     three mantissa patterns at that exponent reserved for NaN. Typically
//     used for gradients, and has a similar dynamic range to float16.
//
// (The FNUZ variants of both -- AMD/GraphCore-oriented, no negative zero, a
// different exponent bias, and a single NaN encoding that collides with
// their own most-negative representable pattern -- are not implemented
// here; the two formats above are the ones broadly implemented by
// NVIDIA/Intel/ARM hardware and the ones the introducing papers focus on.)
//
// Both formats are converted with **saturation**: a float32 value whose
// magnitude exceeds the target format's largest finite value (including
// +-Inf itself) is clamped to that largest finite value rather than mapped
// to a float8 infinity/NaN -- the same design choice quantize_fp16.h makes,
// and the "with saturation" column of the ONNX float8 conversion table
// linked above (this also sidesteps needing to ever emit E5M2's infinity
// encoding, so both formats share one saturating-clamp code path). A value
// that is already NaN maps to the format's canonical NaN encoding.
//
// Rounding is round-to-nearest, ties-to-even (unlike quantize_fp16.h/
// quantize_bf16.h's ties-away-from-zero simplification): float8's mantissa
// is only 2-3 bits wide, so an exact tie between two representable values is
// far more likely to occur on real weight/activation data than it is at
// float16/bfloat16's much finer granularity, and ties-to-even is float8's
// documented standard (RNE, "round nearest even" -- see the same table).
//
// See quantize_fp16.h's doc comment for the shared whole-graph-transform
// design this file otherwise mirrors exactly (constant conversion via
// FetchConstantTensor, keep_io_types boundary-Cast insertion/removal, no
// per-op float8-support checking, top-level-graph-only scope, optional-
// input-with-default-initializer skip). Casting to/from these types needs
// opset >= 19 (onnx's Cast op gained float8 support there).

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

enum class Float8Format {
  kE4M3FN,
  kE5M2,
};

// Which float8 format QuantizeFp8Pass should convert to. Same function-
// local-static parameter-passing pattern as QuantizeFp16KeepIoTypes() --
// set by QuantizeFp8 in onnxsim.cpp immediately before calling
// OptimizeFixed.
inline Float8Format& QuantizeFp8TargetFormat() {
  static Float8Format format = Float8Format::kE4M3FN;
  return format;
}

// Whether QuantizeFp8Pass should keep the graph's own external input/output
// types at float32 (inserting boundary Cast nodes) rather than redeclaring
// them float8 directly. See QuantizeFp16KeepIoTypes()'s comment.
inline bool& QuantizeFp8KeepIoTypes() {
  static bool keep_io_types = true;
  return keep_io_types;
}

constexpr float kFloat8E4M3FNMax = 448.0f;
constexpr float kFloat8E5M2Max = 57344.0f;
constexpr uint8_t kFloat8E4M3FNCanonicalNaN = 0x7Fu;  // S.1111.111
constexpr uint8_t kFloat8E5M2CanonicalNaN = 0x7Du;    // S.11111.01

// Converts `value` to the bit pattern of the nearest representable value in
// the target 8-bit format, round-to-nearest ties-to-even, saturating
// (clamping) rather than producing an infinity or overflow NaN for an
// out-of-range magnitude -- see this file's doc comment.
// `mantissa_bits`/`bias`/`max_value`/`nan_pattern` fully describe the target
// format's layout (the exponent width itself is never needed directly: it
// only bounds shifts, which the `max_value` clamp and `mantissa_bits`
// underflow cutoff already handle), so E4M3FN and E5M2 share this one
// implementation rather than needing near-duplicate copies (unlike
// quantize_fp16.h/quantize_bf16.h, whose bit layouts are dissimilar enough,
// and whose rounding rule is simpler, that duplication was the more
// readable choice there).
inline uint8_t FloatToFloat8Bits(float value, int32_t mantissa_bits,
                                 int32_t bias, float max_value,
                                 uint8_t nan_pattern) {
  if (std::isnan(value)) {
    return nan_pattern;
  }
  if (value > max_value) {
    value = max_value;
  } else if (value < -max_value) {
    value = -max_value;
  }

  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign8 = (bits >> 24) & 0x80u;
  const uint32_t exp32 = (bits >> 23) & 0xFFu;
  const uint32_t mant32 = bits & 0x7FFFFFu;

  if (exp32 == 0 && mant32 == 0) {
    return static_cast<uint8_t>(sign8);  // +-0
  }

  int32_t exp_narrow = static_cast<int32_t>(exp32) - 127 + bias;
  if (exp_narrow <= 0) {
    // Too small for a normalized value in `format` -- subnormal, or
    // underflows to zero. The cutoff below keeps `shift` (and therefore
    // every subsequent shift amount) bounded by 24 -- see the mirroring
    // cutoff in quantize_fp16.h's FloatToFloat16Bits for why this specific
    // bound (`-mantissa_bits`) is exactly where the smallest subnormal
    // rounds to zero.
    if (exp_narrow < -mantissa_bits) {
      return static_cast<uint8_t>(sign8);
    }
    const uint32_t m = mant32 | 0x800000u;  // implicit leading 1 bit
    const uint32_t shift =
        static_cast<uint32_t>(24 - mantissa_bits - exp_narrow);
    uint32_t rounded_mant = m >> shift;
    const uint32_t remainder = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (rounded_mant & 1u) != 0)) {
      rounded_mant += 1u;
    }
    return static_cast<uint8_t>(sign8 | rounded_mant);
  }

  // Normalized case: round the 23-bit mantissa down to `mantissa_bits` bits,
  // ties-to-even.
  const uint32_t shift = static_cast<uint32_t>(23 - mantissa_bits);
  uint32_t rounded_mant = mant32 >> shift;
  const uint32_t remainder = mant32 & ((1u << shift) - 1u);
  const uint32_t halfway = 1u << (shift - 1u);
  if (remainder > halfway ||
      (remainder == halfway && (rounded_mant & 1u) != 0)) {
    rounded_mant += 1u;
  }
  if (rounded_mant & (1u << mantissa_bits)) {
    // Carried into the (implicit) leading bit -- rounded up to exactly the
    // next power of two, so bump the exponent instead. `value` was already
    // clamped to +-max_value above, and max_value is itself exactly
    // representable, so this can only land exactly on `format`'s max
    // exponent, never overflow it (same reasoning as
    // quantize_fp16.h's FloatToFloat16Bits).
    rounded_mant = 0;
    exp_narrow += 1;
  }
  return static_cast<uint8_t>(
      sign8 | (static_cast<uint32_t>(exp_narrow) << mantissa_bits) |
      rounded_mant);
}

inline uint8_t FloatToFloat8Bits(float value, Float8Format format) {
  switch (format) {
    case Float8Format::kE5M2:
      return FloatToFloat8Bits(value, /*mantissa_bits=*/2, /*bias=*/15,
                               kFloat8E5M2Max, kFloat8E5M2CanonicalNaN);
    case Float8Format::kE4M3FN:
    default:
      return FloatToFloat8Bits(value, /*mantissa_bits=*/3, /*bias=*/7,
                               kFloat8E4M3FNMax, kFloat8E4M3FNCanonicalNaN);
  }
}

inline TensorProto_DataType Float8ElemType(Float8Format format) {
  return format == Float8Format::kE5M2 ? TensorProto_DataType_FLOAT8E5M2
                                       : TensorProto_DataType_FLOAT8E4M3FN;
}

// Converts a constant float32 tensor of any rank/shape to `format`, keeping
// the same shape. Like float16/bfloat16, float8 has no dedicated typed field
// in TensorProto -- see ConvertFloatTensorToFp16's comment in
// quantize_fp16.h for why this packs the bit patterns into raw_data (via
// WriteRawDataLittleEndian, endian_read.h) rather than the far less compact
// typed `int32_data` field ONNX's wire format also allows for it.
inline Tensor ConvertFloatTensorToFp8(const Tensor& t, Float8Format format) {
  const std::vector<float> data = ReadFloatTensorFlat(t);
  Tensor out;
  out.elem_type() = Float8ElemType(format);
  out.sizes() = t.sizes();
  std::vector<uint8_t> bits(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    bits[i] = FloatToFloat8Bits(data[i], format);
  }
  out.set_raw_data(WriteRawDataLittleEndian(bits));
  return out;
}

struct QuantizeFp8Pass final : public FullGraphBasedPass {
  explicit QuantizeFp8Pass()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "quantize_fp8"; }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    const bool keep_io_types = QuantizeFp8KeepIoTypes();
    const Float8Format format = QuantizeFp8TargetFormat();
    const TensorProto_DataType target_elem_type = Float8ElemType(format);

    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());
    std::unordered_set<std::string> graph_input_names;
    for (Value* v : graph.inputs()) {
      graph_input_names.insert(v->uniqueName());
    }

    // 1. Convert every constant float32 tensor (a true initializer, or a
    // Constant node's embedded value -- FetchConstantTensor covers both
    // uniformly) to `format`, once per unique value, skipping any that is
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
      Tensor fp8_t = ConvertFloatTensorToFp8(*t, format);
      Value* new_v = graph.addInitializerAndCreateValue(fp8_t);
      tryReplacingAllUsesWith(old_v, new_v);
    }

    // 1.5. Clear stale float32 elemType/shape metadata on every existing
    // node's output (except graph outputs, which step 3 below still needs
    // to see as FLOAT to decide whether to convert/cast them, and sets
    // explicitly itself either way) -- see quantize_fp16.h's own step 1.5
    // for the full rationale: this pass never re-runs shape inference, so a
    // stale float32 declaration left over from an earlier pass (e.g.
    // ``simplify()``) would otherwise round-trip into the exported model's
    // value_info as a wrong type for a tensor now actually `format`, which
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
        value->setElemType(target_elem_type);
        continue;
      }
      // Cast(to=target) right after the input; redirect every existing
      // consumer to the cast's output, leaving the input's own declared
      // type at float32. Snapshot uses() before creating the cast so the
      // cast itself is never among the uses redirected.
      auto use_list = value->uses();
      Node* cast = graph.create(kCast, 1);
      cast = graph.appendNode(cast);
      cast->i_(kto, static_cast<int64_t>(target_elem_type));
      cast->addInput(value);
      cast->output()->setUniqueName(graph.getNextUniqueName());
      cast->output()->setElemType(target_elem_type);
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
        out_v->setElemType(target_elem_type);
        continue;
      }
      // Rename the true producer internally (its real new type is
      // `format`), freeing up the original external name; Cast(to=FLOAT)
      // back to it. Unlike the input case, do NOT redirect out_v's other
      // uses (if any -- e.g. this value also feeds another node internally,
      // not just being a graph output): those should keep consuming out_v
      // directly, now correctly typed, same as everything else in the
      // converted graph. Only the specific graph-output slot(s) that
      // referenced out_v need redirecting to the cast.
      const std::string original_name = out_v->uniqueName();
      out_v->setElemType(target_elem_type);
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
