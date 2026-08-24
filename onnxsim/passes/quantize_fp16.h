// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Converts a model's float32 weights and (by default) internal activations
// to float16 -- a different kind of "quantization" from every other pass in
// this directory: float16 is still an IEEE 754 floating-point format (just a
// narrower one, 5 exponent bits / 10 mantissa bits instead of float32's 8/23)
// rather than an integer scheme, so unlike the INT8/INT4 passes there is no
// scale, no zero-point, and no calibration data of any kind -- every
// float32 value is simply rounded to its nearest representable float16
// value.
//
// Unlike the INT8/INT4 passes (each a narrow per-node pattern match), this
// is a whole-graph transform: every float32 initializer (and every
// float32-valued ``Constant`` node -- both are found the same way, via
// ``FetchConstantTensor``) is converted to float16, and, when
// ``keep_io_types`` is true (the default -- see ``QuantizeFp16KeepIoTypes``),
// the graph's own external input/output types stay float32 by inserting a
// boundary ``Cast``: ``Cast(to=FLOAT16)`` right after each float32 graph
// input (redirecting every existing consumer to the cast result, so the
// input's own declared type is untouched), and ``Cast(to=FLOAT)`` right
// before each float32 graph output (renaming the true producer internally
// and giving the cast the original external output name, so external
// callers see the same name and type as before). With ``keep_io_types``
// false, graph inputs/outputs are simply redeclared float16 directly, no
// casts inserted.
//
// No node's op_type or attributes are touched, and no per-op fp16-support
// checking is done: an ordinary "normal" feedforward graph (every op
// propagating its input dtype to its output dtype, which is true of the
// vast majority of ONNX ops) ends up computing in float16 from the first
// input Cast to the last output Cast purely as a side effect of every value
// along the way now being float16-typed -- this pass never has to trace
// that propagation itself. The corollary: a model containing an op with no
// float16 kernel in the runtime it's deployed on will fail at *execution*
// time, not at conversion time here -- the same limitation every other
// float32-to-float16 model converter (e.g. onnxconverter-common's
// ``convert_float_to_float16``) has.
//
// Only the top-level graph is converted -- nodes inside control-flow
// subgraphs (If/Loop/Scan bodies) are left untouched, a known limitation
// matching several other onnxsim passes (e.g. GraphDiff). An initializer
// whose name is also a graph input (the rarely-used ONNX "optional input
// with a default value" convention) is left alone entirely, both to avoid
// the ambiguity of which type it should end up as and because it is
// vanishingly rare in practice.

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

// Whether QuantizeFp16Pass should keep the graph's own external input/output
// types at float32 (inserting boundary Cast nodes) rather than redeclaring
// them float16 directly. A function-local static, the same pattern
// StaticQuantizationCalibrationRanges() (static_quantize_matmul.h) uses to
// pass a parameter into a pass that OptimizeFixed's pass-name-list interface
// has no other way to carry -- set by QuantizeFp16 in onnxsim.cpp
// immediately before calling OptimizeFixed.
inline bool& QuantizeFp16KeepIoTypes() {
  static bool keep_io_types = true;
  return keep_io_types;
}

// The largest finite magnitude a float16 value can represent. Values outside
// [-kFloat16Max, kFloat16Max] (including +-Inf itself) are clamped to it
// rather than rounded to a float16 infinity -- so this pass never silently
// introduces a new infinity into the graph's numeric data. A value already
// NaN stays NaN.
constexpr float kFloat16Max = 65504.0f;

// Converts `value` to the bit pattern of the nearest representable IEEE 754
// binary16 (float16) value, rounding to nearest (ties away from zero -- a
// simpler, still-standard tie-breaking rule than hardware's usual
// ties-to-even, and not a meaningfully different result for real weight/
// activation data, whose bit patterns essentially never land exactly
// halfway between two representable float16 values).
inline uint16_t FloatToFloat16Bits(float value) {
  if (std::isnan(value)) {
    return 0x7E00u;  // A canonical quiet NaN; any input NaN payload collapses
                     // to it.
  }
  if (value > kFloat16Max) {
    value = kFloat16Max;
  } else if (value < -kFloat16Max) {
    value = -kFloat16Max;
  }

  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exp32 = (bits >> 23) & 0xFFu;
  const uint32_t mant32 = bits & 0x7FFFFFu;

  if (exp32 == 0 && mant32 == 0) {
    return static_cast<uint16_t>(sign);  // +-0
  }

  int32_t exp16 = static_cast<int32_t>(exp32) - 127 + 15;
  if (exp16 <= 0) {
    // Too small for a normalized float16 -- subnormal, or underflows to zero.
    if (exp16 < -10) {
      return static_cast<uint16_t>(sign);
    }
    // Add the implicit leading 1 bit, then shift right into subnormal
    // position, rounding to nearest.
    const uint32_t m = mant32 | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp16);
    uint32_t half_mant = m >> shift;
    const uint32_t remainder = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder >= halfway) {
      half_mant += 1u;
    }
    return static_cast<uint16_t>(sign | half_mant);
  }

  // Normalized case: round the 23-bit mantissa down to float16's 10 bits.
  uint32_t rounded =
      mant32 + 0x00001000u;  // bias: half an ULP at the kept precision
  if (rounded & 0x00800000u) {
    // The rounded mantissa carried into the (implicit) leading bit -- it
    // rounded up to exactly the next power of two, so bump the exponent
    // instead. `value` was already clamped to +-kFloat16Max above, so this
    // can only land exactly on float16's max exponent, never overflow it.
    rounded = 0;
    exp16 += 1;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp16) << 10) |
                               (rounded >> 13));
}

// Converts a constant float32 tensor of any rank/shape to float16, keeping
// the same shape. float16 has no dedicated typed field in TensorProto (like
// INT8/UINT8/etc., its values live in `int32_data`/`Tensor::int32s()`, each
// entry the 16-bit bit pattern zero-extended to int32 -- see
// third_party/onnx-optimizer/third_party/onnx/onnx/common/ir_pb_converter.cc's
// FLOAT16 handling), unlike INT4, which needs the raw_data workaround other
// onnxsim passes use.
inline Tensor ConvertFloatTensorToFp16(const Tensor& t) {
  const std::vector<float> data = ReadFloatTensorFlat(t);
  Tensor out;
  out.elem_type() = TensorProto_DataType_FLOAT16;
  out.sizes() = t.sizes();
  out.int32s().resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    out.int32s()[i] = static_cast<int32_t>(FloatToFloat16Bits(data[i]));
  }
  return out;
}

struct QuantizeFp16Pass final : public FullGraphBasedPass {
  explicit QuantizeFp16Pass()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "quantize_fp16"; }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    const bool keep_io_types = QuantizeFp16KeepIoTypes();

    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());
    std::unordered_set<std::string> graph_input_names;
    for (Value* v : graph.inputs()) {
      graph_input_names.insert(v->uniqueName());
    }

    // 1. Convert every constant float32 tensor (a true initializer, or a
    // Constant node's embedded value -- FetchConstantTensor covers both
    // uniformly) to float16, once per unique value, skipping any that is
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
      Tensor fp16_t = ConvertFloatTensorToFp16(*t);
      Value* new_v = graph.addInitializerAndCreateValue(fp16_t);
      tryReplacingAllUsesWith(old_v, new_v);
    }

    // 1.5. Every *interior* activation value (every existing node's output --
    // not yet the boundary Cast nodes steps 2/3 below add, which set their
    // own output's type explicitly and correctly) now silently computes in
    // float16 as a side effect of step 1's retyped weights/constants,
    // exactly as this file's own header comment describes -- but its
    // declared elemType/shape metadata is whatever an *earlier* pass (e.g.
    // onnxsim's own shape inference, if this model went through
    // ``simplify()`` first) set it to, still float32, and this pass never
    // re-runs shape inference itself to correct it (see the header comment).
    // Left alone, that stale metadata round-trips into the exported model's
    // value_info as an outright *wrong* declaration (still float32) for a
    // tensor the graph now actually produces as float16 -- unlike a merely
    // *missing* value_info entry, which every conformant consumer (ONNX
    // Runtime included) is expected to -- and does -- infer for itself, a
    // wrong one is exactly the kind of type mismatch ONNX Runtime's own
    // stricter load-time type-checking rejects outright. Clear it rather
    // than guess a replacement: this pass doesn't model individual ops'
    // real type-propagation rules (e.g. `Cast`'s output type depends only on
    // its own `to` attribute, never its input, so blindly relabeling every
    // float32-declared output float16 here would just trade one wrong
    // declaration for another), so wiping the stale metadata for every
    // existing node's float32-declared output is the only *always-correct*
    // move available without doing full type inference -- whatever
    // eventually reads this model performs it fresh instead.
    //
    // Graph outputs are skipped here even when they coincide with a node's
    // output (the common case): step 3 below still needs to see each one's
    // *original* elemType()==FLOAT to decide whether to convert/cast it, and
    // sets its final type explicitly itself either way.
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
        value->setElemType(TensorProto_DataType_FLOAT16);
        continue;
      }
      // Cast(to=FLOAT16) right after the input; redirect every existing
      // consumer to the cast's output, leaving the input's own declared
      // type at float32. Snapshot uses() before creating the cast so the
      // cast itself is never among the uses redirected.
      auto use_list = value->uses();
      Node* cast = graph.create(kCast, 1);
      cast = graph.appendNode(cast);
      cast->i_(kto, static_cast<int64_t>(TensorProto_DataType_FLOAT16));
      cast->addInput(value);
      cast->output()->setUniqueName(graph.getNextUniqueName());
      cast->output()->setElemType(TensorProto_DataType_FLOAT16);
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
        out_v->setElemType(TensorProto_DataType_FLOAT16);
        continue;
      }
      // Rename the true producer internally (its real new type is float16),
      // freeing up the original external name; Cast(to=FLOAT) back to it.
      // Unlike the input case, do NOT redirect out_v's other uses (if any --
      // e.g. this value also feeds another node internally, not just being
      // a graph output): those should keep consuming out_v directly, now
      // correctly float16, same as everything else in the converted graph.
      // Only the specific graph-output slot(s) that referenced out_v need
      // redirecting to the cast.
      const std::string original_name = out_v->uniqueName();
      out_v->setElemType(TensorProto_DataType_FLOAT16);
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
