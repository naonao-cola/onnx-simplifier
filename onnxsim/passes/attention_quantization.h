// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Attention computation quantization -- C++ port of
// attention_quantization.py's own apply_attention_quantization. See that
// module's own docstring for the full rationale: unlike every other
// quantizer in this repo (which targets a weight-bearing MatMul/Gemm, or
// KV-cache tensors specifically), this pass quantizes the attention
// *computation* itself -- the ``QK^T`` score matmul's own Q/K operands, the
// ``softmax(QK^T)@V`` value-weighted sum's own V operand, and the Softmax
// output (the attention probabilities) that feeds it. None of these four
// tensors is a constant weight, so none of this repo's weight-quantization
// machinery (constant-tensor matching, fold-to-initializer) applies here --
// this is a genuine new-graph-nodes rewrite, the same shape
// ibert_softmax.h/quarot.h already establish for this codebase.
//
// Matches the common **decomposed** attention subgraph most ONNX exports
// still produce (not the newer, opset-23+ fused ``Attention`` operator
// fuse_attention.h already recognizes for a *different* purpose --
// fusion, not quantization):
//
//   scores  = MatMul(Q, Kt)                  -- Kt: K, transposed
//   scaled  = Mul(scores, scale)  [optional]  -- e.g. 1/sqrt(head_dim)
//   masked  = Add(scaled, mask)   [optional]  -- e.g. causal mask
//   probs   = Softmax(masked, axis=-1)
//   out     = MatMul(probs, V)
//
// Anchored on the Softmax node itself (matching ibert_softmax.h's own
// single-op-type-anchor convention), walking backward through at most 2
// optional Mul/Div/Add hops to find the score MatMul (mirroring
// attention_quantization.py's own _find_matmul_producer exactly: only the
// *first* input of each intervening Mul/Div/Add is followed -- the
// scale/mask operand itself is never a valid path back to the score
// MatMul), and forward through the Softmax output's own uses to find the
// first MatMul consuming it at input position 0 (the second attention
// matmul, whose other operand is V).
//
// Q, K, and V each get the same data-free, per-token dynamic INT8 round
// trip quarot.h's own activation quantization already establishes for this
// codebase (``scale = max(|x|, axis=-1, eps-floored) / 127``, then
// round-clip-dequant, all computed at graph-run time, no calibration
// statistics stored -- kept in float32 to simulate the precision loss
// rather than emitting a true integer matmul, exactly like every other
// activation-quantization pass in this repo). The Softmax output gets a
// *fixed*-scale UINT8-range round trip instead (``scale = 1/255,
// zero_point = 0``, no data-dependent computation at all): unlike every
// other activation this repo quantizes, a Softmax output's range is
// *guaranteed* to lie in [0, 1] for any input at all, so its scale needs no
// runtime max-reduction. Score computation itself (``MatMul(Q, Kt)``) and
// the softmax normalization are left running in float -- only the four
// tensors *crossing* a matmul boundary are quantized, matching every other
// onnxsim quantizer's own convention of touching operands, not
// recomputing an op's own math in lower precision.
//
// Before:
//   scores = MatMul(Q, Kt)
//   ...
//   probs  = Softmax(masked, axis=-1)
//   out    = MatMul(probs, V)
// After (opset 18+ only -- ReduceMax's axes-as-input form, matching
// attention_quantization.py's own opset gate exactly):
//   Qdq    = round_trip_int8_per_token(Q)
//   Kdq    = round_trip_int8_per_token(Kt)
//   scores = MatMul(Qdq, Kdq)
//   ...
//   probs  = Softmax(masked, axis=-1)
//   Pdq    = round_trip_uint8_fixed_scale(probs)
//   Vdq    = round_trip_int8_per_token(V)
//   out    = MatMul(Pdq, Vdq)
//
// SCOPE NARROWING, beyond attention_quantization.py's own scope (which this
// port otherwise matches exactly): this port additionally declines to match
// when Q, Kt, V, or the Softmax output has a KNOWN, non-FLOAT elemType.
// attention_quantization.py has no such check at all -- it builds its new
// Div/Mul nodes' constants as plain np.float32 regardless of Q/K/V's own
// actual dtype, so a non-float32 model would silently produce a
// type-mismatched (checker-rejected or runtime-rejected) graph on the
// Python side. An UNKNOWN elemType (UNDEFINED -- the common case for an
// intermediate activation with no explicit value_info, unless a shape-
// inference pass already ran) is treated as "assume float" and matched
// anyway, the same as Python's own unconditional assumption -- an earlier
// version of this port required a statically-KNOWN FLOAT dtype instead,
// which silently declined the overwhelming majority of real, unannotated
// attention subgraphs (see IsFloatOrUnknown below). Declining only on a
// KNOWN non-float dtype still catches the real type-mismatch risk this
// note originally called out, without that false-negative cost.
//
// Unlike attention_quantization.py's own apply_attention_quantization, this
// port hardcodes epsilon (1e-12) rather than exposing it as a parameter --
// several other *_cpp ports in this repo already establish that a C++ port
// need not mirror every optional knob its Python counterpart has. Per-match
// scalar constants (eps, 127, -127, axes=[-1], the Softmax-output scale/
// bounds) are recreated fresh for every matched Softmax node rather than
// shared across the whole model -- the same per-match constant-creation
// convention ibert_softmax.h's own ln2/quadratic-coefficient constants and
// weight_only_quantize_mxfp4_matmul.h's own per-match codebook already
// establish (a pass instance is reused across future calls via
// RegisterOrReplace's std::call_once, so it must never cache a Value*/
// Node* pointer into a specific graph across matches).
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond the scope narrowing already
// noted above -- this is a closed-form, deterministic elementwise
// round-trip with no RNG and no accumulation step beyond an ordinary
// per-token ReduceMax, so this port is expected to track the Python
// port's own float64/float32 numpy implementation closely, up to ordinary
// floating-point summation-order differences in that reduction.
// apply_attention_quantization and this port's _cpp counterpart remain
// independently-correct, non-interchangeable entry points.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace attention_quantization_detail {

constexpr double kEpsilon = 1e-12;
constexpr int kMaxScaleMaskHops = 2;

// Walks back from `v` through at most `hops_left` Mul/Div/Add nodes
// (following each one's own *first* input only -- the scale/mask operand,
// never the divisor/mask itself), looking for the MatMul that produced the
// raw attention scores. Direct transcription of
// attention_quantization.py's own _find_matmul_producer; a value with no
// real producer (a graph input/initializer, kind() == kParam) naturally
// fails every kind check below and returns nullptr, the same as Python's
// own producer_by_output.get(name) returning None.
inline Node* FindQKMatMulProducer(Value* v, int hops_left) {
  Node* node = v->node();
  if (node->kind() == kMatMul) {
    return node;
  }
  if (hops_left <= 0) {
    return nullptr;
  }
  if (node->kind() != kMul && node->kind() != kDiv && node->kind() != kAdd) {
    return nullptr;
  }
  if (node->inputs().empty()) {
    return nullptr;
  }
  return FindQKMatMulProducer(node->input(0), hops_left - 1);
}

// First MatMul among `softmax_out`'s own uses that consumes it at input
// position 0 (the second attention matmul; its other operand is V) --
// direct transcription of attention_quantization.py's own
// `next((c for c in consumers if c.op_type == "MatMul" and c.input[0] ==
// softmax_out), None)`.
inline Node* FindOutMatMulConsumer(Value* softmax_out) {
  for (const auto& use : softmax_out->uses()) {
    if (use.user->kind() == kMatMul && use.offset == 0 &&
        use.user->inputs().size() == 2) {
      return use.user;
    }
  }
  return nullptr;
}

}  // namespace attention_quantization_detail

struct AttentionQuantizationMatch {
  Node* qk_matmul = nullptr;
  Node* out_matmul = nullptr;
};

// True unless `v`'s own elemType is KNOWN and is something other than
// FLOAT. An intermediate activation's own elemType is UNDEFINED unless the
// model carries explicit value_info for it (typically only after an
// explicit onnx.shape_inference pass) or this graph's own IR importer ran
// per-node type inference -- neither is guaranteed for an arbitrary input
// model, and attention_quantization.py's own matcher never checks dtype at
// all (see this header's own top-of-file comment). Treating "unknown" as
// "assume float, like every other onnxsim activation-quantization pass"
// rather than declining keeps this port matching the same common,
// not-fully-shape-inferred models the Python reference already handles;
// declining only on a genuinely KNOWN non-float dtype still catches the
// real type-mismatch risk that scope-narrowing note originally called out.
inline bool IsFloatOrUnknown(Value* v) {
  return v->elemType() == TensorProto_DataType_UNDEFINED ||
         v->elemType() == TensorProto_DataType_FLOAT;
}

// Anchored on a Softmax node `n`: walks back (through at most 2 optional
// Mul/Div/Add hops) to find the score MatMul, and forward through `n`'s
// own output uses to find the consuming out-projection MatMul. See this
// header's own top-of-file comment for the FLOAT-dtype scope narrowing
// beyond attention_quantization.py's own matching.
inline bool MatchAttentionQuantization(Node* n, AttentionQuantizationMatch& m) {
  if (n->kind() != kSoftmax || n->inputs().size() != 1 ||
      n->outputs().size() != 1) {
    return false;
  }

  Node* qk = attention_quantization_detail::FindQKMatMulProducer(
      n->input(0), attention_quantization_detail::kMaxScaleMaskHops);
  if (qk == nullptr || qk->inputs().size() != 2) {
    return false;
  }
  if (!IsFloatOrUnknown(qk->input(0)) || !IsFloatOrUnknown(qk->input(1))) {
    return false;
  }

  if (!IsFloatOrUnknown(n->output())) {
    return false;
  }
  Node* out_mm =
      attention_quantization_detail::FindOutMatMulConsumer(n->output());
  if (out_mm == nullptr) {
    return false;
  }
  if (!IsFloatOrUnknown(out_mm->input(1))) {
    return false;
  }

  m.qk_matmul = qk;
  m.out_matmul = out_mm;
  return true;
}

// Attention computation quantization -- matches the decomposed
// MatMul(Q,Kt) -> [Mul/Div] -> [Add] -> Softmax -> MatMul(_,V) subgraph via
// MatchAttentionQuantization, then replaces Q/Kt/V with a data-free
// per-token dynamic INT8 round trip and the Softmax output with a
// fixed-scale round trip (see this header's own top-of-file comment).
struct AttentionQuantization final : public PredicateBasedPass {
  explicit AttentionQuantization()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "attention_quantization"; }

  bool patternMatchPredicate(Node* n) override {
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 18) {
      return false;
    }
    AttentionQuantizationMatch m;
    return MatchAttentionQuantization(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 18) {
      return false;
    }
    AttentionQuantizationMatch m;
    if (!MatchAttentionQuantization(n, m)) {
      return false;
    }
    Node* qk = m.qk_matmul;
    Node* out_mm = m.out_matmul;

    // Per-match scalar constants -- see this header's own top-of-file
    // comment for why these are recreated fresh per matched Softmax node
    // rather than shared across the whole model.
    auto make_scalar_f32 = [&](float value) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_FLOAT;
      t.floats() = {value};
      return graph.addInitializerAndCreateValue(t);
    };
    Value* eps_v = make_scalar_f32(
        static_cast<float>(attention_quantization_detail::kEpsilon));
    Value* i127_v = make_scalar_f32(127.0f);
    Value* neg127_v = make_scalar_f32(-127.0f);
    Value* probs_scale_v = make_scalar_f32(1.0f / 255.0f);
    Value* probs_zero_v = make_scalar_f32(0.0f);
    Value* probs_max_v = make_scalar_f32(255.0f);

    Tensor axes_t;
    axes_t.elem_type() = TensorProto_DataType_INT64;
    axes_t.sizes() = {1};
    axes_t.int64s() = {-1};
    Value* axes_v = graph.addInitializerAndCreateValue(axes_t);

    // Data-free, per-token round-to-nearest INT8 activation quantization --
    // simulated via an immediate dequantize (kept in float32) rather than a
    // true packed INT8 tensor, since `x` isn't constant. Mirrors quarot.h's
    // own identical Abs/ReduceMax/Clip/Div/Round/Clip/Mul chain exactly
    // (scale = max(|x|, axis=-1, eps-floored) / 127), just with a 127
    // (not 7-bit) clip bound. Every new node's output only gets its
    // elemType set, never its sizes -- left for the next shape-inference
    // pass, the same convention quarot.h's own identical chain uses.
    auto quantize_per_token_int8 = [&](Value* x, Node* insert_before) {
      auto make_node = [&](Symbol kind, const std::vector<Value*>& inputs) {
        Node* node = graph.create(kind, 1);
        for (Value* v : inputs) {
          node->addInput(v);
        }
        node->insertBefore(insert_before);
        node->output()->setElemType(TensorProto_DataType_FLOAT);
        return node;
      };

      Node* abs_n = make_node(Symbol("Abs"), {x});
      Node* max_n = graph.create(kReduceMax, 1);
      max_n->addInput(abs_n->output());
      max_n->addInput(axes_v);
      max_n->i_(kkeepdims, 1);
      max_n->insertBefore(insert_before);
      max_n->output()->setElemType(TensorProto_DataType_FLOAT);

      Node* safe_max_n = graph.create(Symbol("Clip"), 1);
      safe_max_n->addInput(max_n->output());
      safe_max_n->addInput(eps_v);
      safe_max_n->insertBefore(insert_before);
      safe_max_n->output()->setElemType(TensorProto_DataType_FLOAT);

      Node* scale_n = make_node(kDiv, {safe_max_n->output(), i127_v});
      Node* scaled_n = make_node(kDiv, {x, scale_n->output()});
      Node* rounded_n = make_node(Symbol("Round"), {scaled_n->output()});

      Node* clipped_n = graph.create(Symbol("Clip"), 1);
      clipped_n->addInput(rounded_n->output());
      clipped_n->addInput(neg127_v);
      clipped_n->addInput(i127_v);
      clipped_n->insertBefore(insert_before);
      clipped_n->output()->setElemType(TensorProto_DataType_FLOAT);

      Node* dequant_n =
          make_node(kMul, {clipped_n->output(), scale_n->output()});
      return dequant_n->output();
    };

    // Fixed-scale UINT8-range round trip for the Softmax output: its range
    // is [0, 1] for any input at all, so scale=1/255, zero_point=0 needs no
    // data-dependent computation -- see this header's own top-of-file
    // comment.
    auto quantize_probs_fixed_scale = [&](Value* x, Node* insert_before) {
      auto make_node = [&](Symbol kind, const std::vector<Value*>& inputs) {
        Node* node = graph.create(kind, 1);
        for (Value* v : inputs) {
          node->addInput(v);
        }
        node->insertBefore(insert_before);
        node->output()->setElemType(TensorProto_DataType_FLOAT);
        return node;
      };

      Node* scaled_n = make_node(kDiv, {x, probs_scale_v});
      Node* rounded_n = make_node(Symbol("Round"), {scaled_n->output()});
      Node* clipped_n = graph.create(Symbol("Clip"), 1);
      clipped_n->addInput(rounded_n->output());
      clipped_n->addInput(probs_zero_v);
      clipped_n->addInput(probs_max_v);
      clipped_n->insertBefore(insert_before);
      clipped_n->output()->setElemType(TensorProto_DataType_FLOAT);

      Node* dequant_n = make_node(kMul, {clipped_n->output(), probs_scale_v});
      return dequant_n->output();
    };

    // Q then K, both inserted directly before `qk`, matching
    // attention_quantization.py's own insertion order exactly.
    Value* q_dq = quantize_per_token_int8(qk->input(0), qk);
    Value* k_dq = quantize_per_token_int8(qk->input(1), qk);
    qk->replaceInput(0, q_dq);
    qk->replaceInput(1, k_dq);

    // probs then V, both inserted directly before `out_mm`, matching
    // attention_quantization.py's own insertion order exactly.
    Value* probs_dq = quantize_probs_fixed_scale(n->output(), out_mm);
    Value* v_dq = quantize_per_token_int8(out_mm->input(1), out_mm);
    out_mm->replaceInput(0, probs_dq);
    out_mm->replaceInput(1, v_dq);

    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
