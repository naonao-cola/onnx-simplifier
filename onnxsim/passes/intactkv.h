// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// IntactKV (Liu, Zhang, Wang, Jin, Sun, Gu, Zeng, Zhu, Wei, Cheng, Chen,
// Zhang, 2024, "IntactKV: Improving Large Language Model Quantization by
// Keeping Pivot Tokens Intact", https://arxiv.org/abs/2403.01241) -- C++
// port of intactkv.py's own apply_intactkv. See that module's own
// docstring for the full rationale: unlike every other KV-cache module in
// this repo (onnxsim.kv_cache_quantization/rotatekv/gear, none yet ported
// to C++), IntactKV is not a quantizer at all -- it is a *companion* pass
// that splits a KV-cache stream's own fixed-length leading "pivot" prefix
// (attention-sink tokens, disproportionately sensitive to quantization
// error) out into its own always-exact stream, so that a following
// KV-cache quantizer can quantize only the remaining, still-growing "rest"
// of the cache and leave the pivots at exact float precision forever.
//
// Matches the exact same structural pattern intactkv.py's own import of
// onnxsim.kv_cache_quantization's `_find_kv_cache_candidates` does (see
// this header's own SCOPE NARROWING note below for exactly what piece of
// that shared matcher is reimplemented here, directly against
// onnx-optimizer's own Node/Value IR rather than raw protobuf, since
// kv_cache_quantization.py itself is not yet ported to C++ and this port
// depends on none of it):
//   `Concat(past, new, axis=seq)` where `past` is a float32 graph input
//   consumed by nothing else, and the Concat's own output is directly a
//   graph output.
//
// Before (illustrated for Key; Value is handled identically, and every
// candidate matched in a graph -- Key and Value together -- is rewritten
// the same way, one match at a time):
//   past_key: graph input, float32 [..., seq_past, head_dim]
//   new_key:  float32 [..., seq_new, head_dim]
//   present_key = Concat(past_key, new_key, axis=seq)   -- graph output
// After:
//   past_key_pivot:   NEW graph input, float32 [..., num_pivot_tokens,
//                     head_dim] -- fixed size, holds the pivot tokens' own
//                     exact Key, set once, never revised
//   past_key_rest:    past_key, renamed in place -- float32
//                     [..., seq_past - num_pivot_tokens, head_dim], still
//                     consumed only by the SAME Concat node (also renamed)
//   present_key_rest = Concat(past_key_rest, new_key, axis=seq)  -- NEW
//                     graph output, the SAME Concat node/Value as before,
//                     just renamed -- an ordinary KV-cache stream in its
//                     own right, ready for a following quantizer to match
//   present_key_pivot = Identity(past_key_pivot)  -- NEW node + graph
//                     output: exact passthrough, every step
//   present_key = Concat(present_key_pivot, present_key_rest, axis=seq)
//                     -- NEW node, reusing the ORIGINAL present_key name:
//                     every pre-existing consumer of present_key (the
//                     attention math, and the original graph-output
//                     binding) is retargeted here automatically. This port
//                     calls Value::replaceAllUsesWith(reconstruction) on
//                     the original Concat's own output Value (BEFORE
//                     renaming it to *_rest -- see runTransform's own step
//                     4 comment for why that order matters) -- the
//                     IR-pointer-based analogue of Python's own name-based
//                     rebinding, since onnx-optimizer's graph outputs are
//                     Value* bindings (an entry in the graph's own pseudo
//                     "return" node's input list), not name strings the
//                     way onnx.GraphProto.output is.
//
// SCOPE NARROWING: onnxsim.kv_cache_quantization's own shared
// `_find_kv_cache_candidates` also resolves a `channel_axis` (the tensor's
// last axis) for its own per-channel quantization scale -- IntactKV never
// uses that field, so this port's own candidate match only resolves
// `seq_axis`, the one piece it actually needs. This port also (a)
// requires the matched `past` input to have a statically-known rank
// (`has_sizes()`) and (b) requires the resolved `seq_axis` to actually be
// in `[0, rank)` -- intactkv.py's own matcher has no such checks (an
// entirely unranked KV-cache input, or a `Concat` whose axis resolves
// outside its own operand's rank, are both pathological/malformed-model
// edge cases the Python reference doesn't defend against and would
// silently mishandle rather than reject); this port declines to match
// instead of producing a degenerate or out-of-bounds-indexed model,
// mirroring the same defensive-bound-check precedent already established
// in low_rank_compensation_entry.cpp for raw IR/protobuf shape reads. (c)
// This port hardcodes intactkv.py's own default `num_pivot_tokens=4`
// rather than exposing it as a parameter -- several other *_cpp ports in
// this repo already establish that a C++ port need not mirror every
// optional knob its Python counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE: none -- this rewrite is a closed-form,
// deterministic structural transformation with no fitting/RNG step
// anywhere in it, so (up to the IR-vs-protobuf representational
// differences already called out above, and ordinary floating-point
// non-involvement since no numeric computation happens at all here) this
// port is expected to track intactkv.py's own graph rewrite exactly --
// unlike the k-means/rotation-family ports elsewhere in this repo, which
// document a real, permanent numerical divergence.

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

namespace intactkv_detail {

// intactkv.py's own default `num_pivot_tokens` (see this header's own
// SCOPE NARROWING note on why this port hardcodes it).
constexpr int64_t kNumPivotTokens = 4;

// Whether `name` is currently displayed by any initializer, graph input/
// output, or node input/output in `graph` (top level only -- this pass
// never touches a subgraph, so unlike Graph::isNameUnique this doesn't
// recurse into If/Loop/Scan bodies). A self-contained re-implementation
// against only Graph's own public API: Graph::isNameUnique itself (and its
// own collectAllNames helper) is private, with no public wrapper for a
// single candidate name.
inline bool NameInUse(Graph& graph, const std::string& name) {
  for (const auto& t : graph.initializers()) {
    if (t->name() == name) {
      return true;
    }
  }
  for (Value* v : graph.inputs()) {
    if (v->uniqueName() == name) {
      return true;
    }
  }
  for (Value* v : graph.outputs()) {
    if (v->uniqueName() == name) {
      return true;
    }
  }
  for (Node* n : graph.nodes()) {
    for (Value* v : n->inputs()) {
      if (v->uniqueName() == name) {
        return true;
      }
    }
    for (Value* v : n->outputs()) {
      if (v->uniqueName() == name) {
        return true;
      }
    }
  }
  return false;
}

// Unique-name generation matching onnxsim.bias_correction._unique_name's
// own base/base_1/base_2/... scheme, against NameInUse above.
inline std::string UniqueName(Graph& graph, const std::string& base) {
  std::string name = base;
  int i = 0;
  while (NameInUse(graph, name)) {
    ++i;
    name = base + "_" + std::to_string(i);
  }
  return name;
}

inline bool IsGraphInput(Graph& graph, Value* v) {
  for (Value* in : graph.inputs()) {
    if (in == v) {
      return true;
    }
  }
  return false;
}

inline bool IsGraphOutput(Graph& graph, Value* v) {
  for (Value* out : graph.outputs()) {
    if (out == v) {
      return true;
    }
  }
  return false;
}

struct Candidate {
  Value* past_value = nullptr;
  int64_t seq_axis = 0;
};

// Reimplements the one piece of onnxsim.kv_cache_quantization's own
// `_find_kv_cache_candidates` IntactKV itself needs (see this header's own
// top-of-file SCOPE NARROWING note), directly against the IR: `n` must be
// a 2-input, 1-output Concat whose own output is a graph output, exactly
// one of whose two inputs is a float32 graph input consumed by nothing
// else, with a statically-known rank and an axis attribute that resolves
// in-bounds.
inline bool FindCandidate(Node* n, Candidate& out) {
  if (n->kind() != kConcat || n->inputs().size() != 2 ||
      n->outputs().size() != 1) {
    return false;
  }
  Graph& graph = *n->owningGraph();
  if (!IsGraphOutput(graph, n->output())) {
    return false;
  }
  if (!n->hasAttribute(kaxis)) {
    return false;
  }

  Value* a = n->input(0);
  Value* b = n->input(1);
  Value* past_value = nullptr;
  if (a->elemType() == TensorProto_DataType_FLOAT && IsGraphInput(graph, a) &&
      a->uses().size() == 1) {
    past_value = a;
  } else if (b->elemType() == TensorProto_DataType_FLOAT &&
             IsGraphInput(graph, b) && b->uses().size() == 1) {
    past_value = b;
  } else {
    return false;
  }
  if (!past_value->has_sizes()) {
    return false;
  }

  const int64_t rank = static_cast<int64_t>(past_value->sizes().size());
  const int64_t axis = n->i(kaxis);
  const int64_t seq_axis = axis < 0 ? axis + rank : axis;
  if (seq_axis < 0 || seq_axis >= rank) {
    return false;
  }

  out.past_value = past_value;
  out.seq_axis = seq_axis;
  return true;
}

}  // namespace intactkv_detail

// IntactKV -- matches a `Concat(past, new, axis=seq)` KV-cache stream (see
// this header's own top-of-file comment for the exact match/rewrite), and
// splits it into a fixed, always-exact `*_pivot` stream plus an ordinary
// `*_rest` KV-cache stream ready for a following quantizer.
struct IntactKv final : public PredicateBasedPass {
  explicit IntactKv()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "intactkv"; }

  bool patternMatchPredicate(Node* n) override {
    intactkv_detail::Candidate c;
    return intactkv_detail::FindCandidate(n, c);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    intactkv_detail::Candidate c;
    if (!intactkv_detail::FindCandidate(n, c)) {
      return false;
    }

    Value* past_value = c.past_value;
    Value* present_value = n->output();

    const std::string past_base = past_value->uniqueName();
    const std::string present_base = present_value->uniqueName();
    const std::vector<Dimension> past_sizes = past_value->sizes();

    // 1. New pivot graph input: past's own rank/dims, seq axis fixed to
    //    kNumPivotTokens elements.
    std::vector<Dimension> pivot_sizes = past_sizes;
    pivot_sizes[static_cast<size_t>(c.seq_axis)] =
        Dimension(intactkv_detail::kNumPivotTokens);
    Value* pivot_value = graph.addInput();
    pivot_value->setUniqueName(
        intactkv_detail::UniqueName(graph, past_base + "_pivot"));
    pivot_value->setElemType(TensorProto_DataType_FLOAT);
    pivot_value->setSizes(pivot_sizes);

    // 2. Rename past (in place) to "_rest" -- still consumed only by n,
    //    which keeps computing exactly Concat(past_rest, new, axis=seq).
    past_value->setUniqueName(
        intactkv_detail::UniqueName(graph, past_base + "_rest"));

    // 3. Build (but do not yet wire/attach) the reconstruction Concat's
    //    own output Value.
    Node* reconstruct_node = graph.create(kConcat, 1);
    Value* reconstructed = reconstruct_node->output();

    // 4. Retarget every PRE-EXISTING consumer of present_* (the original
    //    graph-output binding -- onnx-optimizer's graph outputs are just
    //    the graph's own pseudo return node's ordinary inputs, so they are
    //    covered by the very same use-rewiring loop as any real downstream
    //    consumer node) onto the reconstruction, while present_value STILL
    //    holds its ORIGINAL name/sizes. This ordering is required, not
    //    cosmetic: Value::replaceAllUsesWith (third_party/onnx/onnx/common
    //    /ir.h) special-cases a `this` that is currently a graph output --
    //    it stamps `this`'s name (captured at CALL time) onto `newValue`,
    //    then bumps `this` to an internal auto-generated name, so that the
    //    value now occupying the output slot keeps a stable, meaningful
    //    name. Renaming present_value to "_rest" before this call (as an
    //    earlier version of this pass did) would leak that placeholder
    //    name onto the reconstruction instead of the real present_base
    //    name, and clobber present_value's own "_rest" rename with an
    //    unrelated auto-generated one. Calling it here, first, correctly
    //    transfers present_base's own name/sizes/elemType onto
    //    `reconstructed` and rewires every real use in one step.
    present_value->replaceAllUsesWith(reconstructed);

    // 5. NOW rename/resize present_value -- already displaced from every
    //    use, including the graph-output slot, by step 4 above -- into its
    //    own final "_rest" stream identity, then register it as a new
    //    graph output. Its declared shape becomes past's own dims with the
    //    seq axis cleared to unknown: the rest stream's own length is no
    //    longer the fixed/symbolic value the ORIGINAL present declaration
    //    carried, which was sized for the whole pivot+rest stream.
    present_value->setUniqueName(
        intactkv_detail::UniqueName(graph, present_base + "_rest"));
    std::vector<Dimension> rest_sizes = past_sizes;
    rest_sizes[static_cast<size_t>(c.seq_axis)] = Dimension();
    present_value->setElemType(TensorProto_DataType_FLOAT);
    present_value->setSizes(rest_sizes);
    graph.registerOutput(present_value);  // new "_rest" stream output

    // 6. present_*_pivot = Identity(past_*_pivot): exact passthrough,
    //    every step -- the pivot tokens are set once and never revised.
    Node* identity_node = graph.create(Symbol("Identity"), 1);
    identity_node->addInput(pivot_value);
    identity_node->insertAfter(n);
    Value* identity_out = identity_node->output();
    identity_out->setUniqueName(
        intactkv_detail::UniqueName(graph, present_base + "_pivot"));
    identity_out->setElemType(TensorProto_DataType_FLOAT);
    identity_out->setSizes(pivot_sizes);
    graph.registerOutput(identity_out);

    // 7. present_* (the ORIGINAL name/binding, now produced here instead
    //    of by n) = Concat(pivot, rest).
    reconstruct_node->addInput(identity_out);
    reconstruct_node->addInput(present_value);
    reconstruct_node->i_(kaxis, c.seq_axis);
    reconstruct_node->insertAfter(identity_node);

    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
