#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "model_metrics.h"

// A static memory allocator for a graph's activation tensors: assigns each
// plannable tensor a byte offset within one shared arena, reusing space from
// tensors whose liveness interval has already ended, so a deployment target
// can allocate a single `arena_bytes`-sized buffer instead of one permanent
// buffer per activation.
//
// This builds directly on model_metrics.h's liveness pass (the one behind
// PeakMemoryFootprint, whose "live from production to last use" convention
// this mirrors exactly). PeakMemoryFootprint already reports the *ideal*
// number -- the peak bytes simultaneously live, i.e. what a perfectly packed
// allocator with zero fragmentation would need; this module is what actually
// produces that allocator's offsets, so `arena_bytes` here is always
// >= the graph's PeakMemoryFootprint (same intervals, plus whatever
// fragmentation this greedy placement leaves behind) and
// <= `naive_bytes` (every tensor given its own permanent slot, i.e. no reuse
// at all -- the baseline "compression ratio" is measured against).
//
// On top of that liveness-only reuse, an in-place-aliasing pass (see
// IsInPlaceSafeOp and IsViewOp in memory_planning.cpp) unions a safe op's
// input with its output whenever overwriting the input in place is provably
// correct -- the input is not a weight/graph input/graph output, and this
// node is its only consumer. This covers two categories: ops that can
// *compute* their output by overwriting the input (Relu, Sigmoid, Tanh, ...)
// and pure view ops that reinterpret the same bytes under a different shape
// with no computation at all (Reshape, Flatten, Squeeze, Unsqueeze). A chain
// mixing either kind (e.g. Reshape -> Relu -> Flatten) all collapses into
// one placement group needing a single slot for the whole chain's span,
// rather than one slot per node, so the arena for a long chain stays roughly
// constant instead of growing with its length;
// TestInPlaceAliasingCollapsesWholeChain in memory_planning_test.cpp
// demonstrates this directly. Aliased tensors report the identical (offset,
// size) in `offsets` -- a downstream consumer honours the plan by actually
// running that node's kernel in place (writing its output over the input's own
// buffer, or for a view op simply treating input and output as the same buffer
// under different shape metadata), not merely by treating same-offset as "safe
// to reuse after the fact" the way two disjoint-interval tensors are.
//
// v1 scope, deliberately:
//   * Concrete shapes only. A tensor whose size cannot be resolved to a
//     concrete (non-symbolic) byte count -- unknown shape/dtype, or a
//     dynamic dim_param -- is left out of the plan and listed in
//     `unplanned`, rather than guessed.
//   * Top-level graph only. Control-flow subgraph (If/Loop/Scan) bodies are
//     not visited or planned; a follow-up could extend this to a joint
//     cross-subgraph plan the way PeakMemoryFootprint adds a subgraph's own
//     peak on top of its owning node's live set, but a *shared offset space*
//     across graph scopes is a materially bigger allocator problem than
//     this v1 takes on.
//   * Weights (initializers) are excluded: they stay resident for the whole
//     graph and are not part of an activation arena.
namespace onnxsim {

struct MemoryPlan {
  // name -> (offset, size in bytes) within the shared arena, one entry per
  // planned tensor (graph inputs, node outputs, graph outputs with a
  // concrete size). Two entries' [offset, offset + size) ranges only overlap
  // when their liveness intervals do not -- except a set of tensors the
  // in-place-aliasing pass unioned together (see the module comment above),
  // which report the identical (offset, size) on purpose: they are the same
  // logical storage, not merely two tensors sharing recycled space.
  std::map<std::string, std::pair<int64_t, int64_t>> offsets;
  // Size of the arena this plan requires: the high-water mark of every
  // offset + size above. 0 when there is nothing to plan.
  int64_t arena_bytes = 0;
  // Sum of every *planned* tensor's size, as if each got its own permanent
  // slot (no reuse) -- the baseline `arena_bytes` is compressed against.
  int64_t naive_bytes = 0;
  // Names of activation tensors that could not be planned (unknown shape/
  // dtype, or a symbolic dimension) -- excluded from both `offsets` and
  // `naive_bytes` above. A plan with a non-empty `unplanned` should be
  // treated as a partial lower bound, not as evidence of a small arena.
  std::vector<std::string> unplanned;
};

// Compute a memory plan for `graph`'s top-level activation tensors (graph
// inputs, node outputs, graph outputs) -- see the module comment above for
// what "top-level" and "planned" mean.
MemoryPlan ComputeActivationMemoryPlan(const GraphView& graph);

}  // namespace onnxsim
