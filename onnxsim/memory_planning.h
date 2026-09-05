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
// to reuse after the fact" the way two disjoint-interval tensors are. A third,
// independent category (see IsInPlaceSafeBinaryOp) extends this to binary
// elementwise ops -- Add, Mul, and the like -- which may instead union their
// output with *one* of their two operands ("operand donation"), the candidate
// operand held to the exact same bar as above (checked against each operand
// in turn, aliasing at most one; a broadcast operand's byte size never
// matches the output's, so it is never a candidate).
//
// v1 scope, deliberately:
//   * Concrete shapes only. A tensor whose size cannot be resolved to a
//     concrete (non-symbolic) byte count -- unknown shape/dtype, or a
//     dynamic dim_param -- is left out of the plan and listed in
//     `unplanned`, rather than guessed.
//   * Control-flow subgraphs (If/Loop/Scan bodies) are planned independently,
//     each in its own arena starting at offset 0 (see `subgraph_plans`
//     below) -- not jointly with the owning graph's. The owning graph
//     reserves room in *its own* arena for a subgraph's peak requirement for
//     the span its owning node executes (`subgraph_reserved_bytes`),
//     mirroring PeakMemoryFootprint's "subgraph peak added on top of the
//     live set at that node." A genuinely joint plan -- one *shared* offset
//     space spanning graph scopes, so a subgraph's tensors could in
//     principle also alias something live in the outer scope -- is a
//     materially bigger allocator problem this still doesn't take on.
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

  // Extra bytes reserved within *this* arena (already folded into
  // `arena_bytes` above) for control-flow subgraphs owned by nodes in this
  // graph -- one reservation per node that owns any subgraphs, sized to the
  // sum of that node's subgraphs' own `arena_bytes` (both an `If`'s
  // then_branch and else_branch count, even though only one runs -- the
  // same conservative "add every subgraph" convention PeakMemoryFootprint
  // uses) and live only for the span that node executes. 0 when this graph
  // has no control-flow nodes.
  int64_t subgraph_reserved_bytes = 0;
  // One independently-computed plan per control-flow subgraph body owned by
  // a node in this graph, keyed by "<node's first non-empty output
  // name>#<subgraph index>" -- e.g. an `If` node producing output "y" keys
  // its then_branch as "y#0" and else_branch as "y#1"; a `Loop`/`Scan`
  // node's single body keys as "y#0". Each nested plan is entirely
  // self-contained: its `offsets` start at 0 in its *own* arena, never
  // overlapping this plan's `offsets` or another subgraph's -- see
  // `subgraph_reserved_bytes` above for how the owning graph accounts for
  // the space instead.
  std::map<std::string, MemoryPlan> subgraph_plans;
};

// Compute a memory plan for `graph`'s top-level activation tensors (graph
// inputs, node outputs, graph outputs) -- see the module comment above for
// what "top-level" and "planned" mean.
MemoryPlan ComputeActivationMemoryPlan(const GraphView& graph);

}  // namespace onnxsim
