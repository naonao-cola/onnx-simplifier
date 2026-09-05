#include "memory_planning.h"

#include <algorithm>
#include <set>

namespace onnxsim {

namespace {

struct Interval {
  std::string name;
  int64_t size = 0;
  int start =
      0;  // node index after which the tensor is available (-1 = before node 0)
  int end = 0;  // last node index at which the tensor is still needed
};

// True when `a` and `b` are never simultaneously live, i.e. safe to share
// offset space. Conservative at the boundary: a tensor whose last use is node
// i and one produced by node i itself still count as overlapping (`a.end ==
// b.start` is not `<`), since a node's own output is not guaranteed safe to
// alias with its inputs' storage in general -- that exception is handled
// separately and explicitly by the in-place-aliasing pass below, for the
// specific ops it is actually safe for.
bool Disjoint(const Interval& a, const Interval& b) {
  return a.end < b.start || b.end < a.start;
}

// Ops whose ONNX semantics guarantee a single output that is strictly
// shape- and dtype-identical to their first input, computed purely
// elementwise (output[i] depends only on input[i]) -- so an executor can
// always compute the output by overwriting the input's own storage in
// place, with no numerical difference from writing to a separate buffer.
// This is a curated allowlist, not a schema query (nothing here parses
// attributes or checks the actual input/output types), so it deliberately
// excludes anything with an exception -- e.g. not "Transpose", whose output
// (unless the permutation happens to be the identity, which nothing here
// checks) physically reorders elements rather than computing the same
// index in place.
bool IsInPlaceSafeOp(const std::string& op_type) {
  static const std::set<std::string> kSafeOps = {
      "Relu",     "Sigmoid",  "Tanh",        "LeakyRelu",  "Elu",      "Selu",
      "Softplus", "Softsign", "HardSigmoid", "Clip",       "Neg",      "Abs",
      "Sqrt",     "Exp",      "Log",         "Reciprocal", "Identity", "Erf",
      "Celu",     "Round",    "Ceil",        "Floor",      "Sign",
  };
  return kSafeOps.count(op_type) > 0;
}

// Ops that reinterpret their first input's element sequence under a
// different shape without moving any bytes -- ONNX requires row-major/
// C-contiguous tensor layout, and none of these ops permute element order
// (they only regroup, split, or insert/remove size-1 dimensions), so their
// output is byte-for-byte identical to their input whenever the sizes
// TensorBytes() reports actually agree (checked the same as every other
// candidate, at the call site). This is a *stronger* guarantee than
// IsInPlaceSafeOp's "safe to compute in place" -- these ops need no
// computation at all, the output is a pure view -- but it is folded into
// the same union-find aliasing pass since the resulting placement is
// identical: one shared slot for input and output alike. Not "Transpose"
// (see IsInPlaceSafeOp) or "Expand"/"Tile" (which do move/duplicate bytes).
bool IsViewOp(const std::string& op_type) {
  static const std::set<std::string> kViewOps = {
      "Reshape",
      "Flatten",
      "Squeeze",
      "Unsqueeze",
  };
  return kViewOps.count(op_type) > 0;
}

// Union-find (disjoint-set) over tensor names, path-compressed, used to
// coalesce a chain of in-place-eligible ops (e.g. Relu -> Sigmoid -> Tanh)
// into one placement group that needs a single slot for its whole span
// rather than one slot per tensor. Passes/returns names by value (plain
// std::string copies) rather than references into `parent_`, since Find()
// mutates that same map recursively -- returning a reference into a std::map
// entry that a nested call may go on to overwrite is the kind of aliasing
// hazard worth just not risking for a data structure this small.
class UnionFind {
 public:
  std::string Find(const std::string& name) {
    auto it = parent_.find(name);
    if (it == parent_.end()) {
      parent_[name] = name;
      return name;
    }
    if (it->second == name) return name;
    const std::string root = Find(it->second);
    parent_[name] = root;  // path compression
    return root;
  }

  void Union(const std::string& a, const std::string& b) {
    const std::string ra = Find(a);
    const std::string rb = Find(b);
    if (ra != rb) parent_[ra] = rb;
  }

 private:
  std::map<std::string, std::string> parent_;
};

}  // namespace

MemoryPlan ComputeActivationMemoryPlan(const GraphView& graph) {
  MemoryPlan plan;

  const std::set<std::string> weights(graph.initializers.begin(),
                                      graph.initializers.end());
  const std::set<std::string> graph_inputs(graph.inputs.begin(),
                                           graph.inputs.end());
  const std::set<std::string> graph_outputs(graph.outputs.begin(),
                                            graph.outputs.end());

  // Last node index consuming each tensor; a graph output is "consumed" at
  // `end` so it stays live through the whole pass -- the same liveness
  // convention PeakMemoryFootprint uses (model_metrics.cpp).
  const int end = static_cast<int>(graph.nodes.size());
  std::map<std::string, int> last_use;
  for (int i = 0; i < end; ++i) {
    for (const std::string& name : graph.nodes[i].inputs) {
      if (!name.empty()) last_use[name] = i;
    }
  }
  for (const std::string& out : graph.outputs) last_use[out] = end;

  // Producer node index of each non-weight tensor: -1 for a graph input
  // (available before node 0), else the index of the node that outputs it.
  std::map<std::string, int> produced_at;
  for (const std::string& in : graph.inputs) {
    if (weights.find(in) == weights.end()) produced_at[in] = -1;
  }
  for (int i = 0; i < end; ++i) {
    for (const std::string& name : graph.nodes[i].outputs) {
      if (!name.empty() && weights.find(name) == weights.end()) {
        produced_at[name] = i;
      }
    }
  }

  // How many (node, input-slot) pairs consume each tensor across the whole
  // graph -- the in-place-aliasing pass below only aliases a tensor whose
  // sole consumer is the one candidate node, so overwriting it can never
  // corrupt a read some other node still needs.
  std::map<std::string, int> consumer_count;
  for (const NodeView& node : graph.nodes) {
    for (const std::string& name : node.inputs) {
      if (!name.empty()) ++consumer_count[name];
    }
  }

  // In-place aliasing: union a safe unary/view op's input[0] with its
  // output[0] (see IsInPlaceSafeOp and IsViewOp) whenever doing so is
  // provably safe --
  //   * not a weight, a graph input, or a declared graph output (an
  //     externally-owned buffer, or one the caller reads only after the
  //     whole graph finishes, so overwriting it mid-run would be observed
  //     as corruption rather than reused space);
  //   * consumed exactly once, by this node -- the only read of the
  //     original value;
  //   * input and output agree on size (always true for these ops on a
  //     well-formed model; checked anyway as a defensive belt-and-braces).
  // A chain of such ops unions transitively into one group -- Find() on any
  // member returns the same root -- that needs only a single slot for its
  // entire span, rather than one slot per node. The chain may freely mix
  // both categories (e.g. Reshape -> Relu -> Flatten): the eligibility
  // conditions and the resulting placement are identical either way.
  UnionFind uf;
  for (const NodeView& node : graph.nodes) {
    if (!IsInPlaceSafeOp(node.op_type) && !IsViewOp(node.op_type)) continue;
    if (node.inputs.empty() || node.outputs.size() != 1) continue;
    const std::string& in0 = node.inputs[0];
    const std::string& out0 = node.outputs[0];
    if (in0.empty() || out0.empty()) continue;
    if (weights.count(in0) || graph_inputs.count(in0) ||
        graph_outputs.count(in0)) {
      continue;
    }
    const auto cit = consumer_count.find(in0);
    if (cit == consumer_count.end() || cit->second != 1) continue;

    const auto in_bytes = TensorBytes(in0, graph.shapes, graph.dtypes);
    const auto out_bytes = TensorBytes(out0, graph.shapes, graph.dtypes);
    if (!in_bytes || !out_bytes || in_bytes->is_symbolic() ||
        out_bytes->is_symbolic()) {
      continue;
    }
    if (in_bytes->to_int() != out_bytes->to_int()) continue;

    uf.Union(in0, out0);
  }

  // Pass 1: one Interval per plannable *group* -- a group is a single
  // tensor, or several coalesced by the aliasing pass above -- with
  // everything unplannable going to `unplanned`. An unconsumed intermediate
  // (no `last_use` entry, not a graph output) defaults to staying live
  // through `end`, the same "never freed" fallback the live-set walk behind
  // PeakMemoryFootprint implicitly applies to it. `naive_bytes` is summed
  // per tensor regardless of grouping: it is the no-optimization-at-all
  // baseline, not affected by how well this planner happens to pack things.
  struct GroupSpan {
    int64_t size = 0;
    int start = 0;
    int end = 0;
    bool seen = false;
  };
  std::map<std::string, GroupSpan> groups;  // keyed by union-find root

  for (const auto& [name, start] : produced_at) {
    const auto bytes = TensorBytes(name, graph.shapes, graph.dtypes);
    if (!bytes || bytes->is_symbolic()) {
      plan.unplanned.push_back(name);
      continue;
    }
    const auto it = last_use.find(name);
    const int last = it == last_use.end() ? end : it->second;
    plan.naive_bytes += bytes->to_int();

    GroupSpan& g = groups[uf.Find(name)];
    if (!g.seen) {
      g = {bytes->to_int(), start, last, true};
    } else {
      g.start = std::min(g.start, start);
      g.end = std::max(g.end, last);
      g.size = std::max(g.size, bytes->to_int());
    }
  }

  std::vector<Interval> intervals;
  intervals.reserve(groups.size());
  for (const auto& [root, g] : groups) {
    intervals.push_back({root, g.size, g.start, g.end});
  }

  // Pass 2: greedy best-fit placement, largest group first (ties broken by
  // root name for determinism) -- the standard linear-scan/greedy-by-size
  // register allocation heuristic, which is what makes this
  // "fragmentation-aware" rather than a naive bump allocator. For each
  // group, gather the offset ranges of every already-placed group whose
  // liveness overlaps it (the ranges it must avoid), then take the smallest
  // gap between/around them that fits, falling back to appending after the
  // last blocker.
  std::sort(intervals.begin(), intervals.end(),
            [](const Interval& a, const Interval& b) {
              if (a.size != b.size) return a.size > b.size;
              return a.name < b.name;
            });

  std::vector<Interval> placed;
  std::vector<std::pair<int64_t, int64_t>>
      placed_ranges;  // parallel: [off, off+size)
  std::map<std::string, std::pair<int64_t, int64_t>> group_offsets;

  for (const Interval& iv : intervals) {
    std::vector<std::pair<int64_t, int64_t>> blocking;
    for (std::size_t i = 0; i < placed.size(); ++i) {
      if (!Disjoint(iv, placed[i])) blocking.push_back(placed_ranges[i]);
    }
    std::sort(blocking.begin(), blocking.end());

    int64_t best_offset = -1;
    int64_t best_gap = -1;
    int64_t cursor = 0;
    for (const auto& [b_start, b_end] : blocking) {
      const int64_t gap = b_start - cursor;
      if (gap >= iv.size && (best_gap < 0 || gap < best_gap)) {
        best_offset = cursor;
        best_gap = gap;
      }
      cursor = std::max(cursor, b_end);
    }
    if (best_offset < 0)
      best_offset = cursor;  // no gap fit; append after every blocker

    group_offsets[iv.name] = {best_offset, iv.size};
    placed.push_back(iv);
    placed_ranges.push_back({best_offset, best_offset + iv.size});
    plan.arena_bytes = std::max(plan.arena_bytes, best_offset + iv.size);
  }

  // Expand each group's single placement to every tensor that was coalesced
  // into it -- an aliased tensor and the value(s) it shares storage with all
  // report the identical (offset, size).
  for (const auto& [name, start] : produced_at) {
    (void)start;
    const auto it = group_offsets.find(uf.Find(name));
    if (it != group_offsets.end()) plan.offsets[name] = it->second;
  }

  return plan;
}

}  // namespace onnxsim
