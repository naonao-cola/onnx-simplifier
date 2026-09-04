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
// alias with its inputs' storage in general (only true in-place ops allow
// that, and this allocator has no notion of which ops are in-place-safe).
bool Disjoint(const Interval& a, const Interval& b) {
  return a.end < b.start || b.end < a.start;
}

}  // namespace

MemoryPlan ComputeActivationMemoryPlan(const GraphView& graph) {
  MemoryPlan plan;

  const std::set<std::string> weights(graph.initializers.begin(),
                                      graph.initializers.end());

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

  // Pass 1: one Interval per plannable tensor (known concrete size);
  // everything else goes to `unplanned`. An unconsumed intermediate (no
  // `last_use` entry, not a graph output) defaults to staying live through
  // `end` -- the same "never freed" fallback the live-set walk behind
  // PeakMemoryFootprint implicitly applies to it.
  std::vector<Interval> intervals;
  for (const auto& [name, start] : produced_at) {
    const auto bytes = TensorBytes(name, graph.shapes, graph.dtypes);
    if (!bytes || bytes->is_symbolic()) {
      plan.unplanned.push_back(name);
      continue;
    }
    const auto it = last_use.find(name);
    intervals.push_back({name, bytes->to_int(), start,
                         it == last_use.end() ? end : it->second});
    plan.naive_bytes += bytes->to_int();
  }

  // Pass 2: greedy best-fit placement, largest tensor first (ties broken by
  // name for determinism) -- the standard linear-scan/greedy-by-size register
  // allocation heuristic, which is what makes this "fragmentation-aware"
  // rather than a naive bump allocator. For each tensor, gather the offset
  // ranges of every already-placed tensor whose liveness overlaps it (the
  // ranges it must avoid), then take the smallest gap between/around them
  // that fits, falling back to appending after the last blocker.
  std::sort(intervals.begin(), intervals.end(),
            [](const Interval& a, const Interval& b) {
              if (a.size != b.size) return a.size > b.size;
              return a.name < b.name;
            });

  std::vector<Interval> placed;
  std::vector<std::pair<int64_t, int64_t>>
      placed_ranges;  // parallel: [off, off+size)

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

    plan.offsets[iv.name] = {best_offset, iv.size};
    placed.push_back(iv);
    placed_ranges.push_back({best_offset, best_offset + iv.size});
    plan.arena_bytes = std::max(plan.arena_bytes, best_offset + iv.size);
  }

  return plan;
}

}  // namespace onnxsim
