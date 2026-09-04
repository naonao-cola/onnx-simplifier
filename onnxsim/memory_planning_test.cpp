// Standalone test for the ONNX-free memory-planning core:
//   g++ -std=c++20 memory_planning.cpp model_metrics.cpp sym_expr.cpp \
//       memory_planning_test.cpp -o t && ./t
#include "memory_planning.h"

#include <cassert>
#include <iostream>

#include "sym_expr.h"

namespace {

using onnxsim::ComputeActivationMemoryPlan;
using onnxsim::GraphView;
using onnxsim::MemoryPlan;
using onnxsim::NodeView;
using onnxsim::SymExpr;

// Single Conv: x [1,3,8,8] f32 -> y [1,4,8,8] f32, weight w [4,3,3,3] f32.
// x is read by the same node that produces y, so under the allocator's
// conservative same-node-boundary rule they can never share space: the
// planned arena must be exactly xb + yb, and it must exclude the weight
// (weights stay resident, not part of the activation arena) -- which doubles
// as a cross-check against PeakMemoryFootprint: wb + xb + yb (from
// model_metrics_test.cpp's TestMemAccessAndPeak) minus the resident weight
// wb is exactly this arena.
void TestSingleNodeNoReuse() {
  GraphView g;
  g.shapes["x"] = {SymExpr(1), SymExpr(3), SymExpr(8), SymExpr(8)};
  g.shapes["w"] = {SymExpr(4), SymExpr(3), SymExpr(3), SymExpr(3)};
  g.shapes["y"] = {SymExpr(1), SymExpr(4), SymExpr(8), SymExpr(8)};
  for (const char* n : {"x", "w", "y"}) g.dtypes[n] = 4;  // float32
  g.inputs = {"x"};
  g.outputs = {"y"};
  g.initializers = {"w"};
  NodeView conv;
  conv.op_type = "Conv";
  conv.inputs = {"x", "w"};
  conv.outputs = {"y"};
  g.nodes = {conv};

  const int64_t xb = 1LL * 3 * 8 * 8 * 4;  // 768
  const int64_t yb = 1LL * 4 * 8 * 8 * 4;  // 1024

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.offsets.count("w") == 0);  // weights are never planned
  assert(plan.naive_bytes == xb + yb);
  assert(plan.arena_bytes ==
         xb + yb);  // no reuse possible: only one pair, overlapping
  assert(plan.offsets.at("x").second == xb);
  assert(plan.offsets.at("y").second == yb);
}

// A 3-node Relu chain x -> a -> b -> y (graph output), all tensors the same
// size S=100 bytes. Every tensor overlaps only its immediate neighbor in the
// chain (produced by the node that consumes the previous one), so
// non-adjacent tensors (x/b, x/y, a/y) are free to share offsets while
// adjacent ones (x/a, a/b, b/y) may not -- worked out by hand in the PR
// description: greedy best-fit (processed in name order a, b, x, y, since
// all four are the same size) lands on arena_bytes == 200 (half of the 400
// a naive "one slot per tensor" allocation would need), with a and y sharing
// offset 0 and b and x sharing offset 100.
void TestChainReuse() {
  GraphView g;
  for (const char* n : {"x", "a", "b", "y"}) {
    g.shapes[n] = {SymExpr(25)};
    g.dtypes[n] = 4;  // float32 -> 25*4 = 100 bytes
  }
  g.inputs = {"x"};
  g.outputs = {"y"};

  NodeView n0, n1, n2;
  n0.op_type = n1.op_type = n2.op_type = "Relu";
  n0.inputs = {"x"};
  n0.outputs = {"a"};
  n1.inputs = {"a"};
  n1.outputs = {"b"};
  n2.inputs = {"b"};
  n2.outputs = {"y"};
  g.nodes = {n0, n1, n2};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.naive_bytes == 400);
  assert(plan.arena_bytes == 200);  // 2x compression from reuse

  const auto& off = plan.offsets;
  assert(off.at("a").first == 0);
  assert(off.at("y").first == 0);  // shares a's freed slot
  assert(off.at("b").first == 100);
  assert(off.at("x").first == 100);  // shares b's (later-claimed) slot

  // No two simultaneously-live tensors may share an offset range: check the
  // three adjacent (overlapping) pairs landed at different offsets.
  assert(off.at("x").first != off.at("a").first);
  assert(off.at("a").first != off.at("b").first);
  assert(off.at("b").first != off.at("y").first);
}

// A tensor with a symbolic (dynamic) dimension has no concrete byte size, so
// it must be excluded from the plan (`unplanned`) rather than guessed, and
// must not contribute to naive_bytes/arena_bytes.
void TestSymbolicShapeUnplanned() {
  GraphView g;
  g.shapes["x"] = {SymExpr::Symbol("batch"), SymExpr(8)};
  g.shapes["y"] = {SymExpr::Symbol("batch"), SymExpr(8)};
  g.dtypes["x"] = g.dtypes["y"] = 4;
  g.inputs = {"x"};
  g.outputs = {"y"};
  NodeView relu;
  relu.op_type = "Relu";
  relu.inputs = {"x"};
  relu.outputs = {"y"};
  g.nodes = {relu};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.offsets.empty());
  assert(plan.naive_bytes == 0);
  assert(plan.arena_bytes == 0);
  assert(plan.unplanned.size() == 2);  // both x and y are symbolically sized
}

}  // namespace

int main() {
  TestSingleNodeNoReuse();
  TestChainReuse();
  TestSymbolicShapeUnplanned();
  std::cout << "all memory_planning tests passed\n";
  return 0;
}
