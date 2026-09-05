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
// size S=100 bytes. Relu is in-place-safe (see IsInPlaceSafeOp), so a and b
// are each the sole input of the next Relu and neither is a weight/graph
// input/graph output -- they union into one group with y (Relu(b) -> y
// unions b into y too). x stays its own group: it is a graph input, which
// the aliasing pass never overwrites. The two groups still overlap at the
// x/a boundary under the same-node-boundary conservative rule, so they still
// need separate slots -- arena_bytes is unchanged at 200 (half of the 400 a
// naive "one slot per tensor" allocation would need) -- but now via one
// slot for x and one *shared* slot for {a, b, y}, rather than the
// non-aliased packing's two independently-reused slots.
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
  assert(plan.arena_bytes == 200);  // 2x compression, now via aliasing

  const auto& off = plan.offsets;
  assert(off.at("x").first != off.at("a").first);  // still can't share with x
  assert(off.at("a") == off.at("b"));  // a, b, y are literally one group now
  assert(off.at("b") == off.at("y"));
}

// A pure elementwise chain with no graph-input/output boundary in the
// middle -- x -> Relu -> a -> Sigmoid -> b -> Tanh -> c -> Neg -> d ->
// Identity -> out -- unions a, b, c, d and out into a single group (each is
// the sole consumer of the previous value, and none is a weight/graph
// input/graph output), leaving only x (a graph input, never aliased) in a
// group of its own. So the arena is exactly 2 slots regardless of how long
// the chain is, while naive_bytes keeps growing with every tensor -- the
// asymptotic case aliasing is for, versus TestChainReuse's short chain where
// disjoint-interval reuse alone already got to the same arena size.
void TestInPlaceAliasingCollapsesWholeChain() {
  GraphView g;
  for (const char* n : {"x", "a", "b", "c", "d", "out"}) {
    g.shapes[n] = {SymExpr(25)};
    g.dtypes[n] = 4;  // 100 bytes each
  }
  g.inputs = {"x"};
  g.outputs = {"out"};

  NodeView relu, sigmoid, tanh, neg, identity;
  relu.op_type = "Relu";
  relu.inputs = {"x"};
  relu.outputs = {"a"};
  sigmoid.op_type = "Sigmoid";
  sigmoid.inputs = {"a"};
  sigmoid.outputs = {"b"};
  tanh.op_type = "Tanh";
  tanh.inputs = {"b"};
  tanh.outputs = {"c"};
  neg.op_type = "Neg";
  neg.inputs = {"c"};
  neg.outputs = {"d"};
  identity.op_type = "Identity";
  identity.inputs = {"d"};
  identity.outputs = {"out"};
  g.nodes = {relu, sigmoid, tanh, neg, identity};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.naive_bytes == 600);  // 6 tensors x 100 bytes, no reuse credit
  assert(plan.arena_bytes == 200);  // just x's slot + one shared slot

  const auto& off = plan.offsets;
  assert(off.at("x").first != off.at("a").first);
  assert(off.at("a") == off.at("b"));
  assert(off.at("b") == off.at("c"));
  assert(off.at("c") == off.at("d"));
  assert(off.at("d") == off.at("out"));
}

// A chain of pure view ops -- Reshape -> Squeeze -> Unsqueeze -> Flatten --
// reinterprets the same 100 bytes under a different shape at every step with
// no computation at all, so they union into one group exactly like the
// in-place-safe unary chain above; x (a graph input) stays separate. Real
// Reshape/Squeeze/Unsqueeze also take a second (shape/axes) input, which
// this test's Reshape node includes ("s") -- it is never a shape/dtype
// candidate for anything and correctly plays no role in the plan.
void TestViewOpsAliasWholeChain() {
  GraphView g;
  for (const char* n : {"x", "a", "b", "c", "out"}) {
    g.shapes[n] = {SymExpr(25)};
    g.dtypes[n] = 4;  // 100 bytes each
  }
  g.inputs = {"x"};
  g.outputs = {"out"};

  NodeView reshape, squeeze, unsqueeze, flatten;
  reshape.op_type = "Reshape";
  reshape.inputs = {"x",
                    "s"};  // "s" (the shape tensor) is deliberately unmodeled
  reshape.outputs = {"a"};
  squeeze.op_type = "Squeeze";
  squeeze.inputs = {"a"};
  squeeze.outputs = {"b"};
  unsqueeze.op_type = "Unsqueeze";
  unsqueeze.inputs = {"b"};
  unsqueeze.outputs = {"c"};
  flatten.op_type = "Flatten";
  flatten.inputs = {"c"};
  flatten.outputs = {"out"};
  g.nodes = {reshape, squeeze, unsqueeze, flatten};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.naive_bytes == 500);  // 5 tensors x 100 bytes (x, a, b, c, out)
  assert(plan.arena_bytes == 200);  // x's slot + one shared slot

  const auto& off = plan.offsets;
  assert(off.at("x").first != off.at("a").first);
  assert(off.at("a") == off.at("b"));
  assert(off.at("b") == off.at("c"));
  assert(off.at("c") == off.at("out"));
}

// View ops (IsViewOp) and in-place-safe compute ops (IsInPlaceSafeOp) freely
// mix in the same union-find group: x -> Reshape -> a -> Relu -> b ->
// Flatten -> out. x is a graph input and stays separate (Reshape's own
// aliasing is blocked the same way any other op's would be); a, b and out
// all end up in one group despite two different op categories producing the
// edges between them.
void TestViewAndInPlaceOpsShareOneGroup() {
  GraphView g;
  for (const char* n : {"x", "a", "b", "out"}) {
    g.shapes[n] = {SymExpr(25)};
    g.dtypes[n] = 4;  // 100 bytes each
  }
  g.inputs = {"x"};
  g.outputs = {"out"};

  NodeView reshape, relu, flatten;
  reshape.op_type = "Reshape";
  reshape.inputs = {"x", "s"};
  reshape.outputs = {"a"};
  relu.op_type = "Relu";
  relu.inputs = {"a"};
  relu.outputs = {"b"};
  flatten.op_type = "Flatten";
  flatten.inputs = {"b"};
  flatten.outputs = {"out"};
  g.nodes = {reshape, relu, flatten};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.naive_bytes == 400);  // 4 tensors x 100 bytes
  assert(plan.arena_bytes == 200);  // x's slot + one shared slot

  const auto& off = plan.offsets;
  assert(off.at("x").first != off.at("a").first);
  assert(off.at("a") == off.at("b"));
  assert(off.at("b") == off.at("out"));
}

// `a` feeds two separate in-place-eligible ops (Neg -> y1, Sigmoid -> y2).
// Without the "sole consumer" guard in the aliasing pass, both would try to
// alias `a` away, incorrectly merging y1 and y2 into the same slot despite
// both being live simultaneously (both are graph outputs, live through the
// end) -- corrupting whichever one a real executor computed second.
void TestInPlaceAliasingBlockedByMultipleConsumers() {
  GraphView g;
  for (const char* n : {"x", "a", "y1", "y2"}) {
    g.shapes[n] = {SymExpr(4)};
    g.dtypes[n] = 4;  // 16 bytes each
  }
  g.inputs = {"x"};
  g.outputs = {"y1", "y2"};

  NodeView relu, neg, sigmoid;
  relu.op_type = "Relu";
  relu.inputs = {"x"};
  relu.outputs = {"a"};
  neg.op_type = "Neg";
  neg.inputs = {"a"};
  neg.outputs = {"y1"};
  sigmoid.op_type = "Sigmoid";
  sigmoid.inputs = {"a"};
  sigmoid.outputs = {"y2"};
  g.nodes = {relu, neg, sigmoid};

  const MemoryPlan plan = ComputeActivationMemoryPlan(g);
  assert(plan.unplanned.empty());
  assert(plan.offsets.at("y1").first != plan.offsets.at("y2").first);
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
  TestInPlaceAliasingCollapsesWholeChain();
  TestViewOpsAliasWholeChain();
  TestViewAndInPlaceOpsShareOneGroup();
  TestInPlaceAliasingBlockedByMultipleConsumers();
  TestSymbolicShapeUnplanned();
  std::cout << "all memory_planning tests passed\n";
  return 0;
}
