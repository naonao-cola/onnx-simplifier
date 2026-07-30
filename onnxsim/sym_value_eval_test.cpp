// Standalone test:
//   g++ -std=c++17 sym_expr.cpp sym_value_eval.cpp sym_value_eval_test.cpp -o t
//   ./t
// (Builds identically under Emscripten: em++ -std=c++17 ...)
//
// Exercises the M1 symbolic value evaluator (issue #532) op-by-op and on the
// end-to-end shape-scaffolding chains from the PoC / tests/test_simple.py,
// entirely without ONNX -- the graph is assembled from the plain SymGraph
// structs the real adapter will fill.
#include "sym_value_eval.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using onnxsim::SymAttr;
using onnxsim::SymExpr;
using onnxsim::SymGraph;
using onnxsim::SymNode;
using onnxsim::SymTensor;

namespace {

// --- tiny builders so the graphs below read like the ONNX helper API --------

SymTensor Vec(std::vector<int64_t> xs) {
  std::vector<SymExpr> d;
  for (int64_t x : xs) d.emplace_back(x);
  return SymTensor::Vector(std::move(d));
}
SymTensor Scal(int64_t x) { return SymTensor::Scalar(SymExpr(x)); }

SymAttr AInt(std::string name, int64_t v) {
  SymAttr a;
  a.name = std::move(name);
  a.i = v;
  return a;
}
SymAttr AInts(std::string name, std::vector<int64_t> v) {
  SymAttr a;
  a.name = std::move(name);
  a.ints = std::move(v);
  return a;
}
SymAttr ATensor(std::string name, SymTensor t) {
  SymAttr a;
  a.name = std::move(name);
  a.t = std::move(t);
  return a;
}

SymNode N(std::string op, std::vector<std::string> in,
          std::vector<std::string> out, std::vector<SymAttr> attrs = {}) {
  SymNode n;
  n.op_type = std::move(op);
  n.input = std::move(in);
  n.output = std::move(out);
  n.attribute = std::move(attrs);
  return n;
}

// A resolved value equals this list of sympy-style strings, with this rank.
bool Is(const std::map<std::string, SymTensor>& values, const std::string& name,
        std::vector<std::string> expect, bool scalar) {
  auto it = values.find(name);
  if (it == values.end()) return false;
  const SymTensor& t = it->second;
  if (t.scalar != scalar) return false;
  if (t.data.size() != expect.size()) return false;
  for (std::size_t i = 0; i < expect.size(); ++i)
    if (t.data[i].str() != expect[i]) return false;
  return true;
}

const SymExpr batch = SymExpr::Symbol("batch");

}  // namespace

int main() {
  // === Fully-static Shape -> Gather collapses even with a dynamic batch =====
  // (tests/test_simple.py::test_partial_shape_evaluation_gather)
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3), SymExpr(4), SymExpr(5)};
    g.initializer["indices"] = Vec({1, 2, 3});
    g.node = {N("Shape", {"x"}, {"s"}),
              N("Gather", {"s", "indices"}, {"gout"}, {AInt("axis", 0)})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "s", {"batch", "3", "4", "5"}, /*scalar=*/false));
    assert(Is(v, "gout", {"3", "4", "5"}, false));
  }

  // === Gather that reads the dynamic dim stays symbolic =====================
  // The value is still produced -- the consumer decides foldability -- but it
  // carries the symbol, so the fully-known folder will not fold it.
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3), SymExpr(4), SymExpr(5)};
    g.initializer["idx0"] = Vec({0});
    g.node = {N("Shape", {"x"}, {"s"}),
              N("Gather", {"s", "idx0"}, {"b"}, {AInt("axis", 0)})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "b", {"batch"}, false));
    assert(v.at("b").data[0].is_symbolic());
  }

  // === Shape -> Gather(batch) -> Concat -> [batch, 60] ======================
  // The single-symbolic-entry shape the #526 Reshape rewrite turns into
  // [-1,60]. Also exercises the scalar-index -> Unsqueeze -> Concat variant.
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3), SymExpr(4), SymExpr(5)};
    g.initializer["idx0"] = Scal(0);  // scalar index -> scalar Gather result
    g.initializer["sixty"] = Vec({60});
    g.node = {N("Shape", {"x"}, {"s"}),
              N("Gather", {"s", "idx0"}, {"bscalar"}, {AInt("axis", 0)}),
              N("Unsqueeze", {"bscalar"}, {"bvec"}, {AInts("axes", {0})}),
              N("Concat", {"bvec", "sixty"}, {"newshape"}, {AInt("axis", 0)})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "bscalar", {"batch"}, /*scalar=*/true));
    assert(Is(v, "bvec", {"batch"}, false));
    assert(Is(v, "newshape", {"batch", "60"}, false));
  }

  // === Value survives a flattening Reshape ==================================
  // (tests/test_simple.py::test_data_propagation_through_reshape)
  //   Shape(x) -> Reshape(., [-1]) -> Gather([1, 2]) == [3, 4]
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3), SymExpr(4)};
    g.initializer["flat"] = Vec({-1});
    g.initializer["indices"] = Vec({1, 2});
    g.node = {N("Shape", {"x"}, {"s"}), N("Reshape", {"s", "flat"}, {"s2"}),
              N("Gather", {"s2", "indices"}, {"gout"}, {AInt("axis", 0)})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "s2", {"batch", "3", "4"}, false));
    assert(Is(v, "gout", {"3", "4"}, false));
  }

  // === Div: Reshape's -1 resolved as total // non_deferred =================
  //   ReduceProd(Shape([batch, 1024, 128])) / 131072  ==  batch
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(1024), SymExpr(128)};
    g.initializer["den"] = Vec({1024 * 128});
    g.node = {N("Shape", {"x"}, {"s"}),
              N("ReduceProd", {"s"}, {"total"}, {AInt("keepdims", 1)}),
              N("Div", {"total", "den"}, {"q"})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "total", {"131072*batch"}, false));
    assert(Is(v, "q", {"batch"}, false));
  }

  // A Div that does not divide evenly is undecidable -> no value emitted.
  {
    SymGraph g;
    g.shape["x"] = {batch};
    g.initializer["two"] = Vec({2});
    g.node = {N("Shape", {"x"}, {"s"}), N("Div", {"s", "two"}, {"q"})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(v.find("q") == v.end());
  }

  // === Equal + Where over decidable comparisons ============================
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3)};
    g.initializer["idx1"] = Vec({1});
    g.initializer["three"] = Vec({3});
    g.initializer["a"] = Vec({7});
    g.initializer["b"] = Vec({9});
    g.node = {N("Shape", {"x"}, {"s"}),
              N("Gather", {"s", "idx1"}, {"dim1"}, {AInt("axis", 0)}),
              N("Equal", {"dim1", "three"}, {"cond"}),
              N("Where", {"cond", "a", "b"}, {"sel"})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "cond", {"1"}, false));  // 3 == 3
    assert(Is(v, "sel", {"7"}, false));
  }

  // A Where whose condition is undecidable (symbolic Equal) emits nothing.
  {
    SymGraph g;
    g.shape["x"] = {batch};
    g.initializer["five"] = Vec({5});
    g.initializer["a"] = Vec({7});
    g.initializer["b"] = Vec({9});
    g.node = {N("Shape", {"x"}, {"bdim"}),  // [batch]
              N("Equal", {"bdim", "five"}, {"cond"}),
              N("Where", {"cond", "a", "b"}, {"sel"})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(v.find("cond") == v.end());
    assert(v.find("sel") == v.end());
  }

  // === Expand broadcasts shape data ========================================
  {
    SymGraph g;
    g.initializer["v"] = Scal(5);
    g.initializer["shape"] = Vec({3});
    g.node = {N("Expand", {"v", "shape"}, {"e"})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "e", {"5", "5", "5"}, false));
  }

  // === ConstantOfShape fills a length from another value ===================
  {
    SymGraph g;
    g.initializer["shape"] = Vec({4});
    g.node = {
        N("ConstantOfShape", {"shape"}, {"z"}),  // default fill 0
        N("ConstantOfShape", {"shape"}, {"ones"}, {ATensor("value", Scal(1))})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "z", {"0", "0", "0", "0"}, false));
    assert(Is(v, "ones", {"1", "1", "1", "1"}, false));
  }

  // === Range / Tile / Slice / ReduceProd / Size / Cast / Squeeze ===========
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(6)};
    g.initializer["start"] = Scal(0);
    g.initializer["limit"] = Scal(4);
    g.initializer["delta"] = Scal(1);
    g.initializer["reps"] = Vec({2});
    g.initializer["data"] = Vec({10, 11, 12, 13});
    g.node = {
        N("Range", {"start", "limit", "delta"}, {"r"}),  // [0,1,2,3]
        N("Tile", {"r", "reps"}, {"tiled"}),             // [0,1,2,3,0,1,2,3]
        N("Slice", {"data"}, {"sl"},                     // data[1:3] -> [11,12]
          {AInts("starts", {1}), AInts("ends", {3}), AInts("axes", {0})}),
        N("Size", {"x"}, {"sz"}),                          // batch*6
        N("Cast", {"data"}, {"casted"}, {AInt("to", 6)}),  // INT32 -> unchanged
    };
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "r", {"0", "1", "2", "3"}, false));
    assert(Is(v, "tiled", {"0", "1", "2", "3", "0", "1", "2", "3"}, false));
    assert(Is(v, "sl", {"11", "12"}, false));
    assert(Is(v, "sz", {"6*batch"}, /*scalar=*/true));
    assert(Is(v, "casted", {"10", "11", "12", "13"}, false));
  }

  // Slice with a negative step reverses; Squeeze [1] -> scalar.
  {
    SymGraph g;
    g.initializer["data"] = Vec({0, 1, 2, 3});
    g.initializer["one"] = Vec({1});
    g.node = {
        N("Slice", {"data"}, {"rev"},
          {AInts("starts", {3}), AInts("ends", {-5}), AInts("axes", {0}),
           AInts("steps", {-1})}),
        N("Squeeze", {"one"}, {"scl"}),
    };
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "rev", {"3", "2", "1", "0"}, false));
    assert(Is(v, "scl", {"1"}, /*scalar=*/true));
  }

  // Cast to a float type is not modelled -> unresolved.
  {
    SymGraph g;
    g.initializer["data"] = Vec({3, 4});
    g.node = {N("Cast", {"data"}, {"f"}, {AInt("to", 1)})};  // FLOAT
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(v.find("f") == v.end());
  }

  // === Min / Max: decidable only on a concrete difference ==================
  {
    SymGraph g;
    g.shape["x"] = {batch};
    g.initializer["three"] = Vec({3});
    g.initializer["five"] = Vec({5});
    g.node = {
        N("Min", {"three", "five"}, {"mn"}),  // 3
        N("Max", {"three", "five"}, {"mx"}),  // 5
        N("Shape", {"x"}, {"bdim"}),
        N("Max", {"bdim", "five"}, {"undec"}),  // max(batch, 5) undecidable
    };
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "mn", {"3"}, false));
    assert(Is(v, "mx", {"5"}, false));
    assert(v.find("undec") == v.end());
  }

  // === Add / Sub / Mul with broadcasting ===================================
  {
    SymGraph g;
    g.shape["x"] = {batch, SymExpr(3)};
    g.initializer["one"] = Scal(1);
    g.node = {
        N("Shape", {"x"}, {"s"}),          // [batch, 3]
        N("Add", {"s", "one"}, {"plus"}),  // [batch + 1, 4]
        N("Mul", {"s", "s"}, {"sq"}),      // [batch**2, 9]
    };
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "plus", {"batch + 1", "4"}, false));
    assert(Is(v, "sq", {"batch**2", "9"}, false));
  }

  // === Constant node seeds (value_ints) ====================================
  {
    SymGraph g;
    g.node = {N("Constant", {}, {"c"}, {AInts("value_ints", {2, 3})})};
    const auto v = onnxsim::EvaluateSymbolicValues(g);
    assert(Is(v, "c", {"2", "3"}, false));
  }

  std::cout << "sym_value_eval_test: all assertions passed\n";
  return 0;
}
