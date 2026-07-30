#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sym_expr.h"

namespace onnxsim {

// M1 of issue #532: a dependency-free *symbolic value evaluator* for the shape
// scaffolding (`Shape -> Gather -> Unsqueeze -> Concat -> Reshape`, plus
// `Where`/`Equal`/`Div`/`Expand`) that ONNX data propagation cannot fold on a
// graph with a dynamic dimension.
//
// ONNX data propagation carries a tensor's value as a TensorShapeProto whose
// entries are either a concrete int or an *opaque* dim_param string; it cannot
// do arithmetic on a symbol, so a Reshape target like `[batch, 1024, 128]`, or
// any `Div`/`Equal`/`Where` over a symbolic dim, stalls the whole chain. This
// evaluator instead keeps each dynamic dim as a `SymExpr` (an
// integer-coefficient polynomial in dim symbols) and computes the shape algebra
// with the M0 primitives (`operator+/-/*`, `TryExactDivide`, `TryEqual`).
//
// Like `SymExpr`, this is pure standard C++ with no ONNX / SymEngine / sympy
// dependency, so it builds unchanged under Emscripten/WASM and is unit-testable
// on its own (see sym_value_eval_test.cpp). It deliberately touches no onnx::
// type: the caller (the `_EvalPartialShape` adapter, wired in a later
// milestone) pre-extracts nodes, initializers and tensor shapes into the plain
// structs below.
//
// Design rule for every handler: an op contributes an output value only when it
// is *fully decidable*. Any undecidable step (a symbolic Div that does not
// divide evenly, a Where whose condition cannot be resolved, an out-of-range
// index) leaves the output absent, so a downstream consumer sees "unknown" and
// keeps the runtime computation rather than folding something wrong. Final
// graphs remain gated by onnxsim's own equivalence check.

// A symbolic integer shape-data value: a rank-0 (scalar) or rank-1 (vector)
// tensor whose entries are `SymExpr`. Rank 0 vs rank 1 matters -- a Gather with
// a scalar index yields a scalar, Unsqueeze/Squeeze convert between the two,
// and Concat requires rank 1 -- so it is tracked explicitly. Higher-rank shape
// data does not occur in the scaffolding chains this pass targets; such tensors
// are simply left unevaluated.
struct SymTensor {
  std::vector<SymExpr> data;  // row-major; length == 1 when `scalar`
  bool scalar = false;        // rank 0 (a single number) vs rank 1 (a vector)

  static SymTensor Scalar(SymExpr v) {
    SymTensor t;
    t.data.push_back(std::move(v));
    t.scalar = true;
    return t;
  }
  static SymTensor Vector(std::vector<SymExpr> v) {
    SymTensor t;
    t.data = std::move(v);
    t.scalar = false;
    return t;
  }

  int64_t size() const { return static_cast<int64_t>(data.size()); }

  // The concrete length of the *dim axis* for a rank-1 value; a scalar has no
  // axis. Used only where a rank-1 value is expected.
  bool is_vector() const { return !scalar; }
};

// A node attribute, pre-extracted from the ONNX NodeProto into the few forms
// the handlers read (INT, INTS, TENSOR). Only one field is populated per
// attribute.
struct SymAttr {
  std::string name;
  std::optional<int64_t> i;   // AttributeProto::INT
  std::vector<int64_t> ints;  // AttributeProto::INTS
  std::optional<SymTensor>
      t;  // AttributeProto::TENSOR (Constant / ConstantOfShape)
};

// One node of the graph, in topological order (ONNX graphs already are). An
// empty input name is the ONNX "optional input omitted" sentinel and is treated
// as absent.
struct SymNode {
  std::string op_type;
  std::vector<std::string> input;
  std::vector<std::string> output;
  std::vector<SymAttr> attribute;

  const SymAttr* attr(const std::string& name) const {
    for (const auto& a : attribute)
      if (a.name == name) return &a;
    return nullptr;
  }
};

// Everything the evaluator needs, all dependency-free.
struct SymGraph {
  std::vector<SymNode> node;  // topologically sorted
  std::map<std::string, SymTensor>
      initializer;  // integer initializers as values

  // The shape of a tensor, one `SymExpr` per dimension, seeding `Shape(name)`:
  //   dim_value d   -> SymExpr(d)
  //   dim_param "s" -> SymExpr::Symbol("s")     (the dynamic dim symbol)
  // A tensor absent from this map has no known shape, so `Shape(name)` on it is
  // not evaluable and is skipped.
  std::map<std::string, std::vector<SymExpr>> shape;
};

// Walk the graph once in topological order, maintaining a `name -> SymTensor`
// value table seeded from initializers, `Constant` nodes and `Shape()` of
// tensors with a known (possibly symbolic) shape. Returns every tensor whose
// value the pass could resolve. The result may contain values that are still
// symbolic (e.g. `[batch, 60]`); that is exactly what the `_EvalPartialShape`
// Reshape rewrite consumes -- a single symbolic entry becomes the Reshape `-1`
// slot.
std::map<std::string, SymTensor> EvaluateSymbolicValues(const SymGraph& graph);

}  // namespace onnxsim
