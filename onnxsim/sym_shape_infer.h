#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sym_expr.h"
#include "sym_value_eval.h"  // SymNode, SymAttr, SymTensor

namespace onnxsim {

// M2 of issue #532: dependency-free *symbolic activation-shape inference*.
//
// M1 (sym_value_eval) can fold the shape scaffolding only once `Shape(t)` on an
// intermediate tensor `t` returns a clean symbolic shape. ONNX shape inference
// drops the dynamic-dim symbol on most intermediates (it leaves a bare unnamed
// dimension), so `Shape(t)` comes back opaque and the scaffolding stalls. This
// pass re-derives each tensor's shape as a vector of `SymExpr` dims, *carrying
// the batch/sequence symbol* through the compute ops (`Conv`, `MatMul`, `Gemm`,
// pooling, broadcast arithmetic, `Softmax`, `LayerNormalization`, `Transpose`,
// `Concat`, reductions, ...) using ONNX's own shape rules expressed over
// `SymExpr`. Its output is exactly the `SymGraph::shape` seed M1 consumes.
//
// Per the PoC in #532 this is the high-leverage half: with ONNX-only shapes a
// value evaluator resolves ~1-6 Reshapes; once intermediates carry the symbol
// it resolves nearly all of them.
//
// Like M1 this touches no onnx:: type and has no sympy/SymEngine dependency, so
// it builds under Emscripten/WASM and is unit-testable on its own
// (sym_shape_infer_test.cpp). The `_InferShapes` adapter that fills ShapeGraph
// from a ModelProto and feeds the result into M1 is the follow-up milestone M3.
//
// A dimension is a `SymExpr`: a concrete `dim_value` d becomes `SymExpr(d)`, a
// `dim_param` "s" becomes `SymExpr::Symbol("s")`. A dim that a rule cannot
// compute exactly (e.g. a strided `Conv` over a symbolic spatial length, which
// is not a polynomial) is given a *fresh distinct symbol* ("unk_N") rather than
// dropped, so the rest of the shape -- crucially the batch dim a downstream
// `Reshape` targets -- stays usable. Fresh symbols are never merged, so a shape
// is only ever left less resolved, never wrongly equated.
using SymShape = std::vector<SymExpr>;

// Seeds for the pass. `value_info` holds every already-known shape (graph
// inputs with their dim_param symbols, and the shapes of all
// initializers/weights); `initializer` holds integer initializer *values*,
// needed by the shape rules whose output rank/dims depend on a data input
// (`Reshape` target, `Squeeze` axes, `Slice` starts/ends, `Expand` shape, ...).
struct ShapeGraph {
  std::vector<SymNode> node;  // topologically sorted
  std::map<std::string, SymShape>
      value_info;  // known shapes (inputs + weights)
  std::map<std::string, SymTensor> initializer;  // integer initializer values
};

// Walk the graph once in topological order, inferring a `SymShape` for every
// tensor the implemented rules can reach, seeded from `value_info`. A tensor
// whose shape (or rank) cannot be determined is simply absent from the result;
// `Shape()` on it in M1 then stays unevaluated rather than guessing.
std::map<std::string, SymShape> InferSymbolicShapes(const ShapeGraph& graph);

}  // namespace onnxsim
