#pragma once

// Partial shape evaluation (issue #139): folding shape-family subgraphs
// (Shape/Gather/Slice/Concat/..., and single-dynamic-dim Reshape targets)
// into Constant nodes using ONNX's own data-propagating shape inference plus
// onnxsim's native symbolic evaluator (issue #532). See this file's own
// .cpp for the two entry points' shared design note.

#include <onnx/onnx_pb.h>

namespace onnx {
class Graph;
}  // namespace onnx

// Runs ordinary (non-data-propagating) ONNX shape inference over `model` in
// place.
void _InferShapes(onnx::ModelProto& model);

// Rewrite every node of `model` whose lone output is provably a fully static
// shape-derived value into a `Constant` node, so the ordinary constant folder
// and optimizer can take it from there. Mutates `model` in place; a no-op
// leaves it byte-for-byte unchanged.
void _EvalPartialShape(onnx::ModelProto& model);

// Graph-native counterpart of _EvalPartialShape, reached without a
// ModelProto <-> Graph round trip. Returns whether anything changed.
bool _EvalPartialShapeOnGraph(onnx::Graph& g);

// Unset the recurrent initial states of RNN/GRU/LSTM nodes in `graph` (and
// its nested subgraphs) that are provably all zeros (issue #314), so the
// batch-dependent subgraph that used to compute them becomes dead.
void EliminateZeroRnnInitialState(onnx::Graph& graph);
