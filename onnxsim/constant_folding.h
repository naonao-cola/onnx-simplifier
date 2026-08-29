#pragma once

// Constant folding: partitioning a graph's nodes into foldable/non-foldable,
// running the foldable ones through a ModelExecutor, and materializing their
// outputs -- as a plain initializer when the fold traces back purely to graph
// initializers, or as a fresh Constant node otherwise (e.g. a fold that
// consumed a pre-existing Constant node's embedded value), so that
// provenance stays visible in the output model. Two parallel implementations
// live here: one that works on onnx::ModelProto (used by the legacy Simplify
// path), and a Graph-native one that walks onnx::Graph directly (see this
// file's own top-of-.cpp comment on the Graph-native section for why both
// exist).

#include <onnx/onnx_pb.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxsim.h"

namespace onnx {
class Graph;
}  // namespace onnx

// Shared simplification configuration. Read by constant folding (this file)
// and by Optimize()/OptAndShapeOnGraph() (onnxsim.cpp) to keep both in sync
// on how initializers are treated and which optimizer passes run.
struct Config {
  std::vector<std::string> optimizer_passes;
  // default value is max
  size_t tensor_size_threshold = -1;
  // Whether graph initializers are treated as constant tensors. When false,
  // constant folding does not seed its constant set with initializer names, so
  // nodes rooted only at initializers are left unfolded (see GetConstantNodes),
  // and the onnx optimizer is told to leave initializer-backed weights alone
  // too (see Optimize). Constant nodes are always constant.
  bool initializers_as_constants = true;
};
extern Config config;

// Correct the determinism metadata of operators ONNX mis-annotates, so
// IsDeterministic (constant_folding.cpp) can fold them. Simplify() calls this
// once, before the first folding round.
void FixupSchemaDeterminism();

// Fold every foldable constant subexpression of `model` (into initializers or
// Constant nodes, see this file's own top-of-file comment), dropping the
// folded nodes and sweeping up initializers left unused by the fold.
onnx::ModelProto _FoldConstant(const ModelExecutor& executor,
                               onnx::ModelProto model);

// Graph-native counterpart of _FoldConstant. Returns whether anything folded.
bool _FoldConstantOnGraph(const ModelExecutor& executor, onnx::Graph& g,
                          int64_t ir_version);

// Runs onnxsim's own optimizer-pass fixed point once over `model` (via an
// onnx::ModelProto <-> Graph round trip), honoring `config`.
onnx::ModelProto Optimize(onnx::ModelProto& model);
