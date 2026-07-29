#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <map>
#include <string>

// A lightweight, dependency-free counterpart to the Python ``ModelInfo`` for
// the non-Python bindings (the CLI binary, the C ABI / Rust wrapper, and the
// WASM converter). It reports the two metrics that can be derived from a
// ``ModelProto`` alone, without ONNX shape inference:
//
//   * ``op_nums`` -- how many of each op type the model contains, recursing
//   into
//     control-flow subgraphs. Initializers are folded into the ``Constant``
//     count, exactly as the Python ``ModelInfo`` does, so a weight tensor reads
//     as a constant.
//   * ``model_size`` -- the serialized byte size of the graph plus any tensor
//     data kept in external files (read from the external-data metadata, so
//     weights on disk are counted without being materialized).
//
// The heavier metrics the Python table also shows (MACs / FLOPs and the memory
// figures) require full shape inference and a symbolic-math dependency, so they
// are intentionally out of scope for these bindings.
struct ModelInfo {
  std::map<std::string, int64_t> op_nums;
  int64_t model_size = 0;
};

// Compute the op counts and model size of ``model``.
ModelInfo GetModelInfo(const onnx::ModelProto& model);

// Render an ASCII table comparing an original and a simplified model, mirroring
// the layout of the Python ``print_simplifying_info``: one row per op type (the
// sorted union of both models' ops) followed by a ``Model Size`` row, with
// columns for the original and simplified values. A metric that improved (a
// dropped op count or a smaller model) is flagged with a trailing ``*``, since
// these bindings print to plain terminals where the Python rich-text colouring
// is unavailable. The returned string ends with a newline.
std::string FormatSimplifyingInfo(const onnx::ModelProto& model_ori,
                                  const onnx::ModelProto& model_opt);
