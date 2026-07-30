#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <map>
#include <string>

#include "model_metrics.h"

// A dependency-free counterpart to the Python ``ModelInfo`` for the non-Python
// bindings (the CLI binary, the C ABI / Rust wrapper, and the WASM converter).
// It reports:
//
//   * ``op_nums`` -- how many of each op type the model contains, recursing
//     into control-flow subgraphs. Initializers are folded into the
//     ``Constant`` count, exactly as the Python ``ModelInfo`` does, so a weight
//     tensor reads as a constant.
//   * ``model_size`` -- the serialized byte size of the graph plus any tensor
//     data kept in external files (read from the external-data metadata, so
//     weights on disk are counted without being materialized).
//   * ``macs`` / FLOPs -- multiply-accumulates of the compute-dominant
//     operators (Conv, ConvTranspose, Gemm, MatMul, Attention, and the
//     quantized twins), from ONNX shape inference. A dynamic dimension
//     (dim_param, e.g. "batch") stays symbolic via ``SymExpr``, so these may be
//     a formula rather than a number. Nodes whose shapes cannot be inferred
//     contribute 0. FLOPs are 2 * MACs.
//   * ``mem_access`` / ``memory_footprint`` -- forward-pass memory traffic
//     (every input and output touched) and the peak bytes resident at once
//     (weights plus live activations), from the same inferred shapes.
//
// Unlike the Python ``ModelInfo`` this does not expand/inline function bodies,
// so compute inside a function op is not yet counted (a follow-up); the op
// count still lists the function op itself.
struct ModelInfo {
  std::map<std::string, int64_t> op_nums;
  int64_t model_size = 0;
  onnxsim::SymExpr macs;
  onnxsim::SymExpr mem_access;
  onnxsim::SymExpr memory_footprint;

  // FLOPs are two per MAC (one multiply, one add).
  onnxsim::SymExpr Flops() const { return macs * onnxsim::SymExpr(2); }
};

// Compute the metrics of ``model``. The model is not modified. When
// ``run_shape_inference`` is true (the default) shape inference runs on an
// internal copy to populate the shapes the compute/memory metrics need; pass
// false when ``model`` already carries the value_info (e.g. a caller that
// inferred shapes itself, with data propagation) to avoid the extra pass.
ModelInfo GetModelInfo(const onnx::ModelProto& model,
                       bool run_shape_inference = true);

// Write onnxsim's model-info metrics into ``model``'s ``metadata_props`` in
// place, mirroring the Python ``model_info.annotate_metadata`` so downstream
// tools (e.g. the browser inference panel) can read them back:
//
//   * model level: ``onnxsim.macs``, ``onnxsim.flops``, ``onnxsim.mem_access``,
//     ``onnxsim.memory_footprint``, ``onnxsim.compute_density`` and
//     ``onnxsim.model_size``.
//   * node level: ``onnxsim.macs`` / ``onnxsim.flops`` / ``onnxsim.mem_access``
//     for each node of the top-level graph.
//
// Values are strings: a concrete metric as its plain number, a symbolic one
// (dynamic dims) as its factored formula (e.g. "512*batch"). Shapes are taken
// from a shape-inferred copy, so ``model`` keeps its graph structure and only
// gains ``metadata_props``. An existing entry with the same key is overwritten.
void AnnotateModelInfo(onnx::ModelProto& model);

// Render an ASCII table comparing an original and a simplified model, mirroring
// the layout of the Python ``print_simplifying_info``: one row per op type (the
// sorted union of both models' ops), then ``Model Size``, ``MACs``, ``FLOPs``,
// ``Memory Access``, ``Memory Footprint`` and ``Compute Density``, with columns
// for the original and simplified values. A metric that improved (a dropped op
// count, or a smaller size / MACs / memory figure) is flagged with a trailing
// ``*``, since these bindings print to plain terminals where the Python
// rich-text colouring is unavailable. Compute density is reported without a
// flag (a change there is not strictly better or worse). The returned string
// ends with a newline.
std::string FormatSimplifyingInfo(const onnx::ModelProto& model_ori,
                                  const onnx::ModelProto& model_opt);
