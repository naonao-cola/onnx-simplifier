#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

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
  // Count of the top-level graph's own initializers -- not the recursive,
  // subgraph-including tally that ``op_nums["Constant"]`` folds initializers
  // into. Reported as its own "Initializers" row by ``FormatSimplifyingInfo``
  // so a change here (e.g. weights folded into a fused node, or duplicate
  // initializers deduplicated) is visible without it being muddied by actual
  // ``Constant`` nodes in the same count.
  int64_t initializer_count = 0;
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

// Reduce `model`'s top-level graph to the ``GraphView`` model_metrics.h (and
// its dependents, e.g. memory_planning.h) operate on: node connectivity plus
// every fully-known tensor shape/dtype. Shares the exact same shape-inference
// step ``GetModelInfo`` uses for its own metrics (best-effort: falls back to
// `model`'s existing value_info if inference throws), so a caller building a
// GraphView-based report sees the same shapes ``GetModelInfo`` would.
onnxsim::GraphView GetGraphView(const onnx::ModelProto& model,
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
// sorted union of both models' ops), then ``Model Size``, ``Initializers``,
// ``MACs``, ``FLOPs``, ``Memory Access``, ``Memory Footprint`` and
// ``Compute Density``, with columns
// for the original and simplified values. A metric that improved (a dropped op
// count, or a smaller size / MACs / memory figure) is flagged with a trailing
// ``*``, since these bindings print to plain terminals where the Python
// rich-text colouring is unavailable. Compute density is reported without a
// flag (a change there is not strictly better or worse). The returned string
// ends with a newline.
std::string FormatSimplifyingInfo(const onnx::ModelProto& model_ori,
                                  const onnx::ModelProto& model_opt);

// A node's identity for diffing (see ``DiffGraphs``): enough to describe it on
// a diff line, not the full ``NodeProto`` -- attributes are not compared, so a
// node whose only change is an attribute value is not reported as "changed".
struct NodeDiffEntry {
  std::string op_type;
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

// Node- and value-level diff between an original and simplified graph (see
// ``DiffGraphs``), the C++ counterpart of the Python ``model_info.GraphDiff``
// (used by onnxsim's non-Python bindings: the CLI binary, the C ABI / Rust
// wrapper, and the WASM converter).
struct GraphDiff {
  std::vector<NodeDiffEntry> removed_nodes;
  std::vector<NodeDiffEntry> added_nodes;
  // (before, after) pairs that produce the same output(s) but whose op_type or
  // inputs changed, e.g. a Conv whose bias input was folded away.
  std::vector<std::pair<NodeDiffEntry, NodeDiffEntry>> changed_nodes;
  std::vector<std::string> removed_values;
  std::vector<std::string> added_values;
};

// Diff ``model_ori``'s top-level graph against ``model_opt``'s, matched by
// name rather than position, so the result reflects what simplification
// actually did to named values instead of a meaningless positional diff.
//
// Nodes are matched by their output tensor name(s) -- unique within a graph by
// the ONNX spec, and the identity a downstream consumer actually depends on --
// so a node is reported as "changed" (rather than removed + added) only when
// simplification kept the same output name(s) but altered the op_type or
// inputs. This is precisely why onnxsim tries to preserve value names across
// simplification passes where possible: it is what keeps this diff (and any
// other name-keyed tooling) meaningful instead of turning every fused/folded
// node into an unrelated remove+add pair.
//
// Only the top-level graph is compared -- nodes inside control-flow subgraphs
// (If/Loop/Scan bodies) are not matched across models, since names are only
// required to be unique within their own graph scope.
GraphDiff DiffGraphs(const onnx::ModelProto& model_ori,
                     const onnx::ModelProto& model_opt);

// Render the node- and value-level diff between ``model_ori`` and
// ``model_opt`` (see ``DiffGraphs``) as plain ASCII text, in a unified-diff
// style: ``-`` for what simplification removed, ``+`` for what it added, ``~``
// for a node kept under the same output name(s) but changed. Each section is
// capped at ``limit`` entries (with a "... and N more" line) so a large
// model's diff stays readable. The returned string ends with a newline.
std::string FormatGraphDiff(const onnx::ModelProto& model_ori,
                            const onnx::ModelProto& model_opt,
                            size_t limit = 50);

// Records a summary of what simplification changed structurally between
// ``model_ori`` and ``sim_model`` -- nodes removed by fusion/elimination,
// nodes changed in place under the same output name(s) (see ``DiffGraphs``),
// values that disappeared, and how many nodes' ``doc_string`` did not survive
// (as a plain count: see ``CountDroppedDocStrings``'s doc comment for why a
// capped list doesn't make sense for free-form prose) -- as ``metadata_props``
// on ``sim_model``, namespaced "onnxsim." like the rest of onnxsim's own
// metadata (``AnnotateModelInfo``, ``RecordSimplifyOptionsMetadata``). This
// persists the same diff ``FormatGraphDiff`` renders as a CLI report onto the
// model itself, so a downstream consumer can see what was lost to
// fusion/constant folding without needing the original model on hand. Each
// list is capped at ``limit`` entries (with a paired ``*.count`` key holding
// the true total) to keep the metadata compact on large graphs.
void RecordSimplifyDiffMetadata(onnx::ModelProto& sim_model,
                                const onnx::ModelProto& model_ori,
                                size_t limit = 20);

// Joins up to `limit` entries of `items` with ", ", appending a "... (+N
// more)" marker when there were more -- a single-line summary suitable for
// one metadata_props value (as opposed to ``FormatGraphDiff``'s multi-line
// report). Shared by ``RecordSimplifyDiffMetadata`` above and by other
// metadata-recording call sites (e.g. onnxsim.cpp's node-naming and
// function-inlining metadata) that need the same "capped list, one line"
// shape.
std::string JoinCapped(const std::vector<std::string>& items, size_t limit);

// Writes ``key`` (``JoinCapped(items, limit)``) and ``key + ".count"``
// (``items.size()``, uncapped) into ``model``'s metadata_props, overwriting
// any existing entries under those two keys. This is the common "capped
// list plus its true count" shape used by every list-valued onnxsim.*
// metadata entry -- ``RecordSimplifyDiffMetadata``'s three categories below,
// and onnxsim.cpp's own node-naming / function-inlining metadata.
void RecordCappedListMetadata(onnx::ModelProto& model, const std::string& key,
                              const std::vector<std::string>& items,
                              size_t limit);
