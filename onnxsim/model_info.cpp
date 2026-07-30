#include "model_info.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "model_metrics.h"
#include "onnx/shape_inference/implementation.h"

namespace {

using onnxsim::DTypeMap;
using onnxsim::GraphView;
using onnxsim::Metrics;
using onnxsim::NodeView;
using onnxsim::Shape;
using onnxsim::ShapeMap;
using onnxsim::SymExpr;
using onnxsim::SymRatio;

// Recursively tally op types, descending into every subgraph carried by a node
// attribute (the ``g`` of e.g. an ``If`` branch, or the ``graphs`` of
// ``Loop``). Every initializer of each graph is counted as a ``Constant``,
// matching the Python ``ModelInfo.get_info``.
void CountGraphOps(const onnx::GraphProto& graph,
                   std::map<std::string, int64_t>& op_nums) {
  for (const auto& node : graph.node()) {
    op_nums[node.op_type()] += 1;
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CountGraphOps(attr.g(), op_nums);
      }
      for (const auto& subgraph : attr.graphs()) {
        CountGraphOps(subgraph, op_nums);
      }
    }
  }
  op_nums["Constant"] += graph.initializer_size();
}

// Bytes of a single tensor's data when it lives in an external file, read from
// the ``length`` entry of its ``external_data`` record. Zero for tensors held
// inline (whose bytes are already part of the graph's serialized size).
int64_t TensorExternalSize(const onnx::TensorProto& tensor) {
  if (tensor.data_location() != onnx::TensorProto::EXTERNAL) {
    return 0;
  }
  for (const auto& entry : tensor.external_data()) {
    if (entry.key() == "length") {
      try {
        return std::stoll(entry.value());
      } catch (...) {
        return 0;
      }
    }
  }
  return 0;
}

// Total external-data bytes across every tensor a graph holds -- initializers
// as well as tensors carried in node attributes -- recursing into subgraphs.
// This mirrors Python's ``_external_data_size`` and never loads the data
// itself.
int64_t ExternalDataSize(const onnx::GraphProto& graph) {
  int64_t total = 0;
  for (const auto& initializer : graph.initializer()) {
    total += TensorExternalSize(initializer);
  }
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_t()) {
        total += TensorExternalSize(attr.t());
      }
      for (const auto& tensor : attr.tensors()) {
        total += TensorExternalSize(tensor);
      }
      if (attr.has_g()) {
        total += ExternalDataSize(attr.g());
      }
      for (const auto& subgraph : attr.graphs()) {
        total += ExternalDataSize(subgraph);
      }
    }
  }
  return total;
}

// Format a byte count with 1024-based binary units, matching the Python
// ``human_readable_size`` (e.g. 1536 -> "1.5KiB").
std::string HumanReadableSize(int64_t num) {
  double value = static_cast<double>(num);
  static const char* const kUnits[] = {"",   "Ki", "Mi", "Gi",
                                       "Ti", "Pi", "Ei", "Zi"};
  char buf[64];
  for (const char* unit : kUnits) {
    if (std::fabs(value) < 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.1f%sB", value, unit);
      return buf;
    }
    value /= 1024.0;
  }
  std::snprintf(buf, sizeof(buf), "%.1fYiB", value);
  return buf;
}

int64_t OpCount(const std::map<std::string, int64_t>& op_nums,
                const std::string& key) {
  auto it = op_nums.find(key);
  return it != op_nums.end() ? it->second : 0;
}

// --- ONNX glue for the metric core -------------------------------------------
// The heavy lifting (MAC counters, memory liveness) lives in model_metrics.*,
// which is free of ONNX types. These helpers reduce a shape-inferred
// ``GraphProto`` to the ``GraphView`` that core operates on.

// A tensor's shape as a list of SymExpr dims, or nullopt when it is not fully
// usable: no tensor type, no shape (unknown rank), any dimension that is
// neither a fixed value nor a dim_param, or rank 0 (a scalar -- matching the
// Python ``_known`` rule that a metric-bearing shape has rank >= 1).
std::optional<Shape> ExtractShape(const onnx::TypeProto& type) {
  if (!type.has_tensor_type()) return std::nullopt;
  const auto& tensor_type = type.tensor_type();
  if (!tensor_type.has_shape()) return std::nullopt;
  Shape shape;
  for (const auto& dim : tensor_type.shape().dim()) {
    if (dim.has_dim_value()) {
      shape.push_back(SymExpr(dim.dim_value()));
    } else if (!dim.dim_param().empty()) {
      shape.push_back(SymExpr::Symbol(dim.dim_param()));
    } else {
      return std::nullopt;  // dimension is entirely unknown
    }
  }
  if (shape.empty()) return std::nullopt;
  return shape;
}

// Record a value_info's shape (if fully known) and element size (if fixed) into
// the maps the core reads.
void AddValueInfo(const onnx::ValueInfoProto& value_info, ShapeMap& shapes,
                  DTypeMap& dtypes) {
  if (auto shape = ExtractShape(value_info.type())) {
    shapes[value_info.name()] = std::move(*shape);
  }
  if (value_info.type().has_tensor_type()) {
    if (auto esize = onnxsim::ElemSize(
            static_cast<int>(value_info.type().tensor_type().elem_type()))) {
      dtypes[value_info.name()] = *esize;
    }
  }
}

// Reduce a graph (and its control-flow subgraphs) to a GraphView. Shapes and
// dtypes are inherited by value so a subgraph sees the tensors captured from
// its enclosing scope, exactly as the Python metrics thread them down.
GraphView BuildGraphView(const onnx::GraphProto& graph, ShapeMap shapes,
                         DTypeMap dtypes) {
  GraphView view;
  view.shapes = std::move(shapes);
  view.dtypes = std::move(dtypes);

  for (const auto& value_info : graph.input()) {
    AddValueInfo(value_info, view.shapes, view.dtypes);
  }
  for (const auto& value_info : graph.output()) {
    AddValueInfo(value_info, view.shapes, view.dtypes);
  }
  for (const auto& value_info : graph.value_info()) {
    AddValueInfo(value_info, view.shapes, view.dtypes);
  }
  for (const auto& initializer : graph.initializer()) {
    if (initializer.dims_size() > 0) {  // rank-0 scalars carry no metric shape
      Shape shape;
      for (int i = 0; i < initializer.dims_size(); ++i) {
        shape.push_back(SymExpr(initializer.dims(i)));
      }
      view.shapes[initializer.name()] = std::move(shape);
    }
    if (auto esize =
            onnxsim::ElemSize(static_cast<int>(initializer.data_type()))) {
      view.dtypes[initializer.name()] = *esize;
    }
  }

  for (const auto& value_info : graph.input()) {
    view.inputs.push_back(value_info.name());
  }
  for (const auto& value_info : graph.output()) {
    view.outputs.push_back(value_info.name());
  }
  for (const auto& initializer : graph.initializer()) {
    view.initializers.push_back(initializer.name());
  }

  for (const auto& node : graph.node()) {
    NodeView node_view;
    node_view.op_type = node.op_type();
    for (const auto& name : node.input()) node_view.inputs.push_back(name);
    for (const auto& name : node.output()) node_view.outputs.push_back(name);
    for (const auto& attr : node.attribute()) {
      if (attr.type() == onnx::AttributeProto::INT) {
        node_view.attr_ints[attr.name()] = attr.i();
      }
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        node_view.subgraphs.push_back(
            BuildGraphView(attr.g(), view.shapes, view.dtypes));
      }
      for (const auto& subgraph : attr.graphs()) {
        node_view.subgraphs.push_back(
            BuildGraphView(subgraph, view.shapes, view.dtypes));
      }
    }
    view.nodes.push_back(std::move(node_view));
  }
  return view;
}

// The string form of a metric for metadata_props, mirroring Python's
// ``_metric_str``: a concrete value as its plain integer, a symbolic one as its
// factored formula (e.g. "512*batch").
std::string MetricStr(const SymExpr& value) {
  return value.is_symbolic() ? value.str_factored()
                             : std::to_string(value.to_int());
}

// The string form of compute density (FLOP/Byte) for metadata_props. 0 when no
// traffic is known; a decimal when concrete; the simplified formula when the
// dynamic dims do not cancel.
std::string DensityStr(const ModelInfo& info) {
  if (info.mem_access.representative() == 0) return "0";
  const SymRatio ratio(info.Flops(), info.mem_access);
  if (ratio.is_symbolic()) return ratio.str();
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", ratio.representative());
  return buf;
}

// Set ``key`` -> ``value`` in a proto's metadata_props, overwriting any
// existing entry with that key. Works for any message that has a repeated
// StringStringEntryProto metadata_props (ModelProto and NodeProto here).
template <typename Proto>
void SetMetadata(Proto& proto, const std::string& key,
                 const std::string& value) {
  for (auto& entry : *proto.mutable_metadata_props()) {
    if (entry.key() == key) {
      entry.set_value(value);
      return;
    }
  }
  auto* entry = proto.add_metadata_props();
  entry->set_key(key);
  entry->set_value(value);
}

}  // namespace

void AnnotateModelInfo(onnx::ModelProto& model) {
  // Model-level totals (and model_size) come from the model as given, before we
  // add any metadata_props, so the reported size matches the Python annotator.
  const ModelInfo info = GetModelInfo(model);

  SetMetadata(model, "onnxsim.macs", MetricStr(info.macs));
  SetMetadata(model, "onnxsim.flops", MetricStr(info.Flops()));
  SetMetadata(model, "onnxsim.mem_access", MetricStr(info.mem_access));
  SetMetadata(model, "onnxsim.memory_footprint",
              MetricStr(info.memory_footprint));
  SetMetadata(model, "onnxsim.compute_density", DensityStr(info));
  SetMetadata(model, "onnxsim.model_size", std::to_string(info.model_size));

  // Per-node metrics need tensor shapes; infer them on a copy (best-effort) and
  // reduce to a GraphView. Shape inference only adds value_info -- it never
  // reorders or changes the node list -- so view.nodes[i] lines up with the
  // model's node i, and we write the per-node metadata onto the original nodes.
  onnx::ModelProto inferred = model;
  try {
    onnx::shape_inference::InferShapes(inferred);
  } catch (...) {
  }
  const GraphView view =
      BuildGraphView(inferred.graph(), ShapeMap{}, DTypeMap{});
  auto* graph = model.mutable_graph();
  if (view.nodes.size() == static_cast<std::size_t>(graph->node_size())) {
    for (int i = 0; i < graph->node_size(); ++i) {
      const SymExpr macs = onnxsim::NodeMacs(view.nodes[i], view.shapes);
      const SymExpr mem =
          onnxsim::NodeMemAccess(view.nodes[i], view.shapes, view.dtypes);
      onnx::NodeProto* node = graph->mutable_node(i);
      SetMetadata(*node, "onnxsim.macs", MetricStr(macs));
      SetMetadata(*node, "onnxsim.flops", MetricStr(macs * SymExpr(2)));
      SetMetadata(*node, "onnxsim.mem_access", MetricStr(mem));
    }
  }
}

ModelInfo GetModelInfo(const onnx::ModelProto& model,
                       bool run_shape_inference) {
  ModelInfo info;
  CountGraphOps(model.graph(), info.op_nums);
  // ByteSizeLong() (not the 32-bit ByteSize()) so models above 2GB do not
  // overflow; external tensor data is then added from metadata. Op counts and
  // size come from the model as given -- not the shape-inferred copy below,
  // whose extra value_info would inflate the serialized size.
  info.model_size = static_cast<int64_t>(model.graph().ByteSizeLong()) +
                    ExternalDataSize(model.graph());

  // The compute/memory metrics need tensor shapes. By default run shape
  // inference on a copy (it mutates in place). Best-effort: if it throws (e.g.
  // models > 2GB), fall back to whatever value_info the model already carries
  // -- unshaped nodes then contribute 0, mirroring the Python warning path.
  // When the caller already inferred shapes (e.g. with data propagation), skip
  // the pass and read the model's existing value_info directly.
  onnx::ModelProto inferred;
  const onnx::GraphProto* graph = &model.graph();
  if (run_shape_inference) {
    inferred = model;
    try {
      onnx::shape_inference::InferShapes(inferred);
    } catch (...) {
    }
    graph = &inferred.graph();
  }
  const GraphView view = BuildGraphView(*graph, ShapeMap{}, DTypeMap{});
  const Metrics metrics = onnxsim::ComputeMetrics(view);
  info.macs = metrics.macs;
  info.mem_access = metrics.mem_access;
  info.memory_footprint = onnxsim::PeakMemoryFootprint(view);
  return info;
}

std::string FormatSimplifyingInfo(const onnx::ModelProto& model_ori,
                                  const onnx::ModelProto& model_opt) {
  const ModelInfo ori = GetModelInfo(model_ori);
  const ModelInfo opt = GetModelInfo(model_opt);

  // Each row is {name, original, simplified}; the simplified cell gets a
  // trailing " *" when the metric improved (a smaller count / size).
  std::vector<std::array<std::string, 3>> rows;
  rows.push_back({"", "Original Model", "Simplified Model"});

  std::set<std::string> keys;
  for (const auto& entry : ori.op_nums) keys.insert(entry.first);
  for (const auto& entry : opt.op_nums) keys.insert(entry.first);
  for (const auto& key : keys) {
    const int64_t o = OpCount(ori.op_nums, key);
    const int64_t s = OpCount(opt.op_nums, key);
    std::string simplified = std::to_string(s);
    if (s < o) simplified += " *";
    rows.push_back({key, std::to_string(o), simplified});
  }

  std::string size_cell = HumanReadableSize(opt.model_size);
  if (opt.model_size < ori.model_size) size_cell += " *";
  rows.push_back({"Model Size", HumanReadableSize(ori.model_size), size_cell});

  // Symbolic metric rows: a smaller representative magnitude (every dynamic dim
  // -> 1) counts as the improvement, since "<" on a genuine formula is not
  // decidable.
  auto add_metric = [&](const std::string& name, const SymExpr& o,
                        const SymExpr& s, std::string (*fmt)(const SymExpr&)) {
    std::string cell = fmt(s);
    if (s.representative() < o.representative()) cell += " *";
    rows.push_back({name, fmt(o), cell});
  };
  add_metric("MACs", ori.macs, opt.macs, onnxsim::HumanReadableNum);
  add_metric("FLOPs", ori.Flops(), opt.Flops(), onnxsim::HumanReadableNum);
  add_metric("Memory Access", ori.mem_access, opt.mem_access,
             onnxsim::HumanReadableSize);
  add_metric("Memory Footprint", ori.memory_footprint, opt.memory_footprint,
             onnxsim::HumanReadableSize);

  // Compute density (FLOP/Byte) is a ratio and not strictly better-or-worse, so
  // it is shown without a flag. Zero traffic -> 0 to avoid dividing by zero.
  auto density_cell = [](const ModelInfo& mi) -> std::string {
    if (mi.mem_access.representative() == 0) {
      return onnxsim::HumanReadableDensity(SymRatio(SymExpr(0), SymExpr(1)));
    }
    return onnxsim::HumanReadableDensity(SymRatio(mi.Flops(), mi.mem_access));
  };
  rows.push_back({"Compute Density", density_cell(ori), density_cell(opt)});

  std::array<size_t, 3> width = {0, 0, 0};
  for (const auto& row : rows) {
    for (size_t c = 0; c < 3; ++c) {
      width[c] = std::max(width[c], row[c].size());
    }
  }

  auto border = [&]() {
    std::string line = "+";
    for (size_t c = 0; c < 3; ++c) {
      line.append(width[c] + 2, '-');
      line.push_back('+');
    }
    line.push_back('\n');
    return line;
  };
  auto render = [&](const std::array<std::string, 3>& row) {
    std::string line = "|";
    for (size_t c = 0; c < 3; ++c) {
      line.push_back(' ');
      line.append(row[c]);
      line.append(width[c] - row[c].size(), ' ');
      line.append(" |");
    }
    line.push_back('\n');
    return line;
  };

  std::string out;
  out += border();
  out += render(rows.front());
  out += border();
  for (size_t i = 1; i < rows.size(); ++i) {
    out += render(rows[i]);
  }
  out += border();
  return out;
}
