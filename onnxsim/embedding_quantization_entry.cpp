// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See embedding_quantization_entry.h for the full rationale and
// onnxsim/embedding_quantization.py for the technique this ports.

#include "embedding_quantization_entry.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"

namespace {

// MSB-first bit weights -- matches onnxsim.embedding_quantization's own
// _BIT_WEIGHTS exactly (numpy.packbits(bitorder="big")'s own convention:
// the FIRST of 8 consecutive elements is the most significant bit).
constexpr int64_t kBitWeights[8] = {128, 64, 32, 16, 8, 4, 2, 1};

// Writes `data`/`bytes` into `t` as a fresh raw-data tensor of `data_type`
// and `dims` -- the same generic raw-initializer helper every calibration-
// driven *_entry.cpp in this repo already establishes (see e.g.
// spqr_entry.cpp's own identical SetRawInitializer).
void SetRawInitializer(onnx::TensorProto* t, const std::string& name,
                       int32_t data_type, const std::vector<int64_t>& dims,
                       const void* data, size_t bytes, size_t elem_size) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(static_cast<onnx::TensorProto::DataType>(data_type));
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(bytes, '\0');
  std::memcpy(raw.data(), data, bytes);
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      bytes, elem_size);
  }
  t->set_raw_data(std::move(raw));
}

// Returns the index into `graph.output()` of the FLOAT tensor to binarize,
// or -1 if it can't be resolved unambiguously -- direct transcription of
// quantize_embedding_binary.py's own _resolve_output: a non-empty
// `output_name` must name an existing FLOAT output; an empty one (this
// port's own stand-in for Python's `output_name=None`) requires the graph
// to have *exactly one* FLOAT output.
int ResolveOutput(const onnx::GraphProto& graph,
                  const std::string& output_name) {
  if (!output_name.empty()) {
    for (int i = 0; i < graph.output_size(); ++i) {
      if (graph.output(i).name() == output_name) {
        return graph.output(i).type().tensor_type().elem_type() ==
                       onnx::TensorProto::FLOAT
                   ? i
                   : -1;
      }
    }
    return -1;
  }

  int only_float = -1;
  int float_count = 0;
  for (int i = 0; i < graph.output_size(); ++i) {
    if (graph.output(i).type().tensor_type().elem_type() ==
        onnx::TensorProto::FLOAT) {
      ++float_count;
      only_float = i;
    }
  }
  return float_count == 1 ? only_float : -1;
}

// The resolved output's own last-dimension extent, or -1 when it isn't
// statically known (no dims at all, or the last one is a symbolic
// dim_param rather than a concrete dim_value) -- mirrors
// quantize_embedding_binary.py's own _static_last_dim exactly.
int64_t StaticLastDim(const onnx::ValueInfoProto& value_info) {
  const auto& dims = value_info.type().tensor_type().shape().dim();
  if (dims.empty() || !dims[dims.size() - 1].has_dim_value()) {
    return -1;
  }
  return dims[dims.size() - 1].dim_value();
}

}  // namespace

onnx::ModelProto ApplyEmbeddingQuantizationBinary(
    const onnx::ModelProto& model, const std::string& output_name) {
  bool opset_ge_13 = false;
  for (const auto& opset : model.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 13) {
      opset_ge_13 = true;
      break;
    }
  }
  if (!opset_ge_13) {
    return model;
  }

  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  const int idx = ResolveOutput(*graph, output_name);
  if (idx < 0) {
    return out;
  }
  onnx::ValueInfoProto* target = graph->mutable_output(idx);
  const int64_t embed_dim = StaticLastDim(*target);
  if (embed_dim < 0 || embed_dim % 8 != 0) {
    return out;
  }
  // Snapshot the leading (batch/sequence) dims before `target` is mutated
  // in place below -- symbolic dims (dim_param) and concrete ones
  // (dim_value) are both copied through unchanged, exactly like Python's
  // own `list(target.type.tensor_type.shape.dim[:-1])`.
  const auto& all_dims = target->type().tensor_type().shape().dim();
  std::vector<onnx::TensorShapeProto_Dimension> leading_dims(
      all_dims.begin(), all_dims.end() - 1);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention every *_entry.cpp in this
  // repo already uses.
  std::unordered_set<std::string> taken_names;
  for (const auto& t : graph->initializer()) {
    taken_names.insert(t.name());
  }
  for (const auto& vi : graph->input()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : graph->output()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : graph->value_info()) {
    taken_names.insert(vi.name());
  }
  for (const auto& n : graph->node()) {
    if (!n.name().empty()) {
      taken_names.insert(n.name());
    }
    for (const auto& s : n.input()) {
      taken_names.insert(s);
    }
    for (const auto& s : n.output()) {
      taken_names.insert(s);
    }
  }
  auto unique_name = [&](const std::string& base) {
    std::string name = base;
    int i = 0;
    while (taken_names.count(name) != 0) {
      ++i;
      name = base + "_" + std::to_string(i);
    }
    taken_names.insert(name);
    return name;
  };

  const std::string x = target->name();
  const std::string prefix = x + "_bin";

  const float zero_f = 0.0f;
  const std::string zero_name = unique_name(prefix + "_zero");
  SetRawInitializer(graph->add_initializer(), zero_name,
                    onnx::TensorProto::FLOAT, {}, &zero_f, sizeof(zero_f),
                    sizeof(float));

  const std::string weights_name = unique_name(prefix + "_weights");
  SetRawInitializer(graph->add_initializer(), weights_name,
                    onnx::TensorProto::INT64, {8}, kBitWeights,
                    sizeof(kBitWeights), sizeof(int64_t));

  const int64_t group_shape_vals[2] = {embed_dim / 8, 8};
  const std::string group_shape_name = unique_name(prefix + "_group_shape");
  SetRawInitializer(graph->add_initializer(), group_shape_name,
                    onnx::TensorProto::INT64, {2}, group_shape_vals,
                    sizeof(group_shape_vals), sizeof(int64_t));

  const int64_t neg_one = -1;
  const std::string last_axis_name = unique_name(prefix + "_last_axis");
  SetRawInitializer(graph->add_initializer(), last_axis_name,
                    onnx::TensorProto::INT64, {1}, &neg_one, sizeof(neg_one),
                    sizeof(int64_t));

  const int64_t zero_i64 = 0;
  const std::string slice_start_name = unique_name(prefix + "_slice_start");
  SetRawInitializer(graph->add_initializer(), slice_start_name,
                    onnx::TensorProto::INT64, {1}, &zero_i64, sizeof(zero_i64),
                    sizeof(int64_t));
  const std::string slice_end_name = unique_name(prefix + "_slice_end");
  SetRawInitializer(graph->add_initializer(), slice_end_name,
                    onnx::TensorProto::INT64, {1}, &neg_one, sizeof(neg_one),
                    sizeof(int64_t));

  // Appends every new node directly at the END of the graph -- mirrors
  // quantize_embedding_binary.py's own `graph.node.extend(new_nodes)`
  // exactly (unlike most other *_entry.cpp ports in this repo, this
  // rewrite never needs to insert BEFORE an existing node, since it
  // targets a graph OUTPUT rather than rewiring an existing node's own
  // input in place -- so no SwapElements/pointer-identity dance is needed
  // here at all).
  auto add_node = [&](const std::string& op_type,
                      const std::vector<std::string>& inputs,
                      const std::string& out_suffix) -> std::string {
    onnx::NodeProto* n = graph->add_node();
    n->set_op_type(op_type);
    for (const auto& in : inputs) {
      n->add_input(in);
    }
    const std::string out_name = unique_name(prefix + "_" + out_suffix);
    n->add_output(out_name);
    return out_name;
  };
  auto add_int_attr = [](onnx::NodeProto* n, const std::string& name,
                         int64_t value) {
    onnx::AttributeProto* attr = n->add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto::INT);
    attr->set_i(value);
  };

  const std::string greater_out =
      add_node("Greater", {x, zero_name}, "greater");
  const std::string bits_i64 = add_node("Cast", {greater_out}, "bits_i64");
  {
    onnx::NodeProto* n = graph->mutable_node(graph->node_size() - 1);
    add_int_attr(n, "to", onnx::TensorProto::INT64);
  }

  const std::string shape_full = add_node("Shape", {x}, "shape");
  const std::string shape_prefix = add_node(
      "Slice", {shape_full, slice_start_name, slice_end_name}, "shape_prefix");
  const std::string new_shape =
      add_node("Concat", {shape_prefix, group_shape_name}, "new_shape");
  {
    onnx::NodeProto* n = graph->mutable_node(graph->node_size() - 1);
    add_int_attr(n, "axis", 0);
  }

  const std::string reshaped =
      add_node("Reshape", {bits_i64, new_shape}, "reshaped");
  const std::string weighted =
      add_node("Mul", {reshaped, weights_name}, "weighted");
  const std::string packed_i64 =
      add_node("ReduceSum", {weighted, last_axis_name}, "packed_i64");
  {
    onnx::NodeProto* n = graph->mutable_node(graph->node_size() - 1);
    add_int_attr(n, "keepdims", 0);
  }
  const std::string packed_u8 = add_node("Cast", {packed_i64}, "packed_u8");
  {
    onnx::NodeProto* n = graph->mutable_node(graph->node_size() - 1);
    add_int_attr(n, "to", onnx::TensorProto::UINT8);
  }

  // Rebinds the resolved output declaration itself: its own producer
  // becomes `packed_u8` (by name), its own dtype UINT8, and its own shape
  // the snapshotted leading dims plus a final `embed_dim / 8` -- mirrors
  // `target.name = packed_u8; target.type.tensor_type.elem_type =
  // onnx.TensorProto.UINT8; ...` exactly.
  target->set_name(packed_u8);
  target->mutable_type()->mutable_tensor_type()->set_elem_type(
      onnx::TensorProto::UINT8);
  onnx::TensorShapeProto* shape =
      target->mutable_type()->mutable_tensor_type()->mutable_shape();
  shape->clear_dim();
  for (const auto& d : leading_dims) {
    *shape->add_dim() = d;
  }
  shape->add_dim()->set_dim_value(embed_dim / 8);

  return out;
}
