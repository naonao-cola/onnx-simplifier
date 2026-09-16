// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See kv_cache_quantization_entry.h for the full rationale (including why
// this follows llm_int8_entry.h's own single-model, protobuf-level,
// calibration-driven shape) and onnxsim/kv_cache_quantization.py for the
// technique this ports.

#include "kv_cache_quantization_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

// --- Tensor <-> flat buffers, protobuf level --------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers (FLOAT32
// only -- this pass, like its own Python reference, never widens to
// FLOAT16/BFLOAT16).

std::vector<float> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(out.data(), t.raw_data().data(), out.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                        out.size() * sizeof(float),
                                        sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return out;
}

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

void AddIntAttribute(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

// --- Candidate matching, protobuf level -------------------------------
//
// Direct transcription of kv_cache_quantization.py's own
// _find_kv_cache_candidates (see kv_cache_quantization_entry.h's own
// top-of-file comment for why this is NOT shared with
// passes/intactkv.h's own IR-level reimplementation of the same
// structural pattern): `Concat(past, new, axis=seq)` where `past` is a
// float32 graph input consumed by nothing else, and the Concat's own
// output is directly a graph output.
struct Candidate {
  onnx::NodeProto* concat_node;
  std::string past_name;
  std::string present_name;
  std::string new_name;
  bool new_is_first_input;
  int64_t seq_axis;
  int64_t channel_axis;
};

std::vector<Candidate> FindKvCacheCandidates(onnx::GraphProto* graph) {
  std::unordered_set<std::string> output_names;
  for (const auto& o : graph->output()) {
    output_names.insert(o.name());
  }

  std::unordered_map<std::string, int64_t> float_input_rank;
  for (const auto& inp : graph->input()) {
    if (inp.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT) {
      continue;
    }
    float_input_rank[inp.name()] =
        static_cast<int64_t>(inp.type().tensor_type().shape().dim_size());
  }

  std::unordered_map<std::string, int64_t> consumer_count;
  for (const auto& node : graph->node()) {
    for (const auto& in : node.input()) {
      consumer_count[in] += 1;
    }
  }

  std::vector<Candidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (node->op_type() != "Concat" || node->input_size() != 2) {
      continue;
    }
    if (node->output_size() != 1 || output_names.count(node->output(0)) == 0) {
      continue;
    }
    const std::string& a = node->input(0);
    const std::string& b = node->input(1);
    std::string past_name, new_name;
    bool new_is_first = false;
    if (float_input_rank.count(a) != 0 && consumer_count[a] == 1) {
      past_name = a;
      new_name = b;
      new_is_first = false;
    } else if (float_input_rank.count(b) != 0 && consumer_count[b] == 1) {
      past_name = b;
      new_name = a;
      new_is_first = true;
    } else {
      continue;
    }

    const onnx::AttributeProto* axis_attr = nullptr;
    for (const auto& attr : node->attribute()) {
      if (attr.name() == "axis") {
        axis_attr = &attr;
        break;
      }
    }
    if (axis_attr == nullptr) {
      continue;
    }

    const int64_t rank = float_input_rank[past_name];
    int64_t seq_axis = axis_attr->i();
    if (seq_axis < 0) {
      seq_axis += rank;
    }
    const int64_t channel_axis = rank - 1;
    if (seq_axis == channel_axis) {
      continue;  // No distinct channel axis left to quantize per-channel
                 // on.
    }

    candidates.push_back({node, past_name, node->output(0), new_name,
                          new_is_first, seq_axis, channel_axis});
  }
  return candidates;
}

bool IsValueStyle(const std::string& present_name,
                  const std::vector<std::string>& value_output_names) {
  if (!value_output_names.empty()) {
    return std::find(value_output_names.begin(), value_output_names.end(),
                     present_name) != value_output_names.end();
  }
  return present_name.find(".value") != std::string::npos;
}

// --- Calibration: per-channel (last-axis) abs-max ----------------------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// llm_int8_entry.cpp's own ComputeChannelAbsmax, generalized to an
// arbitrary rank (`new_name`'s own activation collapses every leading
// dimension into rows, keeping only the last axis as the channel to
// reduce over -- mirrors kv_cache_quantization.py's own
// `flat = arr.reshape(-1, arr.shape[-1])` exactly).
struct AbsMaxAcc {
  std::vector<double> values;  // [channels]; empty until first observed.
  int64_t channels = -1;
  bool ok = false;
};

void AccumulatePerChannelAbsMax(
    std::unordered_map<std::string, AbsMaxAcc>& acc,
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  if (probe_names.empty()) {
    return;
  }

  onnx::ModelProto probe_model = model;
  std::unordered_set<std::string> existing_outputs;
  for (const auto& o : probe_model.graph().output()) {
    existing_outputs.insert(o.name());
  }
  for (const auto& name : probe_names) {
    if (existing_outputs.insert(name).second) {
      probe_model.mutable_graph()->add_output()->set_name(name);
    }
  }

  std::unordered_map<std::string, size_t> output_index;
  for (int i = 0; i < probe_model.graph().output_size(); ++i) {
    output_index.emplace(probe_model.graph().output(i).name(),
                         static_cast<size_t>(i));
  }
  const auto& graph_inputs = probe_model.graph().input();

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> input_dls;
    std::vector<const DLManagedTensor*> input_ptrs;
    input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
    input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
    for (const auto& gi : graph_inputs) {
      auto it = batch.find(gi.name());
      if (it == batch.end()) {
        throw std::invalid_argument(
            "ApplyKvCacheQuantization: calibration batch is missing "
            "required graph input '" +
            gi.name() + "'");
      }
      input_dls.emplace_back(
          onnxsim::dlpack::FromTensorProtoBorrowing(it->second));
      input_ptrs.push_back(input_dls.back().get());
    }

    std::vector<DLManagedTensorPtr> outputs =
        executor.Run(probe_model, input_ptrs);

    for (const auto& name : probe_names) {
      auto oit = output_index.find(name);
      if (oit == output_index.end() || oit->second >= outputs.size()) {
        continue;  // Defensive -- every probe name was added as an output
                   // above.
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT) {
        continue;
      }
      if (tp.dims_size() == 0) {
        continue;  // Mirrors `if arr.ndim == 0: continue`.
      }
      const int64_t channels = tp.dims(tp.dims_size() - 1);
      if (channels <= 0) {
        continue;
      }
      int64_t rows = 1;
      for (int d = 0; d < tp.dims_size() - 1; ++d) {
        rows *= tp.dims(d);
      }

      AbsMaxAcc& a = acc[name];
      if (!a.ok) {
        a.values.assign(static_cast<size_t>(channels), 0.0);
        a.channels = channels;
        a.ok = true;
      } else if (a.channels != channels) {
        continue;  // Channel width changed mid-calibration; keep the
                   // first width (numpy's own np.maximum broadcast would
                   // fail instead).
      }

      const std::vector<float> data = ReadFloatTensor(tp);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < channels; ++c) {
          const double v = std::fabs(
              static_cast<double>(data[static_cast<size_t>(r * channels + c)]));
          if (v > a.values[static_cast<size_t>(c)]) {
            a.values[static_cast<size_t>(c)] = v;
          }
        }
      }
    }
  }
}

// Appends a node at the very end, then walks it back down to exactly
// `target_index` via adjacent pointer-slot swaps -- matches
// low_rank_compensation_entry.cpp's/billm_entry.cpp's own identical
// pattern: RepeatedPtrField::SwapElements swaps POINTER SLOTS, never
// moving or freeing the pointed-to messages, so a NodeProto* obtained
// earlier (e.g. a Candidate's own `concat_node`) stays valid and
// locatable by pointer identity across any number of these calls --
// unlike a content-swap (Message::Swap), which this file deliberately
// does not use.
onnx::NodeProto* AppendAt(onnx::GraphProto* graph, const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::string& output, const std::string& name,
                          int target_index) {
  auto* nodes = graph->mutable_node();
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type(op_type);
  for (const auto& in : inputs) {
    node->add_input(in);
  }
  node->add_output(output);
  node->set_name(name);
  for (int i = nodes->size() - 1; i > target_index; --i) {
    nodes->SwapElements(i, i - 1);
  }
  return node;
}

// Re-finds `concat_node`'s own CURRENT index by pointer identity -- an
// earlier candidate's own insertions can have shifted it since
// `FindKvCacheCandidates` ran, so this cannot be cached across
// candidates.
int FindNodeIndex(onnx::GraphProto* graph, const onnx::NodeProto* target) {
  auto* nodes = graph->mutable_node();
  for (int i = 0; i < nodes->size(); ++i) {
    if (nodes->Mutable(i) == target) {
      return i;
    }
  }
  return -1;
}

// Rewires every NODE (other than `concat_node` itself) whose own input
// equals `present_name` to `dequant_name` instead -- mirrors
// kv_cache_quantization.py's own `_rewire_consumers` exactly. Run BEFORE
// any new node referencing `present_name` (the dequantize/Cast node
// below) is inserted, so there is nothing for this loop to incorrectly
// self-reference. The graph's own OUTPUT binding for `present_name`
// itself is untouched: it keeps resolving to the Concat node's own
// (dtype-mutated) output by name, unchanged -- only real downstream NODE
// consumers move to the newly reconstructed float value.
void RewireConsumers(onnx::GraphProto* graph,
                     const onnx::NodeProto* concat_node,
                     const std::string& present_name,
                     const std::string& dequant_name) {
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (node == concat_node) {
      continue;
    }
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == present_name) {
        node->set_input(j, dequant_name);
      }
    }
  }
}

// Static, calibrated, per-channel (Key-style) rewrite -- see
// kv_cache_quantization_entry.h's own top-of-file diagram. Direct
// transcription of kv_cache_quantization.py's own _apply_channel_style.
void ApplyChannelStyle(
    onnx::GraphProto* graph, const Candidate& c,
    const std::vector<double>& channel_absmax,
    std::unordered_set<std::string>& taken_names,
    std::unordered_map<std::string, onnx::ValueInfoProto*>& input_by_name,
    std::unordered_map<std::string, onnx::ValueInfoProto*>& output_by_name) {
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

  const int64_t num_channels = static_cast<int64_t>(channel_absmax.size());
  std::vector<float> scale(static_cast<size_t>(num_channels));
  for (int64_t i = 0; i < num_channels; ++i) {
    scale[static_cast<size_t>(i)] = static_cast<float>(
        std::max(channel_absmax[static_cast<size_t>(i)], 1e-12) / 127.0);
  }
  std::vector<int8_t> zp(static_cast<size_t>(num_channels), 0);

  const std::string scale_name = unique_name(c.present_name + "_kv_scale");
  const std::string zp_name = unique_name(c.present_name + "_kv_zero_point");
  SetRawInitializer(graph->add_initializer(), scale_name,
                    onnx::TensorProto::FLOAT, {num_channels}, scale.data(),
                    scale.size() * sizeof(float), sizeof(float));
  SetRawInitializer(graph->add_initializer(), zp_name, onnx::TensorProto::INT8,
                    {num_channels}, zp.data(), zp.size() * sizeof(int8_t),
                    sizeof(int8_t));

  // past_key/past_key_values.*: FLOAT -> INT8 (same shape).
  input_by_name[c.past_name]
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto::INT8);

  const std::string new_q_name = unique_name(c.new_name + "_kv_q");
  const std::string quantize_node_name =
      unique_name(c.new_name + "_kv_quantize_node");

  // Rewire Concat's "new" input to the now-quantized tensor; the "past"
  // input already reads the (now INT8) graph input as-is, so Concat's own
  // output is INT8 -- exactly present_key's new dtype.
  if (c.new_is_first_input) {
    c.concat_node->set_input(0, new_q_name);
  } else {
    c.concat_node->set_input(1, new_q_name);
  }

  output_by_name[c.present_name]
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto::INT8);

  const std::string dequant_name = unique_name(c.present_name + "_kv_f");
  const std::string dequant_node_name =
      unique_name(c.present_name + "_kv_dequantize_node");

  RewireConsumers(graph, c.concat_node, c.present_name, dequant_name);

  const int insertion_point = FindNodeIndex(graph, c.concat_node);
  onnx::NodeProto* qnode =
      AppendAt(graph, "QuantizeLinear", {c.new_name, scale_name, zp_name},
               new_q_name, quantize_node_name, insertion_point);
  AddIntAttribute(qnode, "axis", c.channel_axis);
  // concat_node is now shifted to insertion_point + 1; the dequant node
  // lands directly after it.
  onnx::NodeProto* dqnode =
      AppendAt(graph, "DequantizeLinear", {c.present_name, scale_name, zp_name},
               dequant_name, dequant_node_name, insertion_point + 2);
  AddIntAttribute(dqnode, "axis", c.channel_axis);
}

// Copies `src`'s own dims into `dst`, forcing the `channel_axis`-th one to
// a concrete size of 1 -- mirrors kv_cache_quantization.py's own
// past_scale_input shape-construction loop exactly (a dim with neither
// dim_value nor dim_param stays entirely unset, the same as the source).
void CopyShapeChannelAxisForcedToOne(const onnx::TensorShapeProto& src,
                                     int64_t channel_axis,
                                     onnx::TensorShapeProto* dst) {
  for (int i = 0; i < src.dim_size(); ++i) {
    onnx::TensorShapeProto_Dimension* d = dst->add_dim();
    if (i == channel_axis) {
      d->set_dim_value(1);
    } else if (src.dim(i).has_dim_value()) {
      d->set_dim_value(src.dim(i).dim_value());
    } else if (src.dim(i).has_dim_param()) {
      d->set_dim_param(src.dim(i).dim_param());
    }
  }
}

// Data-free, per-token (Value-style) rewrite -- see
// kv_cache_quantization_entry.h's own top-of-file diagram. Direct
// transcription of kv_cache_quantization.py's own _apply_value_style.
//
// Unlike the Python reference (which leaves most of these new nodes'
// own `.name` field empty, naming only the scale-Concat node), every new
// node here gets a real, unique name -- a harmless embellishment with no
// effect on graph semantics (ONNX node names are optional debug metadata,
// never used for wiring), the same convention this session's own
// billm_entry.cpp already establishes for an identical situation.
void ApplyValueStyle(
    onnx::GraphProto* graph, const Candidate& c,
    std::unordered_set<std::string>& taken_names,
    std::unordered_map<std::string, onnx::ValueInfoProto*>& input_by_name,
    std::unordered_map<std::string, onnx::ValueInfoProto*>& output_by_name) {
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

  const std::string prefix = c.present_name + "_kv";
  onnx::ValueInfoProto* past_input = input_by_name[c.past_name];
  onnx::ValueInfoProto* present_output = output_by_name[c.present_name];
  const int64_t past_rank =
      static_cast<int64_t>(past_input->type().tensor_type().shape().dim_size());

  // New past_*_scale graph input: same rank/leading dims as past_* (read
  // before past_input's own dtype is mutated below), channel axis forced
  // to size 1.
  const std::string past_scale_name = unique_name(c.past_name + "_scale");
  onnx::ValueInfoProto* past_scale_input = graph->add_input();
  past_scale_input->set_name(past_scale_name);
  onnx::TypeProto::Tensor* pst =
      past_scale_input->mutable_type()->mutable_tensor_type();
  pst->set_elem_type(onnx::TensorProto::FLOAT);
  CopyShapeChannelAxisForcedToOne(past_input->type().tensor_type().shape(),
                                  c.channel_axis, pst->mutable_shape());

  past_input->mutable_type()->mutable_tensor_type()->set_elem_type(
      onnx::TensorProto::INT8);

  auto add_scalar_f32 = [&](const std::string& base, float v) {
    const std::string name = unique_name(base);
    SetRawInitializer(graph->add_initializer(), name, onnx::TensorProto::FLOAT,
                      {}, &v, sizeof(float), sizeof(float));
    return name;
  };
  const std::string eps_name = add_scalar_f32(prefix + "_eps", 1e-12f);
  const std::string div127_name = add_scalar_f32(prefix + "_127", 127.0f);
  const std::string clip_min_name =
      add_scalar_f32(prefix + "_clip_min", -128.0f);
  const std::string clip_max_name =
      add_scalar_f32(prefix + "_clip_max", 127.0f);

  const std::string axes_name = unique_name(prefix + "_reduce_axes");
  const int64_t axes_val = c.channel_axis;
  SetRawInitializer(graph->add_initializer(), axes_name,
                    onnx::TensorProto::INT64, {1}, &axes_val, sizeof(int64_t),
                    sizeof(int64_t));

  const std::string abs_name = unique_name(prefix + "_abs");
  const std::string max_name = unique_name(prefix + "_max");
  const std::string safe_max_name = unique_name(prefix + "_safe_max");
  const std::string new_scale_name = unique_name(prefix + "_new_scale");
  const std::string scaled_name = unique_name(prefix + "_scaled");
  const std::string rounded_name = unique_name(prefix + "_rounded");
  const std::string clipped_name = unique_name(prefix + "_clipped");
  const std::string new_q_name = unique_name(c.new_name + "_kv_q");

  if (c.new_is_first_input) {
    c.concat_node->set_input(0, new_q_name);
  } else {
    c.concat_node->set_input(1, new_q_name);
  }
  present_output->mutable_type()->mutable_tensor_type()->set_elem_type(
      onnx::TensorProto::INT8);

  // present_*_scale: NEW graph output, grows in lockstep with present_*
  // itself (same seq_axis Concat, same two operands' relative order).
  const std::string present_scale_name = unique_name(c.present_name + "_scale");
  onnx::ValueInfoProto* present_scale_output = graph->add_output();
  present_scale_output->set_name(present_scale_name);
  onnx::TypeProto::Tensor* pso =
      present_scale_output->mutable_type()->mutable_tensor_type();
  pso->set_elem_type(onnx::TensorProto::FLOAT);
  {
    onnx::TensorShapeProto* shp = pso->mutable_shape();
    for (int64_t i = 0; i < past_rank; ++i) {
      shp->add_dim();
    }
    shp->mutable_dim(static_cast<int>(c.channel_axis))->set_dim_value(1);
  }

  const std::string present_f32_name = unique_name(prefix + "_present_f32");
  const std::string dequant_name = unique_name(c.present_name + "_kv_f");

  RewireConsumers(graph, c.concat_node, c.present_name, dequant_name);

  const int insertion_point = FindNodeIndex(graph, c.concat_node);

  // pre_nodes (8, inserted before the Concat, in the reference's own
  // order): Abs, ReduceMax, Clip, Div, Div, Round, Clip, Cast.
  AppendAt(graph, "Abs", {c.new_name}, abs_name,
           unique_name(prefix + "_abs_node"), insertion_point + 0);
  onnx::NodeProto* rmax =
      AppendAt(graph, "ReduceMax", {abs_name, axes_name}, max_name,
               unique_name(prefix + "_reducemax_node"), insertion_point + 1);
  AddIntAttribute(rmax, "keepdims", 1);
  AppendAt(graph, "Clip", {max_name, eps_name}, safe_max_name,
           unique_name(prefix + "_safe_max_node"), insertion_point + 2);
  AppendAt(graph, "Div", {safe_max_name, div127_name}, new_scale_name,
           unique_name(prefix + "_new_scale_node"), insertion_point + 3);
  AppendAt(graph, "Div", {c.new_name, new_scale_name}, scaled_name,
           unique_name(prefix + "_scaled_node"), insertion_point + 4);
  AppendAt(graph, "Round", {scaled_name}, rounded_name,
           unique_name(prefix + "_rounded_node"), insertion_point + 5);
  AppendAt(graph, "Clip", {rounded_name, clip_min_name, clip_max_name},
           clipped_name, unique_name(prefix + "_clipped_node"),
           insertion_point + 6);
  onnx::NodeProto* castq =
      AppendAt(graph, "Cast", {clipped_name}, new_q_name,
               unique_name(prefix + "_cast_q_node"), insertion_point + 7);
  AddIntAttribute(castq, "to", onnx::TensorProto::INT8);

  // concat_node is now shifted to insertion_point + 8. post_nodes (3,
  // inserted directly after it): Concat(scale), Cast(present->float),
  // Mul.
  onnx::NodeProto* cscale = AppendAt(
      graph, "Concat", {past_scale_name, new_scale_name}, present_scale_name,
      unique_name(prefix + "_concat_scale_node"), insertion_point + 9);
  AddIntAttribute(cscale, "axis", c.seq_axis);
  onnx::NodeProto* castf =
      AppendAt(graph, "Cast", {c.present_name}, present_f32_name,
               unique_name(prefix + "_cast_f32_node"), insertion_point + 10);
  AddIntAttribute(castf, "to", onnx::TensorProto::FLOAT);
  AppendAt(graph, "Mul", {present_f32_name, present_scale_name}, dequant_name,
           unique_name(prefix + "_dequant_mul_node"), insertion_point + 11);
}

}  // namespace

onnx::ModelProto ApplyKvCacheQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    const std::vector<std::string>& value_output_names) {
  onnx::ModelProto out = model;

  // QuantizeLinear/DequantizeLinear's own per-channel `axis`, and
  // ReduceMax's `axes`-as-input, both need opset >= 13 -- mirrors
  // quantize_kv_cache's own `_has_min_opset(model, 13)` gate exactly.
  bool opset_ge_13 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 13) {
      opset_ge_13 = true;
      break;
    }
  }
  if (!opset_ge_13) {
    return out;
  }

  onnx::GraphProto* graph = out.mutable_graph();
  std::vector<Candidate> candidates = FindKvCacheCandidates(graph);
  if (candidates.empty()) {
    return out;
  }

  // Value-style needs ReduceMax's axes-as-input form, which only arrived
  // at opset 18 -- a stream matched as Value-style below opset 18 is left
  // completely untouched, mirroring the reference exactly.
  bool opset_ge_18 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 18) {
      opset_ge_18 = true;
      break;
    }
  }

  std::vector<Candidate> channel_candidates;
  std::vector<Candidate> value_candidates;
  for (const auto& c : candidates) {
    if (IsValueStyle(c.present_name, value_output_names)) {
      if (opset_ge_18) {
        value_candidates.push_back(c);
      }
      // else: leave this stream untouched -- see comment above.
    } else {
      channel_candidates.push_back(c);
    }
  }

  std::unordered_map<std::string, AbsMaxAcc> absmax_acc;
  if (!channel_candidates.empty()) {
    std::unordered_set<std::string> probe_names;
    for (const auto& c : channel_candidates) {
      probe_names.insert(c.new_name);
    }
    AccumulatePerChannelAbsMax(absmax_acc, executor, model, probe_names,
                               calibration_data);
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention every calibration-driven
  // *_entry.cpp in this repo already uses.
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

  std::unordered_map<std::string, onnx::ValueInfoProto*> input_by_name;
  for (int i = 0; i < graph->input_size(); ++i) {
    input_by_name[graph->input(i).name()] = graph->mutable_input(i);
  }
  std::unordered_map<std::string, onnx::ValueInfoProto*> output_by_name;
  for (int i = 0; i < graph->output_size(); ++i) {
    output_by_name[graph->output(i).name()] = graph->mutable_output(i);
  }

  for (const auto& c : channel_candidates) {
    auto it = absmax_acc.find(c.new_name);
    if (it == absmax_acc.end() || !it->second.ok) {
      continue;  // This stream's activation never appeared in any batch.
    }
    ApplyChannelStyle(graph, c, it->second.values, taken_names, input_by_name,
                      output_by_name);
  }
  for (const auto& c : value_candidates) {
    ApplyValueStyle(graph, c, taken_names, input_by_name, output_by_name);
  }

  return out;
}
