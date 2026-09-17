// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See ptq4vit_entry.h for the full rationale (including why this
// captures raw per-tensor VALUES rather than a reduction, unlike every
// other calibration-driven pass in this codebase) and onnxsim/ptq4vit.py
// for the technique this ports.

#include "ptq4vit_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

constexpr double kEps = 1e-12;

// --- Tensor <-> flat float buffer -------------------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helper.
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

void SetScalarF32Initializer(onnx::TensorProto* t, const std::string& name,
                             float value) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  std::string raw(sizeof(float), '\0');
  std::memcpy(raw.data(), &value, sizeof(float));
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      sizeof(float), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// Round-half-to-even (banker's rounding), matching numpy's own `round`.
// Transcribed from llm_int8_entry.cpp's own RoundHalfToEven.
double RoundHalfToEven(double v) {
  const double f = std::floor(v);
  const double d = v - f;
  if (d < 0.5) {
    return f;
  }
  if (d > 0.5) {
    return f + 1.0;
  }
  const double half = f / 2.0;
  return (half == std::floor(half)) ? f : f + 1.0;
}

// --- The search itself, transcribed verbatim from ptq4vit.py ---------------

double SingleUniformQuantizeDequantizeMse(const std::vector<double>& v,
                                          double lo, double hi,
                                          int64_t n_levels) {
  const double scale =
      std::max(hi - lo, kEps) / static_cast<double>(n_levels - 1);
  double sum_sq = 0.0;
  for (double x : v) {
    double q = RoundHalfToEven((x - lo) / scale);
    q = std::clamp(q, 0.0, static_cast<double>(n_levels - 1));
    const double dq = q * scale + lo;
    const double d = x - dq;
    sum_sq += d * d;
  }
  return sum_sq / static_cast<double>(v.size());
}

double TwinQuantizeDequantizeMse(const std::vector<double>& v, double lo,
                                 double split, double hi, int64_t n_levels) {
  const double scale_lo =
      std::max(split - lo, kEps) / static_cast<double>(n_levels - 1);
  const double scale_hi =
      std::max(hi - split, kEps) / static_cast<double>(n_levels - 1);
  double sum_sq = 0.0;
  for (double x : v) {
    double dq;
    if (x <= split) {
      double q = RoundHalfToEven((x - lo) / scale_lo);
      q = std::clamp(q, 0.0, static_cast<double>(n_levels - 1));
      dq = q * scale_lo + lo;
    } else {
      double q = RoundHalfToEven((x - split) / scale_hi);
      q = std::clamp(q, 0.0, static_cast<double>(n_levels - 1));
      dq = q * scale_hi + split;
    }
    const double d = x - dq;
    sum_sq += d * d;
  }
  return sum_sq / static_cast<double>(v.size());
}

// Transcribed from ptq4vit.py's own _search_twin_split exactly, including
// its own `num_candidates=97` default (not exposed by
// apply_ptq4vit_quantization's own signature, so not exposed here
// either -- see this file's own header comment).
std::pair<bool, double> SearchTwinSplit(const std::vector<double>& raw_values,
                                        double lo, double hi, int64_t n_levels,
                                        int64_t num_candidates = 97) {
  std::vector<double> v;
  v.reserve(raw_values.size());
  for (double x : raw_values) {
    if (std::isfinite(x)) {
      v.push_back(x);
    }
  }
  if (static_cast<int64_t>(v.size()) < 2 * n_levels || hi - lo <= kEps) {
    return {false, 0.0};
  }

  const double baseline_mse =
      SingleUniformQuantizeDequantizeMse(v, lo, hi, 2 * n_levels);

  bool found = false;
  double best_t = 0.0;
  double best_mse = baseline_mse;
  for (int64_t i = 1; i <= num_candidates; ++i) {
    const double t = lo + (hi - lo) * static_cast<double>(i) /
                              static_cast<double>(num_candidates + 1);
    const double mse = TwinQuantizeDequantizeMse(v, lo, t, hi, n_levels);
    if (mse < best_mse) {
      best_mse = mse;
      best_t = t;
      found = true;
    }
  }
  return {found, best_t};
}

// --- Candidate matching ------------------------------------------------------
//
// Transcribed from ptq4vit.py's own _find_softmax_targets/
// _find_gelu_targets exactly.

std::vector<int> FindSoftmaxTargets(const onnx::GraphProto& graph) {
  std::vector<int> out;
  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).op_type() == "Softmax") {
      out.push_back(i);
    }
  }
  return out;
}

std::vector<int> FindGeluTargets(const onnx::GraphProto& graph) {
  std::unordered_map<std::string, std::vector<int>> consumers_by_input;
  for (int i = 0; i < graph.node_size(); ++i) {
    for (const auto& in : graph.node(i).input()) {
      consumers_by_input[in].push_back(i);
    }
  }
  auto first_consumer_of_type = [&](const std::string& output,
                                    const std::string& op_type) -> int {
    auto it = consumers_by_input.find(output);
    if (it == consumers_by_input.end()) {
      return -1;
    }
    for (int idx : it->second) {
      if (graph.node(idx).op_type() == op_type) {
        return idx;
      }
    }
    return -1;
  };

  std::vector<int> targets;
  for (int i = 0; i < graph.node_size(); ++i) {
    if (graph.node(i).op_type() == "Gelu") {
      targets.push_back(i);
    }
  }
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& n = graph.node(i);
    if (n.op_type() != "Erf" || n.input_size() != 1 || n.output_size() < 1) {
      continue;
    }
    const int add_idx = first_consumer_of_type(n.output(0), "Add");
    if (add_idx < 0) {
      continue;
    }
    const int mul1_idx =
        first_consumer_of_type(graph.node(add_idx).output(0), "Mul");
    if (mul1_idx < 0) {
      continue;
    }
    const int mul2_idx =
        first_consumer_of_type(graph.node(mul1_idx).output(0), "Mul");
    if (mul2_idx < 0) {
      continue;
    }
    targets.push_back(mul2_idx);
  }
  return targets;
}

// --- Graph rewrite: twin quantize/dequantize splice -------------------------
//
// Inserts a fresh node at position `index` (shifting later nodes right) --
// transcribed from llm_int8_entry.cpp's own InsertEmptyNodeAt.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

// Splices PTQ4ViT's own twin-uniform quantize/dequantize round trip in
// right after `target_output`'s own producer node, rewiring every
// existing consumer of `target_output` to read the round trip's own
// final `Where` output instead -- transcribed from ptq4vit.py's own
// _insert_twin_quantize exactly (including relying on topological order
// to find consumers safely by post-insertion index -- see this file's
// own header comment).
void InsertTwinQuantize(onnx::GraphProto* graph,
                        const std::string& target_output, double lo,
                        double split, double hi, int64_t n_levels,
                        const std::string& tag,
                        std::unordered_set<std::string>& taken_names) {
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

  int producer_index = -1;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).output_size() >= 1 &&
        graph->node(i).output(0) == target_output) {
      producer_index = i;
      break;
    }
  }
  if (producer_index < 0) {
    return;  // Defensive -- the caller only ever passes a real node output.
  }

  // Old consumers, by ORIGINAL index -- every one is guaranteed to sit
  // after `producer_index` by the topological-order invariant this
  // function relies on (see this file's own header comment), so their
  // post-insertion live index is simply `original_index + num_new_nodes`.
  std::vector<int> old_consumer_indices;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (i == producer_index) {
      continue;
    }
    for (const auto& in : graph->node(i).input()) {
      if (in == target_output) {
        old_consumer_indices.push_back(i);
        break;
      }
    }
  }

  const std::string prefix = "ptq4vit_" + tag + "_";

  const std::string lo_name = unique_name(prefix + "lo");
  const std::string split_name = unique_name(prefix + "split");
  const std::string scale_lo_name = unique_name(prefix + "scale_lo");
  const std::string scale_hi_name = unique_name(prefix + "scale_hi");
  const std::string zero_name = unique_name(prefix + "zero");
  const std::string max_level_name = unique_name(prefix + "max_level");

  const double scale_lo_v =
      std::max(split - lo, kEps) / static_cast<double>(n_levels - 1);
  const double scale_hi_v =
      std::max(hi - split, kEps) / static_cast<double>(n_levels - 1);

  SetScalarF32Initializer(graph->add_initializer(), lo_name,
                          static_cast<float>(lo));
  SetScalarF32Initializer(graph->add_initializer(), split_name,
                          static_cast<float>(split));
  SetScalarF32Initializer(graph->add_initializer(), scale_lo_name,
                          static_cast<float>(scale_lo_v));
  SetScalarF32Initializer(graph->add_initializer(), scale_hi_name,
                          static_cast<float>(scale_hi_v));
  SetScalarF32Initializer(graph->add_initializer(), zero_name, 0.0f);
  SetScalarF32Initializer(graph->add_initializer(), max_level_name,
                          static_cast<float>(n_levels - 1));

  struct NewNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::string output;
    std::string name;
  };
  std::vector<NewNode> new_nodes;
  auto add_node = [&](const std::string& op_type,
                      const std::vector<std::string>& inputs,
                      const std::string& suffix) {
    NewNode n;
    n.op_type = op_type;
    n.inputs = inputs;
    n.output = unique_name(prefix + suffix);
    n.name = unique_name(prefix + suffix + "_node");
    new_nodes.push_back(n);
    return new_nodes.back().output;
  };

  const std::string mask_name =
      add_node("Less", {target_output, split_name}, "mask");

  const std::string shifted_lo =
      add_node("Sub", {target_output, lo_name}, "shifted_lo");
  const std::string scaled_lo =
      add_node("Div", {shifted_lo, scale_lo_name}, "scaled_lo");
  const std::string rounded_lo = add_node("Round", {scaled_lo}, "rounded_lo");
  const std::string clipped_lo =
      add_node("Clip", {rounded_lo, zero_name, max_level_name}, "clipped_lo");
  const std::string dq_lo_scaled =
      add_node("Mul", {clipped_lo, scale_lo_name}, "dq_lo_scaled");
  const std::string dq_lo = add_node("Add", {dq_lo_scaled, lo_name}, "dq_lo");

  const std::string shifted_hi =
      add_node("Sub", {target_output, split_name}, "shifted_hi");
  const std::string scaled_hi =
      add_node("Div", {shifted_hi, scale_hi_name}, "scaled_hi");
  const std::string rounded_hi = add_node("Round", {scaled_hi}, "rounded_hi");
  const std::string clipped_hi =
      add_node("Clip", {rounded_hi, zero_name, max_level_name}, "clipped_hi");
  const std::string dq_hi_scaled =
      add_node("Mul", {clipped_hi, scale_hi_name}, "dq_hi_scaled");
  const std::string dq_hi =
      add_node("Add", {dq_hi_scaled, split_name}, "dq_hi");

  const std::string result_name =
      add_node("Where", {mask_name, dq_lo, dq_hi}, "result");

  const int count = static_cast<int>(new_nodes.size());
  for (int i = 0; i < count; ++i) {
    InsertEmptyNodeAt(graph, producer_index + 1 + i);
  }
  for (int i = 0; i < count; ++i) {
    const NewNode& spec = new_nodes[static_cast<size_t>(i)];
    onnx::NodeProto* n = graph->mutable_node(producer_index + 1 + i);
    n->set_op_type(spec.op_type);
    for (const auto& s : spec.inputs) {
      n->add_input(s);
    }
    n->add_output(spec.output);
    n->set_name(spec.name);
  }

  for (int original_idx : old_consumer_indices) {
    onnx::NodeProto* n = graph->mutable_node(original_idx + count);
    for (int i = 0; i < n->input_size(); ++i) {
      if (n->input(i) == target_output) {
        n->set_input(i, result_name);
      }
    }
  }
}

}  // namespace

onnx::ModelProto ApplyPtq4Vit(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t n_levels) {
  bool opset_ge_11 = false;
  for (const auto& opset : model.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 11) {
      opset_ge_11 = true;
      break;
    }
  }
  if (!opset_ge_11) {
    return model;
  }

  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_set<std::string> graph_output_names;
  for (const auto& o : graph->output()) {
    graph_output_names.insert(o.name());
  }

  // Captured as output NAMES, once, up front -- not node indices: every
  // InsertTwinQuantize call below inserts new nodes and shifts every
  // later node's own index, so re-deriving a target's output name via a
  // stale original index mid-loop would silently read the WRONG node
  // once any earlier candidate in the same loop has already been
  // rewritten. ptq4vit.py's own equivalent loop instead holds live
  // NodeProto object references (`node.output[0]`), which stay valid
  // across mutation by Python object identity; capturing the name once
  // up front is this port's own equivalent, index-free way to get the
  // same stability.
  std::vector<std::string> softmax_target_names;
  for (int idx : FindSoftmaxTargets(*graph)) {
    const std::string& name = graph->node(idx).output(0);
    if (!graph_output_names.count(name)) {
      softmax_target_names.push_back(name);
    }
  }
  std::vector<std::string> gelu_target_names;
  for (int idx : FindGeluTargets(*graph)) {
    const std::string& name = graph->node(idx).output(0);
    if (!graph_output_names.count(name)) {
      gelu_target_names.push_back(name);
    }
  }

  std::vector<std::string> candidate_names;
  candidate_names.reserve(softmax_target_names.size() +
                          gelu_target_names.size());
  candidate_names.insert(candidate_names.end(), softmax_target_names.begin(),
                         softmax_target_names.end());
  candidate_names.insert(candidate_names.end(), gelu_target_names.begin(),
                         gelu_target_names.end());
  if (candidate_names.empty()) {
    return out;
  }

  // Probe: capture every matched tensor's own REAL VALUES (concatenated
  // flat double buffer per name), not a reduction -- see this file's own
  // header comment for why this differs from every other calibration-
  // driven pass in this codebase.
  onnx::ModelProto probe_model = out;
  std::unordered_set<std::string> existing_outputs;
  for (const auto& o : probe_model.graph().output()) {
    existing_outputs.insert(o.name());
  }
  for (const auto& name : candidate_names) {
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

  std::unordered_map<std::string, std::vector<double>> collected;
  for (const auto& name : candidate_names) {
    collected.emplace(name, std::vector<double>{});
  }

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> input_dls;
    std::vector<const DLManagedTensor*> input_ptrs;
    input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
    input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
    for (const auto& gi : graph_inputs) {
      auto it = batch.find(gi.name());
      if (it == batch.end()) {
        throw std::invalid_argument(
            "ApplyPtq4Vit: calibration batch is missing required graph "
            "input '" +
            gi.name() + "'");
      }
      input_dls.emplace_back(
          onnxsim::dlpack::FromTensorProtoBorrowing(it->second));
      input_ptrs.push_back(input_dls.back().get());
    }

    std::vector<DLManagedTensorPtr> outputs =
        executor.Run(probe_model, input_ptrs);

    for (const auto& name : candidate_names) {
      auto oit = output_index.find(name);
      if (oit == output_index.end() || oit->second >= outputs.size()) {
        continue;
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT) {
        continue;  // FLOAT32-only scope -- see this file's own header
                   // comment.
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      if (data.empty()) {
        continue;
      }
      std::vector<double>& acc = collected[name];
      acc.reserve(acc.size() + data.size());
      for (float v : data) {
        acc.push_back(static_cast<double>(v));
      }
    }
  }

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

  for (size_t i = 0; i < softmax_target_names.size(); ++i) {
    const std::string& name = softmax_target_names[i];
    const auto it = collected.find(name);
    if (it == collected.end() || it->second.empty()) {
      continue;
    }
    const auto [found, split] = SearchTwinSplit(it->second, 0.0, 1.0, n_levels);
    if (!found) {
      continue;
    }
    InsertTwinQuantize(graph, name, 0.0, split, 1.0, n_levels,
                       "softmax" + std::to_string(i), taken_names);
  }

  for (size_t i = 0; i < gelu_target_names.size(); ++i) {
    const std::string& name = gelu_target_names[i];
    const auto it = collected.find(name);
    if (it == collected.end() || it->second.empty()) {
      continue;
    }
    const std::vector<double>& v = it->second;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double x : v) {
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
    const auto [found, split] = SearchTwinSplit(v, lo, hi, n_levels);
    if (!found) {
      continue;
    }
    InsertTwinQuantize(graph, name, lo, split, hi, n_levels,
                       "gelu" + std::to_string(i), taken_names);
  }

  return out;
}
