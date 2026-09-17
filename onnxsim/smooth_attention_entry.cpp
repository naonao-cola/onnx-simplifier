// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See smooth_attention_entry.h for the full rationale (including why this
// transcribes attention_quantization.h's own matcher to the protobuf
// level rather than reusing it directly, and why this follows
// llm_int8_entry.h's own protobuf-level, single-model, calibration-driven
// shape) and onnxsim/qoq.py for the technique this ports.

#include "smooth_attention_entry.h"

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

// --- Attention subgraph matching, protobuf level ----------------------------
//
// Transcribed from passes/attention_quantization.h's own
// FindQKMatMulProducer/FindOutMatMulConsumer (Node/Value IR), which
// themselves transcribe attention_quantization.py's own
// _find_matmul_producer/_find_attention_candidates exactly -- see this
// file's own header comment for why a protobuf-level transcription
// (rather than reuse) is this codebase's established convention here.

constexpr int kMaxScaleMaskHops = 2;

// Walks back from tensor `name` through at most `hops_left` Mul/Div/Add
// nodes (following each one's own *first* input only), looking for the
// MatMul that produced the raw attention scores. `producer_by_output`
// mirrors qoq.py's own `producer_by_output` exactly (last-node-wins,
// built by a single forward pass over graph.node).
const onnx::NodeProto* FindQkMatmulProducer(
    const std::string& name,
    const std::unordered_map<std::string, const onnx::NodeProto*>&
        producer_by_output,
    int hops_left) {
  auto it = producer_by_output.find(name);
  if (it == producer_by_output.end()) {
    return nullptr;
  }
  const onnx::NodeProto* node = it->second;
  if (node->op_type() == "MatMul") {
    return node;
  }
  if (hops_left <= 0) {
    return nullptr;
  }
  if (node->op_type() != "Mul" && node->op_type() != "Div" &&
      node->op_type() != "Add") {
    return nullptr;
  }
  if (node->input_size() == 0) {
    return nullptr;
  }
  return FindQkMatmulProducer(node->input(0), producer_by_output,
                              hops_left - 1);
}

// First MatMul among `softmax_out`'s own uses (in graph node order) that
// consumes it at input position 0 -- mirrors qoq.py's own `next((c for c
// in consumers if c.op_type == "MatMul" and c.input[0] == softmax_out),
// None)` exactly.
const onnx::NodeProto* FindOutMatmulConsumer(
    const std::string& softmax_out,
    const std::unordered_map<std::string, std::vector<const onnx::NodeProto*>>&
        consumers_by_input) {
  auto it = consumers_by_input.find(softmax_out);
  if (it == consumers_by_input.end()) {
    return nullptr;
  }
  for (const onnx::NodeProto* c : it->second) {
    if (c->op_type() == "MatMul" && c->input_size() >= 1 &&
        c->input(0) == softmax_out) {
      return c;
    }
  }
  return nullptr;
}

struct AttentionCandidate {
  int qk_matmul_index;
  int softmax_index;
};

std::vector<AttentionCandidate> FindAttentionCandidates(
    const onnx::GraphProto& graph) {
  std::unordered_map<std::string, const onnx::NodeProto*> producer_by_output;
  std::unordered_map<std::string, int> index_by_node_ptr_output;
  std::unordered_map<const onnx::NodeProto*, int> node_index;
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& n = graph.node(i);
    node_index[&n] = i;
    for (const auto& out : n.output()) {
      producer_by_output[out] = &n;
    }
  }
  std::unordered_map<std::string, std::vector<const onnx::NodeProto*>>
      consumers_by_input;
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& n = graph.node(i);
    for (const auto& in : n.input()) {
      consumers_by_input[in].push_back(&n);
    }
  }

  std::vector<AttentionCandidate> candidates;
  std::unordered_set<int> seen_matmuls;
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& n = graph.node(i);
    if (n.op_type() != "Softmax" || n.input_size() < 1) {
      continue;
    }
    const onnx::NodeProto* qk =
        FindQkMatmulProducer(n.input(0), producer_by_output, kMaxScaleMaskHops);
    if (qk == nullptr || qk->input_size() < 2) {
      continue;
    }
    if (n.output_size() < 1) {
      continue;
    }
    const onnx::NodeProto* out_mm =
        FindOutMatmulConsumer(n.output(0), consumers_by_input);
    if (out_mm == nullptr) {
      continue;
    }
    const int qk_idx = node_index[qk];
    if (!seen_matmuls.insert(qk_idx).second) {
      continue;
    }
    candidates.push_back({qk_idx, i});
  }
  return candidates;
}

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

// Transcribed from llm_int8_entry.cpp's own identical helper.
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

// Inserts a fresh node at position `index` (shifting later nodes right) --
// transcribed from llm_int8_entry.cpp's own InsertEmptyNodeAt.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

// --- Calibration: Kt's own per-(second-to-last-axis)-channel absmax --------
//
// Kt (Key, transposed) has shape [..., head_dim, seq_k]: the channel axis
// is the SECOND-TO-LAST one, not the last one every other
// ComputeChannelAbsmax in this codebase (llm_int8_entry.cpp's,
// rptq_entry.cpp's) reduces over -- so this implements its own reduction
// rather than reusing theirs (see this file's own header comment).
// Mirrors qoq.py's own `np.moveaxis(arr, -2, -1).reshape(-1,
// arr.shape[-2])` exactly via flat-index arithmetic: for a row-major
// tensor of shape [..., C, L], element at flat index `idx` belongs to
// channel `c = (idx / L) % C`.
std::unordered_map<std::string, std::vector<double>>
ComputeSecondToLastAxisAbsmax(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  std::unordered_map<std::string, std::vector<double>> result;
  if (probe_names.empty()) {
    return result;
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
            "ApplySmoothAttention: calibration batch is missing required "
            "graph input '" +
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
        continue;
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 2) {
        continue;  // arr.ndim < 2: never observed as a plain-enough probe.
      }
      const int64_t rank = tp.dims_size();
      const int64_t c = tp.dims(static_cast<int>(rank - 2));
      const int64_t l = tp.dims(static_cast<int>(rank - 1));
      if (c <= 0 || l <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      auto& acc = result[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(c), 0.0);
      } else if (acc.size() != static_cast<size_t>(c)) {
        continue;  // Channel width changed mid-calibration; keep first.
      }
      for (int64_t idx = 0, total = static_cast<int64_t>(data.size());
           idx < total; ++idx) {
        const int64_t ch = (idx / l) % c;
        const double v =
            std::abs(static_cast<double>(data[static_cast<size_t>(idx)]));
        double& slot = acc[static_cast<size_t>(ch)];
        if (v > slot) {
          slot = v;
        }
      }
    }
  }
  return result;
}

}  // namespace

onnx::ModelProto ApplySmoothAttention(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double epsilon) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::vector<AttentionCandidate> candidates = FindAttentionCandidates(*graph);
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(graph->node(c.qk_matmul_index).input(1));
  }
  const std::unordered_map<std::string, std::vector<double>> k_absmax =
      ComputeSecondToLastAxisAbsmax(executor, out, probe_names,
                                    calibration_data);

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

  // Sorted by the score MatMul's own original node index, ascending --
  // insertion only ever happens right before that node, so processing in
  // ascending order lets a running `shift` (2 nodes inserted per
  // processed candidate) account for every earlier insertion exactly,
  // the same bookkeeping llm_int8_entry.cpp's own `net_insertions` uses.
  std::sort(candidates.begin(), candidates.end(),
            [](const AttentionCandidate& a, const AttentionCandidate& b) {
              return a.qk_matmul_index < b.qk_matmul_index;
            });

  int shift = 0;
  for (const auto& c : candidates) {
    const int live_qk_index = c.qk_matmul_index + shift;
    const onnx::NodeProto& qk = graph->node(live_qk_index);
    const std::string q_name = qk.input(0);
    const std::string kt_name = qk.input(1);

    auto ait = k_absmax.find(kt_name);
    if (ait == k_absmax.end()) {
      continue;  // Kt never observed as a rank >= 2 probe; skip.
    }
    const std::vector<double>& absmax = ait->second;
    const int64_t head_dim = static_cast<int64_t>(absmax.size());

    std::vector<float> s(absmax.size());
    for (size_t i = 0; i < absmax.size(); ++i) {
      s[i] = static_cast<float>(std::max(absmax[i], epsilon));
    }

    const std::string s_row_name = unique_name(q_name + "_smooth_attn_s");
    SetRawInitializer(graph->add_initializer(), s_row_name,
                      onnx::TensorProto::FLOAT, {head_dim}, s.data(),
                      s.size() * sizeof(float), sizeof(float));

    const std::string q_scaled_name = unique_name(q_name + "_smooth_attn_q");
    const std::string q_mul_node_name =
        unique_name(q_name + "_smooth_attn_q_node");

    const std::string s_col_name = unique_name(kt_name + "_smooth_attn_s_col");
    SetRawInitializer(graph->add_initializer(), s_col_name,
                      onnx::TensorProto::FLOAT, {head_dim, 1}, s.data(),
                      s.size() * sizeof(float), sizeof(float));

    const std::string kt_scaled_name = unique_name(kt_name + "_smooth_attn_k");
    const std::string k_div_node_name =
        unique_name(kt_name + "_smooth_attn_k_node");

    // Mul then Div, both inserted directly before `qk` -- mirrors
    // qoq.py's own insertion order exactly.
    InsertEmptyNodeAt(graph, live_qk_index);
    onnx::NodeProto* mul_node = graph->mutable_node(live_qk_index);
    mul_node->set_op_type("Mul");
    mul_node->add_input(q_name);
    mul_node->add_input(s_row_name);
    mul_node->add_output(q_scaled_name);
    mul_node->set_name(q_mul_node_name);

    InsertEmptyNodeAt(graph, live_qk_index + 1);
    onnx::NodeProto* div_node = graph->mutable_node(live_qk_index + 1);
    div_node->set_op_type("Div");
    div_node->add_input(kt_name);
    div_node->add_input(s_col_name);
    div_node->add_output(kt_scaled_name);
    div_node->set_name(k_div_node_name);

    onnx::NodeProto* qk_mut = graph->mutable_node(live_qk_index + 2);
    qk_mut->set_input(0, q_scaled_name);
    qk_mut->set_input(1, kt_scaled_name);

    shift += 2;
  }

  return out;
}
