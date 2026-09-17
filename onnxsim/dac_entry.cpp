// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See dac_entry.h for the full rationale and onnxsim/d2quant.py's own
// apply_dac for the technique this ports. The dual-model probing loop
// below is transcribed from norm_tweaking_entry.cpp's own identical
// helpers (RunOneModel/AddProbeOutputs/SameDims/AllNames/UniqueName) --
// see norm_tweaking_entry.h's own top-of-file comment for why they aren't
// shared via a common header.

#include "dac_entry.h"

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

std::vector<double> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> raw_floats(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(raw_floats.data(), t.raw_data().data(),
                raw_floats.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(
          reinterpret_cast<uint8_t*>(raw_floats.data()),
          raw_floats.size() * sizeof(float), sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      raw_floats[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return std::vector<double>(raw_floats.begin(), raw_floats.end());
}

void SetFloatInitializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<double>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::vector<float> as_float(data.begin(), data.end());
  std::string raw(as_float.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), as_float.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

std::unordered_set<std::string> AllNames(const onnx::GraphProto& graph) {
  std::unordered_set<std::string> taken;
  for (const auto& t : graph.initializer()) {
    taken.insert(t.name());
  }
  for (const auto& vi : graph.input()) {
    taken.insert(vi.name());
  }
  for (const auto& vi : graph.output()) {
    taken.insert(vi.name());
  }
  for (const auto& vi : graph.value_info()) {
    taken.insert(vi.name());
  }
  for (const auto& n : graph.node()) {
    if (!n.name().empty()) {
      taken.insert(n.name());
    }
    for (const auto& s : n.input()) {
      taken.insert(s);
    }
    for (const auto& s : n.output()) {
      taken.insert(s);
    }
  }
  return taken;
}

std::string UniqueName(const std::string& base,
                       std::unordered_set<std::string>& taken) {
  std::string name = base;
  int i = 0;
  while (taken.count(name) != 0) {
    ++i;
    name = base + "_" + std::to_string(i);
  }
  taken.insert(name);
  return name;
}

std::unordered_map<std::string, size_t> AddProbeOutputs(
    onnx::ModelProto* probe_model, const std::vector<std::string>& names) {
  std::unordered_set<std::string> existing;
  for (const auto& o : probe_model->graph().output()) {
    existing.insert(o.name());
  }
  for (const auto& name : names) {
    if (existing.insert(name).second) {
      probe_model->mutable_graph()->add_output()->set_name(name);
    }
  }
  std::unordered_map<std::string, size_t> output_index;
  for (int i = 0; i < probe_model->graph().output_size(); ++i) {
    output_index.emplace(probe_model->graph().output(i).name(),
                         static_cast<size_t>(i));
  }
  return output_index;
}

std::vector<DLManagedTensorPtr> RunOneModel(
    const ModelExecutor& executor, const onnx::ModelProto& probe_model,
    const std::unordered_map<std::string, onnx::TensorProto>& batch,
    const char* fn_name) {
  const auto& graph_inputs = probe_model.graph().input();
  std::vector<DLManagedTensorPtr> input_dls;
  std::vector<const DLManagedTensor*> input_ptrs;
  input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
  input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
  for (const auto& gi : graph_inputs) {
    auto it = batch.find(gi.name());
    if (it == batch.end()) {
      throw std::invalid_argument(std::string(fn_name) +
                                  ": calibration batch is missing required "
                                  "graph input '" +
                                  gi.name() + "'");
    }
    input_dls.emplace_back(
        onnxsim::dlpack::FromTensorProtoBorrowing(it->second));
    input_ptrs.push_back(input_dls.back().get());
  }
  return executor.Run(probe_model, input_ptrs);
}

bool SameDims(const onnx::TensorProto& a, const onnx::TensorProto& b) {
  if (a.dims_size() != b.dims_size()) {
    return false;
  }
  for (int i = 0; i < a.dims_size(); ++i) {
    if (a.dims(i) != b.dims(i)) {
      return false;
    }
  }
  return true;
}

// Mirrors d2quant.py's own _apply_ln_bias_correction exactly: an existing
// FLOAT, correctly-shaped bias initializer is overwritten IN PLACE (same
// name, same referencing node input -- no rewiring needed); otherwise a
// brand new bias initializer is created and wired in (appended if the
// node had no bias input at all).
void ApplyLnBiasCorrection(
    onnx::GraphProto* graph, onnx::NodeProto* ln_node,
    const std::vector<double>& correction,
    std::unordered_map<std::string, int>& initializer_index,
    std::unordered_set<std::string>& taken_names) {
  const int64_t channels = static_cast<int64_t>(correction.size());
  if (ln_node->input_size() >= 3 && !ln_node->input(2).empty()) {
    auto bit = initializer_index.find(ln_node->input(2));
    if (bit != initializer_index.end()) {
      onnx::TensorProto* bias_init = graph->mutable_initializer(bit->second);
      if (bias_init->data_type() == onnx::TensorProto::FLOAT &&
          bias_init->dims_size() == 1 && bias_init->dims(0) == channels) {
        const std::vector<double> old_bias = ReadFloatTensor(*bias_init);
        std::vector<double> new_bias(static_cast<size_t>(channels));
        for (int64_t c = 0; c < channels; ++c) {
          new_bias[static_cast<size_t>(c)] = old_bias[static_cast<size_t>(c)] +
                                             correction[static_cast<size_t>(c)];
        }
        // Copy the name into a local FIRST: SetFloatInitializer's own
        // t->Clear() would otherwise clear it out from under a
        // reference/pointer straight into bias_init's own name field
        // (t IS bias_init here, an in-place update).
        const std::string original_name = bias_init->name();
        SetFloatInitializer(bias_init, original_name, {channels}, new_bias);
        return;
      }
    }
    return;  // Mirrors the Python reference's own early `return` when the
             // existing bias input doesn't resolve to a well-formed FLOAT
             // initializer of the right width.
  }

  const std::string new_name =
      UniqueName(ln_node->output(0) + "_dac_bias", taken_names);
  onnx::TensorProto* new_bias = graph->add_initializer();
  SetFloatInitializer(new_bias, new_name, {channels}, correction);
  initializer_index[new_name] = graph->initializer_size() - 1;
  if (ln_node->input_size() >= 3) {
    ln_node->set_input(2, new_name);
  } else {
    ln_node->add_input(new_name);
  }
}

}  // namespace

onnx::ModelProto ApplyDac(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double min_expected_error_reduction, double correction_threshold) {
  const onnx::GraphProto& q_graph = quantized_model.graph();
  const onnx::GraphProto& f_graph = float_model.graph();

  std::unordered_set<std::string> quantized_ln_outputs;
  for (const auto& n : q_graph.node()) {
    if (n.op_type() == "LayerNormalization" && n.output_size() > 0) {
      quantized_ln_outputs.insert(n.output(0));
    }
  }

  std::vector<std::string> names;
  {
    std::unordered_set<std::string> seen;
    for (const auto& n : f_graph.node()) {
      if (n.op_type() != "LayerNormalization" || n.output_size() == 0) {
        continue;
      }
      if (!quantized_ln_outputs.count(n.output(0))) {
        continue;
      }
      if (seen.insert(n.output(0)).second) {
        names.push_back(n.output(0));
      }
    }
  }
  if (names.empty()) {
    return quantized_model;
  }

  onnx::ModelProto float_probe = float_model;
  onnx::ModelProto quantized_probe = quantized_model;
  const auto f_output_index = AddProbeOutputs(&float_probe, names);
  const auto q_output_index = AddProbeOutputs(&quantized_probe, names);

  std::unordered_map<std::string, std::vector<double>> sums, sumsqs;
  std::unordered_map<std::string, int64_t> counts;

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> f_outputs =
        RunOneModel(executor, float_probe, batch, "ApplyDac");
    std::vector<DLManagedTensorPtr> q_outputs =
        RunOneModel(executor, quantized_probe, batch, "ApplyDac");

    for (const auto& name : names) {
      auto fit = f_output_index.find(name);
      auto qit = q_output_index.find(name);
      if (fit == f_output_index.end() || fit->second >= f_outputs.size() ||
          qit == q_output_index.end() || qit->second >= q_outputs.size()) {
        continue;
      }
      const onnx::TensorProto f_tp =
          onnxsim::dlpack::ToTensorProto(f_outputs[fit->second]->dl_tensor);
      const onnx::TensorProto q_tp =
          onnxsim::dlpack::ToTensorProto(q_outputs[qit->second]->dl_tensor);
      if (f_tp.data_type() != onnx::TensorProto::FLOAT ||
          q_tp.data_type() != onnx::TensorProto::FLOAT) {
        continue;
      }
      if (!SameDims(f_tp, q_tp) || f_tp.dims_size() == 0) {
        continue;  // Mirrors `if f.shape != q.shape or f.ndim == 0: continue`.
      }
      const int64_t channels = f_tp.dims(f_tp.dims_size() - 1);
      if (channels <= 0) {
        continue;
      }
      int64_t numel = 1;
      for (int64_t d : f_tp.dims()) {
        numel *= d;
      }
      const int64_t rows = numel / channels;

      if (sums.count(name) &&
          static_cast<int64_t>(sums[name].size()) != channels) {
        continue;
      }
      if (!sums.count(name)) {
        sums[name].assign(static_cast<size_t>(channels), 0.0);
        sumsqs[name].assign(static_cast<size_t>(channels), 0.0);
        counts[name] = 0;
      }

      const std::vector<double> f_data = ReadFloatTensor(f_tp);
      const std::vector<double> q_data = ReadFloatTensor(q_tp);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < channels; ++c) {
          const double diff = f_data[static_cast<size_t>(r * channels + c)] -
                              q_data[static_cast<size_t>(r * channels + c)];
          sums[name][static_cast<size_t>(c)] += diff;
          sumsqs[name][static_cast<size_t>(c)] += diff * diff;
        }
      }
      counts[name] += rows;
    }
  }

  onnx::ModelProto out = quantized_model;
  onnx::GraphProto* graph = out.mutable_graph();
  std::unordered_set<std::string> taken_names = AllNames(*graph);
  std::unordered_map<std::string, int> initializer_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    initializer_index.emplace(graph->initializer(i).name(), i);
  }
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->op_type() == "LayerNormalization" && n->output_size() > 0) {
      node_by_output[n->output(0)] = n;
    }
  }

  for (const auto& name : names) {
    auto sit = sums.find(name);
    if (sit == sums.end()) {
      continue;
    }
    const int64_t n = counts[name];
    if (n <= 0) {
      continue;
    }
    const std::vector<double>& total = sit->second;
    const std::vector<double>& totalsq = sumsqs[name];
    const int64_t channels = static_cast<int64_t>(total.size());

    std::vector<double> correction(static_cast<size_t>(channels));
    double max_abs = 0.0;
    for (int64_t c = 0; c < channels; ++c) {
      const double mu = total[static_cast<size_t>(c)] / static_cast<double>(n);
      const double var = std::max(
          totalsq[static_cast<size_t>(c)] / static_cast<double>(n) - mu * mu,
          0.0);
      const double expected_reduction = (mu * mu) / (mu * mu + var + 1e-12);
      const double corr =
          (expected_reduction >= min_expected_error_reduction) ? mu : 0.0;
      correction[static_cast<size_t>(c)] = corr;
      max_abs = std::max(max_abs, std::fabs(corr));
    }
    if (max_abs <= correction_threshold) {
      continue;
    }

    auto node_it = node_by_output.find(name);
    if (node_it == node_by_output.end()) {
      continue;
    }
    onnx::NodeProto* ln_node = node_it->second;
    if (ln_node->input_size() < 2) {
      continue;
    }
    auto gamma_it = initializer_index.find(ln_node->input(1));
    if (gamma_it == initializer_index.end()) {
      continue;
    }
    const onnx::TensorProto& gamma_init = graph->initializer(gamma_it->second);
    if (gamma_init.data_type() != onnx::TensorProto::FLOAT ||
        gamma_init.dims_size() != 1 || gamma_init.dims(0) != channels) {
      continue;
    }

    ApplyLnBiasCorrection(graph, ln_node, correction, initializer_index,
                          taken_names);
  }

  return out;
}
