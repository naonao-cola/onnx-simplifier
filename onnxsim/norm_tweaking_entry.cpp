// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See norm_tweaking_entry.h for the full rationale and
// onnxsim/norm_tweaking.py for the technique this ports.

#include "norm_tweaking_entry.h"

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

// Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
// base_1, base_2, ...) -- same convention every calibration-driven
// *_entry.cpp in this codebase already uses.
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

// Adds a graph output for every name in `names` not already present, and
// returns a {name -> output index} map for those names. Mirrors
// bias_correction.py's own _add_probe_outputs.
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

// Runs `executor` on `probe_model` for one calibration batch, throwing
// std::invalid_argument (mirrors every other calibration-driven pass's
// own identical guard) when `batch` is missing one of `probe_model`'s own
// graph inputs.
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

}  // namespace

onnx::ModelProto ApplyNormTweaking(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double eps) {
  const onnx::GraphProto& q_graph = quantized_model.graph();
  const onnx::GraphProto& f_graph = float_model.graph();

  // quantized_by_output: LayerNormalization nodes only, last-wins --
  // mirrors apply_norm_tweaking's own `quantized_by_output` dict.
  std::unordered_map<std::string, int> quantized_by_output;
  for (int i = 0; i < q_graph.node_size(); ++i) {
    const auto& n = q_graph.node(i);
    if (n.op_type() == "LayerNormalization" && n.output_size() > 0) {
      quantized_by_output[n.output(0)] = i;
    }
  }
  std::unordered_map<std::string, int> q_init_index;
  for (int i = 0; i < q_graph.initializer_size(); ++i) {
    q_init_index.emplace(q_graph.initializer(i).name(), i);
  }

  // Candidate LayerNorm output names, first-seen order, deduplicated --
  // mirrors apply_norm_tweaking's own `candidates`/`names` list.
  std::vector<std::string> names;
  {
    std::unordered_set<std::string> seen;
    for (int i = 0; i < f_graph.node_size(); ++i) {
      const auto& n = f_graph.node(i);
      if (n.op_type() != "LayerNormalization" || n.output_size() == 0) {
        continue;
      }
      auto qit = quantized_by_output.find(n.output(0));
      if (qit == quantized_by_output.end()) {
        continue;
      }
      const onnx::NodeProto& q_node = q_graph.node(qit->second);
      if (q_node.input_size() < 2) {
        continue;
      }
      auto sit = q_init_index.find(q_node.input(1));
      if (sit == q_init_index.end()) {
        continue;
      }
      if (q_graph.initializer(sit->second).dims_size() != 1) {
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

  std::unordered_map<std::string, std::vector<double>> f_sum, f_sumsq, q_sum,
      q_sumsq;
  std::unordered_map<std::string, int64_t> counts;

  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> f_outputs =
        RunOneModel(executor, float_probe, batch, "ApplyNormTweaking");
    std::vector<DLManagedTensorPtr> q_outputs =
        RunOneModel(executor, quantized_probe, batch, "ApplyNormTweaking");

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

      if (f_sum.count(name) &&
          static_cast<int64_t>(f_sum[name].size()) != channels) {
        continue;  // Channel width changed mid-calibration; keep the first.
      }
      if (!f_sum.count(name)) {
        f_sum[name].assign(static_cast<size_t>(channels), 0.0);
        f_sumsq[name].assign(static_cast<size_t>(channels), 0.0);
        q_sum[name].assign(static_cast<size_t>(channels), 0.0);
        q_sumsq[name].assign(static_cast<size_t>(channels), 0.0);
        counts[name] = 0;
      }

      const std::vector<double> f_data = ReadFloatTensor(f_tp);
      const std::vector<double> q_data = ReadFloatTensor(q_tp);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < channels; ++c) {
          const double fv = f_data[static_cast<size_t>(r * channels + c)];
          const double qv = q_data[static_cast<size_t>(r * channels + c)];
          f_sum[name][static_cast<size_t>(c)] += fv;
          f_sumsq[name][static_cast<size_t>(c)] += fv * fv;
          q_sum[name][static_cast<size_t>(c)] += qv;
          q_sumsq[name][static_cast<size_t>(c)] += qv * qv;
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
    auto cit = counts.find(name);
    if (cit == counts.end() || cit->second <= 0) {
      continue;  // Mirrors `if name not in counts: continue`.
    }
    const int64_t n = cit->second;
    const std::vector<double>& fs = f_sum[name];
    const std::vector<double>& fss = f_sumsq[name];
    const std::vector<double>& qs = q_sum[name];
    const std::vector<double>& qss = q_sumsq[name];
    const int64_t channels = static_cast<int64_t>(fs.size());

    auto node_it = node_by_output.find(name);
    if (node_it == node_by_output.end()) {
      continue;
    }
    onnx::NodeProto* q_node = node_it->second;
    if (q_node->input_size() < 2) {
      continue;
    }
    auto sit = initializer_index.find(q_node->input(1));
    if (sit == initializer_index.end()) {
      continue;
    }
    onnx::TensorProto& scale_init = *graph->mutable_initializer(sit->second);
    if (scale_init.dims_size() != 1 || scale_init.dims(0) != channels) {
      continue;  // Defensive: candidate matching already checked this on
                 // the ORIGINAL quantized_model; re-check on the working
                 // copy in case another candidate's own edit somehow
                 // touched the same initializer (never happens in
                 // practice -- LayerNorm scale/bias initializers are
                 // never shared across nodes in a well-formed graph, but
                 // raw protobuf reads warrant the same explicit bound
                 // check low_rank_compensation_entry.cpp's own analogous
                 // comment explains).
      continue;
    }
    const std::vector<double> old_scale = ReadFloatTensor(scale_init);

    std::vector<double> alpha(static_cast<size_t>(channels));
    std::vector<double> beta(static_cast<size_t>(channels));
    for (int64_t c = 0; c < channels; ++c) {
      const double mu_f = fs[static_cast<size_t>(c)] / static_cast<double>(n);
      const double var_f = std::max(
          fss[static_cast<size_t>(c)] / static_cast<double>(n) - mu_f * mu_f,
          0.0);
      const double sigma_f = std::sqrt(var_f);
      const double mu_q = qs[static_cast<size_t>(c)] / static_cast<double>(n);
      const double var_q = std::max(
          qss[static_cast<size_t>(c)] / static_cast<double>(n) - mu_q * mu_q,
          0.0);
      const double sigma_q = std::sqrt(var_q);
      const double a = sigma_f / (sigma_q + eps);
      alpha[static_cast<size_t>(c)] = a;
      beta[static_cast<size_t>(c)] = mu_f - a * mu_q;
    }

    std::vector<double> new_scale(static_cast<size_t>(channels));
    for (int64_t c = 0; c < channels; ++c) {
      new_scale[static_cast<size_t>(c)] =
          old_scale[static_cast<size_t>(c)] * alpha[static_cast<size_t>(c)];
    }
    const std::string new_scale_name =
        UniqueName(name + "_norm_tweak_scale", taken_names);
    SetFloatInitializer(graph->add_initializer(), new_scale_name, {channels},
                        new_scale);
    // Re-find q_node by pointer identity: add_initializer() above cannot
    // invalidate mutable_node() pointers (separate RepeatedPtrFields), but
    // this mirrors gptq_entry.cpp's own defensive re-lookup convention
    // for clarity.
    q_node->set_input(1, new_scale_name);

    std::vector<double> new_bias(static_cast<size_t>(channels));
    if (q_node->input_size() >= 3 && !q_node->input(2).empty()) {
      auto bit = initializer_index.find(q_node->input(2));
      if (bit != initializer_index.end()) {
        const onnx::TensorProto& bias_init = graph->initializer(bit->second);
        if (bias_init.dims_size() == 1 && bias_init.dims(0) == channels) {
          const std::vector<double> old_bias = ReadFloatTensor(bias_init);
          for (int64_t c = 0; c < channels; ++c) {
            new_bias[static_cast<size_t>(c)] =
                alpha[static_cast<size_t>(c)] *
                    old_bias[static_cast<size_t>(c)] +
                beta[static_cast<size_t>(c)];
          }
        } else {
          new_bias = beta;
        }
      } else {
        new_bias = beta;
      }
    } else {
      new_bias = beta;
    }
    const std::string new_bias_name =
        UniqueName(name + "_norm_tweak_bias", taken_names);
    SetFloatInitializer(graph->add_initializer(), new_bias_name, {channels},
                        new_bias);
    if (q_node->input_size() >= 3) {
      q_node->set_input(2, new_bias_name);
    } else {
      q_node->add_input(new_bias_name);
    }
  }

  return out;
}
