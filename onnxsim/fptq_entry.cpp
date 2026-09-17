// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See fptq_entry.h for the full rationale (including why this reuses
// smoothquant_entry.cpp's own MatchMatMulLike/ComputeChannelAbsmax/
// node-insertion machinery near-verbatim, changing only the scale
// computation) and onnxsim/fptq.py for the technique this ports. See
// gptaq_entry.cpp's own top-of-file comment for why these helpers are
// duplicated here rather than shared via a common header (matches this
// codebase's established convention: every *_entry.cpp is self-contained).

#include "fptq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Identical to smoothquant_entry.cpp's own MatchMatMulLike (which is
// exactly what onnxsim.smoothquant._match_matmul_like -- reused unmodified
// by fptq.py -- checks).
struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
  bool weight_transposed;
};

std::optional<MatMulLikeMatch> MatchMatMulLike(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), false};
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return std::nullopt;
    }
    bool has_trans_a = false, has_alpha = false, has_beta = false;
    int64_t trans_a = 0, trans_b = 0;
    double alpha = 1.0, beta = 1.0;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transA") {
        trans_a = attr.i();
        has_trans_a = true;
      } else if (attr.name() == "alpha") {
        alpha = attr.f();
        has_alpha = true;
      } else if (attr.name() == "beta") {
        beta = attr.f();
        has_beta = true;
      } else if (attr.name() == "transB") {
        trans_b = attr.i();
      }
    }
    if (has_trans_a && trans_a != 0) {
      return std::nullopt;
    }
    if (has_alpha && alpha != 1.0) {
      return std::nullopt;
    }
    if (num_inputs == 3 && has_beta && beta != 1.0) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), trans_b != 0};
  }
  return std::nullopt;
}

// --- Tensor <-> flat float buffer, protobuf level ---------------------------

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

void SetFloatInitializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<float>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// --- Calibration: per-channel activation absmax ----------------------------
//
// Identical to smoothquant_entry.cpp's own ComputeChannelAbsmax.
std::unordered_map<std::string, std::vector<double>> ComputeChannelAbsmax(
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
            "ApplyFptq: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() != 2) {
        continue;
      }
      const int64_t k = tp.dims(1);
      if (k <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      auto& acc = result[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(k), 0.0);
      }
      for (int64_t flat = 0, total = static_cast<int64_t>(data.size());
           flat < total; ++flat) {
        const double v =
            std::abs(static_cast<double>(data[static_cast<size_t>(flat)]));
        double& slot = acc[static_cast<size_t>(flat % k)];
        if (v > slot) {
          slot = v;
        }
      }
    }
  }
  return result;
}

}  // namespace

onnx::ModelProto ApplyFptq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha, double outlier_ratio_threshold, double epsilon) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  struct Candidate {
    int node_index;
    std::string x_name;
    std::string w_name;
    bool weight_transposed;
  };
  std::vector<Candidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    auto m = MatchMatMulLike(graph->node(i));
    if (!m) {
      continue;
    }
    auto it = init_index.find(m->w_name);
    if (it == init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_init = graph->initializer(it->second);
    if (w_init.data_type() != onnx::TensorProto::FLOAT ||
        w_init.dims_size() != 2) {
      continue;
    }
    candidates.push_back({i, m->x_name, m->w_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  onnx::ModelProto probe_model = out;
  const std::unordered_map<std::string, std::vector<double>> act_absmax =
      ComputeChannelAbsmax(executor, probe_model, probe_names,
                           calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...) so freshly minted names match the Python
  // reference's own one-for-one.
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

  // Candidates are processed in forward node order (matching the Python
  // reference, whose _unique_name sequence depends on it). Each rewrite
  // inserts exactly one Mul node immediately before its own MatMul, so a
  // candidate's live index is its original index plus the number of Muls
  // already inserted by earlier candidates -- indices are never held across
  // mutations any other way (no NodeProto pointers survive an insert).
  int64_t insertions = 0;
  for (const auto& c : candidates) {
    auto acts_it = act_absmax.find(c.x_name);
    if (acts_it == act_absmax.end()) {
      continue;  // Never observed as a plain 2-D tensor; skip.
    }
    const std::vector<double>& acts = acts_it->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (acts.size() != static_cast<size_t>(k)) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // act_channel[j] = max(acts[j], eps); weight_channel[j] =
    // max(|w_nk[:, j]|.max(), eps) -- mirrors the Python reference's own
    // two lines exactly, computed in float64 throughout.
    std::vector<double> act_channel(static_cast<size_t>(k));
    std::vector<double> weight_channel(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      act_channel[static_cast<size_t>(j)] =
          std::max(acts[static_cast<size_t>(j)], epsilon);
      double wmax = 0.0;
      for (int64_t i = 0; i < n_rows; ++i) {
        const double v = std::abs(at_nk(i, j));
        if (v > wmax) {
          wmax = v;
        }
      }
      weight_channel[static_cast<size_t>(j)] = std::max(wmax, epsilon);
    }

    // ref = geometric mean of act_channel (exp(mean(log(.)))), floored at
    // eps; outlier_ratio = max(act_channel) / ref -- mirrors the Python
    // reference's own `ref`/`outlier_ratio` exactly.
    double log_sum = 0.0;
    double act_max = 0.0;
    for (double v : act_channel) {
      log_sum += std::log(v);
      act_max = std::max(act_max, v);
    }
    double ref = std::exp(log_sum / static_cast<double>(k));
    ref = std::max(ref, epsilon);
    const double outlier_ratio = act_max / ref;

    // s_j -- FPTQ's own logarithmic equalization on an "intractable"
    // layer (see fptq_entry.h's own header comment), SmoothQuant's own
    // power-law scale otherwise -- mirrors the Python reference's own
    // if/else exactly.
    std::vector<double> s(static_cast<size_t>(k));
    if (outlier_ratio >= outlier_ratio_threshold) {
      for (int64_t j = 0; j < k; ++j) {
        const double ratio = act_channel[static_cast<size_t>(j)] / ref;
        s[static_cast<size_t>(j)] = ref * std::log2(1.0 + ratio);
      }
    } else {
      for (int64_t j = 0; j < k; ++j) {
        s[static_cast<size_t>(j)] =
            std::pow(act_channel[static_cast<size_t>(j)], alpha) /
            std::pow(weight_channel[static_cast<size_t>(j)], 1.0 - alpha);
      }
    }
    for (int64_t j = 0; j < k; ++j) {
      s[static_cast<size_t>(j)] = std::max(s[static_cast<size_t>(j)], epsilon);
    }

    // Weight columns rescaled in place (same initializer, same position --
    // mirrors `w_init.CopyFrom(from_array(w_new, name=w_name))`), stored
    // float32 (mirrors `.astype(np.float32)`).
    std::vector<float> w_new(static_cast<size_t>(dim0 * dim1));
    for (int64_t i0 = 0; i0 < dim0; ++i0) {
      for (int64_t j0 = 0; j0 < dim1; ++j0) {
        const int64_t i = c.weight_transposed ? i0 : j0;
        const int64_t j = c.weight_transposed ? j0 : i0;
        const double v = at_nk(i, j) * s[static_cast<size_t>(j)];
        w_new[static_cast<size_t>(i0 * dim1 + j0)] = static_cast<float>(v);
      }
    }
    SetFloatInitializer(graph->mutable_initializer(init_index[c.w_name]),
                        c.w_name, {dim0, dim1}, w_new);

    // `1 / s` scale initializer (float32), then the Mul node inserted
    // immediately before the MatMul, whose activation input is rewired to
    // the scaled tensor -- mirrors apply_fptq's own naming
    // ("_fptq_inv_scale"/"_fptq_scaled"/"_fptq_mul") and unique_name call
    // order exactly.
    std::vector<float> inv_s(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      inv_s[static_cast<size_t>(j)] =
          static_cast<float>(1.0 / s[static_cast<size_t>(j)]);
    }
    const std::string scale_name = unique_name(c.x_name + "_fptq_inv_scale");
    SetFloatInitializer(graph->add_initializer(), scale_name, {k}, inv_s);
    const std::string scaled_name = unique_name(c.x_name + "_fptq_scaled");
    const std::string mul_name = unique_name(c.x_name + "_fptq_mul");

    // RepeatedPtrField<NodeProto>::Add() appends, then std::rotate moves it
    // into place -- InsertBeforeNode is not exposed on the generated API.
    // (No pointer into graph->node() is held across this: only indices.)
    const int live_index = c.node_index + static_cast<int>(insertions);
    onnx::NodeProto* mul = graph->add_node();
    int last = graph->node_size() - 1;
    for (int i = last; i > live_index; --i) {
      graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
    }
    mul = graph->mutable_node(live_index);
    mul->set_op_type("Mul");
    mul->add_input(c.x_name);
    mul->add_input(scale_name);
    mul->add_output(scaled_name);
    mul->set_name(mul_name);
    graph->mutable_node(live_index + 1)->set_input(0, scaled_name);
    ++insertions;
  }

  return out;
}
