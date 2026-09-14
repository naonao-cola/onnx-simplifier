// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See outlier_suppression_entry.h for the full rationale (including why
// this follows smoothquant_entry.h's own protobuf-level,
// calibration-driven shape) and onnxsim/outlier_suppression.py for the
// technique this ports.

#include "outlier_suppression_entry.h"

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
// Transcribed from smoothquant_entry.cpp's own MatchMatMulLike (which
// itself mirrors onnxsim.smoothquant._match_matmul_like, the exact matcher
// onnxsim.outlier_suppression reuses for its consumers): a MatMul, or a
// Gemm with transA=0, alpha=1 and (when it has a bias) beta=1. The bias
// itself, when present, is simply ignored -- this pass never reads or
// rewrites it, only the activation (input 0) and weight (input 1)
// operands.
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
//
// Transcribed from smoothquant_entry.cpp's own ReadFloatTensor/
// SetFloatInitializer (FLOAT32 only -- this pass, like its own Python
// reference onnxsim.outlier_suppression, never widens to FLOAT16/BFLOAT16),
// reusing dlpack_bridge.h's kRawDataIsHostOrder/SwapElementBytes for the
// raw_data little-endian convention every onnx::TensorProto.raw_data()
// must hold.

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

// --- Calibration: per-channel activation absmax, any rank >= 1 ------------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// smoothquant_entry.cpp's own ComputeChannelAbsmax, widened to exactly
// what onnxsim.outlier_suppression needs: the absmax reduced over every
// leading axis (`np.abs(x).max(axis=tuple(range(x.ndim - 1)))`), so any
// rank >= 1 resolves (a scalar never does) -- mirroring `if x.ndim < 1:
// continue`. Elementwise maximum across batches, FLOAT32 only.
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
            "ApplyOutlierSuppression: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 1) {
        continue;
      }
      const int64_t channels = tp.dims(static_cast<int>(tp.dims_size() - 1));
      if (channels <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      // First observation assigns (rather than max-ing against zero),
      // matching `m if name not in act_absmax else np.maximum(...)`.
      auto& acc = result[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(channels), 0.0);
      }
      for (int64_t flat = 0, total = static_cast<int64_t>(data.size());
           flat < total; ++flat) {
        const double v =
            std::abs(static_cast<double>(data[static_cast<size_t>(flat)]));
        double& slot = acc[static_cast<size_t>(flat % channels)];
        if (v > slot) {
          slot = v;
        }
      }
    }
  }
  return result;
}

}  // namespace

onnx::ModelProto ApplyOutlierSuppression(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha, double epsilon) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }
  std::unordered_set<std::string> graph_output_names;
  for (const auto& o : graph->output()) {
    graph_output_names.insert(o.name());
  }

  struct Consumer {
    std::string w_name;
    bool weight_transposed;
  };
  struct Candidate {
    std::string ln_out;
    std::string gamma_name;
    bool has_beta;
    std::string beta_name;
    int64_t channels;
    std::vector<Consumer> consumers;
  };
  std::vector<Candidate> candidates;
  for (int li = 0; li < graph->node_size(); ++li) {
    const onnx::NodeProto& ln = graph->node(li);
    if (ln.op_type() != "LayerNormalization" || ln.input_size() < 2 ||
        ln.output_size() < 1) {
      continue;
    }
    auto git = init_index.find(ln.input(1));
    if (git == init_index.end()) {
      continue;
    }
    const onnx::TensorProto& gamma_init = graph->initializer(git->second);
    if (gamma_init.data_type() != onnx::TensorProto::FLOAT ||
        gamma_init.dims_size() != 1) {
      continue;
    }
    const int64_t channels = gamma_init.dims(0);
    bool has_beta = false;
    std::string beta_name;
    // Mirrors `if len(ln.input) >= 3 and ln.input[2]:` -- an empty third
    // input is ONNX's "optional input not present", not a name to look up.
    if (ln.input_size() >= 3 && !ln.input(2).empty()) {
      auto bit = init_index.find(ln.input(2));
      if (bit == init_index.end()) {
        continue;  // Malformed/non-constant bias -- decline, don't guess.
      }
      const onnx::TensorProto& beta_init = graph->initializer(bit->second);
      // The Python reference checks dims only, not dtype; this port
      // additionally requires FLOAT (see the header's own scope note).
      if (beta_init.dims_size() != 1 || beta_init.dims(0) != channels ||
          beta_init.data_type() != onnx::TensorProto::FLOAT) {
        continue;
      }
      has_beta = true;
      beta_name = ln.input(2);
    }

    const std::string ln_out = ln.output(0);
    if (graph_output_names.count(ln_out) != 0) {
      continue;  // An external consumer would see the scaled-down value.
    }

    std::vector<Consumer> consumers;
    bool declined = false;
    for (int ni = 0; ni < graph->node_size(); ++ni) {
      const onnx::NodeProto& node = graph->node(ni);
      bool uses = false;
      for (const auto& s : node.input()) {
        if (s == ln_out) {
          uses = true;
          break;
        }
      }
      if (!uses) {
        continue;
      }
      auto m = MatchMatMulLike(node);
      if (!m) {
        declined = true;  // A non-MatMul/Gemm consumer -- untouched, not
                          // partially migrated.
        break;
      }
      if (m->x_name != ln_out) {
        declined = true;  // ln_out feeds a weight/bias slot, not the
                          // activation input.
        break;
      }
      auto wit = init_index.find(m->w_name);
      if (wit == init_index.end()) {
        declined = true;
        break;
      }
      const onnx::TensorProto& w_init = graph->initializer(wit->second);
      if (w_init.data_type() != onnx::TensorProto::FLOAT ||
          w_init.dims_size() != 2) {
        declined = true;
        break;
      }
      const int64_t k_dim =
          m->weight_transposed ? w_init.dims(1) : w_init.dims(0);
      if (k_dim != channels) {
        declined = true;
        break;
      }
      consumers.push_back({m->w_name, m->weight_transposed});
    }
    if (declined || consumers.empty()) {
      continue;
    }
    candidates.push_back({ln_out, ln.input(1), has_beta, beta_name, channels,
                          std::move(consumers)});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.ln_out);
  }
  onnx::ModelProto probe_model = out;
  const std::unordered_map<std::string, std::vector<double>> act_absmax =
      ComputeChannelAbsmax(executor, probe_model, probe_names,
                           calibration_data);

  // No new nodes or tensors are ever created here (the scale folds into
  // existing gamma/bias/weight initializers), so -- unlike ApplySmoothQuant
  // -- no taken-names bookkeeping and no index adjustment is needed: every
  // rewrite is in place and node indices never shift.
  for (const auto& c : candidates) {
    auto acts_it = act_absmax.find(c.ln_out);
    if (acts_it == act_absmax.end() ||
        acts_it->second.size() != static_cast<size_t>(c.channels)) {
      continue;  // Never observed, or a rank/shape mismatch; skip.
    }
    const std::vector<double>& acts = acts_it->second;
    const int64_t k = static_cast<int64_t>(acts.size());

    // weight_channel maximized over every compensated consumer's own
    // column max-abs (floored at epsilon), then the shared SmoothQuant
    // closed form -- mirrors the Python reference's own lines exactly,
    // computed in float64 throughout like its own astype(np.float64).
    std::vector<double> weight_channel(static_cast<size_t>(k), epsilon);
    for (const auto& cons : c.consumers) {
      const onnx::TensorProto& w_init =
          graph->initializer(init_index[cons.w_name]);
      const int64_t dim0 = w_init.dims(0);
      const int64_t dim1 = w_init.dims(1);
      const int64_t n_rows = cons.weight_transposed ? dim0 : dim1;
      const std::vector<float> flat = ReadFloatTensor(w_init);
      for (int64_t j = 0; j < k; ++j) {
        double wmax = 0.0;
        for (int64_t i = 0; i < n_rows; ++i) {
          // w_nk[i, j], row-major over [N, K]: the stored layout is
          // [N, K] itself when transposed, else [K, N].
          const float v = cons.weight_transposed
                              ? flat[static_cast<size_t>(i * k + j)]
                              : flat[static_cast<size_t>(j * n_rows + i)];
          const double a = std::abs(static_cast<double>(v));
          if (a > wmax) {
            wmax = a;
          }
        }
        double& slot = weight_channel[static_cast<size_t>(j)];
        if (wmax > slot) {
          slot = wmax;
        }
      }
    }
    std::vector<double> s(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      const double a = std::max(acts[static_cast<size_t>(j)], epsilon);
      const double w = weight_channel[static_cast<size_t>(j)];
      double sj = std::pow(a, alpha) / std::pow(w, 1.0 - alpha);
      s[static_cast<size_t>(j)] = std::max(sj, epsilon);
    }

    // gamma /= s (and bias /= s when present), in place under the same
    // names -- mirrors `gamma_init.CopyFrom(from_array(..., name=...))`.
    {
      const onnx::TensorProto& gamma_init =
          graph->initializer(init_index[c.gamma_name]);
      const std::vector<float> flat = ReadFloatTensor(gamma_init);
      std::vector<float> gamma_new(flat.size());
      for (size_t i = 0; i < flat.size(); ++i) {
        gamma_new[i] = static_cast<float>(static_cast<double>(flat[i]) /
                                          s[i % static_cast<size_t>(k)]);
      }
      SetFloatInitializer(graph->mutable_initializer(init_index[c.gamma_name]),
                          c.gamma_name, {k}, gamma_new);
    }
    if (c.has_beta) {
      const onnx::TensorProto& beta_init =
          graph->initializer(init_index[c.beta_name]);
      const std::vector<float> flat = ReadFloatTensor(beta_init);
      std::vector<float> beta_new(flat.size());
      for (size_t i = 0; i < flat.size(); ++i) {
        beta_new[i] = static_cast<float>(static_cast<double>(flat[i]) /
                                         s[i % static_cast<size_t>(k)]);
      }
      SetFloatInitializer(graph->mutable_initializer(init_index[c.beta_name]),
                          c.beta_name, {k}, beta_new);
    }

    // Every compensated consumer's weight rows *= s, in place.
    for (const auto& cons : c.consumers) {
      const onnx::TensorProto& w_init =
          graph->initializer(init_index[cons.w_name]);
      const int64_t dim0 = w_init.dims(0);
      const int64_t dim1 = w_init.dims(1);
      const std::vector<float> flat = ReadFloatTensor(w_init);
      std::vector<float> w_new(flat.size());
      for (int64_t i0 = 0; i0 < dim0; ++i0) {
        for (int64_t j0 = 0; j0 < dim1; ++j0) {
          // s indexes the reduction (K) axis: j0 for a transposed [N, K]
          // weight, i0 for a plain [K, N] one.
          const int64_t j = cons.weight_transposed ? j0 : i0;
          const double v =
              static_cast<double>(flat[static_cast<size_t>(i0 * dim1 + j0)]);
          w_new[static_cast<size_t>(i0 * dim1 + j0)] =
              static_cast<float>(v * s[static_cast<size_t>(j)]);
        }
      }
      SetFloatInitializer(graph->mutable_initializer(init_index[cons.w_name]),
                          cons.w_name, {dim0, dim1}, w_new);
    }
  }

  return out;
}
