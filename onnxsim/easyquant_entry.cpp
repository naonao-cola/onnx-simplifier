// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See easyquant_entry.h for the full rationale and onnxsim/easyquant.py
// for the technique this ports.
//
// MatchMatMulLike/InsertEmptyNodeAt/the taken_names+unique_name machinery
// are transcribed from llm_int8_entry.cpp's own identical helpers (see
// that file's own top-of-file comment for the underlying
// passes/quantize_matmul_common.h precedent) -- see gptaq_entry.cpp's own
// top-of-file comment for why they aren't shared via a common header
// (matches this codebase's established convention: every *_entry.cpp is
// self-contained).

#include "easyquant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

// --- Tensor <-> flat buffers, protobuf level --------------------------------

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

// --- Round-half-to-even and the plain W8A8 quantize-dequantize round trip --
//
// Mirrors onnxsim.easyquant's own `_quantize_round_trip` (numpy `round` is
// itself round-half-to-even) exactly: `clip(round(v / s), -127, 127) * s`.

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

double QuantizeRoundTrip(double v, double scale) {
  double q = RoundHalfToEven(v / scale);
  q = std::min(127.0, std::max(-127.0, q));
  return q * scale;
}

// --- EasyQuant's own coordinate-descent scale search ------------------------
//
// Mirrors onnxsim.easyquant's own `_search_scales` exactly -- see
// easyquant_entry.h's own top-of-file comment for the technique. `w_nk` is
// [N, K] (output channels first), `x` is [S, K] (S concatenated
// calibration rows).

struct ScaleSearchResult {
  std::vector<double> w_scale;  // [N]
  double a_scale = 0.0;
};

ScaleSearchResult SearchScales(const std::vector<std::vector<double>>& w_nk,
                               const std::vector<std::vector<double>>& x,
                               int64_t num_iterations, int64_t num_candidates,
                               double search_span) {
  const size_t n = w_nk.size();
  const size_t k = w_nk.empty() ? 0 : w_nk[0].size();
  const size_t s = x.size();
  constexpr double kEps = 1e-12;

  std::vector<double> base_w_scale(n);
  for (size_t i = 0; i < n; ++i) {
    double max_abs = 0.0;
    for (size_t j = 0; j < k; ++j) {
      max_abs = std::max(max_abs, std::fabs(w_nk[i][j]));
    }
    base_w_scale[i] = std::max(max_abs, kEps) / 127.0;
  }
  double max_abs_x = 0.0;
  for (const auto& row : x) {
    for (double v : row) {
      max_abs_x = std::max(max_abs_x, std::fabs(v));
    }
  }
  const double base_a_scale = std::max(max_abs_x, kEps) / 127.0;

  // y_float = x @ w_nk^T -- [S, N].
  std::vector<std::vector<double>> y_float(s, std::vector<double>(n, 0.0));
  for (size_t si = 0; si < s; ++si) {
    for (size_t ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (size_t ki = 0; ki < k; ++ki) {
        acc += x[si][ki] * w_nk[ni][ki];
      }
      y_float[si][ni] = acc;
    }
  }

  std::vector<double> multipliers;
  {
    const int64_t count = std::max<int64_t>(num_candidates, 1);
    for (int64_t i = 0; i < count; ++i) {
      const double t =
          count == 1 ? 0.0
                     : static_cast<double>(i) / static_cast<double>(count - 1);
      const double m = (1.0 - search_span) + t * (2.0 * search_span);
      if (m > 0.0) {
        multipliers.push_back(m);
      }
    }
  }

  ScaleSearchResult result;
  result.w_scale = base_w_scale;
  result.a_scale = base_a_scale;

  std::vector<std::vector<double>> x_q(s, std::vector<double>(k));
  std::vector<double> w_q_row(k);
  std::vector<double> y_q_col(s);
  std::vector<std::vector<double>> w_q(n, std::vector<double>(k));
  std::vector<std::vector<double>> x_q_try(s, std::vector<double>(k));

  for (int64_t iter = 0; iter < num_iterations; ++iter) {
    // --- Weight step: fix the activation scale, grid-search each output
    // channel's own weight scale independently.
    for (size_t si = 0; si < s; ++si) {
      for (size_t ki = 0; ki < k; ++ki) {
        x_q[si][ki] = QuantizeRoundTrip(x[si][ki], result.a_scale);
      }
    }
    for (size_t ni = 0; ni < n; ++ni) {
      bool have_best = false;
      double best_mse = 0.0;
      double best_scale = result.w_scale[ni];
      for (double m : multipliers) {
        const double cand_scale = base_w_scale[ni] * m;
        for (size_t ki = 0; ki < k; ++ki) {
          w_q_row[ki] = QuantizeRoundTrip(w_nk[ni][ki], cand_scale);
        }
        double sq_err = 0.0;
        for (size_t si = 0; si < s; ++si) {
          double acc = 0.0;
          for (size_t ki = 0; ki < k; ++ki) {
            acc += x_q[si][ki] * w_q_row[ki];
          }
          y_q_col[si] = acc;
          const double d = y_q_col[si] - y_float[si][ni];
          sq_err += d * d;
        }
        const double mse = s == 0 ? 0.0 : sq_err / static_cast<double>(s);
        if (!have_best || mse < best_mse) {
          have_best = true;
          best_mse = mse;
          best_scale = cand_scale;
        }
      }
      result.w_scale[ni] = best_scale;
    }

    // --- Activation step: fix the just-updated weight scale, grid-search
    // the single per-tensor activation scale against the whole output's
    // cosine similarity.
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t ki = 0; ki < k; ++ki) {
        w_q[ni][ki] = QuantizeRoundTrip(w_nk[ni][ki], result.w_scale[ni]);
      }
    }
    double sum_sq_y_float = 0.0;
    for (const auto& row : y_float) {
      for (double v : row) {
        sum_sq_y_float += v * v;
      }
    }
    const double y_float_norm = std::sqrt(sum_sq_y_float) + kEps;

    bool have_best_cos = false;
    double best_cos = 0.0;
    double best_a_scale = result.a_scale;
    for (double m : multipliers) {
      const double cand_scale = base_a_scale * m;
      for (size_t si = 0; si < s; ++si) {
        for (size_t ki = 0; ki < k; ++ki) {
          x_q_try[si][ki] = QuantizeRoundTrip(x[si][ki], cand_scale);
        }
      }
      double dot = 0.0;
      double sum_sq_y_q = 0.0;
      for (size_t si = 0; si < s; ++si) {
        for (size_t ni = 0; ni < n; ++ni) {
          double acc = 0.0;
          for (size_t ki = 0; ki < k; ++ki) {
            acc += x_q_try[si][ki] * w_q[ni][ki];
          }
          dot += acc * y_float[si][ni];
          sum_sq_y_q += acc * acc;
        }
      }
      const double norm_y_q = std::sqrt(sum_sq_y_q) + kEps;
      const double cos = dot / (norm_y_q * y_float_norm);
      if (!have_best_cos || cos > best_cos) {
        have_best_cos = true;
        best_cos = cos;
        best_a_scale = cand_scale;
      }
    }
    result.a_scale = best_a_scale;
  }

  return result;
}

// --- Node insertion / unique naming -----------------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers.

void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

}  // namespace

onnx::ModelProto ApplyEasyquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations, int64_t num_candidates, double search_span) {
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
    const onnx::NodeProto& node = graph->node(i);
    if (node.output_size() < 1) {
      continue;
    }
    auto m = MatchMatMulLike(node);
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

  // Probe once, per unique activation name -- mirrors apply_easyquant's
  // own `probe_names`/`activations` dict exactly.
  std::vector<std::string> probe_names;
  {
    std::unordered_set<std::string> seen;
    for (const auto& c : candidates) {
      if (seen.insert(c.x_name).second) {
        probe_names.push_back(c.x_name);
      }
    }
  }
  onnx::ModelProto probe_model = out;
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

  // Per-name accumulated [rows, k] activation parts, one entry per
  // observed batch (dropped, not zero-filled, when rank < 2) -- mirrors
  // onnxsim.bias_correction._activation_rows exactly.
  struct ActivationPart {
    int64_t rows = 0;
    int64_t k = 0;
    std::vector<double> data;
  };
  std::unordered_map<std::string, std::vector<ActivationPart>> parts_by_name;
  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> input_dls;
    std::vector<const DLManagedTensor*> input_ptrs;
    input_dls.reserve(static_cast<size_t>(graph_inputs.size()));
    input_ptrs.reserve(static_cast<size_t>(graph_inputs.size()));
    for (const auto& gi : graph_inputs) {
      auto bit = batch.find(gi.name());
      if (bit == batch.end()) {
        throw std::invalid_argument(
            "ApplyEasyquant: calibration batch is missing required graph "
            "input '" +
            gi.name() + "'");
      }
      input_dls.emplace_back(
          onnxsim::dlpack::FromTensorProtoBorrowing(bit->second));
      input_ptrs.push_back(input_dls.back().get());
    }
    std::vector<DLManagedTensorPtr> outputs =
        executor.Run(probe_model, input_ptrs);
    for (const auto& name : probe_names) {
      auto oit = output_index.find(name);
      if (oit == output_index.end() || oit->second >= outputs.size()) {
        continue;
      }
      const onnx::TensorProto tp =
          onnxsim::dlpack::ToTensorProto(outputs[oit->second]->dl_tensor);
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 2) {
        continue;
      }
      const int64_t k = tp.dims(tp.dims_size() - 1);
      if (k <= 0) {
        continue;
      }
      int64_t numel = 1;
      for (int64_t d : tp.dims()) {
        numel *= d;
      }
      ActivationPart part;
      part.rows = numel / k;
      part.k = k;
      const std::vector<float> flat = ReadFloatTensor(tp);
      part.data.assign(flat.begin(), flat.end());
      parts_by_name[name].push_back(std::move(part));
    }
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly.
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

  // Candidates are processed in forward node order, each insertion
  // shifting later live indices by the net nodes already added -- mirrors
  // llm_int8_entry.cpp's own `net_insertions` bookkeeping (here strictly
  // additive: EasyQuant never deletes the matched node itself, only
  // rewires its two inputs, so `net_insertions` only ever grows).
  int64_t net_insertions = 0;
  for (const auto& c : candidates) {
    auto pit = parts_by_name.find(c.x_name);
    const std::vector<ActivationPart>& parts =
        pit == parts_by_name.end() ? std::vector<ActivationPart>()
                                   : pit->second;
    if (parts.empty()) {
      continue;  // No usable activation (no feature axis); skip.
    }
    const int64_t k_feat = parts[0].k;
    bool k_ok = true;
    int64_t total_rows = 0;
    for (const auto& p : parts) {
      if (p.k != k_feat) {
        k_ok = false;
        break;
      }
      total_rows += p.rows;
    }
    if (!k_ok) {
      continue;
    }
    std::vector<std::vector<double>> x(static_cast<size_t>(total_rows));
    {
      size_t row = 0;
      for (const auto& p : parts) {
        for (int64_t r = 0; r < p.rows; ++r) {
          x[row].assign(p.data.begin() + r * k_feat,
                        p.data.begin() + (r + 1) * k_feat);
          ++row;
        }
      }
    }

    auto wit = init_index.find(c.w_name);
    if (wit == init_index.end()) {
      continue;  // Defensive -- already checked during candidate matching.
    }
    const onnx::TensorProto& w_init = graph->initializer(wit->second);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (k_feat != k) {
      continue;
    }
    const std::vector<float> w_flat = ReadFloatTensor(w_init);
    std::vector<std::vector<double>> w_nk(
        static_cast<size_t>(n_rows),
        std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(i * k + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + i)]);
      }
    }

    const ScaleSearchResult scales =
        SearchScales(w_nk, x, num_iterations, num_candidates, search_span);

    std::vector<double> w_quant_flat;
    w_quant_flat.reserve(static_cast<size_t>(dim0 * dim1));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          w_quant_flat.push_back(QuantizeRoundTrip(
              w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)],
              scales.w_scale[static_cast<size_t>(i)]));
        }
      }
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          w_quant_flat.push_back(QuantizeRoundTrip(
              w_nk[static_cast<size_t>(j)][static_cast<size_t>(i)],
              scales.w_scale[static_cast<size_t>(j)]));
        }
      }
    }

    const int64_t live_index = c.node_index + net_insertions;
    onnx::NodeProto* node = graph->mutable_node(static_cast<int>(live_index));

    const std::string new_w_name = unique_name(w_init.name() + "_easyquant");
    SetFloatInitializer(graph->add_initializer(), new_w_name, {dim0, dim1},
                        w_quant_flat);
    node->set_input(1, new_w_name);

    const std::string prefix = unique_name(node->output(0) + "_easyquant");
    const std::string scale_name = unique_name(prefix + "_act_scale");
    SetFloatInitializer(graph->add_initializer(), scale_name, {},
                        {scales.a_scale});
    const std::string neg_name = unique_name(prefix + "_neg127");
    SetFloatInitializer(graph->add_initializer(), neg_name, {}, {-127.0});
    const std::string pos_name = unique_name(prefix + "_pos127");
    SetFloatInitializer(graph->add_initializer(), pos_name, {}, {127.0});

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
    };
    std::vector<NewNode> new_nodes;
    auto add_op = [&](const std::string& op_type,
                      const std::vector<std::string>& inputs,
                      const std::string& tag) {
      NewNode n;
      n.op_type = op_type;
      n.inputs = inputs;
      n.output = unique_name(prefix + "_" + tag);
      n.name = unique_name(prefix + "_" + tag + "_node");
      new_nodes.push_back(n);
      return new_nodes.back().output;
    };
    const std::string scaled = add_op("Div", {c.x_name, scale_name}, "scaled");
    const std::string rounded = add_op("Round", {scaled}, "rounded");
    const std::string clipped =
        add_op("Clip", {rounded, neg_name, pos_name}, "clipped");
    const std::string dequant = add_op("Mul", {clipped, scale_name}, "dequant");
    node->set_input(0, dequant);

    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, static_cast<int>(live_index) + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* n =
          graph->mutable_node(static_cast<int>(live_index) + i);
      n->set_op_type(spec.op_type);
      for (const auto& in : spec.inputs) {
        n->add_input(in);
      }
      n->add_output(spec.output);
      n->set_name(spec.name);
    }
    net_insertions += count;
  }

  return out;
}
