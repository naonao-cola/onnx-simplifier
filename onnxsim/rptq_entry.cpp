// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See rptq_entry.h for the full rationale (including why this follows
// smoothquant_entry.h's own protobuf-level, calibration-driven shape) and
// onnxsim/rptq.py for the technique this ports.

#include "rptq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
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
// Transcribed from smoothquant_entry.cpp's own MatchMatMulLike, exactly
// what onnxsim.rptq._match_matmul_like (== onnxsim.smoothquant's own) also
// checks.
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

// --- Tensor <-> flat buffers, protobuf level -------------------------------
//
// Transcribed from smoothquant_entry.cpp's own identical helpers (FLOAT32
// only -- this pass, like its own Python reference onnxsim.rptq, never
// widens to FLOAT16/BFLOAT16).

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

void SetInt64Initializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(int64_t), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(int64_t));
  }
  t->set_raw_data(std::move(raw));
}

// --- Calibration: per-channel activation absmax ----------------------------
//
// Transcribed verbatim from smoothquant_entry.cpp's own
// ComputeChannelAbsmax -- rptq.py's own per-channel abs-max capture is the
// exact same shape (`if x.ndim != 2: continue`, elementwise-max across
// batches).
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
            "ApplyRptqReorder: calibration batch is missing "
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

// --- K-means (1-D) ----------------------------------------------------------
//
// Transcribed from rptq.py's own _kmeans_1d: percentile-seeded Lloyd's
// algorithm. See rptq_entry.h's own top-of-file "ACCEPTED, PERMANENT
// DIVERGENCE" note for why this port's RNG (used only to jitter apart
// duplicate seeded centroids) does not reproduce numpy's own PCG64/
// Generator.normal bit-for-bit.
std::vector<int64_t> KMeans1D(const std::vector<double>& values,
                              int64_t num_clusters, std::mt19937_64& rng,
                              int64_t num_iters = 50) {
  const int64_t n = static_cast<int64_t>(values.size());
  const int64_t k = std::min(num_clusters, n);

  std::vector<double> sorted_vals = values;
  std::sort(sorted_vals.begin(), sorted_vals.end());
  // numpy.percentile's own default "linear" interpolation, over
  // np.linspace(0, 100, k) percentile points -- mirrors _kmeans_1d exactly.
  auto percentile = [&](double p) -> double {
    if (n == 1) {
      return sorted_vals[0];
    }
    const double index = (p / 100.0) * static_cast<double>(n - 1);
    const int64_t lower = static_cast<int64_t>(std::floor(index));
    const int64_t upper = static_cast<int64_t>(std::ceil(index));
    if (lower == upper) {
      return sorted_vals[static_cast<size_t>(lower)];
    }
    const double frac = index - static_cast<double>(lower);
    return sorted_vals[static_cast<size_t>(lower)] +
           frac * (sorted_vals[static_cast<size_t>(upper)] -
                   sorted_vals[static_cast<size_t>(lower)]);
  };

  std::vector<double> centroids(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    const double p =
        k > 1 ? 100.0 * static_cast<double>(i) / static_cast<double>(k - 1)
              : 0.0;
    centroids[static_cast<size_t>(i)] = percentile(p);
  }
  // Break ties from duplicate percentile values, same nudge rptq.py uses.
  for (int64_t i = 1; i < k; ++i) {
    if (centroids[static_cast<size_t>(i)] <=
        centroids[static_cast<size_t>(i - 1)]) {
      centroids[static_cast<size_t>(i)] =
          centroids[static_cast<size_t>(i - 1)] +
          1e-9 * (1.0 + std::abs(centroids[static_cast<size_t>(i - 1)]));
    }
  }
  double max_abs_centroid = 0.0;
  for (double c : centroids) {
    max_abs_centroid = std::max(max_abs_centroid, std::abs(c));
  }
  const double jitter_scale = 1e-9 * (1.0 + max_abs_centroid);
  std::normal_distribution<double> jitter(0.0, jitter_scale);
  for (double& c : centroids) {
    c += jitter(rng);
  }

  std::vector<int64_t> assignments(static_cast<size_t>(n), 0);
  for (int64_t iter = 0; iter < num_iters; ++iter) {
    std::vector<int64_t> new_assignments(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      int64_t best = 0;
      double best_dist =
          std::abs(values[static_cast<size_t>(i)] - centroids[0]);
      for (int64_t c = 1; c < k; ++c) {
        const double dist = std::abs(values[static_cast<size_t>(i)] -
                                     centroids[static_cast<size_t>(c)]);
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      new_assignments[static_cast<size_t>(i)] = best;
    }
    const bool unchanged = (iter > 0) && (new_assignments == assignments);
    assignments = std::move(new_assignments);
    if (unchanged) {
      break;
    }
    std::vector<double> sum(static_cast<size_t>(k), 0.0);
    std::vector<int64_t> count(static_cast<size_t>(k), 0);
    for (int64_t i = 0; i < n; ++i) {
      const int64_t c = assignments[static_cast<size_t>(i)];
      sum[static_cast<size_t>(c)] += values[static_cast<size_t>(i)];
      count[static_cast<size_t>(c)] += 1;
    }
    for (int64_t c = 0; c < k; ++c) {
      if (count[static_cast<size_t>(c)] > 0) {
        centroids[static_cast<size_t>(c)] =
            sum[static_cast<size_t>(c)] /
            static_cast<double>(count[static_cast<size_t>(c)]);
      }
    }
  }
  return assignments;
}

// Stable sort of channel indices by cluster id, plus the resulting
// permuted order's own per-cluster [start, end) bounds. Mirrors rptq.py's
// own _reorder_permutation exactly.
void ReorderPermutation(const std::vector<int64_t>& assignments,
                        std::vector<int64_t>* perm,
                        std::vector<std::pair<int64_t, int64_t>>* bounds) {
  const int64_t n = static_cast<int64_t>(assignments.size());
  perm->resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    (*perm)[static_cast<size_t>(i)] = i;
  }
  std::stable_sort(perm->begin(), perm->end(), [&](int64_t a, int64_t b) {
    return assignments[static_cast<size_t>(a)] <
           assignments[static_cast<size_t>(b)];
  });
  bounds->clear();
  int64_t start = 0;
  for (int64_t i = 1; i <= n; ++i) {
    if (i == n ||
        assignments[static_cast<size_t>((*perm)[static_cast<size_t>(i)])] !=
            assignments[static_cast<size_t>(
                (*perm)[static_cast<size_t>(start)])]) {
      bounds->emplace_back(start, i);
      start = i;
    }
  }
}

void AddIntAttribute(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

}  // namespace

RptqReorderResult ApplyRptqReorder(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t seed, int64_t num_clusters) {
  RptqReorderResult result;
  result.model = model;
  onnx::GraphProto* graph = result.model.mutable_graph();

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
    return result;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  onnx::ModelProto probe_model = result.model;
  const std::unordered_map<std::string, std::vector<double>> act_absmax =
      ComputeChannelAbsmax(executor, probe_model, probe_names,
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

  // A single sequentially-advancing RNG across every matched layer, in
  // graph node order -- mirrors rptq.py's own `rng = np.random.default_rng
  // (seed)` created once outside the loop (see this header's own top-of-
  // -file divergence note for why this doesn't reproduce numpy's PCG64
  // bit-for-bit).
  std::mt19937_64 rng(static_cast<uint64_t>(seed));

  int64_t insertions = 0;
  for (const auto& c : candidates) {
    auto acts_it = act_absmax.find(c.x_name);
    if (acts_it == act_absmax.end()) {
      continue;  // Never observed as a plain 2-D tensor; skip.
    }
    const std::vector<double>& absmax = acts_it->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (absmax.size() != static_cast<size_t>(k)) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }

    const std::vector<int64_t> assignments =
        KMeans1D(absmax, num_clusters, rng);
    std::vector<int64_t> perm;
    std::vector<std::pair<int64_t, int64_t>> bounds;
    ReorderPermutation(assignments, &perm, &bounds);

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // w_permuted_nk = w_nk[:, perm] -- mirrors `w_nk[:, perm]` exactly,
    // then reshaped back to the original stored layout.
    std::vector<float> w_new(static_cast<size_t>(dim0 * dim1));
    for (int64_t i0 = 0; i0 < dim0; ++i0) {
      for (int64_t j0 = 0; j0 < dim1; ++j0) {
        const int64_t i = c.weight_transposed ? i0 : j0;
        const int64_t j_dst = c.weight_transposed ? j0 : i0;
        const int64_t j_src = perm[static_cast<size_t>(j_dst)];
        const double v = at_nk(i, j_src);
        w_new[static_cast<size_t>(i0 * dim1 + j0)] = static_cast<float>(v);
      }
    }
    SetFloatInitializer(graph->mutable_initializer(init_index[c.w_name]),
                        c.w_name, {dim0, dim1}, w_new);

    const std::string perm_name = unique_name(c.x_name + "_rptq_perm");
    SetInt64Initializer(graph->add_initializer(), perm_name, {k}, perm);

    const std::string gathered_name = unique_name(c.x_name + "_rptq_reordered");
    const std::string gather_name = unique_name(c.x_name + "_rptq_gather");

    const int live_index = c.node_index + static_cast<int>(insertions);
    onnx::NodeProto* gather = graph->add_node();
    int last = graph->node_size() - 1;
    for (int i = last; i > live_index; --i) {
      graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
    }
    gather = graph->mutable_node(live_index);
    gather->set_op_type("Gather");
    gather->add_input(c.x_name);
    gather->add_input(perm_name);
    gather->add_output(gathered_name);
    gather->set_name(gather_name);
    AddIntAttribute(gather, "axis", -1);
    graph->mutable_node(live_index + 1)->set_input(0, gathered_name);
    ++insertions;

    RptqLayerInfo info;
    info.x_name = c.x_name;
    info.w_name = c.w_name;
    info.gather_output = gathered_name;
    info.permutation = perm;
    info.cluster_bounds = bounds;
    result.layers.push_back(std::move(info));
  }

  return result;
}
