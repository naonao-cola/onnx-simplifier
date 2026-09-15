// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See gptvq_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own protobuf-level, calibration-driven shape) and
// onnxsim/gptvq.py for the technique this ports.

#include "gptvq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike, narrowed to
// exactly what onnxsim.gptvq._match_matmul_like needs: no bias name (this
// pass never touches the node's bias input).
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

// --- Tensor <-> flat float buffer -------------------------------------------
//
// Transcribed from gptq_entry.cpp's own ReadFloatTensor (FLOAT32 only,
// mirroring this pass's own FLOAT-only weight/activation scope).
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

// --- Small dense double-precision linear algebra ----------------------------
//
// Transcribed verbatim from gptq_entry.cpp's own kernels -- see that
// file's own comment for why scalar kernels rather than LAPACK, and the
// accepted numerical scope that follows from it.
using Matrix = std::vector<std::vector<double>>;

Matrix CholeskyLower(const Matrix& a) {
  const size_t n = a.size();
  Matrix l(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double s = a[i][j];
      for (size_t k = 0; k < j; ++k) {
        s -= l[i][k] * l[j][k];
      }
      if (i == j) {
        l[i][j] = std::sqrt(std::max(s, 1e-24));
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return l;
}

Matrix InverseSPD(const Matrix& a) {
  const size_t n = a.size();
  const Matrix l = CholeskyLower(a);
  Matrix inv(n, std::vector<double>(n, 0.0));
  std::vector<double> y(n), x(n);
  for (size_t c = 0; c < n; ++c) {
    for (size_t i = 0; i < n; ++i) {
      double s = (i == c) ? 1.0 : 0.0;
      for (size_t k = 0; k < i; ++k) {
        s -= l[i][k] * y[k];
      }
      y[i] = s / l[i][i];
    }
    for (size_t i = n; i-- > 0;) {
      double s = y[i];
      for (size_t k = i + 1; k < n; ++k) {
        s -= l[k][i] * x[k];
      }
      x[i] = s / l[i][i];
    }
    for (size_t i = 0; i < n; ++i) {
      inv[i][c] = x[i];
    }
  }
  return inv;
}

Matrix InverseHessianCholesky(const Matrix& h, double percdamp) {
  const size_t k = h.size();
  Matrix damped = h;
  double diag_sum = 0.0;
  for (size_t i = 0; i < k; ++i) {
    if (damped[i][i] == 0.0) {
      damped[i][i] = 1.0;
    }
    diag_sum += damped[i][i];
  }
  const double damp =
      std::max(percdamp * diag_sum / static_cast<double>(k), 1e-8);
  for (size_t i = 0; i < k; ++i) {
    damped[i][i] += damp;
  }
  const Matrix h_inv = InverseSPD(damped);
  const Matrix l = CholeskyLower(h_inv);
  Matrix u(k, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = i; j < k; ++j) {
      u[i][j] = l[j][i];
    }
  }
  return u;
}

// --- k-means codebook fitting -----------------------------------------------
//
// Transcribed from onnxsim.aqlm's own _fit_kmeans_codebook (Python):
// ordinary (unweighted) Lloyd's algorithm, centroids initialized from a
// random sample of `data`'s own rows (without replacement; padded by
// repeating the last sampled point if `data` has fewer rows than
// `codebook_size`); an empty cluster keeps its previous centroid rather
// than going undefined. ACCEPTED, PERMANENT DIVERGENCE from the
// reference's own sampling (numpy.random.Generator.choice): this uses a
// partial Fisher-Yates shuffle over std::mt19937_64 instead -- both are
// valid without-replacement samples, not expected to align for the same
// seed (see gptvq_entry.h's own top-of-file note).
Matrix FitKmeansCodebook(const Matrix& data, int64_t codebook_size,
                         int64_t num_iterations, std::mt19937_64& rng) {
  const int64_t num_points = static_cast<int64_t>(data.size());
  const int64_t dim = static_cast<int64_t>(data[0].size());
  const int64_t k = std::min(codebook_size, num_points);

  std::vector<int64_t> order(static_cast<size_t>(num_points));
  for (int64_t i = 0; i < num_points; ++i) {
    order[static_cast<size_t>(i)] = i;
  }
  for (int64_t i = 0; i < k; ++i) {
    std::uniform_int_distribution<int64_t> dist(i, num_points - 1);
    const int64_t j = dist(rng);
    std::swap(order[static_cast<size_t>(i)], order[static_cast<size_t>(j)]);
  }

  Matrix centroids(static_cast<size_t>(codebook_size),
                   std::vector<double>(static_cast<size_t>(dim)));
  for (int64_t c = 0; c < k; ++c) {
    centroids[static_cast<size_t>(c)] =
        data[static_cast<size_t>(order[static_cast<size_t>(c)])];
  }
  for (int64_t c = k; c < codebook_size; ++c) {
    centroids[static_cast<size_t>(c)] = centroids[static_cast<size_t>(k - 1)];
  }

  std::vector<int64_t> assignment(static_cast<size_t>(num_points), 0);
  for (int64_t iter = 0; iter < num_iterations; ++iter) {
    for (int64_t p = 0; p < num_points; ++p) {
      int64_t best = 0;
      double best_dist = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < codebook_size; ++c) {
        double dist = 0.0;
        for (int64_t d = 0; d < dim; ++d) {
          const double diff =
              data[static_cast<size_t>(p)][static_cast<size_t>(d)] -
              centroids[static_cast<size_t>(c)][static_cast<size_t>(d)];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      assignment[static_cast<size_t>(p)] = best;
    }

    Matrix sum(static_cast<size_t>(codebook_size),
               std::vector<double>(static_cast<size_t>(dim), 0.0));
    std::vector<int64_t> count(static_cast<size_t>(codebook_size), 0);
    for (int64_t p = 0; p < num_points; ++p) {
      const int64_t c = assignment[static_cast<size_t>(p)];
      count[static_cast<size_t>(c)] += 1;
      for (int64_t d = 0; d < dim; ++d) {
        sum[static_cast<size_t>(c)][static_cast<size_t>(d)] +=
            data[static_cast<size_t>(p)][static_cast<size_t>(d)];
      }
    }
    for (int64_t c = 0; c < codebook_size; ++c) {
      if (count[static_cast<size_t>(c)] == 0) {
        continue;  // Keep the previous centroid, mirroring the reference.
      }
      for (int64_t d = 0; d < dim; ++d) {
        centroids[static_cast<size_t>(c)][static_cast<size_t>(d)] =
            sum[static_cast<size_t>(c)][static_cast<size_t>(d)] /
            static_cast<double>(count[static_cast<size_t>(c)]);
      }
    }
  }
  return centroids;
}

// --- GPTVQ's Hessian-compensated group quantization -------------------------
//
// Transcribed from onnxsim.gptvq's own _gptvq_quantize_groups: for every
// row of w_nk ([N, K], output channel first) and every consecutive
// vector_dim-wide column group, assigns the group to its nearest fixed
// codebook entry, then propagates the resulting per-column residual into
// every not-yet-quantized column via h's Cholesky-based inverse (see
// InverseHessianCholesky above) -- exactly gptq_entry.cpp's own
// GptqQuantizeColumns loop, except the value each column is corrected
// towards comes from a joint, whole-group codebook lookup rather than an
// independently-rounded scalar.
std::vector<std::vector<int64_t>> GptvqQuantizeGroups(const Matrix& w_nk,
                                                      const Matrix& codebook,
                                                      const Matrix& h,
                                                      double percdamp,
                                                      int64_t vector_dim) {
  const int64_t n = static_cast<int64_t>(w_nk.size());
  const int64_t k = static_cast<int64_t>(w_nk[0].size());
  const int64_t num_groups = k / vector_dim;
  const int64_t num_centroids = static_cast<int64_t>(codebook.size());
  const Matrix hinv = InverseHessianCholesky(h, percdamp);

  std::vector<std::vector<int64_t>> codes(
      static_cast<size_t>(n),
      std::vector<int64_t>(static_cast<size_t>(num_groups), 0));
  Matrix w_work = w_nk;

  for (int64_t g = 0; g < num_groups; ++g) {
    const int64_t col_start = g * vector_dim;

    for (int64_t r = 0; r < n; ++r) {
      int64_t best = 0;
      double best_dist = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < num_centroids; ++c) {
        double dist = 0.0;
        for (int64_t d = 0; d < vector_dim; ++d) {
          const double diff =
              w_work[static_cast<size_t>(r)]
                    [static_cast<size_t>(col_start + d)] -
              codebook[static_cast<size_t>(c)][static_cast<size_t>(d)];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best = c;
        }
      }
      codes[static_cast<size_t>(r)][static_cast<size_t>(g)] = best;

      for (int64_t j = 0; j < vector_dim; ++j) {
        const int64_t col = col_start + j;
        const double q =
            codebook[static_cast<size_t>(best)][static_cast<size_t>(j)];
        const double d =
            hinv[static_cast<size_t>(col)][static_cast<size_t>(col)];
        const double err =
            (w_work[static_cast<size_t>(r)][static_cast<size_t>(col)] - q) / d;
        w_work[static_cast<size_t>(r)][static_cast<size_t>(col)] = q;
        if (col + 1 < k) {
          for (int64_t jj = col + 1; jj < k; ++jj) {
            w_work[static_cast<size_t>(r)][static_cast<size_t>(jj)] -=
                err * hinv[static_cast<size_t>(col)][static_cast<size_t>(jj)];
          }
        }
      }
    }
  }
  return codes;
}

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own ActivationRows/
// AccumulateActivationRows: every observed 2-D-or-higher FLOAT32 activation
// flattened to rows (`reshape(-1, K)`, exact) and concatenated across
// batches. Rank < 2 resolves to no rows at all.
struct ActivationRows {
  std::vector<double> data;  // Concatenated [total_rows, K], row-major.
  int64_t k = -1;
  bool ok = false;
};

void AccumulateActivationRows(
    std::unordered_map<std::string, ActivationRows>& acc,
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
            "ApplyGptvq: calibration batch is missing "
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
        continue;
      }
      const DLTensor& dl = outputs[oit->second]->dl_tensor;
      onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 2) {
        continue;
      }
      const int64_t kk = tp.dims(static_cast<int>(tp.dims_size() - 1));
      if (kk <= 0) {
        continue;
      }
      ActivationRows& rows = acc[name];
      if (!rows.ok) {
        rows.k = kk;
        rows.ok = true;
      } else if (rows.k != kk) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      rows.data.reserve(rows.data.size() + data.size());
      for (float v : data) {
        rows.data.push_back(static_cast<double>(v));
      }
    }
  }
}

void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

void AddIntAttribute(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

void AddIntsAttribute(onnx::NodeProto* node, const std::string& name,
                      const std::vector<int64_t>& values) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INTS);
  for (int64_t v : values) {
    attr->add_ints(v);
  }
}

}  // namespace

onnx::ModelProto ApplyGptvq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    uint64_t seed, int64_t vector_dim, int64_t num_centroids,
    int64_t num_iterations, double percdamp,
    const std::unordered_set<std::string>& skip_names) {
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
    if (skip_names.count(m->w_name)) {
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
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, model, probe_names,
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

  int64_t net_insertions = 0;
  for (const auto& c : candidates) {
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable calibration activation -- leave untouched.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (k % vector_dim != 0 || rows.k != k) {
      continue;
    }
    const int64_t num_rows =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
                std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(r * k + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + r)]);
      }
    }

    Matrix h(static_cast<size_t>(k),
             std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi = rows.data[static_cast<size_t>(r * k + i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * rows.data[static_cast<size_t>(r * k + j)];
        }
      }
    }

    // Groups: w_nk's own row-major flatten split into vector_dim-wide
    // chunks -- mirrors `w_nk.reshape(num_groups, vector_dim)` exactly
    // (K a multiple of vector_dim, so a group never spans two rows).
    const int64_t num_groups_per_row = k / vector_dim;
    const int64_t num_groups = n_rows * num_groups_per_row;
    Matrix groups(static_cast<size_t>(num_groups),
                  std::vector<double>(static_cast<size_t>(vector_dim)));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t g = 0; g < num_groups_per_row; ++g) {
        for (int64_t d = 0; d < vector_dim; ++d) {
          groups[static_cast<size_t>(r * num_groups_per_row + g)]
                [static_cast<size_t>(d)] =
                    w_nk[static_cast<size_t>(r)]
                        [static_cast<size_t>(g * vector_dim + d)];
        }
      }
    }

    // ACCEPTED, PERMANENT DIVERGENCE from gptvq.py's own RNG derivation
    // (see gptvq_entry.h's own identical note): a fresh std::mt19937_64
    // reseeded per matched node, rather than a single
    // numpy.random.Generator sequenced across matches.
    std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ULL *
                                (static_cast<uint64_t>(c.node_index) + 1)));
    const Matrix codebook =
        FitKmeansCodebook(groups, num_centroids, num_iterations, rng);

    const std::vector<std::vector<int64_t>> codes_nk =
        GptvqQuantizeGroups(w_nk, codebook, h, percdamp, vector_dim);

    std::vector<float> codebook_flat(
        static_cast<size_t>(num_centroids * vector_dim));
    for (int64_t cc = 0; cc < num_centroids; ++cc) {
      for (int64_t d = 0; d < vector_dim; ++d) {
        codebook_flat[static_cast<size_t>(cc * vector_dim + d)] =
            static_cast<float>(
                codebook[static_cast<size_t>(cc)][static_cast<size_t>(d)]);
      }
    }
    std::vector<int64_t> codes_flat(
        static_cast<size_t>(n_rows * num_groups_per_row));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t g = 0; g < num_groups_per_row; ++g) {
        codes_flat[static_cast<size_t>(r * num_groups_per_row + g)] =
            codes_nk[static_cast<size_t>(r)][static_cast<size_t>(g)];
      }
    }

    const std::string prefix = c.w_name + "_gptvq";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string codebook_name =
        add_const("codebook", onnx::TensorProto::FLOAT,
                  {num_centroids, vector_dim}, codebook_flat.data(),
                  codebook_flat.size() * sizeof(float), sizeof(float));
    const std::string codes_name =
        add_const("codes", onnx::TensorProto::INT64,
                  {n_rows, num_groups_per_row}, codes_flat.data(),
                  codes_flat.size() * sizeof(int64_t), sizeof(int64_t));
    const std::vector<int64_t> shape_vals = {n_rows, k};
    const std::string shape_name =
        add_const("shape", onnx::TensorProto::INT64, {2}, shape_vals.data(),
                  shape_vals.size() * sizeof(int64_t), sizeof(int64_t));

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
    };
    std::vector<NewNode> new_nodes;
    auto add_node = [&](const std::string& op_type,
                        const std::vector<std::string>& inputs,
                        const std::string& out_suffix) {
      NewNode n;
      n.op_type = op_type;
      n.inputs = inputs;
      n.output = unique_name(prefix + "_" + out_suffix);
      n.name = unique_name(prefix + "_" + out_suffix + "_node");
      const std::string output = n.output;
      new_nodes.push_back(std::move(n));
      return output;
    };

    const std::string gathered =
        add_node("Gather", {codebook_name, codes_name}, "gathered");
    const std::string unblocked =
        add_node("Reshape", {gathered, shape_name}, "unblocked");
    std::string final_name = unblocked;
    if (!c.weight_transposed) {
      final_name = add_node("Transpose", {unblocked}, "transposed");
    }

    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, live_index + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* n = graph->mutable_node(live_index + i);
      n->set_op_type(spec.op_type);
      for (const auto& s : spec.inputs) {
        n->add_input(s);
      }
      n->add_output(spec.output);
      n->set_name(spec.name);
      if (spec.op_type == "Gather") {
        AddIntAttribute(n, "axis", 0);
      } else if (spec.op_type == "Transpose") {
        AddIntsAttribute(n, "perm", {1, 0});
      }
    }
    net_insertions += count;

    onnx::NodeProto* target = graph->mutable_node(live_index + count);
    for (int i = 0; i < target->input_size(); ++i) {
      if (target->input(i) == c.w_name) {
        target->set_input(i, final_name);
      }
    }
  }

  return out;
}
