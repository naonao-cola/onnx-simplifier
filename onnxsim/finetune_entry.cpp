// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See finetune_entry.h for the full rationale (including why this
// follows gptq_entry.h's own two-model, protobuf-level, calibration-
// driven shape while using an unrelated closed-form ridge-regression
// solve rather than any Hessian-compensated rounding scheme) and
// onnxsim/finetune.py for the technique this ports.

#include "finetune_entry.h"

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

// --- MatMul/vanilla-Gemm matching, protobuf level -----------------------
//
// Transcribed from onnxsim.smoothquant's own _match_matmul_like (which
// finetune.py itself imports and reuses) -- byte-identical to
// bwa_ptq_entry.cpp's own MatchMatMulLike; duplicated here per this
// codebase's established "no shared dependency between independently
// tested *_entry.cpp TUs" convention.
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

// --- Tensor <-> flat float buffer, protobuf level -----------------------

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

void SetRawInitializer(onnx::TensorProto* t, int32_t data_type,
                       const std::vector<int64_t>& dims, const void* data,
                       size_t bytes, size_t elem_size) {
  const std::string name = t->name();
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

// --- Candidate matching (two-model join by node output name) -----------
//
// Direct transcription of finetune.py's own _find_matmul_finetune_candidates.

struct Candidate {
  std::string x_name;                 // Into original_model.
  const onnx::TensorProto* w_orig;    // Into original_model's initializers.
  const onnx::TensorProto* w_pruned;  // Into pruned_model's initializers.
  const onnx::TensorProto* b_orig;    // Nullable.
  const onnx::TensorProto* b_pruned;  // Nullable.
  bool weight_transposed;
};

std::vector<Candidate> FindMatmulFinetuneCandidates(
    const onnx::ModelProto& original_model,
    const onnx::ModelProto& pruned_model) {
  const onnx::GraphProto& og = original_model.graph();
  const onnx::GraphProto& pg = pruned_model.graph();

  std::unordered_map<std::string, int> orig_by_output;
  for (int i = 0; i < og.node_size(); ++i) {
    if (og.node(i).output_size() > 0) {
      orig_by_output[og.node(i).output(0)] = i;
    }
  }
  std::unordered_map<std::string, int> pruned_by_output;
  for (int i = 0; i < pg.node_size(); ++i) {
    if (pg.node(i).output_size() > 0) {
      pruned_by_output[pg.node(i).output(0)] = i;
    }
  }
  std::unordered_map<std::string, const onnx::TensorProto*> orig_init;
  for (const auto& t : og.initializer()) {
    orig_init[t.name()] = &t;
  }
  std::unordered_map<std::string, const onnx::TensorProto*> pruned_init;
  for (const auto& t : pg.initializer()) {
    pruned_init[t.name()] = &t;
  }

  std::vector<Candidate> candidates;
  for (const auto& [out_name, pi] : pruned_by_output) {
    const onnx::NodeProto& pn = pg.node(pi);
    auto oit = orig_by_output.find(out_name);
    if (oit == orig_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& on = og.node(oit->second);
    if (on.op_type() != pn.op_type()) {
      continue;
    }
    auto p_match = MatchMatMulLike(pn);
    auto o_match = MatchMatMulLike(on);
    if (!p_match || !o_match) {
      continue;
    }
    if (p_match->weight_transposed != o_match->weight_transposed) {
      continue;
    }

    auto wp_it = pruned_init.find(p_match->w_name);
    auto wo_it = orig_init.find(o_match->w_name);
    if (wp_it == pruned_init.end() || wo_it == orig_init.end()) {
      continue;
    }
    const onnx::TensorProto* w_pruned = wp_it->second;
    const onnx::TensorProto* w_orig = wo_it->second;
    if (w_pruned->data_type() != onnx::TensorProto::FLOAT ||
        w_orig->data_type() != onnx::TensorProto::FLOAT ||
        w_pruned->dims_size() != 2 || w_orig->dims_size() != 2) {
      continue;
    }

    const onnx::TensorProto* b_pruned = nullptr;
    if (pn.input_size() == 3) {
      auto it = pruned_init.find(pn.input(2));
      if (it != pruned_init.end()) {
        b_pruned = it->second;
      }
    }
    const onnx::TensorProto* b_orig = nullptr;
    if (on.input_size() == 3) {
      auto it = orig_init.find(on.input(2));
      if (it != orig_init.end()) {
        b_orig = it->second;
      }
    }
    if ((b_pruned == nullptr) != (b_orig == nullptr)) {
      continue;  // Bias presence disagrees -- something other than
                 // pruning touched this layer.
    }

    candidates.push_back({o_match->x_name, w_orig, w_pruned, b_orig, b_pruned,
                          p_match->weight_transposed});
  }
  return candidates;
}

// --- Channel correspondence recovery ------------------------------------
//
// Direct transcription of finetune.py's own _find_keep_indices/
// _find_channel_correspondence: `orig_rows`/`pruned_rows` are row-major
// [n_orig, width] / [n_pruned, width] (already float64-widened, exact
// values); a forward two-pointer scan finds each pruned row's own
// originating index, since pruning's own slice always makes the pruned
// matrix an order-preserving subsequence of the original.

std::optional<std::vector<int64_t>> FindKeepIndices(
    const std::vector<double>& orig_rows, int64_t n_orig,
    const std::vector<double>& pruned_rows, int64_t n_pruned, int64_t width) {
  if (n_pruned > n_orig) {
    return std::nullopt;
  }
  std::vector<int64_t> keep(static_cast<size_t>(n_pruned));
  int64_t oi = 0;
  for (int64_t pi = 0; pi < n_pruned; ++pi) {
    bool found = false;
    while (oi < n_orig) {
      bool equal = true;
      const double* orow = orig_rows.data() +
                           static_cast<size_t>(oi) * static_cast<size_t>(width);
      const double* prow = pruned_rows.data() +
                           static_cast<size_t>(pi) * static_cast<size_t>(width);
      for (int64_t c = 0; c < width; ++c) {
        if (orow[c] != prow[c]) {
          equal = false;
          break;
        }
      }
      if (equal) {
        found = true;
        break;
      }
      ++oi;
    }
    if (!found) {
      return std::nullopt;
    }
    keep[static_cast<size_t>(pi)] = oi;
    ++oi;
  }
  return keep;
}

// Transposes a row-major [rows, cols] matrix into a row-major
// [cols, rows] one.
std::vector<double> Transpose(const std::vector<double>& m, int64_t rows,
                              int64_t cols) {
  std::vector<double> out(static_cast<size_t>(rows) *
                          static_cast<size_t>(cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c) * static_cast<size_t>(rows) +
          static_cast<size_t>(r)] =
          m[static_cast<size_t>(r) * static_cast<size_t>(cols) +
            static_cast<size_t>(c)];
    }
  }
  return out;
}

struct Correspondence {
  std::vector<int64_t> keep_out;
  std::vector<int64_t> keep_in;
  bool ok = false;
};

Correspondence FindChannelCorrespondence(const std::vector<double>& w_orig_nk,
                                         int64_t n_orig, int64_t k_orig,
                                         const std::vector<double>& w_pruned_nk,
                                         int64_t n_pruned, int64_t k_pruned) {
  Correspondence res;
  if (k_orig == k_pruned) {
    auto keep_out =
        FindKeepIndices(w_orig_nk, n_orig, w_pruned_nk, n_pruned, k_orig);
    if (!keep_out) {
      return res;
    }
    res.keep_out = *keep_out;
    res.keep_in.resize(static_cast<size_t>(k_orig));
    for (int64_t i = 0; i < k_orig; ++i) {
      res.keep_in[static_cast<size_t>(i)] = i;
    }
    res.ok = true;
    return res;
  }
  if (n_orig == n_pruned) {
    const std::vector<double> orig_t = Transpose(w_orig_nk, n_orig, k_orig);
    const std::vector<double> pruned_t =
        Transpose(w_pruned_nk, n_pruned, k_pruned);
    auto keep_in = FindKeepIndices(orig_t, k_orig, pruned_t, k_pruned, n_orig);
    if (!keep_in) {
      return res;
    }
    res.keep_in = *keep_in;
    res.keep_out.resize(static_cast<size_t>(n_orig));
    for (int64_t i = 0; i < n_orig; ++i) {
      res.keep_out[static_cast<size_t>(i)] = i;
    }
    res.ok = true;
    return res;
  }
  return res;  // Both axes changed -- declined.
}

// --- Small dense linear solve (Gaussian elimination, partial pivoting) --
//
// Solves `a @ x == rhs` for `x` (`a`: n x n, `rhs`: n x m, both row-major
// std::vector<std::vector<double>>). No BLAS/LAPACK dependency, the same
// "hand-written scalar kernel" convention gptq_entry.cpp's own
// CholeskyLower/InverseSPD already establish for this codebase -- see
// finetune_entry.h's own "Accepted numerical scope" note.
std::vector<std::vector<double>> SolveLinearSystem(
    std::vector<std::vector<double>> a, std::vector<std::vector<double>> rhs) {
  const size_t n = a.size();
  const size_t m = rhs.empty() ? 0 : rhs[0].size();
  for (size_t col = 0; col < n; ++col) {
    size_t piv = col;
    double best = std::fabs(a[col][col]);
    for (size_t r = col + 1; r < n; ++r) {
      const double v = std::fabs(a[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != col) {
      std::swap(a[col], a[piv]);
      std::swap(rhs[col], rhs[piv]);
    }
    double diag = a[col][col];
    if (std::fabs(diag) < 1e-300) {
      diag = (diag >= 0.0) ? 1e-300 : -1e-300;
      a[col][col] = diag;
    }
    for (size_t r = col + 1; r < n; ++r) {
      const double factor = a[r][col] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (size_t c = col; c < n; ++c) {
        a[r][c] -= factor * a[col][c];
      }
      for (size_t c = 0; c < m; ++c) {
        rhs[r][c] -= factor * rhs[col][c];
      }
    }
  }

  std::vector<std::vector<double>> x(n, std::vector<double>(m, 0.0));
  for (size_t ii = n; ii-- > 0;) {
    for (size_t c = 0; c < m; ++c) {
      double s = rhs[ii][c];
      for (size_t k = ii + 1; k < n; ++k) {
        s -= a[ii][k] * x[k][c];
      }
      double diag = a[ii][ii];
      if (std::fabs(diag) < 1e-300) {
        diag = (diag >= 0.0) ? 1e-300 : -1e-300;
      }
      x[ii][c] = s / diag;
    }
  }
  return x;
}

// Direct transcription of finetune.py's own _ridge_fit: ridge-regression
// fit of `y ~= x @ w.T [+ b]`, regularized toward `(w0, b0)`.
// `x`: [num_samples, K] row-major, `y`: [num_samples, N] row-major,
// `w0`: [N, K] row-major, `b0`: optional length-N. Returns `(w, b)`
// (`b` populated iff `b0` is).
struct RidgeResult {
  std::vector<double> w;  // N * K, row-major.
  std::optional<std::vector<double>> b;
};

RidgeResult RidgeFit(const std::vector<double>& x, int64_t num_samples,
                     int64_t k, const std::vector<double>& y, int64_t n,
                     const std::vector<double>& w0,
                     const std::optional<std::vector<double>>& b0,
                     double reg_param) {
  const bool has_bias = b0.has_value();
  const int64_t k_aug = has_bias ? k + 1 : k;

  std::vector<std::vector<double>> x_aug(
      static_cast<size_t>(num_samples),
      std::vector<double>(static_cast<size_t>(k_aug)));
  for (int64_t r = 0; r < num_samples; ++r) {
    for (int64_t c = 0; c < k; ++c) {
      x_aug[static_cast<size_t>(r)][static_cast<size_t>(c)] =
          x[static_cast<size_t>(r) * static_cast<size_t>(k) +
            static_cast<size_t>(c)];
    }
    if (has_bias) {
      x_aug[static_cast<size_t>(r)][static_cast<size_t>(k)] = 1.0;
    }
  }
  std::vector<std::vector<double>> w0_aug(
      static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(k_aug)));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t c = 0; c < k; ++c) {
      w0_aug[static_cast<size_t>(i)][static_cast<size_t>(c)] =
          w0[static_cast<size_t>(i) * static_cast<size_t>(k) +
             static_cast<size_t>(c)];
    }
    if (has_bias) {
      w0_aug[static_cast<size_t>(i)][static_cast<size_t>(k)] =
          (*b0)[static_cast<size_t>(i)];
    }
  }

  std::vector<std::vector<double>> gram(
      static_cast<size_t>(k_aug),
      std::vector<double>(static_cast<size_t>(k_aug), 0.0));
  for (int64_t r = 0; r < num_samples; ++r) {
    for (int64_t i = 0; i < k_aug; ++i) {
      const double xi = x_aug[static_cast<size_t>(r)][static_cast<size_t>(i)];
      if (xi == 0.0) {
        continue;
      }
      for (int64_t j = 0; j < k_aug; ++j) {
        gram[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
            xi * x_aug[static_cast<size_t>(r)][static_cast<size_t>(j)];
      }
    }
  }
  double trace = 0.0;
  for (int64_t i = 0; i < k_aug; ++i) {
    trace += gram[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }
  const double lam = reg_param * trace / static_cast<double>(k_aug);

  std::vector<std::vector<double>> a = gram;
  for (int64_t i = 0; i < k_aug; ++i) {
    a[static_cast<size_t>(i)][static_cast<size_t>(i)] += lam;
  }

  std::vector<std::vector<double>> rhs(
      static_cast<size_t>(k_aug),
      std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int64_t r = 0; r < num_samples; ++r) {
    for (int64_t i = 0; i < k_aug; ++i) {
      const double xi = x_aug[static_cast<size_t>(r)][static_cast<size_t>(i)];
      if (xi == 0.0) {
        continue;
      }
      for (int64_t j = 0; j < n; ++j) {
        rhs[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
            xi * y[static_cast<size_t>(r) * static_cast<size_t>(n) +
                   static_cast<size_t>(j)];
      }
    }
  }
  for (int64_t i = 0; i < k_aug; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      rhs[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
          lam * w0_aug[static_cast<size_t>(j)][static_cast<size_t>(i)];
    }
  }

  const std::vector<std::vector<double>> w_aug_t = SolveLinearSystem(a, rhs);

  RidgeResult res;
  res.w.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t c = 0; c < k; ++c) {
      res.w[static_cast<size_t>(i) * static_cast<size_t>(k) +
            static_cast<size_t>(c)] =
          w_aug_t[static_cast<size_t>(c)][static_cast<size_t>(i)];
    }
  }
  if (has_bias) {
    std::vector<double> b(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      b[static_cast<size_t>(i)] =
          w_aug_t[static_cast<size_t>(k)][static_cast<size_t>(i)];
    }
    res.b = std::move(b);
  }
  return res;
}

// --- Calibration: concatenated activation rows ---------------------------
//
// Transcribed from gptq_entry.cpp's own AccumulateActivationRows (probed
// against `original_model` here, matching apply_pruning_finetune's own
// single-probe-point-per-layer design).

struct ActivationRows {
  std::vector<double> data;  // [total_rows, K] row-major.
  int64_t k = -1;
  bool ok = false;
};

void AccumulateActivationRows(
    std::unordered_map<std::string, ActivationRows>& acc,
    const ModelExecutor& executor, const onnx::ModelProto& original_model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  if (probe_names.empty()) {
    return;
  }

  onnx::ModelProto probe_model = original_model;
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
            "ApplyPruningFinetune: calibration batch is missing "
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

}  // namespace

onnx::ModelProto ApplyPruningFinetune(
    const onnx::ModelProto& original_model,
    const onnx::ModelProto& pruned_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double reg_param) {
  const std::vector<Candidate> candidates =
      FindMatmulFinetuneCandidates(original_model, pruned_model);
  if (candidates.empty()) {
    return pruned_model;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, original_model, probe_names,
                           calibration_data);

  std::unordered_map<std::string, std::vector<float>> optimized_w;
  std::unordered_map<std::string, std::vector<float>> optimized_b;

  for (const auto& c : candidates) {
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable activation (no feature axis); skip.
    }
    const ActivationRows& rows = ait->second;

    const int64_t dim0_o = c.w_orig->dims(0);
    const int64_t dim1_o = c.w_orig->dims(1);
    const int64_t dim0_p = c.w_pruned->dims(0);
    const int64_t dim1_p = c.w_pruned->dims(1);
    const std::vector<float> w_orig_flat = ReadFloatTensor(*c.w_orig);
    const std::vector<float> w_pruned_flat = ReadFloatTensor(*c.w_pruned);

    // Normalize to [N, K] (output channel first), mirroring
    // apply_pruning_finetune's own `w_orig_nk = w_orig if
    // weight_transposed else w_orig.T` exactly.
    const int64_t n_orig = c.weight_transposed ? dim0_o : dim1_o;
    const int64_t k_orig = c.weight_transposed ? dim1_o : dim0_o;
    const int64_t n_pruned = c.weight_transposed ? dim0_p : dim1_p;
    const int64_t k_pruned = c.weight_transposed ? dim1_p : dim0_p;

    auto to_nk = [&](const std::vector<float>& flat, int64_t dim0, int64_t dim1,
                     int64_t n, int64_t k) {
      std::vector<double> nk(static_cast<size_t>(n) * static_cast<size_t>(k));
      for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          nk[static_cast<size_t>(i) * static_cast<size_t>(k) +
             static_cast<size_t>(j)] =
              static_cast<double>(
                  c.weight_transposed
                      ? flat[static_cast<size_t>(i) * static_cast<size_t>(k) +
                             static_cast<size_t>(j)]
                      : flat[static_cast<size_t>(j) * static_cast<size_t>(n) +
                             static_cast<size_t>(i)]);
        }
      }
      return nk;
    };
    const std::vector<double> w_orig_nk =
        to_nk(w_orig_flat, dim0_o, dim1_o, n_orig, k_orig);
    const std::vector<double> w_pruned_nk =
        to_nk(w_pruned_flat, dim0_p, dim1_p, n_pruned, k_pruned);

    if (rows.k != k_orig) {
      continue;  // Activation's feature dim doesn't match K -- skip.
    }

    const Correspondence corr = FindChannelCorrespondence(
        w_orig_nk, n_orig, k_orig, w_pruned_nk, n_pruned, k_pruned);
    if (!corr.ok) {
      continue;  // Not a clean subsequence of the original -- decline.
    }

    const int64_t num_rows = static_cast<int64_t>(rows.data.size()) / k_orig;
    const int64_t k_keep = static_cast<int64_t>(corr.keep_in.size());
    const int64_t n_keep = static_cast<int64_t>(corr.keep_out.size());

    // x = x_full[:, keep_in].
    std::vector<double> x(static_cast<size_t>(num_rows) *
                          static_cast<size_t>(k_keep));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t c = 0; c < k_keep; ++c) {
        x[static_cast<size_t>(r) * static_cast<size_t>(k_keep) +
          static_cast<size_t>(c)] =
            rows.data[static_cast<size_t>(r) * static_cast<size_t>(k_orig) +
                      static_cast<size_t>(
                          corr.keep_in[static_cast<size_t>(c)])];
      }
    }

    // y_full = x_full @ w_orig_nk^T [+ b_orig]; y = y_full[:, keep_out].
    std::optional<std::vector<double>> b_orig_vals;
    if (c.b_orig != nullptr) {
      const std::vector<float> bf = ReadFloatTensor(*c.b_orig);
      b_orig_vals = std::vector<double>(bf.begin(), bf.end());
    }
    std::vector<double> y(static_cast<size_t>(num_rows) *
                          static_cast<size_t>(n_keep));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t c = 0; c < n_keep; ++c) {
        const int64_t out_ch = corr.keep_out[static_cast<size_t>(c)];
        double acc = 0.0;
        for (int64_t kk = 0; kk < k_orig; ++kk) {
          acc +=
              rows.data[static_cast<size_t>(r) * static_cast<size_t>(k_orig) +
                        static_cast<size_t>(kk)] *
              w_orig_nk[static_cast<size_t>(out_ch) *
                            static_cast<size_t>(k_orig) +
                        static_cast<size_t>(kk)];
        }
        if (b_orig_vals) {
          acc += (*b_orig_vals)[static_cast<size_t>(out_ch)];
        }
        y[static_cast<size_t>(r) * static_cast<size_t>(n_keep) +
          static_cast<size_t>(c)] = acc;
      }
    }

    std::optional<std::vector<double>> b_pruned_vals;
    if (c.b_pruned != nullptr) {
      const std::vector<float> bf = ReadFloatTensor(*c.b_pruned);
      b_pruned_vals = std::vector<double>(bf.begin(), bf.end());
    }

    const RidgeResult fit = RidgeFit(x, num_rows, k_keep, y, n_keep,
                                     w_pruned_nk, b_pruned_vals, reg_param);

    // w_write = w_new_nk if weight_transposed else w_new_nk.T -- back to
    // the pruned weight's own original storage layout.
    std::vector<float> w_write(static_cast<size_t>(dim0_p) *
                               static_cast<size_t>(dim1_p));
    for (int64_t i = 0; i < n_keep; ++i) {
      for (int64_t j = 0; j < k_keep; ++j) {
        const double v =
            fit.w[static_cast<size_t>(i) * static_cast<size_t>(k_keep) +
                  static_cast<size_t>(j)];
        if (c.weight_transposed) {
          w_write[static_cast<size_t>(i) * static_cast<size_t>(k_keep) +
                  static_cast<size_t>(j)] = static_cast<float>(v);
        } else {
          w_write[static_cast<size_t>(j) * static_cast<size_t>(n_keep) +
                  static_cast<size_t>(i)] = static_cast<float>(v);
        }
      }
    }
    optimized_w[c.w_pruned->name()] = std::move(w_write);

    if (fit.b && c.b_pruned != nullptr) {
      std::vector<float> b_write(fit.b->size());
      for (size_t i = 0; i < fit.b->size(); ++i) {
        b_write[i] = static_cast<float>((*fit.b)[i]);
      }
      optimized_b[c.b_pruned->name()] = std::move(b_write);
    }
  }

  if (optimized_w.empty()) {
    return pruned_model;
  }

  onnx::ModelProto out = pruned_model;
  for (auto& t : *out.mutable_graph()->mutable_initializer()) {
    auto wit = optimized_w.find(t.name());
    if (wit != optimized_w.end()) {
      std::vector<int64_t> dims(t.dims().begin(), t.dims().end());
      SetRawInitializer(&t, onnx::TensorProto::FLOAT, dims, wit->second.data(),
                        wit->second.size() * sizeof(float), sizeof(float));
      continue;
    }
    auto bit = optimized_b.find(t.name());
    if (bit != optimized_b.end()) {
      std::vector<int64_t> dims(t.dims().begin(), t.dims().end());
      SetRawInitializer(&t, onnx::TensorProto::FLOAT, dims, bit->second.data(),
                        bit->second.size() * sizeof(float), sizeof(float));
    }
  }
  return out;
}
