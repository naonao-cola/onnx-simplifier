// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See owq_entry.h for the full rationale (including why this follows
// gptq_entry.h's own protobuf-level, two-model, calibration-driven shape)
// and onnxsim/owq.py for the technique this ports.

#include "owq_entry.h"

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

// --- Candidate matching, two-model -----------------------------------------
//
// Transcribed from gptq_entry.cpp's own FindInt4MatmulCandidates (itself a
// transcription of adaround.py's own _find_int4_matmul_candidates, which
// apply_owq reuses verbatim): joins the float and quantized models by node
// output tensor name. Output-name maps are last-wins with first-seen
// iteration order, mirroring `_node_outputs`' own `m[n.output[0]] = n`
// reassignment exactly.

struct Candidate {
  std::string output_name;
  std::string float_x_name;
  std::string w_float_name;
  std::string wq_name;
  std::string ws_name;
  int64_t block_size;
  bool weight_transposed;
};

int64_t GetIntAttr(const onnx::NodeProto& node, const std::string& name,
                   int64_t fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::INT) {
      return attr.i();
    }
  }
  return fallback;
}

std::vector<Candidate> FindInt4MatmulCandidates(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model,
    std::unordered_map<std::string, int>& q_init_index,
    std::unordered_map<std::string, int>& f_init_index) {
  const onnx::GraphProto& q_graph = quantized_model.graph();
  const onnx::GraphProto& f_graph = float_model.graph();

  for (int i = 0; i < q_graph.initializer_size(); ++i) {
    q_init_index.emplace(q_graph.initializer(i).name(), i);
  }
  for (int i = 0; i < f_graph.initializer_size(); ++i) {
    f_init_index.emplace(f_graph.initializer(i).name(), i);
  }

  std::vector<std::string> q_order;
  std::unordered_map<std::string, int> q_by_output;
  for (int i = 0; i < q_graph.node_size(); ++i) {
    const auto& n = q_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    if (!q_by_output.count(n.output(0))) {
      q_order.push_back(n.output(0));
    }
    q_by_output[n.output(0)] = i;
  }
  std::unordered_map<std::string, int> f_by_output;
  for (int i = 0; i < f_graph.node_size(); ++i) {
    const auto& n = f_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    f_by_output[n.output(0)] = i;
  }

  std::vector<Candidate> candidates;
  for (const auto& out_name : q_order) {
    const onnx::NodeProto& qn = q_graph.node(q_by_output[out_name]);
    if ((qn.op_type() != "MatMul" && qn.op_type() != "Gemm") ||
        qn.input_size() < 2) {
      continue;
    }
    auto fit = f_by_output.find(out_name);
    if (fit == f_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& fn = f_graph.node(fit->second);
    if (fn.op_type() != qn.op_type() || fn.input_size() < 2) {
      continue;
    }

    auto wfit = f_init_index.find(fn.input(1));
    if (wfit == f_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_float_init = f_graph.initializer(wfit->second);
    if (w_float_init.data_type() != onnx::TensorProto::FLOAT ||
        w_float_init.dims_size() != 2) {
      continue;
    }

    auto dqit = q_by_output.find(qn.input(1));
    if (dqit == q_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& dq = q_graph.node(dqit->second);
    if (dq.op_type() != "DequantizeLinear" || dq.input_size() < 2) {
      continue;
    }
    auto wqit = q_init_index.find(dq.input(0));
    auto wsit = q_init_index.find(dq.input(1));
    if (wqit == q_init_index.end() || wsit == q_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& wq_init = q_graph.initializer(wqit->second);
    if (wq_init.data_type() != onnx::TensorProto::INT4 ||
        wq_init.dims_size() != w_float_init.dims_size()) {
      continue;
    }
    bool same_dims = true;
    for (int d = 0; d < wq_init.dims_size(); ++d) {
      if (wq_init.dims(d) != w_float_init.dims(d)) {
        same_dims = false;
        break;
      }
    }
    if (!same_dims) {
      continue;
    }

    const int64_t block_size = GetIntAttr(dq, "block_size", 0);
    if (!block_size) {
      continue;
    }

    const bool weight_transposed =
        qn.op_type() == "Gemm" && GetIntAttr(qn, "transB", 0) != 0;
    candidates.push_back({out_name, fn.input(0), fn.input(1), dq.input(0),
                          dq.input(1), block_size, weight_transposed});
  }
  return candidates;
}

// --- Tensor <-> flat buffers, protobuf level -------------------------------

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

// Same low-nibble-first unpacking low_rank_compensation_entry.cpp's own
// ReadInt4Tensor (and test_adaround_cpp.py's own _int4_codes helper)
// establish: byte i packs codes[2i] in its low nibble, codes[2i+1] in its
// high nibble, each a twos-complement nibble in [-8, 7]. Mirrors owq.py's
// own _unpack_int4 exactly -- OWQ needs quantized_model's own REAL codes
// (not a fresh from-scratch RTN recomputation), so the residual below is
// exact against whatever quantized_model actually contains.
std::vector<double> ReadInt4Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<double> out(static_cast<size_t>(numel));
  const std::string& raw = t.raw_data();
  for (int64_t i = 0; i < numel; ++i) {
    const auto byte = static_cast<uint8_t>(raw[static_cast<size_t>(i / 2)]);
    const uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    const int8_t signed_val = (nibble >= 8) ? static_cast<int8_t>(nibble - 16)
                                            : static_cast<int8_t>(nibble);
    out[static_cast<size_t>(i)] = static_cast<double>(signed_val);
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
// A deliberate, file-local TRANSCRIBED COPY of gptq_entry.cpp's own
// CholeskyLower/InverseSPD/InverseHessianCholesky (see owq_entry.h's own
// top-of-file comment for why this is a local copy rather than a shared
// dependency between the two translation units -- the same convention
// billm_entry.cpp already establishes for its own copy of the identical
// kernels).

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

// Mirrors gptq.py's own _inverse_hessian_cholesky exactly: dead channels
// (zero diagonal) get a fixed diagonal of 1, every diagonal entry is
// damped, and the returned factor U is upper-triangular with
// inverse(h) == U^T @ U.
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

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own AccumulateActivationRows.

struct ActivationRows {
  std::vector<double> data;  // Concatenated [total_rows, K], row-major.
  int64_t k = -1;
  bool ok = false;
};

void AccumulateActivationRows(
    std::unordered_map<std::string, ActivationRows>& acc,
    const ModelExecutor& executor, const onnx::ModelProto& float_model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  if (probe_names.empty()) {
    return;
  }

  onnx::ModelProto probe_model = float_model;
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
            "ApplyOwq: calibration batch is missing "
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
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (numpy would fail to concatenate).
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      rows.data.reserve(rows.data.size() + data.size());
      for (float v : data) {
        rows.data.push_back(static_cast<double>(v));
      }
    }
  }
}

// Round-half-to-even (banker's rounding), matching Python's own built-in
// `round()` -- OWQ's own `num_weak = int(round(outlier_fraction * k))`.
// Transcribed from llm_int8_entry.cpp's own identical helper.
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

}  // namespace

onnx::ModelProto ApplyOwq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double outlier_fraction, double percdamp) {
  std::unordered_map<std::string, int> q_init_index;
  std::unordered_map<std::string, int> f_init_index;
  const std::vector<Candidate> candidates = FindInt4MatmulCandidates(
      float_model, quantized_model, q_init_index, f_init_index);
  if (candidates.empty()) {
    return quantized_model;
  }
  const onnx::GraphProto& f_graph = float_model.graph();

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.float_x_name);
  }
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, float_model, probe_names,
                           calibration_data);

  onnx::ModelProto out = quantized_model;
  onnx::GraphProto* graph = out.mutable_graph();

  // Mirrors owq.py's own `node_by_output = {n.output[0]: n for n in
  // graph.node if n.output}`, rebuilt on the *copy* -- NodeProto* pointers
  // into `graph` stay valid across every later add_node()/SwapElements()
  // call (protobuf's RepeatedPtrField swaps pointer slots, never moves or
  // frees the pointed-to messages), the same convention
  // low_rank_compensation_entry.cpp's own identical map already
  // establishes for this codebase.
  std::unordered_map<std::string, onnx::NodeProto*> node_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->output_size() > 0) {
      node_by_output[n->output(0)] = n;
    }
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention every calibration-driven
  // *_entry.cpp in this repo already uses.
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

  for (const auto& c : candidates) {
    auto ait = activations.find(c.float_x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable activation (no feature axis); skip.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_float_init =
        f_graph.initializer(f_init_index[c.w_float_name]);
    const int64_t dim0 = w_float_init.dims(0);
    const int64_t dim1 = w_float_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }
    if (k % c.block_size != 0) {
      continue;  // Outside quantize_weight_only_int4's own precondition.
    }
    const int64_t num_weak = static_cast<int64_t>(
        RoundHalfToEven(outlier_fraction * static_cast<double>(k)));
    if (num_weak < 1) {
      continue;
    }
    const int64_t num_rows =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_float_init);
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

    // Hessian over every concatenated row: H = X^T X.
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
    const Matrix u = InverseHessianCholesky(h, percdamp);
    // h_inv_diag[j] = sum_i u[i][j]^2 -- inv(h) == u.T @ u, so its own
    // diagonal at j is the squared L2 norm of u's own column j. Mirrors
    // `np.maximum((u**2).sum(axis=0), 1e-12)` exactly.
    std::vector<double> h_inv_diag(static_cast<size_t>(k), 0.0);
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        const double uij = u[static_cast<size_t>(i)][static_cast<size_t>(j)];
        h_inv_diag[static_cast<size_t>(j)] += uij * uij;
      }
    }
    for (int64_t j = 0; j < k; ++j) {
      h_inv_diag[static_cast<size_t>(j)] =
          std::max(h_inv_diag[static_cast<size_t>(j)], 1e-12);
    }

    // Unpack quantized_model's own REAL INT4 codes (not a fresh
    // from-scratch RTN recomputation) -- see owq_entry.h's own top-of-file
    // comment for why that distinction matters here.
    const onnx::GraphProto& q_graph = quantized_model.graph();
    const onnx::TensorProto& wq_init =
        q_graph.initializer(q_init_index[c.wq_name]);
    const std::vector<double> codes_flat = ReadInt4Tensor(wq_init);
    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);

    const int64_t num_groups = k / c.block_size;
    // codes_nk/scale_blocks: [N, K] / [N, num_groups] -- mirrors
    // `codes_nk = codes if weight_transposed else codes.T` and
    // `scale_blocks = scale if weight_transposed else scale.T` exactly.
    std::vector<std::vector<double>> codes_nk(
        static_cast<size_t>(n_rows),
        std::vector<double>(static_cast<size_t>(k)));
    std::vector<std::vector<double>> scale_blocks(
        static_cast<size_t>(n_rows),
        std::vector<double>(static_cast<size_t>(num_groups)));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
              codes_flat[static_cast<size_t>(i * k + j)];
        }
        for (int64_t g = 0; g < num_groups; ++g) {
          scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(g)] =
              static_cast<double>(
                  s_flat[static_cast<size_t>(i * num_groups + g)]);
        }
      }
    } else {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
              codes_flat[static_cast<size_t>(j * n_rows + i)];
        }
        for (int64_t g = 0; g < num_groups; ++g) {
          scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(g)] =
              static_cast<double>(s_flat[static_cast<size_t>(g * n_rows + i)]);
        }
      }
    }

    // w_rtn[i][j] = codes_nk[i][j] * scale_blocks[i][j / block_size] --
    // mirrors `scale_full = np.repeat(scale_blocks, block_size, axis=1);
    // w_rtn = codes_nk * scale_full`. col_error[j] = mean_i (w_nk[i][j] -
    // w_rtn[i][j])^2.
    std::vector<double> col_error(static_cast<size_t>(k), 0.0);
    std::vector<std::vector<double>> delta_full(
        static_cast<size_t>(n_rows),
        std::vector<double>(static_cast<size_t>(k)));
    for (int64_t j = 0; j < k; ++j) {
      const int64_t group = j / c.block_size;
      double sum_sq = 0.0;
      for (int64_t i = 0; i < n_rows; ++i) {
        const double scale =
            scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(group)];
        const double w_rtn =
            codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] * scale;
        const double diff =
            w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] - w_rtn;
        delta_full[static_cast<size_t>(i)][static_cast<size_t>(j)] = diff;
        sum_sq += diff * diff;
      }
      col_error[static_cast<size_t>(j)] =
          n_rows > 0 ? sum_sq / static_cast<double>(n_rows) : 0.0;
    }

    std::vector<double> sensitivity(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      sensitivity[static_cast<size_t>(j)] = col_error[static_cast<size_t>(j)] /
                                            h_inv_diag[static_cast<size_t>(j)];
    }

    // weak_idx = argsort(-sensitivity)[:num_weak], then sorted ascending
    // for storage -- see owq_entry.h's own "Accepted numerical scope" note
    // on std::stable_sort vs np.argsort's own non-guaranteed stability.
    std::vector<int64_t> order(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      order[static_cast<size_t>(j)] = j;
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      return sensitivity[static_cast<size_t>(a)] >
             sensitivity[static_cast<size_t>(b)];
    });
    const int64_t take = std::min<int64_t>(num_weak, k);
    std::vector<int64_t> weak_idx(order.begin(), order.begin() + take);
    std::sort(weak_idx.begin(), weak_idx.end());

    // delta_w = (w_nk - w_rtn)[:, weak_idx] -- [N, num_weak]. Skip entirely
    // (matches `if not np.any(delta_w): continue`) when RTN already
    // reconstructs every rescued column exactly.
    bool any_nonzero = false;
    std::vector<double> delta_w(static_cast<size_t>(n_rows) *
                                static_cast<size_t>(take));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t c2 = 0; c2 < take; ++c2) {
        const double v =
            delta_full[static_cast<size_t>(i)]
                      [static_cast<size_t>(weak_idx[static_cast<size_t>(c2)])];
        delta_w[static_cast<size_t>(i) * static_cast<size_t>(take) +
                static_cast<size_t>(c2)] = v;
        if (v != 0.0) {
          any_nonzero = true;
        }
      }
    }
    if (!any_nonzero) {
      continue;
    }

    const std::string prefix = c.output_name + "_owq";

    // idx_name: INT64 [num_weak], ascending weak column indices.
    std::vector<int64_t> weak_idx64(weak_idx.begin(), weak_idx.end());
    const std::string idx_name = unique_name(prefix + "_weak_idx");
    SetRawInitializer(graph->add_initializer(), idx_name,
                      onnx::TensorProto::INT64, {take}, weak_idx64.data(),
                      weak_idx64.size() * sizeof(int64_t), sizeof(int64_t));

    // delta_w_name: FLOAT32 [num_weak, N] -- delta_w's own transpose
    // ([N, num_weak] -> [num_weak, N]), ready for a right-multiply against
    // XWeak's own [..., num_weak] shape. Mirrors `delta_w.T` exactly.
    std::vector<float> delta_w_t(static_cast<size_t>(take) *
                                 static_cast<size_t>(n_rows));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t c2 = 0; c2 < take; ++c2) {
        delta_w_t[static_cast<size_t>(c2) * static_cast<size_t>(n_rows) +
                  static_cast<size_t>(i)] =
            static_cast<float>(
                delta_w[static_cast<size_t>(i) * static_cast<size_t>(take) +
                        static_cast<size_t>(c2)]);
      }
    }
    const std::string delta_w_name = unique_name(prefix + "_delta_w");
    SetRawInitializer(graph->add_initializer(), delta_w_name,
                      onnx::TensorProto::FLOAT, {take, n_rows},
                      delta_w_t.data(), delta_w_t.size() * sizeof(float),
                      sizeof(float));

    onnx::NodeProto* qn = node_by_output.at(c.output_name);
    const std::string old_output = qn->output(0);
    const std::string pre_name = unique_name(prefix + "_pre_correction");
    qn->set_output(0, pre_name);

    // Re-finds qn's *current* node-list index by pointer identity -- earlier
    // candidates' own insertions can have shifted it since `node_by_output`
    // was built, so this cannot be cached across candidates. Mirrors
    // low_rank_compensation_entry.cpp's own identical pattern:
    // RepeatedPtrField::SwapElements (used by append_at below) swaps
    // POINTER SLOTS, never moving or freeing the pointed-to messages, so a
    // NodeProto* stays valid (and locatable by identity) across a later
    // SwapElements call.
    auto* nodes = graph->mutable_node();
    int qn_index = -1;
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == qn) {
        qn_index = i;
        break;
      }
    }

    // Appends a node at the very end, then walks it back down to exactly
    // `target_index` via adjacent pointer-slot swaps -- matches
    // low_rank_compensation_entry.cpp's own identical pattern, itself
    // matching owq.py's own `graph.node.insert(node_idx + 1, ...)`
    // placement.
    auto append_at =
        [&](const std::string& op_type, const std::vector<std::string>& inputs,
            const std::string& output, const std::string& name,
            int target_index, std::optional<int64_t> axis_attr = std::nullopt) {
          onnx::NodeProto* node = graph->add_node();
          node->set_op_type(op_type);
          for (const auto& in : inputs) {
            node->add_input(in);
          }
          node->add_output(output);
          node->set_name(name);
          if (axis_attr.has_value()) {
            onnx::AttributeProto* attr = node->add_attribute();
            attr->set_name("axis");
            attr->set_type(onnx::AttributeProto::INT);
            attr->set_i(*axis_attr);
          }
          for (int i = nodes->size() - 1; i > target_index; --i) {
            nodes->SwapElements(i, i - 1);
          }
        };

    const std::string x_weak_name = unique_name(prefix + "_x_weak");
    append_at("Gather", {c.float_x_name, idx_name}, x_weak_name,
              unique_name(prefix + "_gather"), qn_index + 1, int64_t{-1});

    const std::string corr_name = unique_name(prefix + "_correction");
    append_at("MatMul", {x_weak_name, delta_w_name}, corr_name,
              unique_name(prefix + "_matmul"), qn_index + 2);

    append_at("Add", {pre_name, corr_name}, old_output,
              unique_name(prefix + "_add"), qn_index + 3);
  }

  return out;
}
