// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See gptaq_entry.h for the full rationale (including why this reuses
// gptq_entry.h's own candidate matching and column-processing/Hessian-
// inversion machinery almost verbatim, adding only the second-model
// probing and the closed-form `Shift` correction) and onnxsim/gptaq.py's
// own module docstring for the technique this ports.
//
// The candidate-matching, small-dense-linear-algebra, and GPTQ column-
// processing helpers below are transcribed from gptq_entry.cpp's own
// identical helpers of the same names (FindInt4MatmulCandidates,
// ReadFloatTensor, CholeskyLower, InverseSPD, InverseHessianCholesky,
// RoundHalfToEven, GptqQuantizeColumns, PackInt4) -- see gptq_entry.h's
// own top-of-file comment for why they aren't shared via a common header
// (matches this codebase's established convention: every *_entry.cpp is
// self-contained). The dual-model probing loop (AddProbeOutputs/
// RunOneModel) is transcribed from dac_entry.cpp's own identical helpers.

#include "gptaq_entry.h"

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

// --- Candidate matching ----------------------------------------------------
//
// Identical to gptq_entry.cpp's own FindInt4MatmulCandidates, plus one
// extra field: the matched quantized-side node's own input[0] (the probe
// point for the corrupted activation X) -- mirrors apply_gptaq's own
// `quant_probe_name` construction (`q_by_output.get(c.output_name).input[0]`)
// exactly.

struct Candidate {
  std::string output_name;
  std::string float_x_name;
  std::string quant_x_name;
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

  // Last-wins values, first-seen order -- mirrors _node_outputs.
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
    candidates.push_back({out_name, fn.input(0), qn.input(0), fn.input(1),
                          dq.input(0), dq.input(1), block_size,
                          weight_transposed});
  }
  return candidates;
}

// --- Tensor <-> flat float buffer -------------------------------------------

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

// --- Small dense double-precision linear algebra ----------------------------
//
// Identical to gptq_entry.cpp's own kernels -- see that file's own
// comment for the rationale (no BLAS/LAPACK dependency; small K x K
// matrices, run once per layer).

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

// GPTQ's own efficient reformulation -- returned U is upper-triangular
// with inverse(h) == U^T @ U.
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

// Identical to gptq_entry.cpp's own GptqQuantizeColumns: GPTQ's
// sequential, Hessian-compensated integer-code search, applied here to
// `w_nk + Shift` (the caller passes the already-shifted weight matrix in
// `w_nk`) rather than the raw float weight -- see this file's own
// ApplyGptaq for why that alone is GPTAQ's entire departure from plain
// GPTQ.
std::vector<std::vector<double>> GptqQuantizeColumns(
    const std::vector<std::vector<double>>& w_nk,
    const std::vector<std::vector<double>>& scale_blocks, int64_t block_size,
    const Matrix& h, double percdamp, int64_t proc_block_size) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const Matrix hinv = InverseHessianCholesky(h, percdamp);

  std::vector<std::vector<double>> codes(n, std::vector<double>(k, 0.0));
  std::vector<std::vector<double>> w_work = w_nk;

  for (int64_t block_start = 0; block_start < static_cast<int64_t>(k);
       block_start += proc_block_size) {
    const int64_t block_end =
        std::min(block_start + proc_block_size, static_cast<int64_t>(k));
    const size_t bs = static_cast<size_t>(block_end - block_start);
    std::vector<std::vector<double>> w1(n, std::vector<double>(bs));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < bs; ++j) {
        w1[i][j] = w_work[i][static_cast<size_t>(block_start) + j];
      }
    }
    std::vector<std::vector<double>> err1(n, std::vector<double>(bs, 0.0));

    for (size_t i = 0; i < bs; ++i) {
      const int64_t k_abs = block_start + static_cast<int64_t>(i);
      const size_t group = static_cast<size_t>(k_abs / block_size);
      for (size_t r = 0; r < n; ++r) {
        const double s = scale_blocks[r][group];
        const double w_col = w1[r][i];
        double code = RoundHalfToEven(w_col / s);
        code = std::min(7.0, std::max(-7.0, code));
        codes[r][static_cast<size_t>(k_abs)] = code;
        const double d = hinv[static_cast<size_t>(block_start) + i]
                             [static_cast<size_t>(block_start) + i];
        const double err = (w_col - code * s) / d;
        err1[r][i] = err;
        for (size_t j = i + 1; j < bs; ++j) {
          w1[r][j] -= err * hinv[static_cast<size_t>(block_start) + i]
                                [static_cast<size_t>(block_start) + j];
        }
      }
    }

    if (block_end < static_cast<int64_t>(k)) {
      for (size_t r = 0; r < n; ++r) {
        for (int64_t j = block_end; j < static_cast<int64_t>(k); ++j) {
          double acc = 0.0;
          for (size_t i = 0; i < bs; ++i) {
            acc += err1[r][i] * hinv[static_cast<size_t>(block_start) + i]
                                    [static_cast<size_t>(j)];
          }
          w_work[r][static_cast<size_t>(j)] -= acc;
        }
      }
    }
  }
  return codes;
}

std::string PackInt4(const std::vector<double>& codes_nk_flat) {
  std::string packed;
  packed.resize(codes_nk_flat.size() / 2);
  for (size_t i = 0; i < packed.size(); ++i) {
    const auto lo =
        static_cast<uint8_t>(static_cast<int64_t>(codes_nk_flat[2 * i]));
    const auto hi =
        static_cast<uint8_t>(static_cast<int64_t>(codes_nk_flat[2 * i + 1]));
    packed[i] = static_cast<char>((lo & 0xF) | ((hi & 0xF) << 4));
  }
  return packed;
}

// Matrix product `a @ b`, `a` [m,k'], `b` [k',n] (m/k'/n implied by
// argument shapes). Small dense helper -- K here is the same "reduction
// dim" scale as the Cholesky/inverse kernels above, so a plain O(m*k'*n)
// triple loop is the right cost/complexity tradeoff (matches this file's
// own no-BLAS convention).
Matrix MatMul(const Matrix& a, const Matrix& b) {
  const size_t m = a.size();
  const size_t kk = a.empty() ? 0 : a[0].size();
  const size_t n = b.empty() ? 0 : b[0].size();
  Matrix out(m, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < kk; ++j) {
      const double aij = a[i][j];
      if (aij == 0.0) {
        continue;
      }
      for (size_t l = 0; l < n; ++l) {
        out[i][l] += aij * b[j][l];
      }
    }
  }
  return out;
}

Matrix Transpose(const Matrix& a) {
  const size_t m = a.size();
  const size_t n = m == 0 ? 0 : a[0].size();
  Matrix out(n, std::vector<double>(m, 0.0));
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      out[j][i] = a[i][j];
    }
  }
  return out;
}

// --- Dual-model calibration probing -----------------------------------------
//
// AddProbeOutputs/RunOneModel are transcribed from dac_entry.cpp's own
// identical helpers (see this file's own top-of-file comment for why they
// aren't shared via a common header).

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

// One [rows, k] activation part (a single batch's own contribution),
// mirroring apply_gptaq's own `_activation_rows` per-array output.
struct ActivationPart {
  int64_t rows = 0;
  int64_t k = 0;
  std::vector<double> data;  // row-major [rows, k]
};

// Accumulates one FLOAT, rank->=2 probe output per batch into `parts`,
// dropping (not zero-filling) any batch whose probed tensor is missing,
// non-FLOAT, or rank < 2 -- mirrors `_activation_rows`'s own per-array
// `if a.ndim < 2: continue` exactly.
void AccumulatePart(std::vector<ActivationPart>& parts,
                    const std::vector<DLManagedTensorPtr>& outputs,
                    const std::unordered_map<std::string, size_t>& index,
                    const std::string& name) {
  auto it = index.find(name);
  if (it == index.end() || it->second >= outputs.size()) {
    return;
  }
  const onnx::TensorProto tp =
      onnxsim::dlpack::ToTensorProto(outputs[it->second]->dl_tensor);
  if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 2) {
    return;
  }
  const int64_t k = tp.dims(tp.dims_size() - 1);
  if (k <= 0) {
    return;
  }
  int64_t numel = 1;
  for (int64_t d : tp.dims()) {
    numel *= d;
  }
  const int64_t rows = numel / k;
  const std::vector<float> flat = ReadFloatTensor(tp);
  ActivationPart part;
  part.rows = rows;
  part.k = k;
  part.data.assign(flat.begin(), flat.end());
  parts.push_back(std::move(part));
}

}  // namespace

onnx::ModelProto ApplyGptaq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp, int64_t proc_block_size) {
  std::unordered_map<std::string, int> q_init_index;
  std::unordered_map<std::string, int> f_init_index;
  const std::vector<Candidate> candidates = FindInt4MatmulCandidates(
      float_model, quantized_model, q_init_index, f_init_index);
  if (candidates.empty()) {
    return quantized_model;
  }
  const onnx::GraphProto& f_graph = float_model.graph();
  const onnx::GraphProto& q_graph = quantized_model.graph();

  // Unique probe names per model -- mirrors apply_gptaq's own
  // `float_probe_names`/`quant_probe_name.values()` sets exactly (probe
  // outputs are added once per unique name; multiple candidates sharing a
  // probe name share the same accumulated parts).
  std::vector<std::string> float_probe_names;
  std::vector<std::string> quant_probe_names;
  {
    std::unordered_set<std::string> f_seen, q_seen;
    for (const auto& c : candidates) {
      if (f_seen.insert(c.float_x_name).second) {
        float_probe_names.push_back(c.float_x_name);
      }
      if (q_seen.insert(c.quant_x_name).second) {
        quant_probe_names.push_back(c.quant_x_name);
      }
    }
  }

  onnx::ModelProto float_probe = float_model;
  onnx::ModelProto quant_probe = quantized_model;
  const auto f_output_index = AddProbeOutputs(&float_probe, float_probe_names);
  const auto q_output_index = AddProbeOutputs(&quant_probe, quant_probe_names);

  std::unordered_map<std::string, std::vector<ActivationPart>>
      float_parts_by_name;
  std::unordered_map<std::string, std::vector<ActivationPart>>
      quant_parts_by_name;
  for (const auto& batch : calibration_data) {
    std::vector<DLManagedTensorPtr> f_outputs =
        RunOneModel(executor, float_probe, batch, "ApplyGptaq");
    std::vector<DLManagedTensorPtr> q_outputs =
        RunOneModel(executor, quant_probe, batch, "ApplyGptaq");
    for (const auto& name : float_probe_names) {
      AccumulatePart(float_parts_by_name[name], f_outputs, f_output_index,
                     name);
    }
    for (const auto& name : quant_probe_names) {
      AccumulatePart(quant_parts_by_name[name], q_outputs, q_output_index,
                     name);
    }
  }

  // Optimized INT4 payloads by quantized-weight initializer name, then
  // one copy-and-rewrite pass -- mirrors apply_gptaq's own `optimized`
  // dict plus final initializer loop exactly.
  std::unordered_map<std::string, std::string> optimized;
  for (const auto& c : candidates) {
    const std::vector<ActivationPart>& float_parts =
        float_parts_by_name[c.float_x_name];
    const std::vector<ActivationPart>& quant_parts =
        quant_parts_by_name[c.quant_x_name];
    // Mirrors `if not x_true_parts or len(x_true_parts) !=
    // len(x_quant_parts): continue`.
    if (float_parts.empty() || float_parts.size() != quant_parts.size()) {
      continue;
    }
    bool shape_ok = true;
    int64_t k = float_parts[0].k;
    for (size_t i = 0; i < float_parts.size(); ++i) {
      if (float_parts[i].rows != quant_parts[i].rows ||
          float_parts[i].k != quant_parts[i].k || float_parts[i].k != k) {
        shape_ok = false;
        break;
      }
    }
    if (!shape_ok) {
      continue;
    }

    const onnx::TensorProto& w_float_init =
        f_graph.initializer(f_init_index[c.w_float_name]);
    const int64_t dim0 = w_float_init.dims(0);
    const int64_t dim1 = w_float_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t kw = c.weight_transposed ? dim1 : dim0;
    if (k != kw) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }
    if (kw % c.block_size != 0) {
      continue;  // Same K-multiple-of-block_size precondition as
                 // gptq_entry.cpp's own ApplyGptq.
    }

    // Concatenated [S, K] rows for both models, in the same batch order
    // -- mirrors `x_true = np.concatenate(x_true_parts, axis=0)` /
    // `x_quant = np.concatenate(x_quant_parts, axis=0)`.
    int64_t total_rows = 0;
    for (const auto& p : float_parts) {
      total_rows += p.rows;
    }
    std::vector<double> x_true;
    std::vector<double> x_quant;
    x_true.reserve(static_cast<size_t>(total_rows * k));
    x_quant.reserve(static_cast<size_t>(total_rows * k));
    for (size_t i = 0; i < float_parts.size(); ++i) {
      x_true.insert(x_true.end(), float_parts[i].data.begin(),
                    float_parts[i].data.end());
      x_quant.insert(x_quant.end(), quant_parts[i].data.begin(),
                     quant_parts[i].data.end());
    }

    const std::vector<float> w_flat = ReadFloatTensor(w_float_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
                std::vector<double>(static_cast<size_t>(kw)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < kw; ++j) {
        w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(i * kw + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + i)]);
      }
    }

    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    const int64_t num_groups = kw / c.block_size;
    Matrix scale_blocks(static_cast<size_t>(n_rows),
                        std::vector<double>(static_cast<size_t>(num_groups)));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_groups; ++g) {
          scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(g)] =
              static_cast<double>(
                  s_flat[static_cast<size_t>(i * num_groups + g)]);
        }
      }
    } else {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_groups; ++g) {
          scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(g)] =
              static_cast<double>(s_flat[static_cast<size_t>(g * n_rows + i)]);
        }
      }
    }

    // H = X^T X, using the CORRUPTED (quantized-model) activation --
    // mirrors `h = x_quant.T @ x_quant` exactly (this is GPTQ's own
    // Hessian, computed from whichever activation actually multiplies the
    // error `e` in the completed-square objective -- see this port's own
    // header comment).
    Matrix h(static_cast<size_t>(kw),
             std::vector<double>(static_cast<size_t>(kw), 0.0));
    for (int64_t r = 0; r < total_rows; ++r) {
      for (int64_t i = 0; i < kw; ++i) {
        const double vi = x_quant[static_cast<size_t>(r * kw + i)];
        for (int64_t j = 0; j < kw; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * x_quant[static_cast<size_t>(r * kw + j)];
        }
      }
    }

    // Shift = (H^{-1} C)^T, C = X^T (δX W^T) -- see this file's own
    // header comment for the closed-form algebra. Computed here as
    // `Shift = W_nk @ (Cross^T @ Hinv)` where `Cross = X^T δX` and
    // `Hinv = Hinv_U^T @ Hinv_U` (Hinv_U from InverseHessianCholesky,
    // upper-triangular with `inverse(H) == Hinv_U^T @ Hinv_U`); algebraic
    // rearrangement of the same computation apply_gptaq's own
    // `hinv_u.T @ (hinv_u @ c_mat)).T` performs, chosen for the same
    // O(K^2 * (S + N)) cost profile as a straightforward transcription
    // (S the calibration row count, N the output-channel count) rather
    // than materializing the intermediate [S, N] `r`/[K, N] `c_mat`
    // matrices the Python reference's own vectorized form does.
    Matrix delta(static_cast<size_t>(total_rows),
                 std::vector<double>(static_cast<size_t>(kw)));
    for (int64_t r = 0; r < total_rows; ++r) {
      for (int64_t i = 0; i < kw; ++i) {
        delta[static_cast<size_t>(r)][static_cast<size_t>(i)] =
            x_true[static_cast<size_t>(r * kw + i)] -
            x_quant[static_cast<size_t>(r * kw + i)];
      }
    }
    Matrix cross(static_cast<size_t>(kw),
                 std::vector<double>(static_cast<size_t>(kw), 0.0));
    for (int64_t r = 0; r < total_rows; ++r) {
      for (int64_t i = 0; i < kw; ++i) {
        const double xi = x_quant[static_cast<size_t>(r * kw + i)];
        if (xi == 0.0) {
          continue;
        }
        for (int64_t j = 0; j < kw; ++j) {
          cross[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              xi * delta[static_cast<size_t>(r)][static_cast<size_t>(j)];
        }
      }
    }
    const Matrix hinv_u = InverseHessianCholesky(h, percdamp);
    const Matrix hinv = MatMul(Transpose(hinv_u), hinv_u);
    // M = Cross^T @ Hinv, Shift = W_nk @ M.
    const Matrix cross_t = Transpose(cross);
    const Matrix m = MatMul(cross_t, hinv);
    const Matrix shift = MatMul(w_nk, m);

    Matrix w_shifted(static_cast<size_t>(n_rows),
                     std::vector<double>(static_cast<size_t>(kw)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < kw; ++j) {
        w_shifted[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] +
            shift[static_cast<size_t>(i)][static_cast<size_t>(j)];
      }
    }

    const std::vector<std::vector<double>> codes_nk = GptqQuantizeColumns(
        w_shifted, scale_blocks, c.block_size, h, percdamp, proc_block_size);

    // Back to the stored layout and nibble-packed -- mirrors
    // `codes_orig = codes_nk if weight_transposed else codes_nk.T` plus
    // `_pack_int4` exactly.
    std::vector<double> codes_flat;
    codes_flat.reserve(static_cast<size_t>(dim0 * dim1));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat.push_back(
              codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      }
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat.push_back(
              codes_nk[static_cast<size_t>(j)][static_cast<size_t>(i)]);
        }
      }
    }
    if (codes_flat.size() % 2 != 0) {
      continue;
    }
    optimized.emplace(c.wq_name, PackInt4(codes_flat));
  }

  if (optimized.empty()) {
    return quantized_model;
  }
  onnx::ModelProto corrected = quantized_model;
  for (auto& t : *corrected.mutable_graph()->mutable_initializer()) {
    auto it = optimized.find(t.name());
    if (it == optimized.end()) {
      continue;
    }
    t.set_raw_data(it->second);
  }
  return corrected;
}
