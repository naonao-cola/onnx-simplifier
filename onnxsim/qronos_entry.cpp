// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See qronos_entry.h for the full rationale (including why this follows
// gptq_entry.h's own two-model, calibration-driven shape but probes
// `executor` once per matched layer rather than once up front) and
// onnxsim/qronos.py for the technique this ports.

#include "qronos_entry.h"

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

// --- Candidate matching ----------------------------------------------------
//
// Transcribed from gptq_entry.cpp's own FindInt4MatmulCandidates
// (itself from adaround.py's own _find_int4_matmul_candidates), with one
// addition: `float_node_index`, the matched node's own position in
// `float_model.graph.node()` -- apply_qronos's own forward-execution
// processing order (a topologically-sorted ONNX graph's node order is a
// valid one), unlike ApplyGptq, which processes candidates in whatever
// order FindInt4MatmulCandidates itself returns them (order-independent
// there, since every layer's Hessian comes from the untouched float
// model regardless of processing order).

struct Candidate {
  std::string output_name;
  std::string float_x_name;
  std::string w_float_name;
  std::string wq_name;
  std::string ws_name;
  int64_t block_size;
  bool weight_transposed;
  int float_node_index;
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
                          dq.input(1), block_size, weight_transposed,
                          fit->second});
  }
  return candidates;
}

// --- Tensor <-> flat float buffer -------------------------------------------
//
// Transcribed from gptq_entry.cpp's own ReadFloatTensor.
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

// Transcribed verbatim from gptq_entry.cpp's own GptqQuantizeColumns.
Matrix GptqQuantizeColumns(const Matrix& w_nk, const Matrix& scale_blocks,
                           int64_t block_size, const Matrix& h, double percdamp,
                           int64_t proc_block_size) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const Matrix hinv = InverseHessianCholesky(h, percdamp);

  Matrix codes(n, std::vector<double>(k, 0.0));
  Matrix w_work = w_nk;

  for (int64_t block_start = 0; block_start < static_cast<int64_t>(k);
       block_start += proc_block_size) {
    const int64_t block_end =
        std::min(block_start + proc_block_size, static_cast<int64_t>(k));
    const size_t bs = static_cast<size_t>(block_end - block_start);
    Matrix w1(n, std::vector<double>(bs));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < bs; ++j) {
        w1[i][j] = w_work[i][static_cast<size_t>(block_start) + j];
      }
    }
    Matrix err1(n, std::vector<double>(bs, 0.0));

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

// Transcribed verbatim from gptq_entry.cpp's own PackInt4.
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

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own ActivationRows/
// AccumulateActivationRows: every observed 2-D-or-higher FLOAT32 activation
// flattened to rows (`reshape(-1, K)`, exact) and concatenated across
// batches. Rank < 2 resolves to no rows at all. Unlike ApplyGptq, which
// calls this once for every candidate's probe name up front, ApplyQronos
// also calls this once per candidate against the *current* working model
// (see the main loop below), mirroring apply_qronos's own per-layer
// `_add_probe_outputs(working, [probe_name])` re-probe exactly.
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
        calibration_data,
    const char* error_prefix) {
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
            std::string(error_prefix) +
            ": calibration batch is missing required graph input '" +
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

onnx::ModelProto ApplyQronos(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp, int64_t proc_block_size) {
  std::unordered_map<std::string, int> q_init_index;
  std::unordered_map<std::string, int> f_init_index;
  std::vector<Candidate> candidates = FindInt4MatmulCandidates(
      float_model, quantized_model, q_init_index, f_init_index);
  if (candidates.empty()) {
    return quantized_model;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.float_node_index < b.float_node_index;
            });
  const onnx::GraphProto& f_graph = float_model.graph();

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.float_x_name);
  }
  std::unordered_map<std::string, ActivationRows> float_activations;
  AccumulateActivationRows(float_activations, executor, float_model,
                           probe_names, calibration_data, "ApplyQronos");

  onnx::ModelProto working = quantized_model;
  bool any_optimized = false;

  for (const auto& c : candidates) {
    auto fait = float_activations.find(c.float_x_name);
    if (fait == float_activations.end() || !fait->second.ok) {
      continue;  // No usable float activation -- leave untouched.
    }
    const ActivationRows& float_rows = fait->second;

    std::unordered_map<std::string, ActivationRows> quant_activations;
    AccumulateActivationRows(quant_activations, executor, working,
                             {c.float_x_name}, calibration_data, "ApplyQronos");
    auto qait = quant_activations.find(c.float_x_name);
    if (qait == quant_activations.end() || !qait->second.ok) {
      continue;  // No usable quantized-side activation -- leave untouched.
    }
    const ActivationRows& quant_rows = qait->second;
    if (quant_rows.k != float_rows.k ||
        quant_rows.data.size() != float_rows.data.size()) {
      continue;  // Not sample-aligned with the float side -- leave
                 // untouched, mirrors apply_qronos's own alignment guard.
    }

    const onnx::TensorProto& w_float_init =
        f_graph.initializer(f_init_index[c.w_float_name]);
    const int64_t dim0 = w_float_init.dims(0);
    const int64_t dim1 = w_float_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (float_rows.k != k) {
      continue;  // Activation's feature dim doesn't match K -- leave
                 // untouched.
    }
    if (k % c.block_size != 0) {
      continue;
    }
    const int64_t num_rows =
        static_cast<int64_t>(float_rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_float_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
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

    const onnx::TensorProto& ws_init =
        working.graph().initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    const int64_t num_groups = k / c.block_size;
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

    // dx = x_quant - x_float; h = x_quant^T @ x_quant -- the Hessian of
    // the *real* (corrupted) input this layer actually sees.
    Matrix dx(static_cast<size_t>(num_rows),
              std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        dx[static_cast<size_t>(r)][static_cast<size_t>(j)] =
            quant_rows.data[static_cast<size_t>(r * k + j)] -
            float_rows.data[static_cast<size_t>(r * k + j)];
      }
    }
    Matrix h(static_cast<size_t>(k),
             std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi = quant_rows.data[static_cast<size_t>(r * k + i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * quant_rows.data[static_cast<size_t>(r * k + j)];
        }
      }
    }

    // dxTxq = dx^T @ x_quant -- [K, K].
    Matrix dx_t_xq(static_cast<size_t>(k),
                   std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double dxi = dx[static_cast<size_t>(r)][static_cast<size_t>(i)];
        if (dxi == 0.0) {
          continue;
        }
        for (int64_t j = 0; j < k; ++j) {
          dx_t_xq[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              dxi * quant_rows.data[static_cast<size_t>(r * k + j)];
        }
      }
    }

    // h_inv = U^T @ U, U = InverseHessianCholesky(h, percdamp).
    const Matrix u = InverseHessianCholesky(h, percdamp);
    Matrix h_inv(static_cast<size_t>(k),
                 std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += u[static_cast<size_t>(kk)][static_cast<size_t>(i)] *
                 u[static_cast<size_t>(kk)][static_cast<size_t>(j)];
        }
        h_inv[static_cast<size_t>(i)][static_cast<size_t>(j)] = acc;
      }
    }

    // w_opt_nk = w_nk - w_nk @ dxTxq @ h_inv.
    Matrix tmp(static_cast<size_t>(n_rows),
               std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += w_nk[static_cast<size_t>(r)][static_cast<size_t>(kk)] *
                 dx_t_xq[static_cast<size_t>(kk)][static_cast<size_t>(j)];
        }
        tmp[static_cast<size_t>(r)][static_cast<size_t>(j)] = acc;
      }
    }
    Matrix w_opt_nk(static_cast<size_t>(n_rows),
                    std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += tmp[static_cast<size_t>(r)][static_cast<size_t>(kk)] *
                 h_inv[static_cast<size_t>(kk)][static_cast<size_t>(j)];
        }
        w_opt_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] =
            w_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] - acc;
      }
    }

    const Matrix codes_nk = GptqQuantizeColumns(
        w_opt_nk, scale_blocks, c.block_size, h, percdamp, proc_block_size);

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
    // Mutate `working` in place -- this candidate's own correction must
    // be visible to every *subsequent* candidate's own re-probe (mirrors
    // apply_qronos's own progressive `working_init[...].raw_data =`
    // assignment exactly).
    working.mutable_graph()
        ->mutable_initializer(q_init_index[c.wq_name])
        ->set_raw_data(PackInt4(codes_flat));
    any_optimized = true;
  }

  if (!any_optimized) {
    return quantized_model;
  }
  return working;
}
