// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See adaround_entry.h for the full rationale (including why this follows
// gptq_entry.h's own two-model, calibration-driven shape, and the
// accepted numerical scope for this being an iterative Adam optimization
// rather than a closed-form computation) and onnxsim/adaround.py for the
// technique this ports.

#include "adaround_entry.h"

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

// Rectified-sigmoid relaxation constants -- transcribed from adaround.py's
// own _ZETA/_GAMMA.
constexpr double kZeta = 1.1;
constexpr double kGamma = -0.1;

// --- Candidate matching ----------------------------------------------------
//
// Transcribed from gptq_entry.cpp's own FindInt4MatmulCandidates (itself
// from adaround.py's own _find_int4_matmul_candidates). No node-order
// bookkeeping is needed: candidates are processed independently here (no
// cross-layer dependency), matching ApplyGptq's own order-independent
// shape.

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

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// transcribed from gptq_entry.cpp's own RoundHalfToEven.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4.
std::string PackInt4(const std::vector<double>& codes_flat) {
  std::string packed;
  packed.resize(codes_flat.size() / 2);
  for (size_t i = 0; i < packed.size(); ++i) {
    const auto lo =
        static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i]));
    const auto hi =
        static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i + 1]));
    packed[i] = static_cast<char>((lo & 0xF) | ((hi & 0xF) << 4));
  }
  return packed;
}

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own ActivationRows/
// AccumulateActivationRows.
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
            "ApplyAdaround: calibration batch is missing "
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

// --- Small dense matmul kernels, nested-vector ("Matrix") style ------------
//
// Matches gptq_entry.cpp's/qronos_entry.cpp's own nested
// std::vector<std::vector<double>> convention rather than tesseraq_entry.
// cpp's flat-array one: this file's per-iteration matmuls are the same
// shape as tesseraq_entry.cpp's own, but the safer, bug-resistant
// double-indexing (a lesson learned the hard way porting TesseraQ -- a
// flat-array row-stride mix-up in that port's own output-repacking loop
// silently corrupted the vast majority of codes for one weight layout;
// nested indexing makes that whole bug class impossible) is worth the
// small performance cost at this port's scale.
using Matrix = std::vector<std::vector<double>>;

// y[S, N] = x[S, K] @ w[N, K]^T.
Matrix YFromXWt(const Matrix& x, const Matrix& w) {
  const size_t num_samples = x.size();
  const size_t k = x[0].size();
  const size_t n_rows = w.size();
  Matrix y(num_samples, std::vector<double>(n_rows, 0.0));
  for (size_t s = 0; s < num_samples; ++s) {
    for (size_t r = 0; r < n_rows; ++r) {
      double acc = 0.0;
      for (size_t c = 0; c < k; ++c) {
        acc += x[s][c] * w[r][c];
      }
      y[s][r] = acc;
    }
  }
  return y;
}

// dl_dw_hat[N, K] = dl_dy[S, N]^T @ x[S, K].
Matrix DlDwHatFromDlDyX(const Matrix& dl_dy, const Matrix& x) {
  const size_t num_samples = dl_dy.size();
  const size_t n_rows = dl_dy[0].size();
  const size_t k = x[0].size();
  Matrix out(n_rows, std::vector<double>(k, 0.0));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      double acc = 0.0;
      for (size_t s = 0; s < num_samples; ++s) {
        acc += dl_dy[s][r] * x[s][c];
      }
      out[r][c] = acc;
    }
  }
  return out;
}

// AdaRound's own Adam loop -- transcribed scalar-loop-for-scalar-loop
// from adaround.py's own _optimize_rounding. See adaround_entry.h's own
// accepted numerical scope note.
Matrix OptimizeRounding(const Matrix& w_nk, const Matrix& scale_nk,
                        const Matrix& x, double n_min, double n_max,
                        int64_t num_iterations, double learning_rate,
                        double reg_param, double warm_start, double beta_start,
                        double beta_end) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_samples = x.size();

  const Matrix y_float = YFromXWt(x, w_nk);

  Matrix floor_base(n_rows, std::vector<double>(k));
  Matrix v(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const double ratio = w_nk[r][c] / scale_nk[r][c];
      const double fb = std::floor(ratio);
      const double frac = std::clamp(ratio - fb, 1e-4, 1.0 - 1e-4);
      const double sig0 =
          std::clamp((frac - kGamma) / (kZeta - kGamma), 1e-4, 1.0 - 1e-4);
      floor_base[r][c] = fb;
      v[r][c] = std::log(sig0 / (1.0 - sig0));
    }
  }

  Matrix m(n_rows, std::vector<double>(k, 0.0));
  Matrix v2(n_rows, std::vector<double>(k, 0.0));
  constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEps = 1e-8;

  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(num_iterations) * warm_start);
  const double n_elems = static_cast<double>(num_samples * n_rows);

  for (int64_t t = 0; t < num_iterations; ++t) {
    Matrix h(n_rows, std::vector<double>(k));
    Matrix dh_dv(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double s = 1.0 / (1.0 + std::exp(-v[r][c]));
        const double raw = s * (kZeta - kGamma) + kGamma;
        const bool active_h = raw > 0.0 && raw < 1.0;
        h[r][c] = std::clamp(raw, 0.0, 1.0);
        dh_dv[r][c] = active_h ? s * (1.0 - s) * (kZeta - kGamma) : 0.0;
      }
    }

    Matrix w_hat(n_rows, std::vector<double>(k));
    std::vector<std::vector<uint8_t>> active_w(n_rows, std::vector<uint8_t>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double raw2 = floor_base[r][c] + h[r][c];
        const double cl = std::clamp(raw2, n_min, n_max);
        const bool act = raw2 > n_min && raw2 < n_max;
        w_hat[r][c] = cl * scale_nk[r][c];
        active_w[r][c] = act ? 1 : 0;
      }
    }

    const Matrix y_hat = YFromXWt(x, w_hat);
    Matrix dl_dy(num_samples, std::vector<double>(n_rows));
    for (size_t s = 0; s < num_samples; ++s) {
      for (size_t r = 0; r < n_rows; ++r) {
        dl_dy[s][r] = 2.0 * (y_hat[s][r] - y_float[s][r]) / n_elems;
      }
    }
    const Matrix dl_dw_hat = DlDwHatFromDlDyX(dl_dy, x);

    Matrix grad(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        const double dl_dh =
            dl_dw_hat[r][c] * (active_w[r][c] ? scale_nk[r][c] : 0.0);
        grad[r][c] = dl_dh * dh_dv[r][c];
      }
    }

    if (t >= warm_start_iters) {
      const double denom = static_cast<double>(
          std::max<int64_t>(1, num_iterations - warm_start_iters - 1));
      const double progress = static_cast<double>(t - warm_start_iters) / denom;
      const double beta = beta_start + (beta_end - beta_start) * progress;
      for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < k; ++c) {
          const double u = 2.0 * h[r][c] - 1.0;
          const double abs_u = std::fabs(u);
          const double sign_u = (u > 0.0) - (u < 0.0);
          const double dreg_dh =
              -2.0 * reg_param * beta * sign_u * std::pow(abs_u, beta - 1.0);
          grad[r][c] += dreg_dh * dh_dv[r][c];
        }
      }
    }

    const double bias_c1 =
        1.0 - std::pow(kAdamBeta1, static_cast<double>(t + 1));
    const double bias_c2 =
        1.0 - std::pow(kAdamBeta2, static_cast<double>(t + 1));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t c = 0; c < k; ++c) {
        m[r][c] = kAdamBeta1 * m[r][c] + (1.0 - kAdamBeta1) * grad[r][c];
        v2[r][c] = kAdamBeta2 * v2[r][c] +
                   (1.0 - kAdamBeta2) * grad[r][c] * grad[r][c];
        v[r][c] -= learning_rate * (m[r][c] / bias_c1) /
                   (std::sqrt(v2[r][c] / bias_c2) + kAdamEps);
      }
    }
  }

  Matrix codes(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const double s = 1.0 / (1.0 + std::exp(-v[r][c]));
      const double raw = s * (kZeta - kGamma) + kGamma;
      const double h_final = std::clamp(raw, 0.0, 1.0);
      codes[r][c] =
          std::clamp(floor_base[r][c] + RoundHalfToEven(h_final), n_min, n_max);
    }
  }
  return codes;
}

}  // namespace

onnx::ModelProto ApplyAdaround(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations, double learning_rate, double reg_param,
    double warm_start, double beta_start, double beta_end) {
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

  constexpr double kNMin = -7.0, kNMax = 7.0;

  std::unordered_map<std::string, std::string> optimized;
  for (const auto& c : candidates) {
    auto ait = activations.find(c.float_x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable activation -- leave untouched.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_float_init =
        f_graph.initializer(f_init_index[c.w_float_name]);
    const int64_t dim0 = w_float_init.dims(0);
    const int64_t dim1 = w_float_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;  // Activation's feature dim doesn't match K -- leave
                 // untouched.
    }
    const int64_t num_samples =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

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

    const onnx::GraphProto& q_graph = quantized_model.graph();
    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    if (k % c.block_size != 0) {
      continue;  // Ragged block -- quantize_weight_only_int4 never
                 // produces this.
    }
    const int64_t num_blocks = k / c.block_size;
    // scale_nk: the block-wise scale, broadcast up to full [N, K] --
    // mirrors `np.repeat(scale_blocks, block_size, axis=1)` exactly.
    Matrix scale_nk(static_cast<size_t>(n_rows),
                    std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        const int64_t blk = j / c.block_size;
        const double s_val =
            c.weight_transposed
                ? static_cast<double>(
                      s_flat[static_cast<size_t>(i * num_blocks + blk)])
                : static_cast<double>(
                      s_flat[static_cast<size_t>(blk * n_rows + i)]);
        scale_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] = s_val;
      }
    }

    Matrix x(static_cast<size_t>(num_samples),
             std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < num_samples; ++r) {
      for (int64_t c2 = 0; c2 < k; ++c2) {
        x[static_cast<size_t>(r)][static_cast<size_t>(c2)] =
            rows.data[static_cast<size_t>(r * k + c2)];
      }
    }

    const Matrix codes_nk = OptimizeRounding(
        w_nk, scale_nk, x, kNMin, kNMax, num_iterations, learning_rate,
        reg_param, warm_start, beta_start, beta_end);

    // Back to the stored layout -- mirrors `codes_orig = codes_nk if
    // weight_transposed else codes_nk.T` exactly. Nested-vector indexing
    // (codes_nk[i][j]) rather than a flat-array stride computation -- see
    // this file's own comment on Matrix for why.
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
