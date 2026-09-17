// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See autoround_entry.h for the full rationale (including why this
// duplicates adaround_entry.cpp's own candidate-matching/tensor-conversion/
// dense-matmul helpers rather than sharing them, and the accepted
// numerical scope for this being a two-parameter-group iterative Adam
// optimization) and onnxsim/autoround.py for the technique this ports.

#include "autoround_entry.h"

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

// Rectified-sigmoid relaxation constants -- transcribed from adaround.py's
// own _ZETA/_GAMMA (autoround.py imports them from there unchanged).
constexpr double kZeta = 1.1;
constexpr double kGamma = -0.1;

// --- Candidate matching ----------------------------------------------------
//
// Identical to adaround_entry.cpp's own FindInt4MatmulCandidates.

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
// Identical to adaround_entry.cpp's own ReadFloatTensor/ReplaceFloatData
// (the latter transcribed from affinequant_entry.cpp's own, since
// AutoRound -- unlike AdaRound -- may also rewrite a scale initializer).

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

std::string FloatVectorToRawData(const std::vector<double>& data) {
  std::vector<float> as_float(data.begin(), data.end());
  std::string raw(as_float.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), as_float.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  return raw;
}

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// transcribed from adaround_entry.cpp's own RoundHalfToEven.
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
// Identical to adaround_entry.cpp's own ActivationRows/
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
            "ApplyAutoround: calibration batch is missing "
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
// Identical to adaround_entry.cpp's own Matrix/YFromXWt/DlDwHatFromDlDyX.

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

// h(v), the rectified sigmoid, and its derivative -- adaround.py's own
// _h_and_dhdv, transcribed elementwise over a Matrix.
void HAndDhDv(const Matrix& v, Matrix& h, Matrix& dh_dv) {
  const size_t n_rows = v.size();
  const size_t k = v[0].size();
  h.assign(n_rows, std::vector<double>(k));
  dh_dv.assign(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const double s = 1.0 / (1.0 + std::exp(-v[r][c]));
      const double raw = s * (kZeta - kGamma) + kGamma;
      const bool active = raw > 0.0 && raw < 1.0;
      h[r][c] = std::clamp(raw, 0.0, 1.0);
      dh_dv[r][c] = active ? s * (1.0 - s) * (kZeta - kGamma) : 0.0;
    }
  }
}

// AdaRound's own initial relaxation point -- adaround.py's own
// _init_relaxation, transcribed. Returns (v, floor_base).
void InitRelaxation(const Matrix& w_nk, const Matrix& scale_nk, Matrix& v,
                    Matrix& floor_base) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  v.assign(n_rows, std::vector<double>(k));
  floor_base.assign(n_rows, std::vector<double>(k));
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
}

// AdaRound's own fixed-scale Adam loop -- adaround_entry.cpp's own
// OptimizeRounding, transcribed verbatim (this TU cannot include that
// anonymous-namespace function). Used here as AutoRound's own safety-net
// candidate (see autoround_entry.h's own docstring).
Matrix OptimizeRoundingFixedScale(const Matrix& w_nk, const Matrix& scale_nk,
                                  const Matrix& x, double n_min, double n_max,
                                  int64_t num_iterations, double learning_rate,
                                  double reg_param, double warm_start,
                                  double beta_start, double beta_end) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_samples = x.size();

  const Matrix y_float = YFromXWt(x, w_nk);

  Matrix v, floor_base;
  InitRelaxation(w_nk, scale_nk, v, floor_base);

  Matrix m(n_rows, std::vector<double>(k, 0.0));
  Matrix v2(n_rows, std::vector<double>(k, 0.0));
  constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEps = 1e-8;

  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(num_iterations) * warm_start);
  const double n_elems = static_cast<double>(num_samples * n_rows);

  for (int64_t t = 0; t < num_iterations; ++t) {
    Matrix h, dh_dv;
    HAndDhDv(v, h, dh_dv);

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

// --- AutoRound's own clip-ratio reparametrization ---------------------------
//
// autoround.py's own _clip_ratio_and_dratio_dc, transcribed elementwise.
void ClipRatioAndDratioDc(const Matrix& c, double cmin, double cmax,
                          Matrix& ratio, Matrix& dratio_dc) {
  const size_t n_rows = c.size();
  const size_t num_blocks = c[0].size();
  ratio.assign(n_rows, std::vector<double>(num_blocks));
  dratio_dc.assign(n_rows, std::vector<double>(num_blocks));
  const double span = cmax - cmin;
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t b = 0; b < num_blocks; ++b) {
      const double s = 1.0 / (1.0 + std::exp(-c[r][b]));
      ratio[r][b] = s * span + cmin;
      dratio_dc[r][b] = s * (1.0 - s) * span;
    }
  }
}

// Broadcasts a [N, num_blocks] block matrix up to [N, K] by repeating each
// block's own value `block_size` times -- mirrors
// `np.repeat(scale_blocks, block_size, axis=1)[:, :k]` exactly.
Matrix BroadcastBlocks(const Matrix& blocks, int64_t block_size, size_t k) {
  const size_t n_rows = blocks.size();
  Matrix out(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < k; ++c) {
      out[r][c] =
          blocks[r][static_cast<size_t>(static_cast<int64_t>(c) / block_size)];
    }
  }
  return out;
}

// autoround.py's own _joint_loop: `num_iterations` Adam steps on both the
// rounding relaxation `v` and the clip-ratio parameter `c` jointly. `v`/`c`
// are updated in place.
void JointLoop(const Matrix& w_nk, const Matrix& scale_blocks,
               int64_t block_size, const Matrix& x, const Matrix& y_float,
               Matrix& v, Matrix& c, double n_min, double n_max,
               int64_t num_iterations, double learning_rate,
               double clip_learning_rate, double reg_param, double warm_start,
               double beta_start, double beta_end, double cmin, double cmax) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_blocks = scale_blocks[0].size();
  const size_t num_samples = x.size();

  Matrix m_v(n_rows, std::vector<double>(k, 0.0));
  Matrix v2_v(n_rows, std::vector<double>(k, 0.0));
  Matrix m_c(n_rows, std::vector<double>(num_blocks, 0.0));
  Matrix v2_c(n_rows, std::vector<double>(num_blocks, 0.0));
  constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEps = 1e-8;

  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(num_iterations) * warm_start);
  const double n_elems = static_cast<double>(num_samples * n_rows);

  for (int64_t t = 0; t < num_iterations; ++t) {
    Matrix clip_ratio, dratio_dc;
    ClipRatioAndDratioDc(c, cmin, cmax, clip_ratio, dratio_dc);

    Matrix scale_eff_blocks(n_rows, std::vector<double>(num_blocks));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t b = 0; b < num_blocks; ++b) {
        scale_eff_blocks[r][b] = scale_blocks[r][b] * clip_ratio[r][b];
      }
    }
    const Matrix scale_eff = BroadcastBlocks(scale_eff_blocks, block_size, k);

    Matrix h, dh_dv;
    HAndDhDv(v, h, dh_dv);

    Matrix ratio_wk(n_rows, std::vector<double>(k));
    Matrix floor_base(n_rows, std::vector<double>(k));
    Matrix code(n_rows, std::vector<double>(k));
    std::vector<std::vector<uint8_t>> active(n_rows, std::vector<uint8_t>(k));
    Matrix w_hat(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t col = 0; col < k; ++col) {
        const double rw = w_nk[r][col] / scale_eff[r][col];
        ratio_wk[r][col] = rw;
        const double fb = std::floor(rw);
        floor_base[r][col] = fb;
        const double raw = fb + h[r][col];
        const bool act = raw > n_min && raw < n_max;
        const double cl = std::clamp(raw, n_min, n_max);
        code[r][col] = cl;
        active[r][col] = act ? 1 : 0;
        w_hat[r][col] = cl * scale_eff[r][col];
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

    // Rounding gradient: identical derivation to AdaRound's own,
    // scale_eff standing in for the (there, fixed) scale.
    Matrix grad_v(n_rows, std::vector<double>(k));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t col = 0; col < k; ++col) {
        const double dl_dh =
            dl_dw_hat[r][col] * (active[r][col] ? scale_eff[r][col] : 0.0);
        grad_v[r][col] = dl_dh * dh_dv[r][col];
      }
    }
    if (t >= warm_start_iters) {
      const double denom = static_cast<double>(
          std::max<int64_t>(1, num_iterations - warm_start_iters - 1));
      const double progress = static_cast<double>(t - warm_start_iters) / denom;
      const double beta = beta_start + (beta_end - beta_start) * progress;
      for (size_t r = 0; r < n_rows; ++r) {
        for (size_t col = 0; col < k; ++col) {
          const double u = 2.0 * h[r][col] - 1.0;
          const double abs_u = std::fabs(u);
          const double sign_u = (u > 0.0) - (u < 0.0);
          const double dreg_dh =
              -2.0 * reg_param * beta * sign_u * std::pow(abs_u, beta - 1.0);
          grad_v[r][col] += dreg_dh * dh_dv[r][col];
        }
      }
    }

    // Clip-ratio gradient: LSQ-style d(w_hat)/d(scale), block-summed then
    // chained through scale_eff_blocks = scale_blocks * clip_ratio(c).
    Matrix dl_ds_eff_blocks(n_rows, std::vector<double>(num_blocks, 0.0));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t col = 0; col < k; ++col) {
        const double dw_hat_ds_eff =
            active[r][col] ? (code[r][col] - ratio_wk[r][col]) : code[r][col];
        const double dl_ds_eff = dl_dw_hat[r][col] * dw_hat_ds_eff;
        const size_t blk =
            static_cast<size_t>(static_cast<int64_t>(col) / block_size);
        dl_ds_eff_blocks[r][blk] += dl_ds_eff;
      }
    }
    Matrix grad_c(n_rows, std::vector<double>(num_blocks));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t b = 0; b < num_blocks; ++b) {
        grad_c[r][b] =
            dl_ds_eff_blocks[r][b] * scale_blocks[r][b] * dratio_dc[r][b];
      }
    }

    const double bias_c1 =
        1.0 - std::pow(kAdamBeta1, static_cast<double>(t + 1));
    const double bias_c2 =
        1.0 - std::pow(kAdamBeta2, static_cast<double>(t + 1));
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t col = 0; col < k; ++col) {
        m_v[r][col] =
            kAdamBeta1 * m_v[r][col] + (1.0 - kAdamBeta1) * grad_v[r][col];
        v2_v[r][col] = kAdamBeta2 * v2_v[r][col] +
                       (1.0 - kAdamBeta2) * grad_v[r][col] * grad_v[r][col];
        v[r][col] -= learning_rate * (m_v[r][col] / bias_c1) /
                     (std::sqrt(v2_v[r][col] / bias_c2) + kAdamEps);
      }
      for (size_t b = 0; b < num_blocks; ++b) {
        m_c[r][b] = kAdamBeta1 * m_c[r][b] + (1.0 - kAdamBeta1) * grad_c[r][b];
        v2_c[r][b] = kAdamBeta2 * v2_c[r][b] +
                     (1.0 - kAdamBeta2) * grad_c[r][b] * grad_c[r][b];
        c[r][b] -= clip_learning_rate * (m_c[r][b] / bias_c1) /
                   (std::sqrt(v2_c[r][b] / bias_c2) + kAdamEps);
      }
    }
  }
}

// autoround.py's own _autoround_results: the deployable values the two
// optimized parameter groups have settled on.
void AutoroundResults(const Matrix& w_nk, const Matrix& scale_blocks,
                      int64_t block_size, const Matrix& v, const Matrix& c,
                      double n_min, double n_max, double cmin, double cmax,
                      Matrix& codes, Matrix& scale_blocks_opt,
                      Matrix& scale_eff) {
  const size_t n_rows = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_blocks = scale_blocks[0].size();

  Matrix clip_ratio, dratio_dc_unused;
  ClipRatioAndDratioDc(c, cmin, cmax, clip_ratio, dratio_dc_unused);
  scale_blocks_opt.assign(n_rows, std::vector<double>(num_blocks));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t b = 0; b < num_blocks; ++b) {
      scale_blocks_opt[r][b] = scale_blocks[r][b] * clip_ratio[r][b];
    }
  }
  scale_eff = BroadcastBlocks(scale_blocks_opt, block_size, k);

  Matrix h, dh_dv_unused;
  HAndDhDv(v, h, dh_dv_unused);
  codes.assign(n_rows, std::vector<double>(k));
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t col = 0; col < k; ++col) {
      const double floor_final = std::floor(w_nk[r][col] / scale_eff[r][col]);
      codes[r][col] =
          std::clamp(floor_final + RoundHalfToEven(h[r][col]), n_min, n_max);
    }
  }
}

}  // namespace

onnx::ModelProto ApplyAutoround(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations, double learning_rate, double clip_learning_rate,
    double reg_param, double warm_start, double beta_start, double beta_end,
    double clip_ratio_min, double clip_ratio_max) {
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

  std::unordered_map<std::string, std::string> optimized_codes;
  std::unordered_map<std::string, std::string> optimized_scale;
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
    Matrix scale_blocks(static_cast<size_t>(n_rows),
                        std::vector<double>(static_cast<size_t>(num_blocks)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(b)] =
            c.weight_transposed
                ? static_cast<double>(
                      s_flat[static_cast<size_t>(i * num_blocks + b)])
                : static_cast<double>(
                      s_flat[static_cast<size_t>(b * n_rows + i)]);
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

    // --- _optimize_rounding_and_clip, transcribed. ---
    const Matrix y_float = YFromXWt(x, w_nk);
    const Matrix scale_nk0 =
        BroadcastBlocks(scale_blocks, c.block_size, static_cast<size_t>(k));
    Matrix v, floor_base_unused;
    InitRelaxation(w_nk, scale_nk0, v, floor_base_unused);
    Matrix c_param(static_cast<size_t>(n_rows),
                   std::vector<double>(static_cast<size_t>(num_blocks), 0.0));

    JointLoop(w_nk, scale_blocks, c.block_size, x, y_float, v, c_param, kNMin,
              kNMax, num_iterations, learning_rate, clip_learning_rate,
              reg_param, warm_start, beta_start, beta_end, clip_ratio_min,
              clip_ratio_max);

    Matrix codes_joint, scale_blocks_joint, scale_eff_joint;
    AutoroundResults(w_nk, scale_blocks, c.block_size, v, c_param, kNMin, kNMax,
                     clip_ratio_min, clip_ratio_max, codes_joint,
                     scale_blocks_joint, scale_eff_joint);

    const Matrix codes_ada_only = OptimizeRoundingFixedScale(
        w_nk, scale_nk0, x, kNMin, kNMax, num_iterations, learning_rate,
        reg_param, warm_start, beta_start, beta_end);

    // _keep_better_of, transcribed: pick whichever candidate has the lower
    // measured reconstruction error on x.
    double loss_joint = 0.0, loss_ada = 0.0;
    {
      Matrix w_hat_joint(codes_joint.size(),
                         std::vector<double>(codes_joint[0].size()));
      for (size_t r = 0; r < w_hat_joint.size(); ++r) {
        for (size_t col = 0; col < w_hat_joint[0].size(); ++col) {
          w_hat_joint[r][col] = codes_joint[r][col] * scale_eff_joint[r][col];
        }
      }
      const Matrix y_hat_joint = YFromXWt(x, w_hat_joint);
      double sum_joint = 0.0;
      for (size_t s = 0; s < y_hat_joint.size(); ++s) {
        for (size_t r = 0; r < y_hat_joint[0].size(); ++r) {
          const double d = y_hat_joint[s][r] - y_float[s][r];
          sum_joint += d * d;
        }
      }
      loss_joint = sum_joint / static_cast<double>(y_hat_joint.size() *
                                                   y_hat_joint[0].size());

      Matrix w_hat_ada(codes_ada_only.size(),
                       std::vector<double>(codes_ada_only[0].size()));
      for (size_t r = 0; r < w_hat_ada.size(); ++r) {
        for (size_t col = 0; col < w_hat_ada[0].size(); ++col) {
          w_hat_ada[r][col] = codes_ada_only[r][col] * scale_nk0[r][col];
        }
      }
      const Matrix y_hat_ada = YFromXWt(x, w_hat_ada);
      double sum_ada = 0.0;
      for (size_t s = 0; s < y_hat_ada.size(); ++s) {
        for (size_t r = 0; r < y_hat_ada[0].size(); ++r) {
          const double d = y_hat_ada[s][r] - y_float[s][r];
          sum_ada += d * d;
        }
      }
      loss_ada =
          sum_ada / static_cast<double>(y_hat_ada.size() * y_hat_ada[0].size());
    }

    const bool use_ada = loss_ada <= loss_joint;
    const Matrix& codes_nk = use_ada ? codes_ada_only : codes_joint;
    const Matrix& scale_blocks_final =
        use_ada ? scale_blocks : scale_blocks_joint;

    // Back to the stored layout -- mirrors `codes_orig = codes_nk if
    // weight_transposed else codes_nk.T` exactly.
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
    optimized_codes.emplace(c.wq_name, PackInt4(codes_flat));

    // Scale, back to the stored layout -- mirrors `scale_orig =
    // scale_blocks_new if weight_transposed else scale_blocks_new.T`.
    std::vector<double> scale_flat;
    scale_flat.reserve(static_cast<size_t>(n_rows * num_blocks));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t b = 0; b < num_blocks; ++b) {
          scale_flat.push_back(scale_blocks_final[static_cast<size_t>(i)]
                                                 [static_cast<size_t>(b)]);
        }
      }
    } else {
      for (int64_t b = 0; b < num_blocks; ++b) {
        for (int64_t i = 0; i < n_rows; ++i) {
          scale_flat.push_back(scale_blocks_final[static_cast<size_t>(i)]
                                                 [static_cast<size_t>(b)]);
        }
      }
    }
    optimized_scale.emplace(c.ws_name, FloatVectorToRawData(scale_flat));
  }

  if (optimized_codes.empty() && optimized_scale.empty()) {
    return quantized_model;
  }
  onnx::ModelProto corrected = quantized_model;
  for (auto& t : *corrected.mutable_graph()->mutable_initializer()) {
    auto cit = optimized_codes.find(t.name());
    if (cit != optimized_codes.end()) {
      t.set_raw_data(cit->second);
      continue;
    }
    auto sit = optimized_scale.find(t.name());
    if (sit != optimized_scale.end()) {
      // Unlike `wq`'s own INT4 initializer (always raw_data-encoded, by
      // construction, from quantize_weight_only_int4 -- see
      // adaround_entry.cpp's own identical unconditional set_raw_data),
      // the FLOAT scale initializer may have arrived with its values in
      // `float_data` instead of `raw_data` (e.g. small enough for
      // onnx.numpy_helper.from_array's own small-tensor convention).
      // TensorProto's value fields are NOT a oneof -- set_raw_data alone
      // would leave a stale `float_data` alongside the new `raw_data`,
      // which onnx.checker rejects ("should contain one and only one
      // value field"). Clear every other value field first, mirroring
      // affinequant_entry.cpp's own ReplaceFloatData.
      t.clear_float_data();
      t.clear_int32_data();
      t.clear_int64_data();
      t.clear_double_data();
      t.set_raw_data(sit->second);
    }
  }
  return corrected;
}
