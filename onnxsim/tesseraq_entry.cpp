// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See tesseraq_entry.h for the full rationale (including why this follows
// gptq_entry.h's own two-model, calibration-driven shape, and the
// accepted numerical scope for this being an iterative Adam optimization
// rather than a closed-form computation) and onnxsim/tesseraq.py for the
// technique this ports.

#include "tesseraq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
// from adaround.py's own _find_int4_matmul_candidates). Unlike
// qronos_entry.cpp's own copy, no float_node_index is needed: candidates
// are processed independently here (no cross-layer dependency), matching
// ApplyGptq's own order-independent shape.

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

void SetRawFloatInitializer(onnx::TensorProto* t,
                            const std::vector<float>& data) {
  std::string raw(data.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
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
            "ApplyTesseraq: calibration batch is missing "
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

// --- Small dense matmul kernels ---------------------------------------------
//
// y[S, N] = x[S, K] @ w[N, K]^T -- every closed-form port in this
// codebase avoids a dense S*N*K matmul; TesseraQ's own reconstruction
// loss needs one per Adam iteration.
void YFromXWt(const std::vector<double>& x, int64_t num_samples, int64_t k,
              const std::vector<double>& w, int64_t n_rows,
              std::vector<double>& y) {
  y.assign(static_cast<size_t>(num_samples * n_rows), 0.0);
  for (int64_t s = 0; s < num_samples; ++s) {
    for (int64_t r = 0; r < n_rows; ++r) {
      double acc = 0.0;
      const double* xs = &x[static_cast<size_t>(s * k)];
      const double* wr = &w[static_cast<size_t>(r * k)];
      for (int64_t c = 0; c < k; ++c) {
        acc += xs[c] * wr[c];
      }
      y[static_cast<size_t>(s * n_rows + r)] = acc;
    }
  }
}

// dl_dw_hat[N, K] = dl_dy[S, N]^T @ x[S, K].
void DlDwHatFromDlDyX(const std::vector<double>& dl_dy, int64_t num_samples,
                      int64_t n_rows, const std::vector<double>& x, int64_t k,
                      std::vector<double>& dl_dw_hat) {
  dl_dw_hat.assign(static_cast<size_t>(n_rows * k), 0.0);
  for (int64_t r = 0; r < n_rows; ++r) {
    for (int64_t c = 0; c < k; ++c) {
      double acc = 0.0;
      for (int64_t s = 0; s < num_samples; ++s) {
        acc += dl_dy[static_cast<size_t>(s * n_rows + r)] *
               x[static_cast<size_t>(s * k + c)];
      }
      dl_dw_hat[static_cast<size_t>(r * k + c)] = acc;
    }
  }
}

struct TesseraqResult {
  std::vector<double> codes_nk;          // [N, K], row-major.
  std::vector<double> scale_blocks_opt;  // [N, num_blocks], row-major.
};

// TesseraQ's own PAR/Adam loop -- transcribed scalar-loop-for-scalar-loop
// from tesseraq.py's own _optimize_tesseraq. See tesseraq_entry.h's own
// accepted numerical scope note.
TesseraqResult OptimizeTesseraq(
    const std::vector<double>& w_nk, int64_t n_rows, int64_t k,
    const std::vector<double>& scale_blocks, int64_t num_blocks,
    int64_t block_size, const std::vector<double>& x, int64_t num_samples,
    double n_min, double n_max, int64_t num_iterations, int64_t par_rounds,
    double learning_rate, double scale_learning_rate, double reg_param,
    double warm_start, double beta_start, double beta_end) {
  const int64_t nk = n_rows * k;
  const int64_t nb = n_rows * num_blocks;

  std::vector<double> y_float;
  YFromXWt(x, num_samples, k, w_nk, n_rows, y_float);

  std::vector<double> floor_base(static_cast<size_t>(nk));
  std::vector<double> v(static_cast<size_t>(nk));
  for (int64_t r = 0; r < n_rows; ++r) {
    for (int64_t c = 0; c < k; ++c) {
      const int64_t idx = r * k + c;
      const int64_t blk = c / block_size;
      const double s0 = scale_blocks[static_cast<size_t>(r * num_blocks + blk)];
      const double ratio = w_nk[static_cast<size_t>(idx)] / s0;
      const double fb = std::floor(ratio);
      const double frac = std::clamp(ratio - fb, 1e-4, 1.0 - 1e-4);
      const double sig0 =
          std::clamp((frac - kGamma) / (kZeta - kGamma), 1e-4, 1.0 - 1e-4);
      floor_base[static_cast<size_t>(idx)] = fb;
      v[static_cast<size_t>(idx)] = std::log(sig0 / (1.0 - sig0));
    }
  }

  std::vector<double> log_delta(static_cast<size_t>(nb), 0.0);
  std::vector<uint8_t> hard_mask(static_cast<size_t>(nk), 0);
  std::vector<double> hard_code(static_cast<size_t>(nk), 0.0);
  std::vector<double> m_v(static_cast<size_t>(nk), 0.0),
      v2_v(static_cast<size_t>(nk), 0.0);
  std::vector<double> m_s(static_cast<size_t>(nb), 0.0),
      v2_s(static_cast<size_t>(nb), 0.0);
  constexpr double kBeta1 = 0.9, kBeta2 = 0.999, kAdamEps = 1e-8;

  par_rounds = std::max<int64_t>(1, par_rounds);
  const int64_t iters_per_round =
      std::max<int64_t>(1, num_iterations / par_rounds);
  const int64_t total_iters = iters_per_round * par_rounds;
  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(total_iters) * warm_start);
  const double n_elems = static_cast<double>(num_samples * n_rows);

  int64_t global_t = 0;
  int64_t hard_count = 0;

  std::vector<double> scale_hat_blocks(static_cast<size_t>(nb));
  std::vector<double> h(static_cast<size_t>(nk)),
      dh_dv(static_cast<size_t>(nk));
  std::vector<double> clipped(static_cast<size_t>(nk)),
      w_hat(static_cast<size_t>(nk));
  std::vector<uint8_t> active(static_cast<size_t>(nk));
  std::vector<double> y_hat, dl_dy(static_cast<size_t>(num_samples * n_rows));
  std::vector<double> dl_dw_hat, grad_v(static_cast<size_t>(nk));
  std::vector<double> grad_log_delta(static_cast<size_t>(nb));
  std::vector<double> h_now(static_cast<size_t>(nk));

  for (int64_t round_idx = 0; round_idx < par_rounds; ++round_idx) {
    for (int64_t iter = 0; iter < iters_per_round; ++iter) {
      for (int64_t i = 0; i < nb; ++i) {
        scale_hat_blocks[static_cast<size_t>(i)] =
            scale_blocks[static_cast<size_t>(i)] *
            std::exp(log_delta[static_cast<size_t>(i)]);
      }
      for (int64_t i = 0; i < nk; ++i) {
        const double s = 1.0 / (1.0 + std::exp(-v[static_cast<size_t>(i)]));
        const double raw = s * (kZeta - kGamma) + kGamma;
        const bool active_h = raw > 0.0 && raw < 1.0;
        h[static_cast<size_t>(i)] = std::clamp(raw, 0.0, 1.0);
        dh_dv[static_cast<size_t>(i)] =
            active_h ? s * (1.0 - s) * (kZeta - kGamma) : 0.0;
      }

      for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t c = 0; c < k; ++c) {
          const int64_t idx = r * k + c;
          const int64_t blk = c / block_size;
          const double raw_val = hard_mask[static_cast<size_t>(idx)]
                                     ? hard_code[static_cast<size_t>(idx)]
                                     : floor_base[static_cast<size_t>(idx)] +
                                           h[static_cast<size_t>(idx)];
          const double cl = std::clamp(raw_val, n_min, n_max);
          const bool act = raw_val > n_min && raw_val < n_max &&
                           !hard_mask[static_cast<size_t>(idx)];
          clipped[static_cast<size_t>(idx)] = cl;
          active[static_cast<size_t>(idx)] = act ? 1 : 0;
          w_hat[static_cast<size_t>(idx)] =
              cl * scale_hat_blocks[static_cast<size_t>(r * num_blocks + blk)];
        }
      }

      YFromXWt(x, num_samples, k, w_hat, n_rows, y_hat);
      for (int64_t i = 0; i < num_samples * n_rows; ++i) {
        dl_dy[static_cast<size_t>(i)] =
            2.0 *
            (y_hat[static_cast<size_t>(i)] - y_float[static_cast<size_t>(i)]) /
            n_elems;
      }
      DlDwHatFromDlDyX(dl_dy, num_samples, n_rows, x, k, dl_dw_hat);

      for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t c = 0; c < k; ++c) {
          const int64_t idx = r * k + c;
          const int64_t blk = c / block_size;
          const double scale_hat_val =
              scale_hat_blocks[static_cast<size_t>(r * num_blocks + blk)];
          const double dl_dh =
              dl_dw_hat[static_cast<size_t>(idx)] *
              (active[static_cast<size_t>(idx)] ? scale_hat_val : 0.0);
          grad_v[static_cast<size_t>(idx)] =
              dl_dh * dh_dv[static_cast<size_t>(idx)];
        }
      }

      if (global_t >= warm_start_iters) {
        const double denom = static_cast<double>(
            std::max<int64_t>(1, total_iters - warm_start_iters - 1));
        const double progress =
            static_cast<double>(global_t - warm_start_iters) / denom;
        const double beta = beta_start + (beta_end - beta_start) * progress;
        for (int64_t i = 0; i < nk; ++i) {
          const double u = 2.0 * h[static_cast<size_t>(i)] - 1.0;
          const double abs_u = std::fabs(u);
          const double sign_u = (u > 0.0) - (u < 0.0);
          const double dreg_dh =
              -2.0 * reg_param * beta * sign_u * std::pow(abs_u, beta - 1.0);
          grad_v[static_cast<size_t>(i)] +=
              dreg_dh * dh_dv[static_cast<size_t>(i)];
        }
      }
      for (int64_t i = 0; i < nk; ++i) {
        if (hard_mask[static_cast<size_t>(i)]) {
          grad_v[static_cast<size_t>(i)] = 0.0;
        }
      }

      std::fill(grad_log_delta.begin(), grad_log_delta.end(), 0.0);
      for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t c = 0; c < k; ++c) {
          const int64_t idx = r * k + c;
          const int64_t blk = c / block_size;
          grad_log_delta[static_cast<size_t>(r * num_blocks + blk)] +=
              dl_dw_hat[static_cast<size_t>(idx)] *
              clipped[static_cast<size_t>(idx)];
        }
      }
      for (int64_t i = 0; i < nb; ++i) {
        grad_log_delta[static_cast<size_t>(i)] *=
            scale_hat_blocks[static_cast<size_t>(i)];
      }

      ++global_t;
      const double bias_c1 =
          1.0 - std::pow(kBeta1, static_cast<double>(global_t));
      const double bias_c2 =
          1.0 - std::pow(kBeta2, static_cast<double>(global_t));
      for (int64_t i = 0; i < nk; ++i) {
        const size_t si = static_cast<size_t>(i);
        m_v[si] = kBeta1 * m_v[si] + (1.0 - kBeta1) * grad_v[si];
        v2_v[si] = kBeta2 * v2_v[si] + (1.0 - kBeta2) * grad_v[si] * grad_v[si];
        v[si] -= learning_rate * (m_v[si] / bias_c1) /
                 (std::sqrt(v2_v[si] / bias_c2) + kAdamEps);
      }
      for (int64_t i = 0; i < nb; ++i) {
        const size_t si = static_cast<size_t>(i);
        m_s[si] = kBeta1 * m_s[si] + (1.0 - kBeta1) * grad_log_delta[si];
        v2_s[si] = kBeta2 * v2_s[si] +
                   (1.0 - kBeta2) * grad_log_delta[si] * grad_log_delta[si];
        log_delta[si] -= scale_learning_rate * (m_s[si] / bias_c1) /
                         (std::sqrt(v2_s[si] / bias_c2) + kAdamEps);
      }
    }  // iters_per_round

    for (int64_t i = 0; i < nk; ++i) {
      const double s = 1.0 / (1.0 + std::exp(-v[static_cast<size_t>(i)]));
      const double raw = s * (kZeta - kGamma) + kGamma;
      h_now[static_cast<size_t>(i)] = std::clamp(raw, 0.0, 1.0);
    }

    std::vector<uint8_t> newly(static_cast<size_t>(nk), 0);
    if (round_idx == par_rounds - 1) {
      for (int64_t i = 0; i < nk; ++i) {
        newly[static_cast<size_t>(i)] =
            hard_mask[static_cast<size_t>(i)] ? 0 : 1;
      }
    } else {
      const double target_fraction =
          static_cast<double>(round_idx + 1) / static_cast<double>(par_rounds);
      const int64_t target_count = static_cast<int64_t>(
          RoundHalfToEven(target_fraction * static_cast<double>(nk)));
      const int64_t to_harden = std::max<int64_t>(0, target_count - hard_count);
      if (to_harden > 0) {
        std::vector<int64_t> soft_idx;
        soft_idx.reserve(static_cast<size_t>(nk - hard_count));
        for (int64_t i = 0; i < nk; ++i) {
          if (!hard_mask[static_cast<size_t>(i)]) {
            soft_idx.push_back(i);
          }
        }
        if (to_harden >= static_cast<int64_t>(soft_idx.size())) {
          for (int64_t idx : soft_idx) {
            newly[static_cast<size_t>(idx)] = 1;
          }
        } else {
          std::vector<std::pair<double, int64_t>> conf;
          conf.reserve(soft_idx.size());
          for (int64_t idx : soft_idx) {
            conf.emplace_back(std::fabs(h_now[static_cast<size_t>(idx)] - 0.5),
                              idx);
          }
          std::nth_element(
              conf.begin(), conf.begin() + to_harden, conf.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
          for (int64_t i = 0; i < to_harden; ++i) {
            newly[static_cast<size_t>(conf[static_cast<size_t>(i)].second)] = 1;
          }
        }
      }
    }
    for (int64_t i = 0; i < nk; ++i) {
      if (newly[static_cast<size_t>(i)]) {
        hard_code[static_cast<size_t>(i)] =
            floor_base[static_cast<size_t>(i)] +
            RoundHalfToEven(h_now[static_cast<size_t>(i)]);
        hard_mask[static_cast<size_t>(i)] = 1;
      }
    }
    hard_count = 0;
    for (int64_t i = 0; i < nk; ++i) {
      hard_count += hard_mask[static_cast<size_t>(i)] ? 1 : 0;
    }
  }  // rounds

  TesseraqResult result;
  result.codes_nk.resize(static_cast<size_t>(nk));
  for (int64_t i = 0; i < nk; ++i) {
    result.codes_nk[static_cast<size_t>(i)] =
        std::clamp(hard_code[static_cast<size_t>(i)], n_min, n_max);
  }
  result.scale_blocks_opt.resize(static_cast<size_t>(nb));
  for (int64_t i = 0; i < nb; ++i) {
    result.scale_blocks_opt[static_cast<size_t>(i)] =
        scale_blocks[static_cast<size_t>(i)] *
        std::exp(log_delta[static_cast<size_t>(i)]);
  }
  return result;
}

}  // namespace

onnx::ModelProto ApplyTesseraq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_bits, int64_t num_iterations, int64_t par_rounds,
    double learning_rate, double scale_learning_rate, double reg_param,
    double warm_start, double beta_start, double beta_end) {
  if (num_bits < 2 || num_bits > 4) {
    throw std::invalid_argument(
        "ApplyTesseraq: num_bits must be between 2 and 4, got " +
        std::to_string(num_bits));
  }

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

  const double n_max = static_cast<double>((int64_t{1} << (num_bits - 1)) - 1);
  const double n_min = -n_max;

  std::unordered_map<std::string, std::string> optimized_codes;
  std::unordered_map<std::string, std::vector<float>> optimized_scale;

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
    if (k % c.block_size != 0) {
      continue;  // Ragged block -- quantize_weight_only_int4 never
                 // produces this.
    }
    const int64_t num_samples =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_float_init);
    std::vector<double> w_nk(static_cast<size_t>(n_rows * k));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(i * k + j)] = static_cast<double>(
            c.weight_transposed ? w_flat[static_cast<size_t>(i * k + j)]
                                : w_flat[static_cast<size_t>(j * n_rows + i)]);
      }
    }

    const onnx::GraphProto& q_graph = quantized_model.graph();
    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    const int64_t num_blocks = k / c.block_size;
    std::vector<double> scale_blocks(static_cast<size_t>(n_rows * num_blocks));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_blocks; ++g) {
          scale_blocks[static_cast<size_t>(i * num_blocks + g)] =
              static_cast<double>(
                  s_flat[static_cast<size_t>(i * num_blocks + g)]);
        }
      }
    } else {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_blocks; ++g) {
          scale_blocks[static_cast<size_t>(i * num_blocks + g)] =
              static_cast<double>(s_flat[static_cast<size_t>(g * n_rows + i)]);
        }
      }
    }

    const TesseraqResult res = OptimizeTesseraq(
        w_nk, n_rows, k, scale_blocks, num_blocks, c.block_size, rows.data,
        num_samples, n_min, n_max, num_iterations, par_rounds, learning_rate,
        scale_learning_rate, reg_param, warm_start, beta_start, beta_end);

    // Back to the stored layout -- mirrors `codes_orig = codes_nk if
    // weight_transposed else codes_nk.T` / `scale_orig` exactly.
    std::vector<double> codes_flat;
    codes_flat.reserve(static_cast<size_t>(dim0 * dim1));
    std::vector<float> scale_flat(
        static_cast<size_t>(dim0 * dim1 / c.block_size));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat.push_back(res.codes_nk[static_cast<size_t>(i * dim1 + j)]);
        }
      }
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_blocks; ++g) {
          scale_flat[static_cast<size_t>(i * num_blocks + g)] =
              static_cast<float>(res.scale_blocks_opt[static_cast<size_t>(
                  i * num_blocks + g)]);
        }
      }
    } else {
      // res.codes_nk is [n_rows, k] = [dim1, dim0] here (w_nk = w.T), so
      // codes_orig[i][j] (the stored [dim0, dim1] layout) reads
      // codes_nk[j][i] -- row stride k (== dim0), NOT dim1.
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          codes_flat.push_back(res.codes_nk[static_cast<size_t>(j * dim0 + i)]);
        }
      }
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < num_blocks; ++g) {
          scale_flat[static_cast<size_t>(g * n_rows + i)] = static_cast<float>(
              res.scale_blocks_opt[static_cast<size_t>(i * num_blocks + g)]);
        }
      }
    }
    if (codes_flat.size() % 2 != 0) {
      continue;
    }
    optimized_codes.emplace(c.wq_name, PackInt4(codes_flat));
    optimized_scale.emplace(c.ws_name, std::move(scale_flat));
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
      SetRawFloatInitializer(&t, sit->second);
    }
  }
  return corrected;
}
