// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See brecq_entry.h for the full rationale (including why this follows
// adaround_entry.h's own two-model, calibration-driven, rectified-sigmoid
// relaxation shape, extended to a jointly-optimized block of layers) and
// onnxsim/brecq.py for the technique this ports.
//
// FindInt4MatmulCandidates/ReadFloatTensor/RoundHalfToEven/PackInt4/
// YFromXWt/DlDwHatFromDlDyX are transcribed from adaround_entry.cpp's own
// identical helpers -- see gptaq_entry.cpp's own top-of-file comment for
// why they aren't shared via a common header (matches this codebase's
// established convention: every *_entry.cpp is self-contained).

#include "brecq_entry.h"

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

using Matrix = std::vector<std::vector<double>>;

constexpr double kZeta = 1.1;
constexpr double kGamma = -0.1;
constexpr double kNMin = -7.0;
constexpr double kNMax = 7.0;

// --- Candidate matching (transcribed from adaround_entry.cpp's own) -------

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

std::unordered_map<std::string, int> BuildOutputIndex(
    const onnx::GraphProto& g) {
  std::unordered_map<std::string, int> m;
  for (int i = 0; i < g.node_size(); ++i) {
    const auto& n = g.node(i);
    if (n.output_size() >= 1) {
      m[n.output(0)] = i;
    }
  }
  return m;
}

// --- Tensor <-> flat float buffer (transcribed from adaround_entry.cpp's own)

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

// --- Per-layer float weight + broadcast scale, [N, K] normalized ----------
//
// Transcribed from brecq.py's own _layer_arrays (itself normalizing
// exactly the way adaround_entry.cpp's own inlined per-candidate loop
// does).

struct LayerArraysResult {
  Matrix w_nk;
  Matrix scale_nk;  // broadcast to full [N, K]
  int64_t dim0 = 0, dim1 = 0;
  bool weight_transposed = false;
};

LayerArraysResult LayerArrays(
    const Candidate& c, const onnx::GraphProto& f_graph,
    const onnx::GraphProto& q_graph,
    std::unordered_map<std::string, int>& f_init_index,
    std::unordered_map<std::string, int>& q_init_index) {
  const onnx::TensorProto& w_float_init =
      f_graph.initializer(f_init_index[c.w_float_name]);
  const int64_t dim0 = w_float_init.dims(0);
  const int64_t dim1 = w_float_init.dims(1);
  const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
  const int64_t k = c.weight_transposed ? dim1 : dim0;

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
      q_graph.initializer(q_init_index[c.ws_name]);
  const std::vector<float> s_flat = ReadFloatTensor(ws_init);
  const int64_t num_blocks = k / c.block_size;
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

  return {w_nk, scale_nk, dim0, dim1, c.weight_transposed};
}

// --- Block discovery ---------------------------------------------------------
//
// Transcribed from brecq.py's own _discover_block_chain.

std::optional<std::pair<std::vector<Candidate>, bool>> DiscoverBlockChain(
    const onnx::GraphProto& f_graph, const std::vector<Candidate>& candidates,
    const std::string& block_input_name, const std::string& block_output_name) {
  std::unordered_map<std::string, const Candidate*> by_input;
  for (const auto& c : candidates) {
    by_input[c.float_x_name] = &c;  // last-writer-wins, matching the
                                    // reference's own dict comprehension.
  }

  std::vector<Candidate> chain;
  std::unordered_set<std::string> seen_outputs;
  std::string cur = block_input_name;
  while (cur != block_output_name) {
    auto it = by_input.find(cur);
    if (it == by_input.end() || seen_outputs.count(it->second->output_name)) {
      break;
    }
    seen_outputs.insert(it->second->output_name);
    chain.push_back(*it->second);
    cur = it->second->output_name;
  }

  if (cur == block_output_name) {
    if (chain.empty()) {
      return std::nullopt;
    }
    return std::make_pair(chain, false);
  }
  if (chain.empty()) {
    return std::nullopt;
  }

  const std::unordered_map<std::string, int> f_by_output =
      BuildOutputIndex(f_graph);
  auto ait = f_by_output.find(block_output_name);
  if (ait != f_by_output.end()) {
    const onnx::NodeProto& add_node = f_graph.node(ait->second);
    if (add_node.op_type() == "Add" && add_node.input_size() == 2) {
      const std::unordered_set<std::string> inputs = {add_node.input(0),
                                                      add_node.input(1)};
      const std::unordered_set<std::string> expected = {cur, block_input_name};
      if (inputs == expected) {
        return std::make_pair(chain, true);
      }
    }
  }
  return std::nullopt;
}

// --- Small dense-matmul helpers, nested-vector ("Matrix") style ------------
//
// Transcribed from adaround_entry.cpp's own identical helpers, plus
// MatMulSK for the block's own backward pass, which adaround's own
// single-layer optimizer never needs.

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

// out[S, K] = a[S, N] @ b[N, K] -- the block's own backward pass, propagating
// the loss gradient from one layer's own input back to the previous layer's
// own output (== dL/dys[layer_idx]).
Matrix MatMulSK(const Matrix& a, const Matrix& b) {
  const size_t s = a.size();
  const size_t n = a[0].size();
  const size_t k = b[0].size();
  Matrix out(s, std::vector<double>(k, 0.0));
  for (size_t si = 0; si < s; ++si) {
    for (size_t ki = 0; ki < k; ++ki) {
      double acc = 0.0;
      for (size_t ni = 0; ni < n; ++ni) {
        acc += a[si][ni] * b[ni][ki];
      }
      out[si][ki] = acc;
    }
  }
  return out;
}

// --- Calibration: per-batch raw activation parts ----------------------------
//
// Unlike every other calibration-driven pass in this codebase (which only
// ever needs the *concatenation* of every batch's own rows for one probe
// name), BRECQ's own block reconstruction needs `block_input_name`'s and
// `block_output_name`'s own rows paired *per batch* before concatenating
// (see brecq.py's own comment on why -- a batch preserves its token axis,
// so pairing by batch index is exact, whereas pairing after a flat
// concatenation could silently misalign samples if the two names ever
// disagreed per batch). So this keeps one entry per batch (valid=false, not
// dropped, when that batch's own output wasn't a usable [*, K] tensor),
// mirroring apply_brecq's own `activations[name]` list -- one raw capture
// per calibration batch, unfiltered -- exactly.

struct RawPart {
  bool valid = false;
  int64_t rows = 0;
  int64_t k = 0;
  std::vector<double> data;  // [rows, k], row-major.
};

std::unordered_map<std::string, std::vector<RawPart>> ProbeRawParts(
    const ModelExecutor& executor, const onnx::ModelProto& float_model,
    const std::vector<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  std::unordered_map<std::string, std::vector<RawPart>> result;
  for (const auto& name : probe_names) {
    result[name] = {};
  }
  if (probe_names.empty()) {
    return result;
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
            "ApplyBrecq: calibration batch is missing "
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
      RawPart part;
      auto oit = output_index.find(name);
      if (oit != output_index.end() && oit->second < outputs.size()) {
        const DLTensor& dl = outputs[oit->second]->dl_tensor;
        onnx::TensorProto tp = onnxsim::dlpack::ToTensorProto(dl);
        if (tp.data_type() == onnx::TensorProto::FLOAT && tp.dims_size() >= 2) {
          const int64_t kk = tp.dims(static_cast<int>(tp.dims_size() - 1));
          if (kk > 0) {
            int64_t numel = 1;
            for (int64_t d : tp.dims()) {
              numel *= d;
            }
            part.valid = true;
            part.k = kk;
            part.rows = numel / kk;
            const std::vector<float> flat = ReadFloatTensor(tp);
            part.data.assign(flat.begin(), flat.end());
          }
        }
      }
      result[name].push_back(std::move(part));
    }
  }
  return result;
}

// --- BRECQ's own joint, Fisher-weighted block Adam loop --------------------
//
// Transcribed from brecq.py's own _optimize_block_rounding. See
// brecq_entry.h's own accepted numerical scope note.

std::vector<Matrix> OptimizeBlockRounding(
    const std::vector<LayerArraysResult>& layers, bool has_residual,
    const Matrix& x0, const Matrix& final_float, int64_t num_iterations,
    double learning_rate, double reg_param, double warm_start,
    double beta_start, double beta_end, double fisher_eps) {
  const size_t num_layers = layers.size();
  const size_t num_samples = x0.size();
  const size_t n_final = final_float[0].size();

  std::vector<Matrix> floor_bases(num_layers), v_list(num_layers);
  for (size_t li = 0; li < num_layers; ++li) {
    const Matrix& w_nk = layers[li].w_nk;
    const Matrix& scale_nk = layers[li].scale_nk;
    const size_t n = w_nk.size();
    const size_t k = w_nk[0].size();
    Matrix fb(n, std::vector<double>(k));
    Matrix v(n, std::vector<double>(k));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        const double ratio = w_nk[i][j] / scale_nk[i][j];
        const double f = std::floor(ratio);
        const double frac = std::clamp(ratio - f, 1e-4, 1.0 - 1e-4);
        const double sig0 =
            std::clamp((frac - kGamma) / (kZeta - kGamma), 1e-4, 1.0 - 1e-4);
        fb[i][j] = f;
        v[i][j] = std::log(sig0 / (1.0 - sig0));
      }
    }
    floor_bases[li] = std::move(fb);
    v_list[li] = std::move(v);
  }

  // Fisher-diagonal proxy: empirical per-element variance across
  // calibration samples, normalized to a mean of 1.
  std::vector<double> mean(n_final, 0.0), var(n_final, 0.0),
      fisher(n_final, 1.0);
  for (size_t j = 0; j < n_final; ++j) {
    double sum = 0.0;
    for (size_t s = 0; s < num_samples; ++s) {
      sum += final_float[s][j];
    }
    mean[j] = sum / static_cast<double>(num_samples);
  }
  for (size_t j = 0; j < n_final; ++j) {
    double sq = 0.0;
    for (size_t s = 0; s < num_samples; ++s) {
      const double d = final_float[s][j] - mean[j];
      sq += d * d;
    }
    var[j] = sq / static_cast<double>(num_samples);
  }
  double mean_var = 0.0;
  for (double v : var) {
    mean_var += v;
  }
  mean_var /= static_cast<double>(n_final);
  if (mean_var > fisher_eps) {
    for (size_t j = 0; j < n_final; ++j) {
      fisher[j] = (var[j] + fisher_eps) / (mean_var + fisher_eps);
    }
  }  // else: fisher stays all-ones, matching np.ones_like(var).

  std::vector<Matrix> m_list(num_layers), v2_list(num_layers);
  for (size_t li = 0; li < num_layers; ++li) {
    m_list[li] = Matrix(v_list[li].size(),
                        std::vector<double>(v_list[li][0].size(), 0.0));
    v2_list[li] = Matrix(v_list[li].size(),
                         std::vector<double>(v_list[li][0].size(), 0.0));
  }
  constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEps = 1e-8;

  const int64_t warm_start_iters =
      static_cast<int64_t>(static_cast<double>(num_iterations) * warm_start);
  const double n_elems =
      static_cast<double>(num_samples) * static_cast<double>(n_final);

  for (int64_t t = 0; t < num_iterations; ++t) {
    std::vector<Matrix> ys;
    ys.reserve(num_layers + 1);
    ys.push_back(x0);
    std::vector<Matrix> h_list(num_layers), dh_dv_list(num_layers),
        w_hat_list(num_layers);
    std::vector<std::vector<std::vector<uint8_t>>> active_list(num_layers);

    for (size_t li = 0; li < num_layers; ++li) {
      const Matrix& w_nk = layers[li].w_nk;
      const Matrix& scale_nk = layers[li].scale_nk;
      const Matrix& fb = floor_bases[li];
      const Matrix& v = v_list[li];
      const size_t n = w_nk.size();
      const size_t k = w_nk[0].size();
      Matrix h(n, std::vector<double>(k)), dh_dv(n, std::vector<double>(k)),
          w_hat(n, std::vector<double>(k));
      std::vector<std::vector<uint8_t>> active(n, std::vector<uint8_t>(k));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          const double s = 1.0 / (1.0 + std::exp(-v[i][j]));
          const double raw = s * (kZeta - kGamma) + kGamma;
          const bool active_h = raw > 0.0 && raw < 1.0;
          h[i][j] = std::clamp(raw, 0.0, 1.0);
          dh_dv[i][j] = active_h ? s * (1.0 - s) * (kZeta - kGamma) : 0.0;
          const double raw2 = fb[i][j] + h[i][j];
          const double cl = std::clamp(raw2, kNMin, kNMax);
          const bool act = raw2 > kNMin && raw2 < kNMax;
          w_hat[i][j] = cl * scale_nk[i][j];
          active[i][j] = act ? 1 : 0;
        }
      }
      h_list[li] = std::move(h);
      dh_dv_list[li] = std::move(dh_dv);
      active_list[li] = std::move(active);
      ys.push_back(YFromXWt(ys.back(), w_hat));
      w_hat_list[li] = std::move(w_hat);
    }

    Matrix final_hat = ys.back();
    if (has_residual) {
      for (size_t s = 0; s < num_samples; ++s) {
        for (size_t j = 0; j < n_final; ++j) {
          final_hat[s][j] += x0[s][j];
        }
      }
    }
    Matrix grad_y(num_samples, std::vector<double>(n_final));
    for (size_t s = 0; s < num_samples; ++s) {
      for (size_t j = 0; j < n_final; ++j) {
        const double diff = final_hat[s][j] - final_float[s][j];
        grad_y[s][j] = 2.0 * fisher[j] * diff / n_elems;
      }
    }

    std::vector<Matrix> grads_v(num_layers);
    for (int64_t li = static_cast<int64_t>(num_layers) - 1; li >= 0; --li) {
      const size_t lidx = static_cast<size_t>(li);
      const Matrix dl_dw_hat = DlDwHatFromDlDyX(grad_y, ys[lidx]);
      const size_t n = dl_dw_hat.size();
      const size_t k = dl_dw_hat[0].size();
      Matrix grad_v(n, std::vector<double>(k));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          const double dl_dh =
              dl_dw_hat[i][j] *
              (active_list[lidx][i][j] ? layers[lidx].scale_nk[i][j] : 0.0);
          grad_v[i][j] = dl_dh * dh_dv_list[lidx][i][j];
        }
      }

      if (t >= warm_start_iters) {
        const double denom = static_cast<double>(
            std::max<int64_t>(1, num_iterations - warm_start_iters - 1));
        const double progress =
            static_cast<double>(t - warm_start_iters) / denom;
        const double beta = beta_start + (beta_end - beta_start) * progress;
        for (size_t i = 0; i < n; ++i) {
          for (size_t j = 0; j < k; ++j) {
            const double u = 2.0 * h_list[lidx][i][j] - 1.0;
            const double abs_u = std::fabs(u);
            const double sign_u = (u > 0.0) - (u < 0.0);
            const double dreg_dh =
                -2.0 * reg_param * beta * sign_u * std::pow(abs_u, beta - 1.0);
            grad_v[i][j] += dreg_dh * dh_dv_list[lidx][i][j];
          }
        }
      }

      grads_v[lidx] = std::move(grad_v);
      if (li > 0) {
        grad_y = MatMulSK(grad_y, w_hat_list[lidx]);
      }
    }

    for (size_t li = 0; li < num_layers; ++li) {
      const size_t n = v_list[li].size();
      const size_t k = v_list[li][0].size();
      const double bias_c1 =
          1.0 - std::pow(kAdamBeta1, static_cast<double>(t + 1));
      const double bias_c2 =
          1.0 - std::pow(kAdamBeta2, static_cast<double>(t + 1));
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          m_list[li][i][j] = kAdamBeta1 * m_list[li][i][j] +
                             (1.0 - kAdamBeta1) * grads_v[li][i][j];
          v2_list[li][i][j] =
              kAdamBeta2 * v2_list[li][i][j] +
              (1.0 - kAdamBeta2) * grads_v[li][i][j] * grads_v[li][i][j];
          v_list[li][i][j] -=
              learning_rate * (m_list[li][i][j] / bias_c1) /
              (std::sqrt(v2_list[li][i][j] / bias_c2) + kAdamEps);
        }
      }
    }
  }

  std::vector<Matrix> codes(num_layers);
  for (size_t li = 0; li < num_layers; ++li) {
    const size_t n = v_list[li].size();
    const size_t k = v_list[li][0].size();
    Matrix c(n, std::vector<double>(k));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < k; ++j) {
        const double s = 1.0 / (1.0 + std::exp(-v_list[li][i][j]));
        const double raw = s * (kZeta - kGamma) + kGamma;
        const double h_final = std::clamp(raw, 0.0, 1.0);
        c[i][j] = std::clamp(floor_bases[li][i][j] + RoundHalfToEven(h_final),
                             kNMin, kNMax);
      }
    }
    codes[li] = std::move(c);
  }
  return codes;
}

}  // namespace

onnx::ModelProto ApplyBrecq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::pair<std::string, std::string>>& blocks,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations, double learning_rate, double reg_param,
    double warm_start, double beta_start, double beta_end, double fisher_eps) {
  std::unordered_map<std::string, int> q_init_index;
  std::unordered_map<std::string, int> f_init_index;
  const std::vector<Candidate> candidates = FindInt4MatmulCandidates(
      float_model, quantized_model, q_init_index, f_init_index);

  struct Discovered {
    std::string block_input_name;
    std::string block_output_name;
    std::vector<Candidate> chain;
    bool has_residual;
  };
  std::vector<Discovered> discovered;
  for (const auto& block : blocks) {
    auto found = DiscoverBlockChain(float_model.graph(), candidates,
                                    block.first, block.second);
    if (found) {
      discovered.push_back(
          {block.first, block.second, std::move(found->first), found->second});
    }
  }
  if (discovered.empty()) {
    return quantized_model;
  }

  std::vector<std::string> probe_names;
  {
    std::unordered_set<std::string> seen;
    for (const auto& d : discovered) {
      if (seen.insert(d.block_input_name).second) {
        probe_names.push_back(d.block_input_name);
      }
      if (seen.insert(d.block_output_name).second) {
        probe_names.push_back(d.block_output_name);
      }
    }
  }
  std::sort(probe_names.begin(), probe_names.end());

  const std::unordered_map<std::string, std::vector<RawPart>> raw_parts =
      ProbeRawParts(executor, float_model, probe_names, calibration_data);

  const onnx::GraphProto& f_graph = float_model.graph();
  const onnx::GraphProto& q_graph = quantized_model.graph();

  std::unordered_map<std::string, std::vector<double>> optimized;
  for (const auto& d : discovered) {
    auto xit = raw_parts.find(d.block_input_name);
    auto fit = raw_parts.find(d.block_output_name);
    if (xit == raw_parts.end() || fit == raw_parts.end()) {
      continue;
    }
    const std::vector<RawPart>& x_parts = xit->second;
    const std::vector<RawPart>& f_parts = fit->second;
    const size_t num_batches = std::min(x_parts.size(), f_parts.size());

    Matrix x0, final_float;
    for (size_t bi = 0; bi < num_batches; ++bi) {
      const RawPart& xp = x_parts[bi];
      const RawPart& fp = f_parts[bi];
      if (!xp.valid || !fp.valid || xp.rows != fp.rows) {
        continue;  // Per-batch pairing failed (or that batch's own probe
                   // wasn't a usable [*, K] tensor) -- skip that batch only.
      }
      for (int64_t r = 0; r < xp.rows; ++r) {
        std::vector<double> xrow(static_cast<size_t>(xp.k));
        for (int64_t c = 0; c < xp.k; ++c) {
          xrow[static_cast<size_t>(c)] =
              xp.data[static_cast<size_t>(r * xp.k + c)];
        }
        x0.push_back(std::move(xrow));
        std::vector<double> frow(static_cast<size_t>(fp.k));
        for (int64_t c = 0; c < fp.k; ++c) {
          frow[static_cast<size_t>(c)] =
              fp.data[static_cast<size_t>(r * fp.k + c)];
        }
        final_float.push_back(std::move(frow));
      }
    }
    if (x0.empty()) {
      continue;  // No usable activation pair; skip this block.
    }

    std::vector<LayerArraysResult> layers;
    layers.reserve(d.chain.size());
    for (const auto& c : d.chain) {
      layers.push_back(
          LayerArrays(c, f_graph, q_graph, f_init_index, q_init_index));
    }
    if (x0[0].size() != layers[0].w_nk[0].size()) {
      continue;  // Activation's feature dim doesn't match the first layer's
                 // own K; skip.
    }

    const std::vector<Matrix> layer_codes = OptimizeBlockRounding(
        layers, d.has_residual, x0, final_float, num_iterations, learning_rate,
        reg_param, warm_start, beta_start, beta_end, fisher_eps);

    for (size_t li = 0; li < d.chain.size(); ++li) {
      const Candidate& c = d.chain[li];
      const LayerArraysResult& la = layers[li];
      const Matrix& codes_nk = layer_codes[li];
      std::vector<double> codes_flat;
      codes_flat.reserve(static_cast<size_t>(la.dim0 * la.dim1));
      if (la.weight_transposed) {
        for (int64_t i = 0; i < la.dim0; ++i) {
          for (int64_t j = 0; j < la.dim1; ++j) {
            codes_flat.push_back(
                codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)]);
          }
        }
      } else {
        for (int64_t i = 0; i < la.dim0; ++i) {
          for (int64_t j = 0; j < la.dim1; ++j) {
            codes_flat.push_back(
                codes_nk[static_cast<size_t>(j)][static_cast<size_t>(i)]);
          }
        }
      }
      if (codes_flat.size() % 2 != 0) {
        continue;
      }
      optimized[c.wq_name] = std::move(codes_flat);
    }
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
    t.set_raw_data(PackInt4(it->second));
  }
  return corrected;
}
