// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See omniquant_entry.h for the full rationale (including why this follows
// adaround_entry.h's own two-model, calibration-driven candidate-matching
// shape but easyquant_entry.h's/daq_entry.h's own grid-search numerical
// character) and onnxsim/omniquant.py for the technique this ports.
//
// FindInt4MatmulCandidates/ReadFloatTensor/RoundHalfToEven/PackInt4/
// ActivationRows/AccumulateActivationRows are transcribed from
// adaround_entry.cpp's own identical helpers; the node-insertion-before
// idiom (InsertEmptyNodeAt, taken_names/unique_name) is transcribed from
// llm_int8_entry.cpp's/easyquant_entry.cpp's own identical helpers; the
// rename-and-insert-Add-after idiom is transcribed from
// bias_correction_entry.cpp's own ApplyBiasCorrections -- see
// gptaq_entry.cpp's own top-of-file comment for why these aren't shared via
// a common header (matches this codebase's established convention: every
// *_entry.cpp is self-contained).

#include "omniquant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

using Matrix = std::vector<std::vector<double>>;

// --- Candidate matching ----------------------------------------------------

struct Candidate {
  std::string output_name;
  std::string float_x_name;
  std::string w_float_name;
  std::string wq_name;
  std::string ws_name;
  int64_t block_size;
  bool weight_transposed;
  int q_node_index;  // node index in quantized_model.graph() at the time of
                     // matching -- used, alongside a running insertion
                     // offset, to locate this candidate's own node again
                     // after earlier candidates' own graph surgery.
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
    const int qi = q_by_output[out_name];
    const onnx::NodeProto& qn = q_graph.node(qi);
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
                          dq.input(1), block_size, weight_transposed, qi});
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

// Overwrites `t`'s own data fields (raw_data/float_data/...) with `data`,
// leaving name/dims/data_type untouched -- mirrors
// `t.CopyFrom(onnx.numpy_helper.from_array(scale, name=t.name))`'s own net
// effect on the tensor's data (dims never change here: the scale tensor's
// own shape is fixed by the candidate's block layout, only its values
// change).
void ReplaceFloatData(onnx::TensorProto* t, const std::vector<double>& data) {
  t->clear_raw_data();
  t->clear_float_data();
  t->clear_int32_data();
  t->clear_int64_data();
  t->clear_double_data();
  std::vector<float> as_float(data.begin(), data.end());
  std::string raw(as_float.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), as_float.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// --- Calibration: concatenated activation rows -----------------------------

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
            "ApplyOmniquant: calibration batch is missing "
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

// --- Small dense-matmul helpers, nested-vector ("Matrix") style ------------

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

std::vector<double> Linspace(double lo, double hi, int64_t n) {
  std::vector<double> out;
  if (n <= 0) {
    return out;
  }
  if (n == 1) {
    out.push_back(lo);
    return out;
  }
  out.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] =
        lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
  }
  return out;
}

// Round-to-nearest block-wise INT4 quantization of `w_nk` ([N, K], output
// channel first), with each block's scale computed from `clip_ratio *
// max(|w| in block) / 7` -- transcribed from omniquant.py's own
// _quantize_blockwise_int4_with_clip.
std::pair<Matrix, Matrix> QuantizeBlockwiseInt4WithClip(const Matrix& w_nk,
                                                        int64_t block_size,
                                                        double clip_ratio) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_blocks = k / static_cast<size_t>(block_size);
  Matrix scale_blocks(n, std::vector<double>(num_blocks, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t blk = 0; blk < num_blocks; ++blk) {
      double max_abs = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const double v =
            std::fabs(w_nk[i][blk * static_cast<size_t>(block_size) +
                              static_cast<size_t>(j)]);
        if (v > max_abs) {
          max_abs = v;
        }
      }
      scale_blocks[i][blk] = std::max(max_abs * clip_ratio, 1e-12) / 7.0;
    }
  }
  Matrix codes_nk(n, std::vector<double>(k));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < k; ++j) {
      const size_t blk = j / static_cast<size_t>(block_size);
      const double s = scale_blocks[i][blk];
      const double q = RoundHalfToEven(w_nk[i][j] / s);
      codes_nk[i][j] = std::clamp(q, -7.0, 7.0);
    }
  }
  return {codes_nk, scale_blocks};
}

Matrix ExpandAndMultiply(const Matrix& codes_nk, const Matrix& scale_blocks,
                         int64_t block_size) {
  const size_t n = codes_nk.size();
  const size_t k = codes_nk[0].size();
  Matrix w_hat(n, std::vector<double>(k));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < k; ++j) {
      w_hat[i][j] =
          codes_nk[i][j] * scale_blocks[i][j / static_cast<size_t>(block_size)];
    }
  }
  return w_hat;
}

// Mirrors omniquant.py's own _reconstruction_error exactly: mean squared
// error of the (optionally shift/scale-transformed, bias-corrected) layer
// output against `y_float`.
double ReconstructionError(const Matrix& x, const Matrix& y_float,
                           const Matrix& w_hat_nk, bool has_let,
                           const std::vector<double>& shift,
                           const std::vector<double>& channel_scale,
                           const std::vector<double>& bias_correction) {
  const size_t s = x.size();
  const size_t n = w_hat_nk.size();
  const size_t k = w_hat_nk[0].size();
  double sq_sum = 0.0;
  std::vector<double> x_row(k);
  for (size_t si = 0; si < s; ++si) {
    if (has_let) {
      for (size_t j = 0; j < k; ++j) {
        x_row[j] = (x[si][j] - shift[j]) / channel_scale[j];
      }
    } else {
      x_row = x[si];
    }
    for (size_t ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (size_t j = 0; j < k; ++j) {
        acc += x_row[j] * w_hat_nk[ni][j];
      }
      if (has_let) {
        acc += bias_correction[ni];
      }
      const double d = y_float[si][ni] - acc;
      sq_sum += d * d;
    }
  }
  const size_t count = s * n;
  return count == 0 ? 0.0 : sq_sum / static_cast<double>(count);
}

// --- Node insertion / unique naming -----------------------------------------
//
// Transcribed from llm_int8_entry.cpp's/easyquant_entry.cpp's own identical
// helpers.

void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

// --- One optimized layer -----------------------------------------------------

struct OmniquantRewrite {
  std::string output_name;
  std::string activation_name;
  std::string wq_name;
  std::string ws_name;
  std::vector<double> codes_flat;  // [dim0 * dim1], original (Wq) layout
  std::vector<double> scale_flat;  // original (Ws) layout
  bool has_let = false;
  std::vector<double> channel_scale;    // [K]
  std::vector<double> shift;            // [K]
  std::vector<double> bias_correction;  // [N]
  int q_node_index = -1;
};

}  // namespace

onnx::ModelProto ApplyOmniquant(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_clip_steps, int64_t num_alpha_steps, double min_clip_ratio) {
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

  std::vector<double> clip_ratios =
      Linspace(min_clip_ratio, 1.0, num_clip_steps);
  std::reverse(clip_ratios.begin(), clip_ratios.end());  // 1.0 first
  const std::vector<double> alphas = Linspace(0.0, 1.0, num_alpha_steps);

  std::vector<OmniquantRewrite> rewrites;
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
      continue;
    }
    if (c.block_size == 0 || k % c.block_size != 0) {
      continue;  // Ragged block -- quantize_weight_only_int4 never produces
                 // this.
    }
    const int64_t num_blocks = k / c.block_size;
    const int64_t num_samples =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);
    if (num_samples <= 0) {
      continue;
    }

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

    Matrix x(static_cast<size_t>(num_samples),
             std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < num_samples; ++r) {
      for (int64_t c2 = 0; c2 < k; ++c2) {
        x[static_cast<size_t>(r)][static_cast<size_t>(c2)] =
            rows.data[static_cast<size_t>(r * k + c2)];
      }
    }

    const Matrix y_float = YFromXWt(x, w_nk);

    // Stage 1: LWC only (no LET) -- clip_ratio=1.0 is always tried first
    // (hardcoded, matching apply_omniquant's own baseline, independent of
    // whatever clip_ratios[0] itself is) and is exactly
    // quantize_weight_only_int4's own scale, so this stage can only match
    // or improve on plain RTN.
    auto [best_codes_nk, best_scale_blocks] =
        QuantizeBlockwiseInt4WithClip(w_nk, c.block_size, 1.0);
    double best_err = ReconstructionError(
        x, y_float,
        ExpandAndMultiply(best_codes_nk, best_scale_blocks, c.block_size),
        false, {}, {}, {});
    double best_clip_ratio = 1.0;
    for (size_t i = 1; i < clip_ratios.size(); ++i) {
      const double clip_ratio = clip_ratios[i];
      auto [codes_nk, scale_blocks] =
          QuantizeBlockwiseInt4WithClip(w_nk, c.block_size, clip_ratio);
      const Matrix w_hat_nk =
          ExpandAndMultiply(codes_nk, scale_blocks, c.block_size);
      const double err =
          ReconstructionError(x, y_float, w_hat_nk, false, {}, {}, {});
      if (err < best_err) {
        best_err = err;
        best_clip_ratio = clip_ratio;
        best_codes_nk = codes_nk;
        best_scale_blocks = scale_blocks;
      }
    }

    // Stage 2: LET (shift + scale) on top of the best LWC ratio found.
    std::vector<double> shift(static_cast<size_t>(k), 0.0);
    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      double sum = 0.0;
      for (int64_t si = 0; si < num_samples; ++si) {
        sum += x[static_cast<size_t>(si)][j];
      }
      shift[j] = sum / static_cast<double>(num_samples);
    }
    Matrix x_centered = x;
    for (auto& row : x_centered) {
      for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
        row[j] -= shift[j];
      }
    }
    std::vector<double> weight_col_absmax(static_cast<size_t>(k), 0.0);
    std::vector<double> act_col_absmax(static_cast<size_t>(k), 0.0);
    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      double wmax = 0.0;
      for (int64_t i = 0; i < n_rows; ++i) {
        wmax = std::max(wmax, std::fabs(w_nk[static_cast<size_t>(i)][j]));
      }
      weight_col_absmax[j] = std::max(wmax, 1e-12);
      double asum = 0.0;
      for (int64_t si = 0; si < num_samples; ++si) {
        asum += std::fabs(x_centered[static_cast<size_t>(si)][j]);
      }
      act_col_absmax[j] =
          std::max(asum / static_cast<double>(num_samples), 1e-12);
    }
    std::vector<double> bias_correction(static_cast<size_t>(n_rows), 0.0);
    for (int64_t i = 0; i < n_rows; ++i) {
      double acc = 0.0;
      for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
        acc += w_nk[static_cast<size_t>(i)][j] * shift[j];
      }
      bias_correction[static_cast<size_t>(i)] = acc;
    }

    bool has_let = false;
    std::vector<double> best_channel_scale, best_shift, best_bias_correction;
    for (size_t ai = 1; ai < alphas.size(); ++ai) {
      const double alpha = alphas[ai];
      std::vector<double> raw(static_cast<size_t>(k));
      double log_sum = 0.0;
      for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
        raw[j] = std::pow(act_col_absmax[j], alpha) *
                 std::pow(weight_col_absmax[j], -(1.0 - alpha));
        log_sum += std::log(raw[j]);
      }
      const double denom = std::exp(log_sum / static_cast<double>(k));
      std::vector<double> channel_scale(static_cast<size_t>(k));
      for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
        channel_scale[j] = raw[j] / denom;
      }

      Matrix w_scaled_nk = w_nk;
      for (auto& row : w_scaled_nk) {
        for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
          row[j] *= channel_scale[j];
        }
      }
      auto [codes_nk, scale_blocks] = QuantizeBlockwiseInt4WithClip(
          w_scaled_nk, c.block_size, best_clip_ratio);
      const Matrix w_hat_nk =
          ExpandAndMultiply(codes_nk, scale_blocks, c.block_size);
      const double err = ReconstructionError(x, y_float, w_hat_nk, true, shift,
                                             channel_scale, bias_correction);
      if (err < best_err) {
        best_err = err;
        best_codes_nk = codes_nk;
        best_scale_blocks = scale_blocks;
        best_channel_scale = channel_scale;
        best_shift = shift;
        best_bias_correction = bias_correction;
        has_let = true;
      }
    }

    OmniquantRewrite r;
    r.output_name = c.output_name;
    r.activation_name = c.float_x_name;
    r.wq_name = c.wq_name;
    r.ws_name = c.ws_name;
    r.q_node_index = c.q_node_index;
    r.has_let = has_let;
    if (has_let) {
      r.channel_scale = std::move(best_channel_scale);
      r.shift = std::move(best_shift);
      r.bias_correction = std::move(best_bias_correction);
    }

    r.codes_flat.reserve(static_cast<size_t>(dim0 * dim1));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          r.codes_flat.push_back(
              best_codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      }
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          r.codes_flat.push_back(
              best_codes_nk[static_cast<size_t>(j)][static_cast<size_t>(i)]);
        }
      }
    }
    if (r.codes_flat.size() % 2 != 0) {
      continue;
    }

    r.scale_flat.reserve(static_cast<size_t>(n_rows * num_blocks));
    if (c.weight_transposed) {
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t blk = 0; blk < num_blocks; ++blk) {
          r.scale_flat.push_back(best_scale_blocks[static_cast<size_t>(i)]
                                                  [static_cast<size_t>(blk)]);
        }
      }
    } else {
      for (int64_t blk = 0; blk < num_blocks; ++blk) {
        for (int64_t i = 0; i < n_rows; ++i) {
          r.scale_flat.push_back(best_scale_blocks[static_cast<size_t>(i)]
                                                  [static_cast<size_t>(blk)]);
        }
      }
    }

    rewrites.push_back(std::move(r));
  }

  if (rewrites.empty()) {
    return quantized_model;
  }

  onnx::ModelProto corrected = quantized_model;

  std::unordered_map<std::string, const std::vector<double>*> codes_by_name;
  std::unordered_map<std::string, const std::vector<double>*> scale_by_name;
  for (const auto& r : rewrites) {
    codes_by_name.emplace(r.wq_name, &r.codes_flat);
    scale_by_name.emplace(r.ws_name, &r.scale_flat);
  }
  for (auto& t : *corrected.mutable_graph()->mutable_initializer()) {
    auto cit = codes_by_name.find(t.name());
    if (cit != codes_by_name.end()) {
      t.set_raw_data(PackInt4(*cit->second));
    }
    auto sit = scale_by_name.find(t.name());
    if (sit != scale_by_name.end()) {
      ReplaceFloatData(&t, *sit->second);
    }
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly.
  onnx::GraphProto* graph = corrected.mutable_graph();
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
  for (const auto& r : rewrites) {
    if (!r.has_let) {
      continue;
    }
    const int64_t live_index = r.q_node_index + net_insertions;
    const std::string& act = r.activation_name;

    const std::string shift_name = unique_name(act + "_omniquant_shift");
    onnx::TensorProto* shift_t = graph->add_initializer();
    shift_t->set_name(shift_name);
    shift_t->set_data_type(onnx::TensorProto::FLOAT);
    shift_t->add_dims(static_cast<int64_t>(r.shift.size()));
    ReplaceFloatData(shift_t, r.shift);

    const std::string centered_name = unique_name(act + "_omniquant_centered");
    InsertEmptyNodeAt(graph, static_cast<int>(live_index));
    {
      onnx::NodeProto* sub_node =
          graph->mutable_node(static_cast<int>(live_index));
      sub_node->set_op_type("Sub");
      sub_node->add_input(act);
      sub_node->add_input(shift_name);
      sub_node->add_output(centered_name);
      sub_node->set_name(unique_name(act + "_omniquant_sub"));
    }

    std::vector<double> inv_scale(r.channel_scale.size());
    for (size_t j = 0; j < inv_scale.size(); ++j) {
      inv_scale[j] = 1.0 / r.channel_scale[j];
    }
    const std::string inv_scale_name =
        unique_name(act + "_omniquant_inv_scale");
    onnx::TensorProto* inv_scale_t = graph->add_initializer();
    inv_scale_t->set_name(inv_scale_name);
    inv_scale_t->set_data_type(onnx::TensorProto::FLOAT);
    inv_scale_t->add_dims(static_cast<int64_t>(inv_scale.size()));
    ReplaceFloatData(inv_scale_t, inv_scale);

    const std::string scaled_name = unique_name(act + "_omniquant_scaled");
    InsertEmptyNodeAt(graph, static_cast<int>(live_index) + 1);
    {
      onnx::NodeProto* mul_node =
          graph->mutable_node(static_cast<int>(live_index) + 1);
      mul_node->set_op_type("Mul");
      mul_node->add_input(centered_name);
      mul_node->add_input(inv_scale_name);
      mul_node->add_output(scaled_name);
      mul_node->set_name(unique_name(act + "_omniquant_mul"));
    }

    onnx::NodeProto* qn = graph->mutable_node(static_cast<int>(live_index) + 2);
    qn->set_input(0, scaled_name);

    const std::string old_output = qn->output(0);
    const std::string base_name =
        unique_name(r.output_name + "_omniquant_base");
    qn->set_output(0, base_name);

    const std::string bias_name =
        unique_name(r.output_name + "_omniquant_bias");
    onnx::TensorProto* bias_t = graph->add_initializer();
    bias_t->set_name(bias_name);
    bias_t->set_data_type(onnx::TensorProto::FLOAT);
    bias_t->add_dims(static_cast<int64_t>(r.bias_correction.size()));
    ReplaceFloatData(bias_t, r.bias_correction);

    InsertEmptyNodeAt(graph, static_cast<int>(live_index) + 3);
    {
      onnx::NodeProto* add_node =
          graph->mutable_node(static_cast<int>(live_index) + 3);
      add_node->set_op_type("Add");
      add_node->add_input(base_name);
      add_node->add_input(bias_name);
      add_node->add_output(old_output);
      add_node->set_name(unique_name(r.output_name + "_omniquant_bias_add"));
    }

    net_insertions += 3;
  }

  return corrected;
}
