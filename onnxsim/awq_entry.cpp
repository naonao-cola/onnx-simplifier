// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See awq_entry.h for the full rationale (including why this follows
// gptq_entry.h's own protobuf-level, calibration-driven shape) and
// onnxsim/awq.py for the technique this ports.

#include "awq_entry.h"

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
// Transcribed from gptq_entry.cpp's own FindInt4MatmulCandidates (which
// itself mirrors adaround.py's _find_int4_matmul_candidates, the exact
// matcher onnxsim.awq reuses): joins the float and quantized models by
// node output tensor name, last-wins with first-seen iteration order.

struct Candidate {
  int node_index;  // Quantized graph node index at match time.
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
    candidates.push_back({qi, out_name, fn.input(0), fn.input(1), dq.input(0),
                          dq.input(1), block_size, weight_transposed});
  }
  return candidates;
}

// --- Tensor <-> flat float buffer -------------------------------------------
//
// FLOAT32 only (see the header's own scope note), reusing
// dlpack_bridge.h's kRawDataIsHostOrder/SwapElementBytes for the
// raw_data little-endian convention.

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

void SetFloatInitializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<float>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::string raw(data.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), data.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// Round-half-to-even (banker's rounding), matching numpy's own `round`
// used by the block quantizer below. Transcribed from
// llm_int8_entry.cpp's own RoundHalfToEven.
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

// Fresh round-to-nearest block-wise INT4 quantization -- mirrors
// awq.py's own _quantize_blockwise_int4 (which itself matches
// TryQuantizeWeightBlockwiseInt4InPlace's scheme): one scale per
// (output channel, block-of-K) group, scale = max(|block|, 1e-12) / 7,
// codes clipped to [-7, 7]. w_nk/scales_out are [N, K] / [N,
// K/block_size]; codes_out holds the clipped codes.
void QuantizeBlockwiseInt4(const std::vector<std::vector<double>>& w_nk,
                           int64_t block_size,
                           std::vector<std::vector<double>>& codes_out,
                           std::vector<std::vector<double>>& scales_out) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t nb = static_cast<size_t>(block_size);
  const size_t num_blocks = k / nb;
  scales_out.assign(n, std::vector<double>(num_blocks, 0.0));
  codes_out.assign(n, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t b = 0; b < num_blocks; ++b) {
      double wmax = 0.0;
      for (size_t j = 0; j < nb; ++j) {
        const double v = std::abs(w_nk[i][b * nb + j]);
        if (v > wmax) {
          wmax = v;
        }
      }
      const double s = std::max(wmax, 1e-12) / 7.0;
      scales_out[i][b] = s;
      for (size_t j = 0; j < nb; ++j) {
        double code = RoundHalfToEven(w_nk[i][b * nb + j] / s);
        codes_out[i][b * nb + j] = std::min(7.0, std::max(-7.0, code));
      }
    }
  }
}

// Same low-nibble-first packing as adaround.py's own _pack_int4.
// Transcribed from gptq_entry.cpp's own PackInt4.
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

// --- Calibration: concatenated activation rows ------------------------------
//
// Probes the float model exactly like gptq_entry.cpp's own
// AccumulateActivationRows (which is what awq.py shares with gptq.py via
// _activation_rows): every observed 2-D-or-higher FLOAT32 activation
// flattened to rows and concatenated along rows across batches.

struct ActivationRows {
  // Concatenated [total_rows, K] in row-major order; K == -1 when
  // nothing was ever observed.
  std::vector<double> data;
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
            "ApplyAwq: calibration batch is missing "
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

// Inserts a fresh node at position `index` (shifting later nodes right).
// The caller re-fetches it via `mutable_node(index)`, since no NodeProto
// pointer is held across a mutation.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

}  // namespace

onnx::ModelProto ApplyAwq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_alpha_steps) {
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

  struct Rewrite {
    std::string output_name;
    std::string activation_name;
    std::string wq_name;
    std::string ws_name;
    std::vector<int8_t> codes;  // Stored layout, row-major.
    std::vector<int64_t> codes_dims;
    std::vector<float> scale;  // Stored layout, row-major.
    std::vector<int64_t> scale_dims;
    std::vector<double> channel_scale;  // [K]; empty means "no Mul needed".
  };
  std::vector<Rewrite> rewrites;

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
      continue;  // Outside the scheme's own K-multiple-of-block_size
                 // precondition (matches _quantize_blockwise_int4's own
                 // reshape assumption).
    }
    const int64_t num_rows =
        static_cast<int64_t>(rows.data.size() / static_cast<size_t>(k));

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

    // AWQ's own saliency signal: each input channel's average activation
    // magnitude across the calibration set.
    std::vector<double> act_magnitude(static_cast<size_t>(k), 0.0);
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        act_magnitude[static_cast<size_t>(j)] +=
            std::abs(rows.data[static_cast<size_t>(r * k + j)]);
      }
    }
    for (int64_t j = 0; j < k; ++j) {
      act_magnitude[static_cast<size_t>(j)] = std::max(
          act_magnitude[static_cast<size_t>(j)] / static_cast<double>(num_rows),
          1e-12);
    }

    // y_float = X @ W^T once -- the reconstruction target every grid
    // point is measured against.
    std::vector<std::vector<double>> y_float(
        static_cast<size_t>(num_rows),
        std::vector<double>(static_cast<size_t>(n_rows), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t n = 0; n < n_rows; ++n) {
        double acc = 0.0;
        for (int64_t j = 0; j < k; ++j) {
          acc += rows.data[static_cast<size_t>(r * k + j)] *
                 w_nk[static_cast<size_t>(n)][static_cast<size_t>(j)];
        }
        y_float[static_cast<size_t>(r)][static_cast<size_t>(n)] = acc;
      }
    }

    auto mean_squared_error =
        [&](const std::vector<std::vector<double>>& y_hat) {
          double acc = 0.0;
          for (int64_t r = 0; r < num_rows; ++r) {
            for (int64_t n = 0; n < n_rows; ++n) {
              const double d =
                  y_float[static_cast<size_t>(r)][static_cast<size_t>(n)] -
                  y_hat[static_cast<size_t>(r)][static_cast<size_t>(n)];
              acc += d * d;
            }
          }
          return acc / static_cast<double>(num_rows * n_rows);
        };

    // alpha == 0 (uniform scale 1, plain round-to-nearest) is always a
    // candidate and always the grid's first point -- evaluated first so
    // best_* never needs a sentinel.
    std::vector<std::vector<double>> best_codes_nk;
    std::vector<std::vector<double>> best_scale_blocks;
    QuantizeBlockwiseInt4(w_nk, c.block_size, best_codes_nk, best_scale_blocks);
    auto expand_scales = [&](const std::vector<std::vector<double>>& blocks) {
      std::vector<std::vector<double>> full(
          static_cast<size_t>(n_rows),
          std::vector<double>(static_cast<size_t>(k)));
      const size_t nb = static_cast<size_t>(c.block_size);
      for (size_t i = 0; i < static_cast<size_t>(n_rows); ++i) {
        for (size_t b = 0; b < blocks[i].size(); ++b) {
          for (size_t j = 0; j < nb; ++j) {
            full[i][b * nb + j] = blocks[i][b];
          }
        }
      }
      return full;
    };
    std::vector<std::vector<double>> best_scale_full =
        expand_scales(best_scale_blocks);
    std::vector<std::vector<double>> y_hat(
        static_cast<size_t>(num_rows),
        std::vector<double>(static_cast<size_t>(n_rows), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t n = 0; n < n_rows; ++n) {
        double acc = 0.0;
        for (int64_t j = 0; j < k; ++j) {
          acc +=
              rows.data[static_cast<size_t>(r * k + j)] *
              (best_codes_nk[static_cast<size_t>(n)][static_cast<size_t>(j)] *
               best_scale_full[static_cast<size_t>(n)][static_cast<size_t>(j)]);
        }
        y_hat[static_cast<size_t>(r)][static_cast<size_t>(n)] = acc;
      }
    }
    double best_err = mean_squared_error(y_hat);
    double best_alpha = 0.0;
    std::vector<double> best_channel_scale;

    // linspace(0, 1, num_alpha_steps)[1:] -- mirrors np.linspace
    // exactly (alpha_i = i / (num - 1)); a single grid point means just
    // the alpha == 0 baseline above.
    for (int64_t step = 1; step < num_alpha_steps; ++step) {
      const double alpha =
          static_cast<double>(step) / static_cast<double>(num_alpha_steps - 1);
      std::vector<double> raw(static_cast<size_t>(k));
      for (int64_t j = 0; j < k; ++j) {
        raw[static_cast<size_t>(j)] =
            std::pow(act_magnitude[static_cast<size_t>(j)], alpha);
      }
      // Geometric-mean normalization -- mirrors raw /
      // exp(mean(log(raw))) exactly.
      double log_sum = 0.0;
      for (int64_t j = 0; j < k; ++j) {
        log_sum += std::log(raw[static_cast<size_t>(j)]);
      }
      const double geo = std::exp(log_sum / static_cast<double>(k));
      std::vector<double> channel_scale(static_cast<size_t>(k));
      for (int64_t j = 0; j < k; ++j) {
        channel_scale[static_cast<size_t>(j)] =
            raw[static_cast<size_t>(j)] / geo;
      }
      std::vector<std::vector<double>> w_scaled_nk(
          static_cast<size_t>(n_rows),
          std::vector<double>(static_cast<size_t>(k)));
      for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t j = 0; j < k; ++j) {
          w_scaled_nk[static_cast<size_t>(n)][static_cast<size_t>(j)] =
              w_nk[static_cast<size_t>(n)][static_cast<size_t>(j)] *
              channel_scale[static_cast<size_t>(j)];
        }
      }
      std::vector<std::vector<double>> codes_nk;
      std::vector<std::vector<double>> scale_blocks;
      QuantizeBlockwiseInt4(w_scaled_nk, c.block_size, codes_nk, scale_blocks);
      const std::vector<std::vector<double>> scale_full =
          expand_scales(scale_blocks);
      for (int64_t r = 0; r < num_rows; ++r) {
        for (int64_t n = 0; n < n_rows; ++n) {
          double acc = 0.0;
          for (int64_t j = 0; j < k; ++j) {
            acc += (rows.data[static_cast<size_t>(r * k + j)] /
                    channel_scale[static_cast<size_t>(j)]) *
                   (codes_nk[static_cast<size_t>(n)][static_cast<size_t>(j)] *
                    scale_full[static_cast<size_t>(n)][static_cast<size_t>(j)]);
          }
          y_hat[static_cast<size_t>(r)][static_cast<size_t>(n)] = acc;
        }
      }
      const double err = mean_squared_error(y_hat);
      if (err < best_err) {
        best_err = err;
        best_alpha = alpha;
        best_codes_nk = codes_nk;
        best_scale_blocks = scale_blocks;
        best_channel_scale = channel_scale;
      }
    }

    // Back to the stored layouts -- mirrors `codes_orig`/`scale_orig`
    // (transpose unless the weight was already [N, K]) exactly.
    // quantize_weight_only_int4 only ever replaces a node's weight input
    // (index 1), never its activation input (index 0) or the node's own
    // identity -- so the quantized graph's matching node still has this
    // exact activation input name.
    rewrites.push_back(Rewrite{});
    Rewrite& out = rewrites.back();
    out.output_name = c.output_name;
    out.activation_name = c.float_x_name;
    out.wq_name = c.wq_name;
    out.ws_name = c.ws_name;
    const int64_t nb_groups = k / c.block_size;
    if (c.weight_transposed) {
      out.codes_dims = {n_rows, k};
      out.scale_dims = {n_rows, nb_groups};
      out.codes.resize(static_cast<size_t>(n_rows * k));
      out.scale.resize(static_cast<size_t>(n_rows * nb_groups));
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          out.codes[static_cast<size_t>(i * k + j)] = static_cast<int8_t>(
              best_codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      }
      for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t g = 0; g < nb_groups; ++g) {
          out.scale[static_cast<size_t>(i * nb_groups + g)] =
              static_cast<float>(best_scale_blocks[static_cast<size_t>(i)]
                                                  [static_cast<size_t>(g)]);
        }
      }
    } else {
      out.codes_dims = {dim0, dim1};
      out.scale_dims = {nb_groups, n_rows};
      out.codes.resize(static_cast<size_t>(dim0 * dim1));
      out.scale.resize(static_cast<size_t>(nb_groups * n_rows));
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          out.codes[static_cast<size_t>(i * dim1 + j)] = static_cast<int8_t>(
              best_codes_nk[static_cast<size_t>(j)][static_cast<size_t>(i)]);
        }
      }
      for (int64_t g = 0; g < nb_groups; ++g) {
        for (int64_t i = 0; i < n_rows; ++i) {
          out.scale[static_cast<size_t>(g * n_rows + i)] =
              static_cast<float>(best_scale_blocks[static_cast<size_t>(i)]
                                                  [static_cast<size_t>(g)]);
        }
      }
    }
    if (best_alpha != 0.0) {
      out.channel_scale = best_channel_scale;
    }
  }

  if (rewrites.empty()) {
    return quantized_model;
  }

  onnx::ModelProto corrected = quantized_model;
  onnx::GraphProto* graph = corrected.mutable_graph();

  std::unordered_map<std::string, std::string> codes_by_name;
  for (const auto& r : rewrites) {
    std::vector<double> codes_flat(r.codes.size());
    for (size_t i = 0; i < r.codes.size(); ++i) {
      codes_flat[i] = static_cast<double>(r.codes[i]);
    }
    codes_by_name.emplace(r.wq_name, PackInt4(codes_flat));
  }
  std::unordered_map<std::string, Rewrite const*> rewrite_by_ws;
  for (const auto& r : rewrites) {
    if (!r.channel_scale.empty()) {
      rewrite_by_ws.emplace(r.ws_name, &r);
    }
  }
  for (auto& t : *graph->mutable_initializer()) {
    auto cit = codes_by_name.find(t.name());
    if (cit != codes_by_name.end()) {
      t.set_raw_data(cit->second);
    }
    auto sit = rewrite_by_ws.find(t.name());
    if (sit != rewrite_by_ws.end()) {
      const Rewrite& r = *sit->second;
      SetFloatInitializer(&t, r.ws_name, r.scale_dims, r.scale);
    }
  }

  // Mul nodes for the improved layers, in rewrite order (each insert
  // shifts later indices by one -- no deletions here, so a running
  // counter suffices, mirroring the reference's own by-identity
  // re-lookup).
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
  std::unordered_map<std::string, int> node_index_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    const auto& n = graph->node(i);
    if (n.output_size() > 0) {
      node_index_by_output[n.output(0)] = i;
    }
  }

  int64_t insertions = 0;
  for (const auto& r : rewrites) {
    if (r.channel_scale.empty()) {
      continue;
    }
    std::vector<float> inv_scale(r.channel_scale.size());
    for (size_t j = 0; j < inv_scale.size(); ++j) {
      inv_scale[j] = static_cast<float>(1.0 / r.channel_scale[j]);
    }
    const std::string scale_name =
        unique_name(r.activation_name + "_awq_inv_scale");
    SetFloatInitializer(graph->add_initializer(), scale_name,
                        {static_cast<int64_t>(inv_scale.size())}, inv_scale);
    const std::string scaled_name =
        unique_name(r.activation_name + "_awq_scaled");
    const std::string mul_name = unique_name(r.activation_name + "_awq_mul");
    const int live_index =
        node_index_by_output[r.output_name] + static_cast<int>(insertions);
    InsertEmptyNodeAt(graph, live_index);
    onnx::NodeProto* mul = graph->mutable_node(live_index);
    mul->set_op_type("Mul");
    mul->add_input(r.activation_name);
    mul->add_input(scale_name);
    mul->add_output(scaled_name);
    mul->set_name(mul_name);
    graph->mutable_node(live_index + 1)->set_input(0, scaled_name);
    ++insertions;
  }

  return corrected;
}
