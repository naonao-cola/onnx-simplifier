// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See quantease_entry.h for the full rationale (including why this follows
// gptq_entry.h's own two-model, calibration-driven shape) and
// onnxsim/quantease.py for the technique this ports.

#include "quantease_entry.h"

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
// Transcribed from gptq_entry.cpp's own identical FindInt4MatmulCandidates
// (itself transcribed from adaround.py's own _find_int4_matmul_candidates,
// which apply_quantease reuses via adaround.py's own
// _find_int4_matmul_candidates import) -- not reused directly since it is
// private to that translation unit's own anonymous namespace.

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
// Transcribed from gptq_entry.cpp's own identical ReadFloatTensor.

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

// Round-half-to-even (banker's rounding), matching numpy's own `round`.
// Transcribed from gptq_entry.cpp's own identical RoundHalfToEven.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4. Transcribed
// from gptq_entry.cpp's own identical PackInt4 -- codes are integral
// doubles in [-7, 7], packed two per byte.
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
// Transcribed from gptq_entry.cpp's own identical AccumulateActivationRows.

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
            "ApplyQuantease: calibration batch is missing "
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

// --- QuantEase's own cyclic coordinate-descent column update ---------------
//
// Transcribed from quantease.py's own _quantease_quantize_columns -- see
// this port's own quantease_entry.h for the full derivation. `w_nk`/
// `scale_blocks` are [N, K] / [N, num_groups]; `h` is the layer's [K, K]
// Hessian. Returns the [N, K] integer code matrix (double-precision,
// integral values in [-7, 7]).
std::vector<std::vector<double>> QuanteaseQuantizeColumns(
    const std::vector<std::vector<double>>& w_nk,
    const std::vector<std::vector<double>>& scale_blocks,
    int64_t quant_block_size, const std::vector<std::vector<double>>& h,
    int64_t num_epochs) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();

  // Fresh round-to-nearest starting point -- same `max(|w| in block,
  // 1e-12) / 7`, round-half-to-even, clamp-to-[-7,7] formula
  // `_quantize_blockwise_int4` (awq.py) uses; its own internal scale is
  // discarded (only used to derive the initial integer codes), mirroring
  // the Python reference's own `codes_nk, _ = _quantize_blockwise_int4(...)`
  // exactly.
  const int64_t num_blocks_init = static_cast<int64_t>(k) / quant_block_size;
  std::vector<std::vector<double>> codes_nk(n, std::vector<double>(k, 0.0));
  for (size_t r = 0; r < n; ++r) {
    for (int64_t b = 0; b < num_blocks_init; ++b) {
      double max_abs = 0.0;
      for (int64_t j = 0; j < quant_block_size; ++j) {
        const size_t c = static_cast<size_t>(b * quant_block_size + j);
        max_abs = std::max(max_abs, std::abs(w_nk[r][c]));
      }
      const double init_scale = std::max(max_abs, 1e-12) / 7.0;
      for (int64_t j = 0; j < quant_block_size; ++j) {
        const size_t c = static_cast<size_t>(b * quant_block_size + j);
        const double code = std::min(
            7.0, std::max(-7.0, RoundHalfToEven(w_nk[r][c] / init_scale)));
        codes_nk[r][c] = code;
      }
    }
  }

  // scale_full[n, kk] = scale_blocks[n, kk / quant_block_size] -- mirrors
  // `np.repeat(scale_blocks, quant_block_size, axis=1)`.
  auto scale_at = [&](size_t r, size_t kk) -> double {
    return scale_blocks[r][kk / static_cast<size_t>(quant_block_size)];
  };

  // w_hat/r use the EXTERNALLY supplied scale, not `_quantize_blockwise_
  // int4`'s own -- mirrors `w_hat = codes_nk * scale_full` exactly (only
  // the starting integer codes come from the fresh quantization above; the
  // value each code represents is always this layer's own already-fixed
  // per-group scale).
  std::vector<std::vector<double>> w_hat(n, std::vector<double>(k, 0.0));
  std::vector<std::vector<double>> r(n, std::vector<double>(k, 0.0));
  for (size_t row = 0; row < n; ++row) {
    for (size_t kk = 0; kk < k; ++kk) {
      const double s = scale_at(row, kk);
      w_hat[row][kk] = codes_nk[row][kk] * s;
      r[row][kk] = w_nk[row][kk] - w_hat[row][kk];
    }
  }

  std::vector<double> h_diag(k);
  for (size_t i = 0; i < k; ++i) {
    h_diag[i] = std::max(h[i][i], 1e-12);
  }

  for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
    for (size_t kk = 0; kk < k; ++kk) {
      const size_t group = kk / static_cast<size_t>(quant_block_size);
      for (size_t row = 0; row < n; ++row) {
        // delta = (r[row, :] @ h[:, kk]) / h_diag[kk].
        double dot = 0.0;
        for (size_t i = 0; i < k; ++i) {
          dot += r[row][i] * h[i][kk];
        }
        const double delta = dot / h_diag[kk];
        const double s = scale_blocks[row][group];
        const double unconstrained = w_hat[row][kk] + delta;
        const double new_code =
            std::min(7.0, std::max(-7.0, RoundHalfToEven(unconstrained / s)));
        const double new_val = new_code * s;
        r[row][kk] -= new_val - w_hat[row][kk];
        w_hat[row][kk] = new_val;
        codes_nk[row][kk] = new_code;
      }
    }
  }

  return codes_nk;
}

}  // namespace

onnx::ModelProto ApplyQuantease(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_epochs) {
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

  std::unordered_map<std::string, std::string> optimized;
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

    const onnx::GraphProto& q_graph = quantized_model.graph();
    const onnx::TensorProto& ws_init =
        q_graph.initializer(q_init_index[c.ws_name]);
    const std::vector<float> s_flat = ReadFloatTensor(ws_init);
    // Scale blocks [N, K/block] (transposed back when the stored layout is
    // [K/block, N]) -- mirrors `scale_blocks = scale[.T]` exactly.
    const int64_t num_groups = k / c.block_size;
    std::vector<std::vector<double>> scale_blocks(
        static_cast<size_t>(n_rows),
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

    // Hessian over every concatenated row: H = X^T X.
    std::vector<std::vector<double>> h(
        static_cast<size_t>(k),
        std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t rr = 0; rr < num_rows; ++rr) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi = rows.data[static_cast<size_t>(rr * k + i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * rows.data[static_cast<size_t>(rr * k + j)];
        }
      }
    }

    const std::vector<std::vector<double>> codes_nk = QuanteaseQuantizeColumns(
        w_nk, scale_blocks, c.block_size, h, num_epochs);

    // Back to the stored layout and nibble-packed -- mirrors
    // `codes_orig = codes_nk[.T]` plus `_pack_int4` exactly.
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
