// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See mixed_precision_entry.h for the full rationale (including why this
// follows llm_int8_entry.h's own protobuf-level, single-model,
// calibration-driven shape) and onnxsim/mixed_precision.py for the
// technique this ports.

#include "mixed_precision_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

using Matrix = std::vector<std::vector<double>>;

// --- MatMul/vanilla-Gemm matching, protobuf level ---------------------------
//
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike, which itself
// mirrors quip_sharp.py's own _match_matmul_like exactly (see that
// file's own comment) -- duplicated here rather than shared, matching
// this codebase's established per-*_entry.cpp self-containment
// convention.
struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
  std::string bias_name;  // Empty when the node has no bias input.
  bool weight_transposed;
};

std::optional<MatMulLikeMatch> MatchMatMulLike(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), "", false};
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return std::nullopt;
    }
    bool has_trans_a = false, has_alpha = false, has_beta = false;
    int64_t trans_a = 0, trans_b = 0;
    double alpha = 1.0, beta = 1.0;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "transA") {
        trans_a = attr.i();
        has_trans_a = true;
      } else if (attr.name() == "alpha") {
        alpha = attr.f();
        has_alpha = true;
      } else if (attr.name() == "beta") {
        beta = attr.f();
        has_beta = true;
      } else if (attr.name() == "transB") {
        trans_b = attr.i();
      }
    }
    if (has_trans_a && trans_a != 0) {
      return std::nullopt;
    }
    if (has_alpha && alpha != 1.0) {
      return std::nullopt;
    }
    std::string bias_name;
    if (num_inputs == 3) {
      if (has_beta && beta != 1.0) {
        return std::nullopt;
      }
      bias_name = node.input(2);
    }
    return MatMulLikeMatch{node.input(0), node.input(1), bias_name,
                           trans_b != 0};
  }
  return std::nullopt;
}

// --- Tensor <-> flat float buffer, raw initializer writing ----------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers.

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

// Round-half-to-even (banker's rounding), matching numpy's own `round`.
// Transcribed from llm_int8_entry.cpp's own RoundHalfToEven.
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

// Transcribed from omniquant_entry.cpp's own QuantizeBlockwiseInt4WithClip
// (clip_ratio always 1.0 here, i.e. plain min/max block-wise INT4 RTN --
// mixed_precision.py's own two call sites both pass clip_ratio=1.0).
std::pair<Matrix, Matrix> QuantizeBlockwiseInt4(const Matrix& w_nk,
                                                int64_t block_size) {
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
      scale_blocks[i][blk] = std::max(max_abs, 1e-12) / 7.0;
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

// Transcribed from mixed_precision.py's own _quantize_blockwise_int8:
// round-to-nearest block-wise INT8 ([-127, 127]) quantization.
std::pair<Matrix, Matrix> QuantizeBlockwiseInt8(const Matrix& w_nk,
                                                int64_t block_size) {
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
      scale_blocks[i][blk] = std::max(max_abs, 1e-12) / 127.0;
    }
  }
  Matrix codes_nk(n, std::vector<double>(k));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < k; ++j) {
      const size_t blk = j / static_cast<size_t>(block_size);
      const double s = scale_blocks[i][blk];
      const double q = RoundHalfToEven(w_nk[i][j] / s);
      codes_nk[i][j] = std::clamp(q, -127.0, 127.0);
    }
  }
  return {codes_nk, scale_blocks};
}

// Low-nibble-first packing -- transcribed from gptq_entry.cpp's own
// PackInt4 (which itself mirrors adaround.py's own _pack_int4).
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
// Transcribed from gptq_entry.cpp's own ActivationRows/
// AccumulateActivationRows (see that file's own comments for the
// probe-injection/batch-iteration/DLPack-crossing shape).

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
        calibration_data) {
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
            "ApplyMixedPrecisionQuantization: calibration batch is missing "
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

// Inserts a fresh node at position `index` (shifting later nodes right) --
// transcribed from llm_int8_entry.cpp's own InsertEmptyNodeAt.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

void AddIntAttribute(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

}  // namespace

onnx::ModelProto ApplyMixedPrecisionQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double high_bits_fraction, int64_t block_size,
    const std::string& sensitivity_metric) {
  if (sensitivity_metric != "hessian_diag" &&
      sensitivity_metric != "full_hessian") {
    throw std::invalid_argument(
        "ApplyMixedPrecisionQuantization: sensitivity_metric must be "
        "'hessian_diag' or 'full_hessian', got '" +
        sensitivity_metric + "'");
  }

  onnx::ModelProto out = model;

  bool opset_ge_21 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 21) {
      opset_ge_21 = true;
      break;
    }
  }
  if (!opset_ge_21) {
    return out;
  }

  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  struct Candidate {
    int node_index;
    std::string x_name;
    std::string w_name;
    std::string bias_name;
    bool weight_transposed;
  };
  std::vector<Candidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    auto m = MatchMatMulLike(graph->node(i));
    if (!m) {
      continue;
    }
    auto it = init_index.find(m->w_name);
    if (it == init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_init = graph->initializer(it->second);
    if (w_init.data_type() != onnx::TensorProto::FLOAT ||
        w_init.dims_size() != 2) {
      continue;
    }
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t k = m->weight_transposed ? dim1 : dim0;
    if (block_size <= 0 || k % block_size != 0) {
      continue;
    }
    candidates.push_back(
        {i, m->x_name, m->w_name, m->bias_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  const bool need_full_hessian = sensitivity_metric == "full_hessian";

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, model, probe_names,
                           calibration_data);

  // Per-candidate sensitivity score -- see header comment for the
  // hessian_diag/full_hessian formulas.
  std::vector<std::optional<double>> sensitivities(candidates.size());
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    const Candidate& c = candidates[idx];
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;
    }
    const ActivationRows& rows = ait->second;
    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;
    }
    const int64_t num_rows =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);
    if (num_rows <= 0) {
      continue;
    }

    const std::vector<float> w_flat = ReadFloatTensor(w_init);
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

    auto [codes_nk, scale_blocks] = QuantizeBlockwiseInt4(w_nk, block_size);
    const int64_t num_groups = k / block_size;
    Matrix err_nk(static_cast<size_t>(n_rows),
                  std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        const int64_t g = j / block_size;
        const double s =
            scale_blocks[static_cast<size_t>(i)][static_cast<size_t>(g)];
        err_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] -
            codes_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] * s;
      }
    }
    (void)num_groups;

    if (need_full_hessian) {
      // Full Hessian H = X^T X / rows, accumulated once per candidate
      // activation name (O(K^2) per row) -- mirrors mixed_precision.py's
      // own `full_h_sum[name] += x_rows.T @ x_rows` exactly.
      Matrix h(static_cast<size_t>(k),
               std::vector<double>(static_cast<size_t>(k), 0.0));
      for (int64_t r = 0; r < num_rows; ++r) {
        for (int64_t i = 0; i < k; ++i) {
          const double vi = rows.data[static_cast<size_t>(r * k + i)];
          if (vi == 0.0) {
            continue;
          }
          for (int64_t j = 0; j < k; ++j) {
            h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                vi * rows.data[static_cast<size_t>(r * k + j)];
          }
        }
      }
      for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] /=
              static_cast<double>(num_rows);
        }
      }
      // sensitivity = mean_n( sum_i sum_j e[n,i] * h[i,j] * e[n,j] )
      double total = 0.0;
      for (int64_t n = 0; n < n_rows; ++n) {
        const std::vector<double>& e = err_nk[static_cast<size_t>(n)];
        double acc = 0.0;
        for (int64_t i = 0; i < k; ++i) {
          if (e[static_cast<size_t>(i)] == 0.0) {
            continue;
          }
          double row_acc = 0.0;
          const std::vector<double>& hi = h[static_cast<size_t>(i)];
          for (int64_t j = 0; j < k; ++j) {
            row_acc += hi[static_cast<size_t>(j)] * e[static_cast<size_t>(j)];
          }
          acc += e[static_cast<size_t>(i)] * row_acc;
        }
        total += acc;
      }
      sensitivities[idx] = total / static_cast<double>(n_rows);
    } else {
      // diag_h[k] = mean_rows(X[:, k]^2); sensitivity =
      // mean_n( sum_k diag_h[k] * e[n,k]^2 ).
      std::vector<double> diag_h(static_cast<size_t>(k), 0.0);
      for (int64_t r = 0; r < num_rows; ++r) {
        for (int64_t i = 0; i < k; ++i) {
          const double v = rows.data[static_cast<size_t>(r * k + i)];
          diag_h[static_cast<size_t>(i)] += v * v;
        }
      }
      for (int64_t i = 0; i < k; ++i) {
        diag_h[static_cast<size_t>(i)] /= static_cast<double>(num_rows);
      }
      double total = 0.0;
      for (int64_t n = 0; n < n_rows; ++n) {
        double acc = 0.0;
        for (int64_t i = 0; i < k; ++i) {
          const double e =
              err_nk[static_cast<size_t>(n)][static_cast<size_t>(i)];
          acc += diag_h[static_cast<size_t>(i)] * e * e;
        }
        total += acc;
      }
      sensitivities[idx] = total / static_cast<double>(n_rows);
    }
  }

  std::vector<size_t> eligible_idx;
  for (size_t i = 0; i < sensitivities.size(); ++i) {
    if (sensitivities[i].has_value()) {
      eligible_idx.push_back(i);
    }
  }
  // Stable descending sort -- matches Python's own `sorted(...,
  // reverse=True)` stability exactly (ties keep original relative order).
  std::stable_sort(eligible_idx.begin(), eligible_idx.end(),
                   [&](size_t a, size_t b) {
                     return *sensitivities[a] > *sensitivities[b];
                   });
  const int64_t num_high_bits = static_cast<int64_t>(RoundHalfToEven(
      high_bits_fraction * static_cast<double>(eligible_idx.size())));
  std::unordered_set<size_t> high_bits_set;
  for (int64_t i = 0;
       i < num_high_bits && i < static_cast<int64_t>(eligible_idx.size());
       ++i) {
    high_bits_set.insert(eligible_idx[static_cast<size_t>(i)]);
  }

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

  std::unordered_set<size_t> eligible_set(eligible_idx.begin(),
                                          eligible_idx.end());

  int64_t net_insertions = 0;
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    if (!eligible_set.count(idx)) {
      continue;
    }
    const Candidate& c = candidates[idx];
    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;

    const std::vector<float> w_flat = ReadFloatTensor(w_init);
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

    const bool use_int8 = high_bits_set.count(idx) != 0;
    Matrix codes_nk, scale_blocks;
    int32_t codes_dtype;
    std::string prefix;
    if (use_int8) {
      std::tie(codes_nk, scale_blocks) =
          QuantizeBlockwiseInt8(w_nk, block_size);
      codes_dtype = onnx::TensorProto::INT8;
      prefix = c.w_name + "_mixedprec_int8";
    } else {
      std::tie(codes_nk, scale_blocks) =
          QuantizeBlockwiseInt4(w_nk, block_size);
      codes_dtype = onnx::TensorProto::INT4;
      prefix = c.w_name + "_mixedprec_int4";
    }
    const int64_t num_groups = k / block_size;

    // codes_kn/scale_kn: [K, N] / [K/block_size, N], row-major -- mirrors
    // `codes_nk.T` / `scale_blocks_nk.T` exactly.
    std::vector<double> codes_kn_flat(static_cast<size_t>(k * n_rows));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < n_rows; ++j) {
        codes_kn_flat[static_cast<size_t>(i * n_rows + j)] =
            codes_nk[static_cast<size_t>(j)][static_cast<size_t>(i)];
      }
    }
    std::vector<float> scale_kn_flat(static_cast<size_t>(num_groups * n_rows));
    for (int64_t g = 0; g < num_groups; ++g) {
      for (int64_t j = 0; j < n_rows; ++j) {
        scale_kn_flat[static_cast<size_t>(g * n_rows + j)] = static_cast<float>(
            scale_blocks[static_cast<size_t>(j)][static_cast<size_t>(g)]);
      }
    }

    const std::string codes_name = unique_name(prefix + "_codes");
    if (codes_dtype == onnx::TensorProto::INT4) {
      onnx::TensorProto* codes_tensor = graph->add_initializer();
      codes_tensor->set_name(codes_name);
      codes_tensor->set_data_type(onnx::TensorProto::INT4);
      codes_tensor->add_dims(k);
      codes_tensor->add_dims(n_rows);
      codes_tensor->set_raw_data(PackInt4(codes_kn_flat));
    } else {
      std::vector<int8_t> codes_i8(codes_kn_flat.size());
      for (size_t i = 0; i < codes_kn_flat.size(); ++i) {
        codes_i8[i] =
            static_cast<int8_t>(static_cast<int64_t>(codes_kn_flat[i]));
      }
      SetRawInitializer(graph->add_initializer(), codes_name,
                        onnx::TensorProto::INT8, {k, n_rows}, codes_i8.data(),
                        codes_i8.size() * sizeof(int8_t), sizeof(int8_t));
    }

    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_groups, n_rows},
                      scale_kn_flat.data(),
                      scale_kn_flat.size() * sizeof(float), sizeof(float));

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
      bool has_int_attrs = false;
      std::vector<std::pair<std::string, int64_t>> int_attrs;
    };
    std::vector<NewNode> new_nodes;
    auto add_node =
        [&](const std::string& op_type, const std::vector<std::string>& inputs,
            const std::string& out_suffix,
            const std::vector<std::pair<std::string, int64_t>>& int_attrs =
                {}) {
          NewNode n;
          n.op_type = op_type;
          n.inputs = inputs;
          n.output = unique_name(prefix + "_" + out_suffix);
          n.name = unique_name(prefix + "_" + out_suffix + "_node");
          n.int_attrs = int_attrs;
          new_nodes.push_back(std::move(n));
          return new_nodes.back().output;
        };

    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string core = add_node("MatMul", {c.x_name, w_dequant}, "core");

    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const std::string old_output = graph->node(live_index).output(0);

    NewNode final_node;
    if (!c.bias_name.empty()) {
      final_node.op_type = "Add";
      final_node.inputs = {core, c.bias_name};
      final_node.output = old_output;
      final_node.name = unique_name(prefix + "_bias_add_node");
    } else {
      final_node.op_type = "Identity";
      final_node.inputs = {core};
      final_node.output = old_output;
      final_node.name = unique_name(prefix + "_identity_node");
    }
    new_nodes.push_back(std::move(final_node));

    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, live_index + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* n = graph->mutable_node(live_index + i);
      n->set_op_type(spec.op_type);
      for (const auto& s : spec.inputs) {
        n->add_input(s);
      }
      n->add_output(spec.output);
      n->set_name(spec.name);
      for (const auto& [attr_name, attr_value] : spec.int_attrs) {
        AddIntAttribute(n, attr_name, attr_value);
      }
    }
    graph->mutable_node()->DeleteSubrange(live_index + count, 1);
    net_insertions += count - 1;
  }

  return out;
}
