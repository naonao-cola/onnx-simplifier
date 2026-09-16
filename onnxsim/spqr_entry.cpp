// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See spqr_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own single-model, calibration-driven shape) and
// onnxsim/spqr.py for the technique this ports.

#include "spqr_entry.h"

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

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike, which itself
// mirrors onnxsim.quip_sharp._match_matmul_like -- the exact matcher
// onnxsim.spqr reuses.
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

// --- Tensor <-> flat buffers, protobuf level -------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers (FLOAT32
// only -- this pass, like its own Python reference onnxsim.spqr, never
// widens to FLOAT16/BFLOAT16).

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

// Round-half-to-even (banker's rounding) -- matches numpy's own `round`
// AND Python's own built-in `round()` (both used by spqr.py: `np.round`
// for the element codes, `round()` for `num_outliers`), unlike
// std::round's half-away-from-zero.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4 (and
// adaround_entry.cpp's own PackInt4, byte-for-byte identical -- not
// reused directly since it is private to that translation unit).
// `codes_flat.size()` must be even (spqr.py's own _pack_int4 shares this
// same implicit assumption -- see spqr_entry.h's own top-of-file comment
// for why this port doesn't add an independent check beyond what the
// Python reference already relies on).
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
// AccumulateActivationRows -- the general (rank >= 2, not just plain 2-D)
// capture onnxsim.bias_correction._activation_rows performs, which
// spqr.py imports directly (see spqr_entry.h's own top-of-file comment
// for why this is NOT llm_int8_entry.cpp's own narrower, 2-D-only
// ComputeChannelAbsmax).
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
            "ApplySpqr: calibration batch is missing "
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
// Transcribed from llm_int8_entry.cpp's own identical helper.
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

// ConstantOfShape's own "value" attribute: a single-element TENSOR
// attribute (here always a [1]-shaped float32 tensor holding 0.0,
// mirroring spqr.py's own
// `value=onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float32))`).
void AddFloatTensorAttribute(onnx::NodeProto* node, const std::string& name,
                             float value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto* t = attr->mutable_t();
  t->set_data_type(onnx::TensorProto::FLOAT);
  t->add_dims(1);
  std::string raw(sizeof(float), '\0');
  std::memcpy(raw.data(), &value, sizeof(float));
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

}  // namespace

onnx::ModelProto ApplySpqr(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, double outlier_fraction) {
  onnx::ModelProto out = model;

  // INT4's tensor type and DequantizeLinear's block_size attribute both
  // need opset >= 21 -- mirrors quantize_weight_only_spqr's own
  // `_has_min_opset(model, 21)` gate exactly.
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
    candidates.push_back(
        {i, m->x_name, m->w_name, m->bias_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  onnx::ModelProto probe_model = out;
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, probe_model, probe_names,
                           calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...) -- same convention every calibration-driven
  // *_entry.cpp in this repo already uses.
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
  for (const auto& c : candidates) {
    auto acts_it = activations.find(c.x_name);
    if (acts_it == activations.end() || acts_it->second.data.empty()) {
      continue;  // Never observed; skip (mirrors `if not acts: continue`).
    }
    const ActivationRows& rows = acts_it->second;
    const int64_t k_obs = rows.k;
    const int64_t total_rows = static_cast<int64_t>(rows.data.size()) / k_obs;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [N, K], output channel first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (k % block_size != 0 || k_obs != k) {
      continue;  // Mirrors `k % block_size != 0 or x.shape[1] != k`.
    }
    const int64_t num_blocks = k / block_size;

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // h_k = mean(X[:, k]^2, axis=0) -- diagonal-Hessian approximation.
    std::vector<double> h_k(static_cast<size_t>(k), 0.0);
    for (int64_t r = 0; r < total_rows; ++r) {
      const double* row = &rows.data[static_cast<size_t>(r * k)];
      for (int64_t j = 0; j < k; ++j) {
        h_k[static_cast<size_t>(j)] += row[j] * row[j];
      }
    }
    for (int64_t j = 0; j < k; ++j) {
      h_k[static_cast<size_t>(j)] /= static_cast<double>(total_rows);
    }

    // sensitivity[n][k] = w_nk[n][k]^2 * h_k[k] -- flat [N*K], row-major
    // over [N, K], matching numpy's own default C-order flatten/unravel.
    std::vector<double> sensitivity(static_cast<size_t>(n_rows * k));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        const double w = at_nk(nn, j);
        sensitivity[static_cast<size_t>(nn * k + j)] =
            w * w * h_k[static_cast<size_t>(j)];
      }
    }

    const int64_t num_outliers = static_cast<int64_t>(
        RoundHalfToEven(outlier_fraction * static_cast<double>(n_rows) *
                        static_cast<double>(k)));

    std::vector<int64_t> outlier_rows, outlier_cols;
    std::vector<uint8_t> is_outlier(static_cast<size_t>(n_rows * k), 0);
    if (num_outliers > 0) {
      std::vector<int64_t> order(static_cast<size_t>(n_rows * k));
      for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int64_t>(i);
      }
      // Full deterministic sort, descending by sensitivity, ties broken
      // ascending by flat index -- see spqr_entry.h's own top-of-file
      // "ACCEPTED, PERMANENT DIVERGENCE" note for why this doesn't try to
      // reproduce numpy's own argpartition order.
      std::partial_sort(
          order.begin(),
          order.begin() + std::min<int64_t>(num_outliers,
                                            static_cast<int64_t>(order.size())),
          order.end(), [&](int64_t a, int64_t b) {
            const double sa = sensitivity[static_cast<size_t>(a)];
            const double sb = sensitivity[static_cast<size_t>(b)];
            if (sa != sb) {
              return sa > sb;
            }
            return a < b;
          });
      const int64_t take =
          std::min<int64_t>(num_outliers, static_cast<int64_t>(order.size()));
      std::vector<int64_t> chosen(order.begin(), order.begin() + take);
      std::sort(chosen.begin(), chosen.end());  // Clean, deterministic
                                                // storage order.
      outlier_rows.reserve(static_cast<size_t>(take));
      outlier_cols.reserve(static_cast<size_t>(take));
      for (int64_t flat_idx : chosen) {
        is_outlier[static_cast<size_t>(flat_idx)] = 1;
        outlier_rows.push_back(flat_idx / k);
        outlier_cols.push_back(flat_idx % k);
      }
    }

    // Block-wise scale, excluded-outlier-aware -- mirrors
    // `abs_masked = np.where(mask_blocks, np.abs(blocks), 0.0);
    // scale_blocks = np.maximum(abs_masked.max(axis=2), 1e-12) / 7.0`
    // exactly (an outlier position's own magnitude is zeroed, not
    // excluded from the reduction, so an all-outlier block still floors
    // at 1e-12 via the same np.maximum).
    std::vector<double> scale_blocks(static_cast<size_t>(n_rows * num_blocks));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        double max_abs = 0.0;
        for (int64_t i = 0; i < block_size; ++i) {
          const int64_t j = b * block_size + i;
          const size_t flat_idx = static_cast<size_t>(nn * k + j);
          if (is_outlier[flat_idx]) {
            continue;  // Zeroed, same effect as np.where's own 0.0.
          }
          max_abs = std::max(max_abs, std::abs(at_nk(nn, j)));
        }
        scale_blocks[static_cast<size_t>(nn * num_blocks + b)] =
            std::max(max_abs, 1e-12) / 7.0;
      }
    }

    // codes_nk/dequant_nk computed for EVERY element (outlier positions
    // included -- their dequantized value is needed to compute the exact
    // correction below).
    std::vector<double> codes_nk(static_cast<size_t>(n_rows * k));
    std::vector<double> dequant_nk(static_cast<size_t>(n_rows * k));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        const int64_t b = j / block_size;
        const double scale =
            scale_blocks[static_cast<size_t>(nn * num_blocks + b)];
        const double q = std::min(
            7.0, std::max(-7.0, RoundHalfToEven(at_nk(nn, j) / scale)));
        const size_t idx = static_cast<size_t>(nn * k + j);
        codes_nk[idx] = q;
        dequant_nk[idx] = q * scale;
      }
    }

    const std::string prefix = c.w_name + "_spqr";

    // codes_kn [K, N] (transpose of codes_nk), packed INT4.
    std::vector<double> codes_kn(static_cast<size_t>(k * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        codes_kn[static_cast<size_t>(j * n_rows + nn)] =
            codes_nk[static_cast<size_t>(nn * k + j)];
      }
    }
    const std::string codes_name = unique_name(prefix + "_codes");
    {
      onnx::TensorProto* t = graph->add_initializer();
      t->Clear();
      t->set_name(codes_name);
      t->set_data_type(onnx::TensorProto::INT4);
      t->add_dims(k);
      t->add_dims(n_rows);
      t->set_raw_data(PackInt4(codes_kn));
    }

    // scale_kn [K/block_size, N] float32 (transpose of scale_blocks).
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        scale_kn[static_cast<size_t>(b * n_rows + nn)] = static_cast<float>(
            scale_blocks[static_cast<size_t>(nn * num_blocks + b)]);
      }
    }
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_blocks, n_rows},
                      scale_kn.data(), scale_kn.size() * sizeof(float),
                      sizeof(float));

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
      std::vector<std::pair<std::string, int64_t>> int_attrs;
      bool has_tensor_attr = false;
      std::string tensor_attr_name;
      float tensor_attr_value = 0.0f;
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

    std::string w_reconstructed = w_dequant;
    const int64_t num_outliers_actual =
        static_cast<int64_t>(outlier_rows.size());
    if (num_outliers_actual > 0) {
      std::vector<int64_t> outlier_indices_kn(
          static_cast<size_t>(num_outliers_actual * 2));
      std::vector<float> outlier_values(
          static_cast<size_t>(num_outliers_actual));
      for (int64_t i = 0; i < num_outliers_actual; ++i) {
        const int64_t row = outlier_rows[static_cast<size_t>(i)];
        const int64_t col = outlier_cols[static_cast<size_t>(i)];
        // [K, N]-layout indices, matching codes_kn/scale_kn's own
        // transposed storage: index[i] = [k_pos, n_pos].
        outlier_indices_kn[static_cast<size_t>(2 * i)] = col;
        outlier_indices_kn[static_cast<size_t>(2 * i + 1)] = row;
        const size_t flat_idx = static_cast<size_t>(row * k + col);
        outlier_values[static_cast<size_t>(i)] =
            static_cast<float>(at_nk(row, col) - dequant_nk[flat_idx]);
      }

      const std::string indices_name = unique_name(prefix + "_outlier_indices");
      SetRawInitializer(
          graph->add_initializer(), indices_name, onnx::TensorProto::INT64,
          {num_outliers_actual, 2}, outlier_indices_kn.data(),
          outlier_indices_kn.size() * sizeof(int64_t), sizeof(int64_t));
      const std::string values_name = unique_name(prefix + "_outlier_values");
      SetRawInitializer(graph->add_initializer(), values_name,
                        onnx::TensorProto::FLOAT, {num_outliers_actual},
                        outlier_values.data(),
                        outlier_values.size() * sizeof(float), sizeof(float));
      const int64_t shape_kn[2] = {k, n_rows};
      const std::string shape_name = unique_name(prefix + "_shape");
      SetRawInitializer(graph->add_initializer(), shape_name,
                        onnx::TensorProto::INT64, {2}, shape_kn,
                        sizeof(shape_kn), sizeof(int64_t));

      const std::string zeros =
          add_node("ConstantOfShape", {shape_name}, "zeros");
      new_nodes.back().has_tensor_attr = true;
      new_nodes.back().tensor_attr_name = "value";
      new_nodes.back().tensor_attr_value = 0.0f;
      const std::string correction = add_node(
          "ScatterND", {zeros, indices_name, values_name}, "correction");
      w_reconstructed =
          add_node("Add", {w_dequant, correction}, "w_reconstructed");
    }

    const std::string core =
        add_node("MatMul", {c.x_name, w_reconstructed}, "core");

    const std::string old_output =
        graph->node(c.node_index + static_cast<int>(net_insertions)).output(0);
    NewNode final_node;
    final_node.output = old_output;
    if (!c.bias_name.empty()) {
      final_node.op_type = "Add";
      final_node.inputs = {core, c.bias_name};
      final_node.name = unique_name(prefix + "_bias_add_node");
    } else {
      final_node.op_type = "Identity";
      final_node.inputs = {core};
      final_node.name = unique_name(prefix + "_identity_node");
    }
    new_nodes.push_back(std::move(final_node));

    // Splice the new nodes in before the old node, then delete the old
    // node itself -- mirrors insert-then-`del` exactly (same pattern
    // llm_int8_entry.cpp/gptq_entry.cpp both already use).
    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, live_index + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* nd = graph->mutable_node(live_index + i);
      nd->set_op_type(spec.op_type);
      for (const auto& s : spec.inputs) {
        nd->add_input(s);
      }
      nd->add_output(spec.output);
      nd->set_name(spec.name);
      for (const auto& [attr_name, attr_value] : spec.int_attrs) {
        AddIntAttribute(nd, attr_name, attr_value);
      }
      if (spec.has_tensor_attr) {
        AddFloatTensorAttribute(nd, spec.tensor_attr_name,
                                spec.tensor_attr_value);
      }
    }
    graph->mutable_node()->DeleteSubrange(live_index + count, 1);
    net_insertions += count - 1;
  }

  return out;
}
