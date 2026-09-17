// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See paroquant_entry.h for the full rationale (including why this follows
// spqr_entry.h's own single-model, calibration-driven shape) and
// onnxsim/paroquant.py for the technique this ports.

#include "paroquant_entry.h"

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
// Transcribed from spqr_entry.cpp's own MatchMatMulLike, which itself
// mirrors onnxsim.quip_sharp._match_matmul_like -- the exact matcher
// onnxsim.paroquant reuses.
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

// Round-half-to-even (banker's rounding) -- matches numpy's own `np.round`
// (used by omniquant.py's own _quantize_blockwise_int4_with_clip, which
// paroquant.py reuses), unlike std::round's half-away-from-zero.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4 -- see
// spinquant_entry.cpp's own identical PackInt4 for the "K*N always even
// in practice here" note (not reused directly since it is private to that
// translation unit).
std::string PackInt4(const std::vector<double>& codes_flat) {
  std::string packed;
  packed.resize((codes_flat.size() + 1) / 2);
  for (size_t i = 0; i * 2 < codes_flat.size(); ++i) {
    const auto lo =
        static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i]));
    uint8_t hi = 0;
    if (2 * i + 1 < codes_flat.size()) {
      hi = static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i + 1]));
    }
    packed[i] = static_cast<char>((lo & 0xF) | ((hi & 0xF) << 4));
  }
  return packed;
}

// --- Block-wise INT4 quantization with clip --------------------------------
//
// Transcribed from omniquant.py's own _quantize_blockwise_int4_with_clip
// (which paroquant.py reuses directly, always with clip_ratio=1.0): each
// block's scale is max(|w| in block) * clip_ratio / 7 (floored at 1e-12),
// codes are round-half-to-even, clamped to [-7, 7]. Used both for the
// per-pair angle-search's own block-local re-quantization (called with
// `k == block_size`, i.e. a single block) and for the final full-width
// quantization (`k` a multiple of `block_size`).
struct BlockwiseInt4 {
  std::vector<double> codes;         // [n, k]
  std::vector<double> scale_blocks;  // [n, num_blocks]
};

BlockwiseInt4 QuantizeBlockwiseInt4WithClip(const std::vector<double>& w_nk,
                                            int64_t n, int64_t k,
                                            int64_t block_size,
                                            double clip_ratio) {
  const int64_t num_blocks = k / block_size;
  BlockwiseInt4 out;
  out.codes.resize(static_cast<size_t>(n * k));
  out.scale_blocks.resize(static_cast<size_t>(n * num_blocks));
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double max_abs = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const int64_t c = b * block_size + j;
        max_abs =
            std::max(max_abs, std::abs(w_nk[static_cast<size_t>(r * k + c)]));
      }
      const double scale = std::max(max_abs * clip_ratio, 1e-12) / 7.0;
      out.scale_blocks[static_cast<size_t>(r * num_blocks + b)] = scale;
      for (int64_t j = 0; j < block_size; ++j) {
        const int64_t c = b * block_size + j;
        const double q = std::min(
            7.0,
            std::max(-7.0, RoundHalfToEven(
                               w_nk[static_cast<size_t>(r * k + c)] / scale)));
        out.codes[static_cast<size_t>(r * k + c)] = q;
      }
    }
  }
  return out;
}

// --- Pairwise (Givens) rotation fit -----------------------------------------
//
// Transcribed from paroquant.py's own _fit_paroquant_pairwise_rotation --
// see paroquant_entry.h's own top-of-file comment for the full algorithm
// description. `w_nk` is [n, k], already SmoothQuant-scaled by the caller.
struct PairwiseRotationResult {
  std::vector<double> r;          // [k, k], block-diagonal Givens blocks.
  std::vector<double> w_rotated;  // [n, k] == w_nk @ r.
};

PairwiseRotationResult FitPairwiseRotation(const std::vector<double>& w_nk,
                                           int64_t n, int64_t k,
                                           int64_t block_size,
                                           int64_t num_angle_steps) {
  PairwiseRotationResult result;
  result.r.assign(static_cast<size_t>(k * k), 0.0);
  for (int64_t i = 0; i < k; ++i) {
    result.r[static_cast<size_t>(i * k + i)] = 1.0;
  }
  result.w_rotated = w_nk;
  std::vector<double>& w_work = result.w_rotated;

  std::vector<double> thetas(static_cast<size_t>(num_angle_steps));
  for (int64_t i = 0; i < num_angle_steps; ++i) {
    const double frac =
        num_angle_steps > 1
            ? static_cast<double>(i) / static_cast<double>(num_angle_steps - 1)
            : 0.0;
    thetas[static_cast<size_t>(i)] = -M_PI / 4.0 + frac * (M_PI / 2.0);
  }

  std::vector<double> block(static_cast<size_t>(n * block_size));
  for (int64_t start = 0; start < k; start += block_size) {
    const int64_t end = start + block_size;
    for (int64_t i = start; i < end; i += 2) {
      const int64_t j = i + 1;
      std::vector<double> col_i(static_cast<size_t>(n)),
          col_j(static_cast<size_t>(n));
      for (int64_t r = 0; r < n; ++r) {
        col_i[static_cast<size_t>(r)] = w_work[static_cast<size_t>(r * k + i)];
        col_j[static_cast<size_t>(r)] = w_work[static_cast<size_t>(r * k + j)];
      }

      double best_err = -1.0;
      double best_theta = 0.0;
      std::vector<double> best_ci = col_i;
      std::vector<double> best_cj = col_j;

      for (double theta : thetas) {
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        for (int64_t r = 0; r < n; ++r) {
          const double new_i = c * col_i[static_cast<size_t>(r)] -
                               s * col_j[static_cast<size_t>(r)];
          const double new_j = s * col_i[static_cast<size_t>(r)] +
                               c * col_j[static_cast<size_t>(r)];
          for (int64_t col = 0; col < block_size; ++col) {
            const int64_t global_col = start + col;
            double v;
            if (global_col == i) {
              v = new_i;
            } else if (global_col == j) {
              v = new_j;
            } else {
              v = w_work[static_cast<size_t>(r * k + global_col)];
            }
            block[static_cast<size_t>(r * block_size + col)] = v;
          }
        }
        const BlockwiseInt4 q = QuantizeBlockwiseInt4WithClip(
            block, n, block_size, block_size, 1.0);
        double sum_sq = 0.0;
        for (int64_t r = 0; r < n; ++r) {
          for (int64_t col = 0; col < block_size; ++col) {
            const size_t idx = static_cast<size_t>(r * block_size + col);
            const double recon =
                q.codes[idx] * q.scale_blocks[static_cast<size_t>(r)];
            const double diff = block[idx] - recon;
            sum_sq += diff * diff;
          }
        }
        const double err = sum_sq / static_cast<double>(n * block_size);
        if (best_err < 0.0 || err < best_err) {
          best_err = err;
          best_theta = theta;
          for (int64_t r = 0; r < n; ++r) {
            best_ci[static_cast<size_t>(r)] =
                c * col_i[static_cast<size_t>(r)] -
                s * col_j[static_cast<size_t>(r)];
            best_cj[static_cast<size_t>(r)] =
                s * col_i[static_cast<size_t>(r)] +
                c * col_j[static_cast<size_t>(r)];
          }
        }
      }

      for (int64_t r = 0; r < n; ++r) {
        w_work[static_cast<size_t>(r * k + i)] =
            best_ci[static_cast<size_t>(r)];
        w_work[static_cast<size_t>(r * k + j)] =
            best_cj[static_cast<size_t>(r)];
      }
      const double c = std::cos(best_theta);
      const double s = std::sin(best_theta);
      result.r[static_cast<size_t>(i * k + i)] = c;
      result.r[static_cast<size_t>(i * k + j)] = s;
      result.r[static_cast<size_t>(j * k + i)] = -s;
      result.r[static_cast<size_t>(j * k + j)] = c;
    }
  }

  return result;
}

// --- Calibration: per-channel activation absmax ----------------------------
//
// Transcribed from smoothquant_entry.cpp's own ComputeChannelAbsmax --
// paroquant.py's own per-channel abs-max capture is the exact same shape
// (`if x.ndim != 2: continue`, elementwise-max across batches).
std::unordered_map<std::string, std::vector<double>> ComputeChannelAbsmax(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  std::unordered_map<std::string, std::vector<double>> result;
  if (probe_names.empty()) {
    return result;
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
            "ApplyParoquant: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() != 2) {
        continue;
      }
      const int64_t k = tp.dims(1);
      if (k <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      auto& acc = result[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(k), 0.0);
      }
      for (int64_t flat = 0, total = static_cast<int64_t>(data.size());
           flat < total; ++flat) {
        const double v =
            std::abs(static_cast<double>(data[static_cast<size_t>(flat)]));
        double& slot = acc[static_cast<size_t>(flat % k)];
        if (v > slot) {
          slot = v;
        }
      }
    }
  }
  return result;
}

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

onnx::ModelProto ApplyParoquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, double alpha, int64_t num_angle_steps, double epsilon) {
  onnx::ModelProto out = model;

  bool opset_ge_21 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 21) {
      opset_ge_21 = true;
      break;
    }
  }
  if (!opset_ge_21 || block_size % 2 != 0) {
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
  const std::unordered_map<std::string, std::vector<double>> act_absmax =
      ComputeChannelAbsmax(executor, probe_model, probe_names,
                           calibration_data);

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
    auto acts_it = act_absmax.find(c.x_name);
    if (acts_it == act_absmax.end()) {
      continue;
    }
    const std::vector<double>& absmax = acts_it->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (k % block_size != 0 || absmax.size() != static_cast<size_t>(k)) {
      continue;
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // s_j = max(|X_j|, eps)**alpha / max(|W_j|, eps)**(1-alpha), floored at
    // eps -- mirrors apply_paroquant's own three lines exactly.
    std::vector<double> s(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      double wmax = 0.0;
      for (int64_t nn = 0; nn < n_rows; ++nn) {
        wmax = std::max(wmax, std::abs(at_nk(nn, j)));
      }
      const double a = std::max(absmax[static_cast<size_t>(j)], epsilon);
      const double w = std::max(wmax, epsilon);
      const double sj = std::pow(a, alpha) / std::pow(w, 1.0 - alpha);
      s[static_cast<size_t>(j)] = std::max(sj, epsilon);
    }

    std::vector<double> w_smooth_nk(static_cast<size_t>(n_rows * k));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        w_smooth_nk[static_cast<size_t>(nn * k + j)] =
            at_nk(nn, j) * s[static_cast<size_t>(j)];
      }
    }

    const PairwiseRotationResult rot = FitPairwiseRotation(
        w_smooth_nk, n_rows, k, block_size, num_angle_steps);

    const BlockwiseInt4 quant = QuantizeBlockwiseInt4WithClip(
        rot.w_rotated, n_rows, k, block_size, 1.0);
    const int64_t num_blocks = k / block_size;

    std::vector<double> codes_kn(static_cast<size_t>(k * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        codes_kn[static_cast<size_t>(j * n_rows + nn)] =
            quant.codes[static_cast<size_t>(nn * k + j)];
      }
    }
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        scale_kn[static_cast<size_t>(b * n_rows + nn)] = static_cast<float>(
            quant.scale_blocks[static_cast<size_t>(nn * num_blocks + b)]);
      }
    }

    const std::string prefix = c.w_name + "_paroquant";

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
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_blocks, n_rows},
                      scale_kn.data(), scale_kn.size() * sizeof(float),
                      sizeof(float));
    std::vector<float> r_float(static_cast<size_t>(k * k));
    for (size_t i = 0; i < r_float.size(); ++i) {
      r_float[i] = static_cast<float>(rot.r[i]);
    }
    const std::string r_name = unique_name(prefix + "_r");
    SetRawInitializer(graph->add_initializer(), r_name,
                      onnx::TensorProto::FLOAT, {k, k}, r_float.data(),
                      r_float.size() * sizeof(float), sizeof(float));
    std::vector<float> inv_s_float(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      inv_s_float[static_cast<size_t>(j)] =
          static_cast<float>(1.0 / s[static_cast<size_t>(j)]);
    }
    const std::string inv_s_name = unique_name(prefix + "_inv_scale");
    SetRawInitializer(graph->add_initializer(), inv_s_name,
                      onnx::TensorProto::FLOAT, {k}, inv_s_float.data(),
                      inv_s_float.size() * sizeof(float), sizeof(float));

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
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

    const std::string x_scaled =
        add_node("Mul", {c.x_name, inv_s_name}, "x_scaled");
    const std::string x_rotated =
        add_node("MatMul", {x_scaled, r_name}, "x_rotated");
    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string core = add_node("MatMul", {x_rotated, w_dequant}, "core");

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

    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, live_index + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* nd = graph->mutable_node(live_index + i);
      nd->set_op_type(spec.op_type);
      for (const auto& s2 : spec.inputs) {
        nd->add_input(s2);
      }
      nd->add_output(spec.output);
      nd->set_name(spec.name);
      for (const auto& [attr_name, attr_value] : spec.int_attrs) {
        AddIntAttribute(nd, attr_name, attr_value);
      }
    }
    graph->mutable_node()->DeleteSubrange(live_index + count, 1);
    net_insertions += count - 1;
  }

  return out;
}
