// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See duquant_entry.h for the full rationale (including why this follows
// spinquant_entry.h's own single-model, calibration-driven shape, and the
// accepted per-node RNG divergence) and onnxsim/duquant.py for the
// technique this ports.

#include "duquant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"
#include "passes/random_orthogonal.h"

namespace {

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Transcribed from spinquant_entry.cpp's own MatchMatMulLike (itself
// transcribed from spqr_entry.cpp's own), the exact matcher
// onnxsim.duquant._match_matmul_like (== onnxsim.quip_sharp's own) checks.
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
// duquant.py reuses), unlike std::round's half-away-from-zero. Transcribed
// from spinquant_entry.cpp's own identical helper.
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
// spinquant_entry.cpp's own identical PackInt4, byte-for-byte, not reused
// directly since it is private to that translation unit).
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
// Transcribed from spinquant_entry.cpp's own identical
// QuantizeBlockwiseInt4WithClip (itself transcribed from
// omniquant.py's own _quantize_blockwise_int4_with_clip, which duquant.py
// reuses directly, always with clip_ratio=1.0).
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

// --- Calibration: per-channel activation absmax ----------------------------
//
// Transcribed from rptq_entry.cpp's own ComputeChannelAbsmax (itself
// transcribed from smoothquant_entry.cpp's own) -- 2-D-only, mirrors
// duquant.py's own `if x.ndim != 2: continue` exactly.
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
            "ApplyDuquant: calibration batch is missing "
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

// --- DuQuant rotation construction: permutation + block-local rotation -----
//
// Transcribed from duquant.py's own _build_duquant_rotation -- see
// duquant_entry.h's own top-of-file comment for the algorithm and this
// port's own accepted RNG divergence.
std::vector<double> BuildDuquantRotation(const std::vector<double>& absmax,
                                         int64_t block_size,
                                         double outlier_fraction,
                                         std::mt19937_64& rng) {
  const int64_t k = static_cast<int64_t>(absmax.size());
  const int64_t num_blocks = k / block_size;
  int64_t num_outliers = std::max<int64_t>(
      1, static_cast<int64_t>(
             std::llround(outlier_fraction * static_cast<double>(k))));
  num_outliers = std::min(num_outliers, k);

  // order = argsort(-absmax): descending by magnitude, stable so ties keep
  // their original channel-index order (a documented tie-break choice --
  // numpy's own default argsort is not guaranteed stable, so exact
  // agreement on ties isn't a goal here either; see this file's own header
  // note).
  std::vector<int64_t> order(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    order[static_cast<size_t>(i)] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return absmax[static_cast<size_t>(a)] > absmax[static_cast<size_t>(b)];
  });

  // Greedily assign each outlier channel (largest first) to whichever
  // block currently holds the least total outlier magnitude.
  std::vector<std::vector<int64_t>> block_slots(
      static_cast<size_t>(num_blocks));
  std::vector<double> block_load(static_cast<size_t>(num_blocks), 0.0);
  for (int64_t i = 0; i < num_outliers; ++i) {
    const int64_t ch = order[static_cast<size_t>(i)];
    const int64_t b = std::min_element(block_load.begin(), block_load.end()) -
                      block_load.begin();
    block_slots[static_cast<size_t>(b)].push_back(ch);
    block_load[static_cast<size_t>(b)] += absmax[static_cast<size_t>(ch)];
  }

  // Fill every block's remaining slots with the non-outlier channels, in
  // their original relative (magnitude) order.
  size_t rest_cursor = static_cast<size_t>(num_outliers);
  for (int64_t b = 0; b < num_blocks; ++b) {
    while (static_cast<int64_t>(block_slots[static_cast<size_t>(b)].size()) <
           block_size) {
      block_slots[static_cast<size_t>(b)].push_back(order[rest_cursor]);
      ++rest_cursor;
    }
  }

  std::vector<int64_t> perm;
  perm.reserve(static_cast<size_t>(k));
  for (int64_t b = 0; b < num_blocks; ++b) {
    for (int64_t ch : block_slots[static_cast<size_t>(b)]) {
      perm.push_back(ch);
    }
  }
  if (perm.size() != static_cast<size_t>(k)) {
    // Defensive only -- duquant.py's own `_build_duquant_rotation` asserts
    // `perm.shape[0] == k` here and would raise on the same pathological
    // input (an outlier_fraction/block_size combination skewed enough,
    // combined with tied abs-max values, that the uncapped greedy
    // assignment above packs more than block_size outliers into a single
    // block). Rather than reproduce an uncaught Python-side crash as C++
    // undefined behavior (out-of-range indexing below), this returns an
    // empty vector; the caller skips the candidate entirely, leaving that
    // layer unquantized -- never exercised by this port's own default
    // parameters or tests, which stay well inside the regime where every
    // block receives at most block_size outliers.
    return {};
  }

  // perm_matrix[i, j] = 1 iff i == perm[j] -- so (x @ perm_matrix)[j] ==
  // x[perm[j]], mirroring `np.eye(k)[:, perm]` exactly.
  std::vector<double> perm_matrix(static_cast<size_t>(k * k), 0.0);
  for (int64_t j = 0; j < k; ++j) {
    perm_matrix[static_cast<size_t>(perm[static_cast<size_t>(j)] * k + j)] =
        1.0;
  }

  std::vector<double> block_rotation(static_cast<size_t>(k * k), 0.0);
  for (int64_t b = 0; b < num_blocks; ++b) {
    const int64_t start = b * block_size;
    const std::vector<float> r =
        onnx::optimization::onnxsim_passes::RandomOrthogonalMatrix(
            block_size, rng);  // [block_size, block_size], row-major.
    for (int64_t i = 0; i < block_size; ++i) {
      for (int64_t j = 0; j < block_size; ++j) {
        block_rotation[static_cast<size_t>((start + i) * k + (start + j))] =
            static_cast<double>(r[static_cast<size_t>(i * block_size + j)]);
      }
    }
  }

  // U = perm_matrix @ block_rotation, [k, k].
  std::vector<double> u(static_cast<size_t>(k * k), 0.0);
  for (int64_t i = 0; i < k; ++i) {
    for (int64_t kk = 0; kk < k; ++kk) {
      const double p = perm_matrix[static_cast<size_t>(i * k + kk)];
      if (p == 0.0) {
        continue;
      }
      for (int64_t j = 0; j < k; ++j) {
        u[static_cast<size_t>(i * k + j)] +=
            p * block_rotation[static_cast<size_t>(kk * k + j)];
      }
    }
  }
  return u;
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

onnx::ModelProto ApplyDuquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t seed, int64_t block_size, double outlier_fraction, double epsilon) {
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
      continue;  // Never observed as a plain 2-D tensor; skip.
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

    // ACCEPTED, PERMANENT DIVERGENCE (see duquant_entry.h's own note): a
    // fresh RNG per matched node, not one sequentially-advancing generator
    // threaded across every layer/block the way duquant.py's own
    // apply_duquant does.
    std::mt19937_64 rng(
        static_cast<uint64_t>(seed) ^
        (0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(c.node_index) + 1)));
    const std::vector<double> u =
        BuildDuquantRotation(absmax, block_size, outlier_fraction, rng);
    if (u.empty()) {
      continue;  // See BuildDuquantRotation's own defensive-only note.
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // w_tilde_nk = w_nk @ u -- [N, K], exact before quantization.
    std::vector<double> w_tilde_nk(static_cast<size_t>(n_rows * k), 0.0);
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t col = 0; col < k; ++col) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += at_nk(nn, kk) * u[static_cast<size_t>(kk * k + col)];
        }
        w_tilde_nk[static_cast<size_t>(nn * k + col)] = acc;
      }
    }

    const BlockwiseInt4 quant =
        QuantizeBlockwiseInt4WithClip(w_tilde_nk, n_rows, k, block_size, 1.0);
    const int64_t num_blocks = k / block_size;

    // Transpose codes/scale to [K, N] layout, ready for a plain MatMul.
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

    const std::string prefix = c.w_name + "_duquant";

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
    std::vector<float> u_float(static_cast<size_t>(k * k));
    for (size_t i = 0; i < u_float.size(); ++i) {
      u_float[i] = static_cast<float>(u[i]);
    }
    const std::string u_name = unique_name(prefix + "_u");
    SetRawInitializer(graph->add_initializer(), u_name,
                      onnx::TensorProto::FLOAT, {k, k}, u_float.data(),
                      u_float.size() * sizeof(float), sizeof(float));
    const float eps_f = static_cast<float>(epsilon);
    const std::string eps_name = unique_name(prefix + "_eps");
    SetRawInitializer(graph->add_initializer(), eps_name,
                      onnx::TensorProto::FLOAT, {}, &eps_f, sizeof(float),
                      sizeof(float));
    const float seven_f = 7.0f;
    const std::string seven_name = unique_name(prefix + "_seven");
    SetRawInitializer(graph->add_initializer(), seven_name,
                      onnx::TensorProto::FLOAT, {}, &seven_f, sizeof(float),
                      sizeof(float));
    const float clip_min_f = -7.0f;
    const std::string clip_min_name = unique_name(prefix + "_clip_min");
    SetRawInitializer(graph->add_initializer(), clip_min_name,
                      onnx::TensorProto::FLOAT, {}, &clip_min_f, sizeof(float),
                      sizeof(float));
    const float clip_max_f = 7.0f;
    const std::string clip_max_name = unique_name(prefix + "_clip_max");
    SetRawInitializer(graph->add_initializer(), clip_max_name,
                      onnx::TensorProto::FLOAT, {}, &clip_max_f, sizeof(float),
                      sizeof(float));
    const int64_t axes_val = -1;
    const std::string axes_name = unique_name(prefix + "_reduce_axes");
    SetRawInitializer(graph->add_initializer(), axes_name,
                      onnx::TensorProto::INT64, {1}, &axes_val, sizeof(int64_t),
                      sizeof(int64_t));

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

    const std::string x_rotated =
        add_node("MatMul", {c.x_name, u_name}, "x_rotated");

    // Data-free, per-token round-to-nearest INT4 activation quantization --
    // mirrors duquant.py's own inline node sequence exactly: scale =
    // max(reduce_max(abs(x_rotated), axis=-1), eps) / 7.
    const std::string abs_name = add_node("Abs", {x_rotated}, "x_abs");
    const std::string max_name = add_node("ReduceMax", {abs_name, axes_name},
                                          "x_max", {{"keepdims", 1}});
    const std::string safe_max_name =
        add_node("Clip", {max_name, eps_name}, "x_safe_max");
    const std::string x_scale =
        add_node("Div", {safe_max_name, seven_name}, "x_scale");
    const std::string x_scaled =
        add_node("Div", {x_rotated, x_scale}, "x_scaled");
    const std::string x_rounded = add_node("Round", {x_scaled}, "x_rounded");
    const std::string x_clipped = add_node(
        "Clip", {x_rounded, clip_min_name, clip_max_name}, "x_clipped");
    const std::string x_dequant =
        add_node("Mul", {x_clipped, x_scale}, "x_dequant");

    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string core = add_node("MatMul", {x_dequant, w_dequant}, "core");

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
      for (const auto& s : spec.inputs) {
        nd->add_input(s);
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
