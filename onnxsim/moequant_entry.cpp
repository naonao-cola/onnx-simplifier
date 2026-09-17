// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See moequant_entry.h for the full rationale (including the FLOAT32-only
// matcher narrowing relative to structured_pruning_entry.cpp's own
// MatchMoeProducer, and the two accepted numerical divergences) and
// onnxsim/moequant.py for the technique this ports.

#include "moequant_entry.h"

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

constexpr char kComMicrosoftDomain[] = "com.microsoft";

// --- MoE node matching, protobuf level --------------------------------------
//
// Transcribed from structured_pruning_entry.cpp's own MatchMoeProducer/
// MoEChain (itself from pruning.py's own _match_moe_producer/_MoEChain,
// which moequant.py's own _find_moe_chains import reuses directly),
// narrowed to FLOAT32-only fc1/fc2 weights -- see moequant_entry.h's own
// scope note for why that narrowing leaves the same set of nodes actually
// rewritten as the wider-matching Python reference.

struct MoEChain {
  int node_index;
  std::string x_name;       // node.input(0) -- hidden_states.
  std::string router_name;  // node.input(1) -- router_probs.
  std::string fc1_w;
  std::optional<std::string> fc1_b;
  std::string fc2_w;
  std::optional<std::string> fc2_b;
  int64_t num_experts;
  int64_t inter_size;
  int64_t hidden_size;
};

using InitMap = std::unordered_map<std::string, int>;
using ConsumerMap = std::unordered_map<std::string, int>;

ConsumerMap ConsumerCounts(const onnx::GraphProto& graph) {
  ConsumerMap counts;
  for (const auto& n : graph.node()) {
    for (const auto& inp : n.input()) {
      if (!inp.empty()) {
        ++counts[inp];
      }
    }
  }
  return counts;
}

const std::unordered_set<std::string>& MoeActivations() {
  static const std::unordered_set<std::string> kOps = {"relu", "identity",
                                                       "silu", "gelu"};
  return kOps;
}

std::string GetStringAttr(const onnx::NodeProto& node, const std::string& name,
                          const std::string& fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::STRING) {
      return attr.s();
    }
  }
  return fallback;
}

int64_t GetIntAttr(const onnx::NodeProto& node, const std::string& name,
                   int64_t fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::INT) {
      return attr.i();
    }
  }
  return fallback;
}

std::optional<std::string> MatchOptionalBias(
    const onnx::NodeProto& node, int index, int64_t dim0, int64_t dim1,
    const onnx::GraphProto& graph, const InitMap& init_index,
    const ConsumerMap& consumers, bool* ok) {
  *ok = true;
  if (node.input_size() <= index || node.input(index).empty()) {
    return std::nullopt;
  }
  const std::string& name = node.input(index);
  auto it = init_index.find(name);
  if (it == init_index.end()) {
    *ok = false;
    return std::nullopt;
  }
  const onnx::TensorProto& t = graph.initializer(it->second);
  auto cit = consumers.find(name);
  const int consumer_count = cit == consumers.end() ? 0 : cit->second;
  if (t.data_type() != onnx::TensorProto::FLOAT || t.dims_size() != 2 ||
      t.dims(0) != dim0 || t.dims(1) != dim1 || consumer_count != 1) {
    *ok = false;
    return std::nullopt;
  }
  return name;
}

std::optional<MoEChain> MatchMoeProducer(const onnx::GraphProto& graph,
                                         int node_index,
                                         const InitMap& init_index,
                                         const ConsumerMap& consumers) {
  const onnx::NodeProto& node = graph.node(node_index);
  if (node.domain() != kComMicrosoftDomain || node.op_type() != "MoE") {
    return std::nullopt;
  }
  const std::string activation = GetStringAttr(node, "activation_type", "relu");
  const int64_t swiglu_fusion = GetIntAttr(node, "swiglu_fusion", 0);
  if (!MoeActivations().count(activation) || swiglu_fusion != 0) {
    return std::nullopt;
  }
  if (node.input_size() > 6 && !node.input(6).empty()) {
    return std::nullopt;  // fc3_experts_weights present -- out of scope.
  }
  if (node.input_size() < 5 || node.input(2).empty() || node.input(4).empty()) {
    return std::nullopt;
  }
  const std::string& fc1_w_name = node.input(2);
  const std::string& fc2_w_name = node.input(4);
  auto fc1_it = init_index.find(fc1_w_name);
  auto fc2_it = init_index.find(fc2_w_name);
  if (fc1_it == init_index.end() || fc2_it == init_index.end()) {
    return std::nullopt;
  }
  const onnx::TensorProto& fc1_w = graph.initializer(fc1_it->second);
  const onnx::TensorProto& fc2_w = graph.initializer(fc2_it->second);
  auto c1 = consumers.find(fc1_w_name);
  auto c2 = consumers.find(fc2_w_name);
  const int fc1_consumers = c1 == consumers.end() ? 0 : c1->second;
  const int fc2_consumers = c2 == consumers.end() ? 0 : c2->second;
  // FLOAT32-only -- see moequant_entry.h's own scope note (narrower than
  // structured_pruning_entry.cpp's own MatchMoeProducer, which also
  // accepts FLOAT16/BFLOAT16).
  if (fc1_w.data_type() != onnx::TensorProto::FLOAT ||
      fc2_w.data_type() != onnx::TensorProto::FLOAT || fc1_w.dims_size() != 3 ||
      fc2_w.dims_size() != 3 || fc1_consumers != 1 || fc2_consumers != 1) {
    return std::nullopt;
  }
  const int64_t num_experts = fc1_w.dims(0);
  const int64_t inter_size = fc1_w.dims(1);
  const int64_t hidden_size = fc1_w.dims(2);
  if (fc2_w.dims(0) != num_experts || fc2_w.dims(1) != hidden_size ||
      fc2_w.dims(2) != inter_size) {
    return std::nullopt;
  }

  bool ok = true;
  const std::optional<std::string> fc1_b = MatchOptionalBias(
      node, 3, num_experts, inter_size, graph, init_index, consumers, &ok);
  if (!ok) {
    return std::nullopt;
  }
  const std::optional<std::string> fc2_b = MatchOptionalBias(
      node, 5, num_experts, hidden_size, graph, init_index, consumers, &ok);
  if (!ok) {
    return std::nullopt;
  }

  return MoEChain{node_index, node.input(0), node.input(1), fc1_w_name,
                  fc1_b,      fc2_w_name,    fc2_b,         num_experts,
                  inter_size, hidden_size};
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

void SetFloatTensorInPlace(onnx::TensorProto* t,
                           const std::vector<double>& data) {
  std::vector<float> f(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    f[i] = static_cast<float>(data[i]);
  }
  std::string raw(f.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), f.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->clear_float_data();
  t->set_raw_data(std::move(raw));
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

// --- Small dense double-precision linear algebra ----------------------------
//
// Transcribed from gptq_entry.cpp's own CholeskyLower/InverseSPD/
// InverseHessianCholesky/GptqQuantizeColumns -- this pass's own
// docstring-documented "reuses GPTQ's own column-update machinery as-is"
// claim, realized as this codebase's established one-copy-per-TU
// transcription rather than a shared header.
using Matrix = std::vector<std::vector<double>>;

Matrix CholeskyLower(const Matrix& a) {
  const size_t n = a.size();
  Matrix l(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double s = a[i][j];
      for (size_t k = 0; k < j; ++k) {
        s -= l[i][k] * l[j][k];
      }
      if (i == j) {
        l[i][j] = std::sqrt(std::max(s, 1e-24));
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return l;
}

Matrix InverseSPD(const Matrix& a) {
  const size_t n = a.size();
  const Matrix l = CholeskyLower(a);
  Matrix inv(n, std::vector<double>(n, 0.0));
  std::vector<double> y(n), x(n);
  for (size_t c = 0; c < n; ++c) {
    for (size_t i = 0; i < n; ++i) {
      double s = (i == c) ? 1.0 : 0.0;
      for (size_t k = 0; k < i; ++k) {
        s -= l[i][k] * y[k];
      }
      y[i] = s / l[i][i];
    }
    for (size_t i = n; i-- > 0;) {
      double s = y[i];
      for (size_t k = i + 1; k < n; ++k) {
        s -= l[k][i] * x[k];
      }
      x[i] = s / l[i][i];
    }
    for (size_t i = 0; i < n; ++i) {
      inv[i][c] = x[i];
    }
  }
  return inv;
}

Matrix InverseHessianCholesky(const Matrix& h, double percdamp) {
  const size_t k = h.size();
  Matrix damped = h;
  double diag_sum = 0.0;
  for (size_t i = 0; i < k; ++i) {
    if (damped[i][i] == 0.0) {
      damped[i][i] = 1.0;
    }
    diag_sum += damped[i][i];
  }
  const double damp =
      std::max(percdamp * diag_sum / static_cast<double>(k), 1e-8);
  for (size_t i = 0; i < k; ++i) {
    damped[i][i] += damp;
  }
  const Matrix h_inv = InverseSPD(damped);
  const Matrix l = CholeskyLower(h_inv);
  Matrix u(k, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = i; j < k; ++j) {
      u[i][j] = l[j][i];
    }
  }
  return u;
}

// Mirrors gptq.py's own _gptq_quantize_columns exactly (see gptq_entry.cpp's
// own identically-named function for the full derivation).
Matrix GptqQuantizeColumns(const Matrix& w_nk, const Matrix& scale_blocks,
                           int64_t block_size, const Matrix& h, double percdamp,
                           int64_t proc_block_size) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const Matrix hinv = InverseHessianCholesky(h, percdamp);

  Matrix codes(n, std::vector<double>(k, 0.0));
  Matrix w_work = w_nk;

  for (int64_t block_start = 0; block_start < static_cast<int64_t>(k);
       block_start += proc_block_size) {
    const int64_t block_end =
        std::min(block_start + proc_block_size, static_cast<int64_t>(k));
    const size_t bs = static_cast<size_t>(block_end - block_start);
    Matrix w1(n, std::vector<double>(bs));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < bs; ++j) {
        w1[i][j] = w_work[i][static_cast<size_t>(block_start) + j];
      }
    }
    Matrix err1(n, std::vector<double>(bs, 0.0));

    for (size_t i = 0; i < bs; ++i) {
      const int64_t k_abs = block_start + static_cast<int64_t>(i);
      const size_t group = static_cast<size_t>(k_abs / block_size);
      for (size_t r = 0; r < n; ++r) {
        const double s = scale_blocks[r][group];
        const double w_col = w1[r][i];
        double code = RoundHalfToEven(w_col / s);
        code = std::min(7.0, std::max(-7.0, code));
        codes[r][static_cast<size_t>(k_abs)] = code;
        const double d = hinv[static_cast<size_t>(block_start) + i]
                             [static_cast<size_t>(block_start) + i];
        const double err = (w_col - code * s) / d;
        err1[r][i] = err;
        for (size_t j = i + 1; j < bs; ++j) {
          w1[r][j] -= err * hinv[static_cast<size_t>(block_start) + i]
                                [static_cast<size_t>(block_start) + j];
        }
      }
    }

    if (block_end < static_cast<int64_t>(k)) {
      for (size_t r = 0; r < n; ++r) {
        for (int64_t j = block_end; j < static_cast<int64_t>(k); ++j) {
          double acc = 0.0;
          for (size_t i = 0; i < bs; ++i) {
            acc += err1[r][i] * hinv[static_cast<size_t>(block_start) + i]
                                    [static_cast<size_t>(j)];
          }
          w_work[r][static_cast<size_t>(j)] -= acc;
        }
      }
    }
  }
  return codes;
}

// Mirrors moequant.py's own _quantize_expert_weight exactly: this module's
// own initial round-to-nearest INT4 block scale (own estimate -- there is
// no existing MoE-targeting int4 rewrite to read a scale back from, unlike
// GptqQuantizeColumns' usual caller), then GPTQ's own column update against
// `h`, returning the dequantized (code * scale) reconstruction.
Matrix QuantizeExpertWeight(const Matrix& w_nk, const Matrix& h,
                            int64_t quant_block_size, double percdamp,
                            int64_t proc_block_size) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const int64_t bs =
      (quant_block_size > 0 && k % static_cast<size_t>(quant_block_size) == 0)
          ? quant_block_size
          : static_cast<int64_t>(k);
  const size_t num_blocks = k / static_cast<size_t>(bs);

  Matrix scale_blocks(n, std::vector<double>(num_blocks));
  for (size_t r = 0; r < n; ++r) {
    for (size_t b = 0; b < num_blocks; ++b) {
      double amax = 0.0;
      for (int64_t j = 0; j < bs; ++j) {
        const double v = std::fabs(
            w_nk[r][b * static_cast<size_t>(bs) + static_cast<size_t>(j)]);
        if (v > amax) {
          amax = v;
        }
      }
      scale_blocks[r][b] = amax == 0.0 ? 1.0 : amax / 7.0;
    }
  }

  const Matrix codes =
      GptqQuantizeColumns(w_nk, scale_blocks, bs, h, percdamp, proc_block_size);

  Matrix recon(n, std::vector<double>(k));
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const size_t blk = c / static_cast<size_t>(bs);
      recon[r][c] = codes[r][c] * scale_blocks[r][blk];
    }
  }
  return recon;
}

// --- Elementwise activation functions ---------------------------------------
double ActRelu(double x) { return std::max(x, 0.0); }
double ActIdentity(double x) { return x; }
double ActSilu(double x) { return x / (1.0 + std::exp(-x)); }
double ActGelu(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

using ActivationFn = double (*)(double);
ActivationFn PickActivation(const std::string& name) {
  if (name == "identity") {
    return ActIdentity;
  }
  if (name == "silu") {
    return ActSilu;
  }
  if (name == "gelu") {
    return ActGelu;
  }
  return ActRelu;
}

// --- Deterministic RNG for EBSS's own weighted-without-replacement pick ----
//
// See moequant_entry.h's own accepted numerical scope note: this does NOT
// reproduce numpy's own `Generator.choice(..., replace=False,
// p=weights)` bit stream, only its distributional intent (weighted
// sampling without replacement). splitmix64, seeded per (chain, expert) so
// results are deterministic and reproducible for a given `seed`+model,
// mirroring this codebase's established per-layer RNG derivation
// convention (e.g. spinquant_entry.cpp/paroquant_entry.cpp's own).
uint64_t SplitMix64Next(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

onnx::ModelProto ApplyMoequant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t quant_block_size, double percdamp, int64_t proc_block_size,
    bool ebss, uint64_t seed) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  InitMap init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }
  const ConsumerMap consumers = ConsumerCounts(*graph);

  std::vector<MoEChain> chains;
  for (int i = 0; i < graph->node_size(); ++i) {
    auto chain = MatchMoeProducer(*graph, i, init_index, consumers);
    if (chain && !chain->x_name.empty() && !chain->router_name.empty()) {
      chains.push_back(std::move(*chain));
    }
  }
  if (chains.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : chains) {
    probe_names.insert(c.x_name);
    probe_names.insert(c.router_name);
  }

  // --- Calibration: concatenated activation rows, same shape as every
  // other calibration-driven pass in this codebase.
  struct ActivationRows {
    std::vector<double> data;
    int64_t k = -1;
    bool ok = false;
  };
  std::unordered_map<std::string, ActivationRows> activations;
  {
    onnx::ModelProto probe_model = out;
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
              "ApplyMoequant: calibration batch is missing "
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
        ActivationRows& rows = activations[name];
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

  std::unordered_set<std::string> touched;
  int64_t chain_ordinal = 0;
  for (const auto& chain : chains) {
    const int64_t this_chain_ordinal = chain_ordinal++;
    if (touched.count(chain.fc1_w) || touched.count(chain.fc2_w)) {
      continue;  // A tied/shared initializer another MoE node already
                 // quantized.
    }
    touched.insert(chain.fc1_w);
    touched.insert(chain.fc2_w);

    auto xit = activations.find(chain.x_name);
    auto rit = activations.find(chain.router_name);
    if (xit == activations.end() || !xit->second.ok ||
        rit == activations.end() || !rit->second.ok) {
      continue;  // No usable calibration activation observed.
    }
    const ActivationRows& x_rows = xit->second;
    const ActivationRows& r_rows = rit->second;
    if (x_rows.k != chain.hidden_size || r_rows.k != chain.num_experts) {
      continue;  // Calibration activation doesn't match this chain's shapes.
    }
    const int64_t x_total_rows =
        static_cast<int64_t>(x_rows.data.size()) / chain.hidden_size;
    const int64_t r_total_rows =
        static_cast<int64_t>(r_rows.data.size()) / chain.num_experts;
    const int64_t n_tokens = std::min(x_total_rows, r_total_rows);
    if (n_tokens <= 0) {
      continue;
    }

    const int64_t hidden = chain.hidden_size;
    const int64_t inter = chain.inter_size;
    const int64_t num_experts = chain.num_experts;

    Matrix x_all(static_cast<size_t>(n_tokens),
                 std::vector<double>(static_cast<size_t>(hidden)));
    for (int64_t t = 0; t < n_tokens; ++t) {
      for (int64_t c = 0; c < hidden; ++c) {
        x_all[static_cast<size_t>(t)][static_cast<size_t>(c)] =
            x_rows.data[static_cast<size_t>(t * hidden + c)];
      }
    }
    Matrix r_all(static_cast<size_t>(n_tokens),
                 std::vector<double>(static_cast<size_t>(num_experts)));
    for (int64_t t = 0; t < n_tokens; ++t) {
      for (int64_t c = 0; c < num_experts; ++c) {
        r_all[static_cast<size_t>(t)][static_cast<size_t>(c)] =
            r_rows.data[static_cast<size_t>(t * num_experts + c)];
      }
    }

    const int64_t k_sel = std::max<int64_t>(
        1, std::min(GetIntAttr(graph->node(chain.node_index), "k", 1),
                    num_experts));

    // Softmax over the last axis, and (for each token) the indices of its
    // top-k=k_sel largest router logits, descending, ties broken by
    // ascending original index -- mirrors `np.argsort(-r_all, axis=1)`'s
    // own effective ordering for this port's purposes (see
    // moequant_entry.h's own accepted numerical scope note on tie-break
    // determinism).
    Matrix affinity_all(static_cast<size_t>(n_tokens),
                        std::vector<double>(static_cast<size_t>(num_experts)));
    std::vector<std::vector<int64_t>> top_k(static_cast<size_t>(n_tokens));
    for (int64_t t = 0; t < n_tokens; ++t) {
      double max_v = r_all[static_cast<size_t>(t)][0];
      for (int64_t e = 1; e < num_experts; ++e) {
        max_v = std::max(max_v,
                         r_all[static_cast<size_t>(t)][static_cast<size_t>(e)]);
      }
      double sum_exp = 0.0;
      std::vector<double> exps(static_cast<size_t>(num_experts));
      for (int64_t e = 0; e < num_experts; ++e) {
        const double ev = std::exp(
            r_all[static_cast<size_t>(t)][static_cast<size_t>(e)] - max_v);
        exps[static_cast<size_t>(e)] = ev;
        sum_exp += ev;
      }
      for (int64_t e = 0; e < num_experts; ++e) {
        affinity_all[static_cast<size_t>(t)][static_cast<size_t>(e)] =
            exps[static_cast<size_t>(e)] / sum_exp;
      }

      std::vector<int64_t> order(static_cast<size_t>(num_experts));
      for (int64_t e = 0; e < num_experts; ++e) {
        order[static_cast<size_t>(e)] = e;
      }
      std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return r_all[static_cast<size_t>(t)][static_cast<size_t>(a)] >
               r_all[static_cast<size_t>(t)][static_cast<size_t>(b)];
      });
      order.resize(static_cast<size_t>(k_sel));
      top_k[static_cast<size_t>(t)] = std::move(order);
    }

    // Ceiling division -- mirrors moequant.py's own `-(-(n_tokens * k) //
    // chain.num_experts)` (Python's `//` floors toward negative infinity,
    // so that trick works there; C++'s `/` truncates toward zero instead,
    // so the usual positive-operands ceiling-division formula is used
    // here rather than transcribing that same trick literally).
    const int64_t target_count =
        (n_tokens * k_sel + num_experts - 1) / num_experts;

    const onnx::TensorProto& fc1_init =
        graph->initializer(init_index[chain.fc1_w]);
    const onnx::TensorProto& fc2_init =
        graph->initializer(init_index[chain.fc2_w]);
    const std::vector<float> fc1_flat = ReadFloatTensor(fc1_init);
    const std::vector<float> fc2_flat = ReadFloatTensor(fc2_init);
    std::vector<double> fc1_b_flat;
    if (chain.fc1_b) {
      const std::vector<float> f =
          ReadFloatTensor(graph->initializer(init_index[*chain.fc1_b]));
      fc1_b_flat.assign(f.begin(), f.end());
    }
    const ActivationFn act_fn = PickActivation(GetStringAttr(
        graph->node(chain.node_index), "activation_type", "relu"));

    std::vector<double> new_fc1(fc1_flat.begin(), fc1_flat.end());
    std::vector<double> new_fc2(fc2_flat.begin(), fc2_flat.end());

    for (int64_t e = 0; e < num_experts; ++e) {
      std::vector<int64_t> routed_idx;
      for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t sel : top_k[static_cast<size_t>(t)]) {
          if (sel == e) {
            routed_idx.push_back(t);
            break;
          }
        }
      }
      if (routed_idx.empty()) {
        continue;  // Never routed to in calibration -- leave float.
      }
      std::vector<double> affinity(routed_idx.size());
      for (size_t i = 0; i < routed_idx.size(); ++i) {
        affinity[i] = affinity_all[static_cast<size_t>(routed_idx[i])]
                                  [static_cast<size_t>(e)];
      }

      if (ebss && target_count > 0 &&
          static_cast<int64_t>(routed_idx.size()) > target_count) {
        // Efraimidis-Spirakis weighted sampling without replacement: key_i
        // = U_i^(1/w_i), keep the `target_count` largest keys. See
        // moequant_entry.h's own accepted numerical scope note.
        uint64_t state =
            seed ^ (static_cast<uint64_t>(this_chain_ordinal + 1) *
                        0x9E3779B97F4A7C15ULL +
                    static_cast<uint64_t>(e + 1) * 0xBF58476D1CE4E5B9ULL);
        std::vector<std::pair<double, size_t>> keyed(routed_idx.size());
        for (size_t i = 0; i < routed_idx.size(); ++i) {
          const uint64_t r = SplitMix64Next(state);
          double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
          u = std::min(std::max(u, 1e-12), 1.0 - 1e-12);
          const double w = std::max(affinity[i], 1e-300);
          keyed[i] = {std::pow(u, 1.0 / w), i};
        }
        std::partial_sort(
            keyed.begin(), keyed.begin() + static_cast<long>(target_count),
            keyed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int64_t> new_routed(static_cast<size_t>(target_count));
        std::vector<double> new_affinity(static_cast<size_t>(target_count));
        for (int64_t i = 0; i < target_count; ++i) {
          const size_t src = keyed[static_cast<size_t>(i)].second;
          new_routed[static_cast<size_t>(i)] = routed_idx[src];
          new_affinity[static_cast<size_t>(i)] = affinity[src];
        }
        routed_idx = std::move(new_routed);
        affinity = std::move(new_affinity);
      }

      const size_t m = routed_idx.size();
      std::vector<double> sqrt_affinity(m);
      for (size_t i = 0; i < m; ++i) {
        sqrt_affinity[i] = std::sqrt(std::max(affinity[i], 0.0));
      }

      // h1 = xe^T @ xe, xe = x_all[routed_idx] * sqrt_affinity.
      Matrix h1(static_cast<size_t>(hidden),
                std::vector<double>(static_cast<size_t>(hidden), 0.0));
      for (size_t i = 0; i < m; ++i) {
        const int64_t t = routed_idx[i];
        const double sa = sqrt_affinity[i];
        for (int64_t a = 0; a < hidden; ++a) {
          const double xa =
              x_all[static_cast<size_t>(t)][static_cast<size_t>(a)] * sa;
          if (xa == 0.0) {
            continue;
          }
          for (int64_t b = 0; b < hidden; ++b) {
            h1[static_cast<size_t>(a)][static_cast<size_t>(b)] +=
                xa * x_all[static_cast<size_t>(t)][static_cast<size_t>(b)] * sa;
          }
        }
      }

      // pre_activation[i, n] = sum_k x_all[t][k] * fc1_w[e][n][k] (+ bias).
      Matrix inter_act(m, std::vector<double>(static_cast<size_t>(inter)));
      for (size_t i = 0; i < m; ++i) {
        const int64_t t = routed_idx[i];
        for (int64_t n = 0; n < inter; ++n) {
          double acc = 0.0;
          const size_t base = static_cast<size_t>((e * inter + n) * hidden);
          for (int64_t kk = 0; kk < hidden; ++kk) {
            acc += x_all[static_cast<size_t>(t)][static_cast<size_t>(kk)] *
                   fc1_flat[base + static_cast<size_t>(kk)];
          }
          if (!fc1_b_flat.empty()) {
            acc += fc1_b_flat[static_cast<size_t>(e * inter + n)];
          }
          inter_act[i][static_cast<size_t>(n)] = act_fn(acc) * sqrt_affinity[i];
        }
      }
      Matrix h2(static_cast<size_t>(inter),
                std::vector<double>(static_cast<size_t>(inter), 0.0));
      for (size_t i = 0; i < m; ++i) {
        for (int64_t a = 0; a < inter; ++a) {
          const double va = inter_act[i][static_cast<size_t>(a)];
          if (va == 0.0) {
            continue;
          }
          for (int64_t b = 0; b < inter; ++b) {
            h2[static_cast<size_t>(a)][static_cast<size_t>(b)] +=
                va * inter_act[i][static_cast<size_t>(b)];
          }
        }
      }

      Matrix fc1_e(static_cast<size_t>(inter),
                   std::vector<double>(static_cast<size_t>(hidden)));
      for (int64_t n = 0; n < inter; ++n) {
        const size_t base = static_cast<size_t>((e * inter + n) * hidden);
        for (int64_t kk = 0; kk < hidden; ++kk) {
          fc1_e[static_cast<size_t>(n)][static_cast<size_t>(kk)] =
              static_cast<double>(fc1_flat[base + static_cast<size_t>(kk)]);
        }
      }
      Matrix fc2_e(static_cast<size_t>(hidden),
                   std::vector<double>(static_cast<size_t>(inter)));
      for (int64_t n = 0; n < hidden; ++n) {
        const size_t base = static_cast<size_t>((e * hidden + n) * inter);
        for (int64_t kk = 0; kk < inter; ++kk) {
          fc2_e[static_cast<size_t>(n)][static_cast<size_t>(kk)] =
              static_cast<double>(fc2_flat[base + static_cast<size_t>(kk)]);
        }
      }

      const Matrix new_fc1_e = QuantizeExpertWeight(fc1_e, h1, quant_block_size,
                                                    percdamp, proc_block_size);
      const Matrix new_fc2_e = QuantizeExpertWeight(fc2_e, h2, quant_block_size,
                                                    percdamp, proc_block_size);

      for (int64_t n = 0; n < inter; ++n) {
        const size_t base = static_cast<size_t>((e * inter + n) * hidden);
        for (int64_t kk = 0; kk < hidden; ++kk) {
          new_fc1[base + static_cast<size_t>(kk)] =
              new_fc1_e[static_cast<size_t>(n)][static_cast<size_t>(kk)];
        }
      }
      for (int64_t n = 0; n < hidden; ++n) {
        const size_t base = static_cast<size_t>((e * hidden + n) * inter);
        for (int64_t kk = 0; kk < inter; ++kk) {
          new_fc2[base + static_cast<size_t>(kk)] =
              new_fc2_e[static_cast<size_t>(n)][static_cast<size_t>(kk)];
        }
      }
    }

    SetFloatTensorInPlace(graph->mutable_initializer(init_index[chain.fc1_w]),
                          new_fc1);
    SetFloatTensorInPlace(graph->mutable_initializer(init_index[chain.fc2_w]),
                          new_fc2);
  }

  return out;
}
