// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See daq_entry.h for the full rationale and onnxsim/daq.py for the
// technique this ports.

#include "daq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "passes/quantize_fp8.h"

namespace {

// --- MatMul/vanilla-Gemm matching, at the raw protobuf level ---------------
//
// Mirrors onnxsim.mx_quantization._match_matmul_like exactly (itself
// documented there as mirroring passes/quantize_matmul_common.h's own
// MatchMatMulLike, which operates on onnx-optimizer's own Node IR rather
// than raw NodeProto -- this TU works directly on onnx::NodeProto, the
// same protobuf-level style every other *_entry.cpp in this codebase
// uses, so the logic is duplicated here rather than shared).

double GetFloatAttr(const onnx::NodeProto& node, const std::string& name,
                    double fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::FLOAT) {
      return static_cast<double>(attr.f());
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

// Returns the node's weight input name if it is a MatMul or a
// transA=0/alpha=1(/beta=1 when it has a bias) Gemm, `""` otherwise.
std::string MatchMatMulLikeWeightName(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return "";
    }
    return node.input(1);
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return "";
    }
    if (GetIntAttr(node, "transA", 0) != 0) {
      return "";
    }
    if (GetFloatAttr(node, "alpha", 1.0) != 1.0) {
      return "";
    }
    if (num_inputs == 3 && GetFloatAttr(node, "beta", 1.0) != 1.0) {
      return "";
    }
    return node.input(1);
  }
  return "";
}

// --- Tensor <-> flat float buffer -------------------------------------------
//
// FLOAT32 only (matching apply_daq's own scope), reusing
// dlpack_bridge.h's kRawDataIsHostOrder/SwapElementBytes for the raw_data
// little-endian convention -- the same helpers every other *_entry.cpp
// in this codebase uses.

std::vector<double> ReadFloatTensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<float> raw_floats(static_cast<size_t>(numel));
  if (t.has_raw_data()) {
    std::memcpy(raw_floats.data(), t.raw_data().data(),
                raw_floats.size() * sizeof(float));
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(
          reinterpret_cast<uint8_t*>(raw_floats.data()),
          raw_floats.size() * sizeof(float), sizeof(float));
    }
  } else {
    for (int64_t i = 0; i < numel; ++i) {
      raw_floats[static_cast<size_t>(i)] = t.float_data(static_cast<int>(i));
    }
  }
  return std::vector<double>(raw_floats.begin(), raw_floats.end());
}

void SetFloatInitializer(onnx::TensorProto* t, const std::string& name,
                         const std::vector<int64_t>& dims,
                         const std::vector<double>& data) {
  t->Clear();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) {
    t->add_dims(d);
  }
  std::vector<float> as_float(data.begin(), data.end());
  std::string raw(as_float.size() * sizeof(float), '\0');
  std::memcpy(raw.data(), as_float.data(), raw.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                      raw.size(), sizeof(float));
  }
  t->set_raw_data(std::move(raw));
}

// --- FP8 E4M3FN round trip --------------------------------------------------
//
// Encode reuses passes/quantize_fp8.h's own verified FloatToFloat8Bits
// (round-to-nearest, ties-to-even, saturating at +-448 -- the real ONNX
// FLOAT8E4M3FN cast semantics, the same ml_dtypes.float8_e4m3fn cast
// daq.py's own _fp8_round_trip uses). Decoding an E4M3FN byte back to a
// double needs no rounding decision at all (unlike encoding), so it is
// implemented directly here from the format's own bit layout (1 sign, 4
// exponent bits, bias 7, 3 mantissa bits; the single reserved NaN pattern
// is exponent=1111 AND mantissa=111 -- every other exponent=1111 mantissa
// value is an ordinary finite number, since E4M3FN has no infinity
// encoding, up to the max finite magnitude 448 at exponent=1111,
// mantissa=110).

constexpr double kFp8Max = 448.0;

double Float8E4M3FNBitsToDouble(uint8_t bits) {
  const uint32_t sign = (bits >> 7) & 0x1u;
  const uint32_t exp = (bits >> 3) & 0xFu;
  const uint32_t mant = bits & 0x7u;
  double value;
  if (exp == 0xFu && mant == 0x7u) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (exp == 0) {
    // Subnormal (mant == 0 is +-0).
    value = (static_cast<double>(mant) / 8.0) * std::ldexp(1.0, 1 - 7);
  } else {
    value = (1.0 + static_cast<double>(mant) / 8.0) *
            std::ldexp(1.0, static_cast<int>(exp) - 7);
  }
  return sign != 0 ? -value : value;
}

double Fp8RoundTrip(double value) {
  using ::onnx::optimization::onnxsim_passes::Float8Format;
  using ::onnx::optimization::onnxsim_passes::FloatToFloat8Bits;
  const double clipped = std::min(std::max(value, -kFp8Max), kFp8Max);
  const uint8_t bits =
      FloatToFloat8Bits(static_cast<float>(clipped), Float8Format::kE4M3FN);
  return Float8E4M3FNBitsToDouble(bits);
}

// --- Delta-fidelity metrics --------------------------------------------------
//
// Mirrors onnxsim.daq's own _cosine_similarity/_sign_preservation_rate/
// _delta_score exactly.

double CosineSimilarity(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  if (!std::isfinite(denom) || denom <= 0.0) {
    return -1.0;
  }
  const double similarity = dot / denom;
  return std::isfinite(similarity) ? similarity : -1.0;
}

double SignPreservationRate(const std::vector<double>& a,
                            const std::vector<double>& b) {
  if (a.empty()) {
    return 0.0;
  }
  int64_t agree = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double sa = a[i] > 0.0 ? 1.0 : (a[i] < 0.0 ? -1.0 : 0.0);
    const double sb = b[i] > 0.0 ? 1.0 : (b[i] < 0.0 ? -1.0 : 0.0);
    if (sa == sb) {
      ++agree;
    }
  }
  return static_cast<double>(agree) / static_cast<double>(a.size());
}

double DeltaScore(const std::vector<double>& delta_w,
                  const std::vector<double>& delta_w_hat,
                  const std::string& metric) {
  if (metric == "cosine") {
    return CosineSimilarity(delta_w, delta_w_hat);
  }
  return SignPreservationRate(delta_w, delta_w_hat);
}

// --- Coarse-to-fine delta-aware scale search ---------------------------------
//
// Mirrors onnxsim.daq's own _search_delta_aware_scale exactly: 9
// log-spaced multipliers over [0.5, 2.0] around the naive absmax scale,
// then 9 linearly-spaced multipliers over [0.9, 1.1] times the coarse
// winner. The coarse winner is kept across both passes (the fine grid is
// not guaranteed to contain it once re-centered).

std::vector<double> Geomspace(double lo, double hi, int count) {
  std::vector<double> out(static_cast<size_t>(count));
  const double log_lo = std::log(lo);
  const double log_hi = std::log(hi);
  for (int i = 0; i < count; ++i) {
    const double t = count == 1 ? 0.0 : static_cast<double>(i) / (count - 1);
    out[static_cast<size_t>(i)] = std::exp(log_lo + t * (log_hi - log_lo));
  }
  return out;
}

std::vector<double> Linspace(double lo, double hi, int count) {
  std::vector<double> out(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const double t = count == 1 ? 0.0 : static_cast<double>(i) / (count - 1);
    out[static_cast<size_t>(i)] = lo + t * (hi - lo);
  }
  return out;
}

double SearchDeltaAwareScale(const std::vector<double>& w_post,
                             const std::vector<double>& w_base,
                             const std::string& metric) {
  std::vector<double> delta_w(w_post.size());
  double max_abs_post = 0.0;
  for (size_t i = 0; i < w_post.size(); ++i) {
    delta_w[i] = w_post[i] - w_base[i];
    max_abs_post = std::max(max_abs_post, std::fabs(w_post[i]));
  }
  const double scale0 = std::max(max_abs_post, 1e-12) / kFp8Max;

  std::vector<double> w_hat(w_post.size());
  std::vector<double> delta_w_hat(w_post.size());
  auto evaluate = [&](double multiplier) {
    const double scale = scale0 * multiplier;
    for (size_t i = 0; i < w_post.size(); ++i) {
      w_hat[i] = Fp8RoundTrip(w_post[i] / scale) * scale;
      delta_w_hat[i] = w_hat[i] - w_base[i];
    }
    return DeltaScore(delta_w, delta_w_hat, metric);
  };

  double best_multiplier = 1.0;
  double best_score = -std::numeric_limits<double>::infinity();
  for (double multiplier : Geomspace(0.5, 2.0, 9)) {
    const double score = evaluate(multiplier);
    if (score > best_score) {
      best_score = score;
      best_multiplier = multiplier;
    }
  }
  for (double multiplier :
       Linspace(best_multiplier * 0.9, best_multiplier * 1.1, 9)) {
    const double score = evaluate(multiplier);
    if (score > best_score) {
      best_score = score;
      best_multiplier = multiplier;
    }
  }
  return scale0 * best_multiplier;
}

}  // namespace

onnx::ModelProto ApplyDaq(const onnx::ModelProto& base_model,
                          const onnx::ModelProto& post_trained_model,
                          const std::string& metric,
                          const std::unordered_set<std::string>& skip_names) {
  if (metric != "cosine" && metric != "sign_preservation") {
    throw std::invalid_argument("unknown metric: " + metric);
  }

  onnx::ModelProto out = post_trained_model;
  onnx::GraphProto* post_graph = out.mutable_graph();
  const onnx::GraphProto& base_graph = base_model.graph();

  std::unordered_map<std::string, int> base_by_output;
  for (int i = 0; i < base_graph.node_size(); ++i) {
    const auto& n = base_graph.node(i);
    if (n.output_size() > 0) {
      base_by_output[n.output(0)] = i;
    }
  }
  std::unordered_map<std::string, int> base_init_index;
  for (int i = 0; i < base_graph.initializer_size(); ++i) {
    base_init_index[base_graph.initializer(i).name()] = i;
  }
  std::unordered_map<std::string, int> post_init_index;
  for (int i = 0; i < post_graph->initializer_size(); ++i) {
    post_init_index[post_graph->initializer(i).name()] = i;
  }

  auto constant_2d_float =
      [](const onnx::GraphProto& graph,
         const std::unordered_map<std::string, int>& init_index,
         const std::string& name) -> const onnx::TensorProto* {
    auto it = init_index.find(name);
    if (it == init_index.end()) {
      return nullptr;
    }
    const onnx::TensorProto& t = graph.initializer(it->second);
    if (t.data_type() != onnx::TensorProto::FLOAT || t.dims_size() != 2) {
      return nullptr;
    }
    return &t;
  };

  struct Match {
    int post_node_index;
    const onnx::TensorProto* w_post;
    const onnx::TensorProto* w_base;
  };
  std::vector<Match> matches;
  for (int i = 0; i < post_graph->node_size(); ++i) {
    const onnx::NodeProto& node = post_graph->node(i);
    if (node.output_size() < 1) {
      continue;
    }
    const std::string w_name = MatchMatMulLikeWeightName(node);
    if (w_name.empty() || skip_names.count(w_name) != 0) {
      continue;
    }
    const onnx::TensorProto* w_post =
        constant_2d_float(*post_graph, post_init_index, w_name);
    if (w_post == nullptr) {
      continue;
    }
    auto base_it = base_by_output.find(node.output(0));
    if (base_it == base_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& base_node = base_graph.node(base_it->second);
    if (base_node.op_type() != node.op_type()) {
      continue;
    }
    const std::string base_w_name = MatchMatMulLikeWeightName(base_node);
    if (base_w_name.empty()) {
      continue;
    }
    const onnx::TensorProto* w_base =
        constant_2d_float(base_graph, base_init_index, base_w_name);
    if (w_base == nullptr || w_base->dims_size() != w_post->dims_size() ||
        w_base->dims(0) != w_post->dims(0) ||
        w_base->dims(1) != w_post->dims(1)) {
      continue;
    }
    matches.push_back({i, w_post, w_base});
  }

  if (matches.empty()) {
    return out;
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly
  // (base, base_1, base_2, ...).
  std::unordered_set<std::string> taken_names;
  for (const auto& t : post_graph->initializer()) {
    taken_names.insert(t.name());
  }
  for (const auto& vi : post_graph->input()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : post_graph->output()) {
    taken_names.insert(vi.name());
  }
  for (const auto& vi : post_graph->value_info()) {
    taken_names.insert(vi.name());
  }
  for (const auto& n : post_graph->node()) {
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

  for (const Match& m : matches) {
    const std::vector<double> w_post = ReadFloatTensor(*m.w_post);
    const std::vector<double> w_base = ReadFloatTensor(*m.w_base);

    double sum_sq_delta = 0.0;
    for (size_t i = 0; i < w_post.size(); ++i) {
      const double d = w_post[i] - w_base[i];
      sum_sq_delta += d * d;
    }
    // No fine-tuning update in this layer -- nothing for a
    // delta-preserving objective to preserve, so leave it alone entirely.
    if (std::sqrt(sum_sq_delta) < 1e-12) {
      continue;
    }

    const double best_scale = SearchDeltaAwareScale(w_post, w_base, metric);
    std::vector<double> w_quant(w_post.size());
    for (size_t i = 0; i < w_post.size(); ++i) {
      w_quant[i] = Fp8RoundTrip(w_post[i] / best_scale) * best_scale;
    }

    const std::string new_name = unique_name(m.w_post->name() + "_daq");
    SetFloatInitializer(post_graph->add_initializer(), new_name,
                        {m.w_post->dims(0), m.w_post->dims(1)}, w_quant);
    post_graph->mutable_node(m.post_node_index)->set_input(1, new_name);
  }

  return out;
}
