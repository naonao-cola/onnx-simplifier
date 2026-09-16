// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See squeezellm_entry.h for the full rationale (including why this
// follows llm_int8_entry.h's own single-model, protobuf-level,
// calibration-driven shape) and onnxsim/squeezellm.py for the technique
// this ports.

#include "squeezellm_entry.h"

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
// Transcribed from squeezellm.py's own _match_matmul_like (itself already
// a direct transcription of passes/quantize_matmul_common.h's
// MatchMatMulLike): a MatMul, or a Gemm with transA=0, alpha=1 and (when
// it has a bias) beta=1. squeezellm.py never touches a Gemm's own bias
// input at all (unlike llm_int8.py's own combining Add), so this matcher
// -- unlike llm_int8_entry.cpp's own MatMulLikeMatch -- does not need to
// capture a bias name.
struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
  bool weight_transposed;
};

std::optional<MatMulLikeMatch> MatchMatMulLike(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), false};
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
    if (num_inputs == 3 && has_beta && beta != 1.0) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1), trans_b != 0};
  }
  return std::nullopt;
}

// --- Tensor <-> flat buffers, protobuf level -------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers (FLOAT32
// only -- this pass, like its own Python reference onnxsim.squeezellm,
// never widens to FLOAT16/BFLOAT16).

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

void AddIntAttribute(onnx::NodeProto* node, const std::string& name,
                     int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INT);
  attr->set_i(value);
}

void AddIntsAttribute(onnx::NodeProto* node, const std::string& name,
                      const std::vector<int64_t>& values) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(onnx::AttributeProto::INTS);
  for (int64_t v : values) {
    attr->add_ints(v);
  }
}

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// transcribed from llm_int8_entry.cpp's own identical helper. Used here
// for `np.linspace(...).round()`'s own k-means centroid-init index
// computation, which can legitimately land exactly on a .5 boundary
// (e.g. `num_levels` evenly dividing `block_size - 1`).
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

// numpy.quantile's own default "linear" interpolation method, over an
// ALREADY-SORTED array -- the same formula lo_bcq.h's own PercentileSorted
// (percentile-scale, p in [0, 100]) already establishes for this
// codebase, restated here in numpy.quantile's own q-in-[0, 1] scale
// (index = q * (n - 1), linear interpolation between the neighboring
// order statistics).
double Quantile(const std::vector<double>& sorted_vals, double q) {
  const int64_t n = static_cast<int64_t>(sorted_vals.size());
  const double index = q * static_cast<double>(n - 1);
  const int64_t lower = static_cast<int64_t>(std::floor(index));
  const int64_t upper = static_cast<int64_t>(std::ceil(index));
  if (lower == upper) {
    return sorted_vals[static_cast<size_t>(lower)];
  }
  const double frac = index - static_cast<double>(lower);
  return sorted_vals[static_cast<size_t>(lower)] +
         frac * (sorted_vals[static_cast<size_t>(upper)] -
                 sorted_vals[static_cast<size_t>(lower)]);
}

// --- Calibration: per-input-channel mean(x_k ** 2) --------------------------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// llm_int8_entry.cpp's own ComputeChannelAbsmax, narrowed to exactly what
// onnxsim.squeezellm needs: accumulate `sum(x_k ** 2)` and the observed
// row count per probe name across every calibration batch (mirroring
// `sq_sum`/`count` accumulation in the Python reference exactly), then
// divide once at the end -- FLOAT32 2-D tensors only.
std::unordered_map<std::string, std::vector<double>>
ComputeActivationSensitivity(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  std::unordered_map<std::string, std::vector<double>> sq_sum;
  std::unordered_map<std::string, int64_t> count;
  if (probe_names.empty()) {
    return {};
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
            "ApplySqueezeLlm: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() != 2) {
        continue;
      }
      const int64_t rows = tp.dims(0);
      const int64_t k = tp.dims(1);
      if (k <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      auto& acc = sq_sum[name];
      if (acc.empty()) {
        acc.assign(static_cast<size_t>(k), 0.0);
      }
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < k; ++c) {
          const double v =
              static_cast<double>(data[static_cast<size_t>(r * k + c)]);
          acc[static_cast<size_t>(c)] += v * v;
        }
      }
      count[name] += rows;
    }
  }

  std::unordered_map<std::string, std::vector<double>> result;
  for (auto& [name, sq] : sq_sum) {
    const int64_t cnt = count[name];
    if (cnt <= 0) {
      continue;
    }
    std::vector<double> sensitivity(sq.size());
    for (size_t i = 0; i < sq.size(); ++i) {
      sensitivity[i] = sq[i] / static_cast<double>(cnt);
    }
    result.emplace(name, std::move(sensitivity));
  }
  return result;
}

// --- Sensitivity-weighted 1-D Lloyd's-algorithm k-means, per group ---------
//
// `values`/`weights` are flat, `num_groups * block_size` elements, group
// `g` occupying `[g * block_size, (g + 1) * block_size)` -- the SAME flat
// layout as the [N, K] weight matrix itself, since a group is exactly one
// contiguous `block_size`-wide run along `K` (see this file's own
// ApplySqueezeLlm for why this flat-index identity holds and needs no
// separate reshape). Mirrors squeezellm.py's own
// _weighted_kmeans_quantize_groups exactly, including its own
// deterministic (no RNG at all) centroid initialization -- see
// squeezellm_entry.h's own top-of-file "NUMERICAL SCOPE" note.
void WeightedKMeansQuantizeGroups(const std::vector<double>& values,
                                  const std::vector<double>& weights,
                                  int64_t num_groups, int64_t block_size,
                                  int64_t num_levels, int64_t num_iterations,
                                  std::vector<int64_t>& codes_out,
                                  std::vector<float>& centroids_out) {
  codes_out.assign(values.size(), 0);
  centroids_out.assign(static_cast<size_t>(num_groups * num_levels), 0.0f);

  // Shared, group-independent initial rank indices: evenly-spaced order
  // statistics of each group's own sorted values (`np.linspace(0,
  // block_size - 1, num_levels).round()`).
  std::vector<int64_t> init_idx(static_cast<size_t>(num_levels));
  for (int64_t l = 0; l < num_levels; ++l) {
    const double t = num_levels > 1 ? static_cast<double>(l) *
                                          static_cast<double>(block_size - 1) /
                                          static_cast<double>(num_levels - 1)
                                    : 0.0;
    int64_t idx = static_cast<int64_t>(RoundHalfToEven(t));
    idx = std::min<int64_t>(std::max<int64_t>(idx, 0), block_size - 1);
    init_idx[static_cast<size_t>(l)] = idx;
  }

  std::vector<double> centroids(static_cast<size_t>(num_levels));
  std::vector<double> sorted_group(static_cast<size_t>(block_size));
  std::vector<int64_t> codes_group(static_cast<size_t>(block_size));
  std::vector<double> wsum(static_cast<size_t>(num_levels));
  std::vector<double> vsum(static_cast<size_t>(num_levels));

  for (int64_t g = 0; g < num_groups; ++g) {
    const double* v = values.data() + g * block_size;
    const double* w = weights.data() + g * block_size;

    sorted_group.assign(v, v + block_size);
    std::sort(sorted_group.begin(), sorted_group.end());
    for (int64_t l = 0; l < num_levels; ++l) {
      centroids[static_cast<size_t>(l)] =
          sorted_group[static_cast<size_t>(init_idx[static_cast<size_t>(l)])];
    }

    for (int64_t iter = 0; iter < num_iterations; ++iter) {
      for (int64_t b = 0; b < block_size; ++b) {
        int64_t best = 0;
        double best_dist = (v[b] - centroids[0]) * (v[b] - centroids[0]);
        for (int64_t l = 1; l < num_levels; ++l) {
          const double d = (v[b] - centroids[static_cast<size_t>(l)]) *
                           (v[b] - centroids[static_cast<size_t>(l)]);
          if (d < best_dist) {
            best_dist = d;
            best = l;
          }
        }
        codes_group[static_cast<size_t>(b)] = best;
      }

      std::fill(wsum.begin(), wsum.end(), 0.0);
      std::fill(vsum.begin(), vsum.end(), 0.0);
      for (int64_t b = 0; b < block_size; ++b) {
        const int64_t l = codes_group[static_cast<size_t>(b)];
        wsum[static_cast<size_t>(l)] += w[b];
        vsum[static_cast<size_t>(l)] += w[b] * v[b];
      }
      // Unsafe (near-zero total weight) levels keep their PREVIOUS
      // centroid value -- mirrors `new_centroids = centroids.copy();
      // new_centroids[safe, level] = ...` exactly: only overwrite when
      // wsum is safely away from zero.
      for (int64_t l = 0; l < num_levels; ++l) {
        if (wsum[static_cast<size_t>(l)] > 1e-12) {
          centroids[static_cast<size_t>(l)] =
              vsum[static_cast<size_t>(l)] / wsum[static_cast<size_t>(l)];
        }
      }
    }

    for (int64_t b = 0; b < block_size; ++b) {
      int64_t best = 0;
      double best_dist = (v[b] - centroids[0]) * (v[b] - centroids[0]);
      for (int64_t l = 1; l < num_levels; ++l) {
        const double d = (v[b] - centroids[static_cast<size_t>(l)]) *
                         (v[b] - centroids[static_cast<size_t>(l)]);
        if (d < best_dist) {
          best_dist = d;
          best = l;
        }
      }
      codes_out[static_cast<size_t>(g * block_size + b)] = best;
    }
    for (int64_t l = 0; l < num_levels; ++l) {
      centroids_out[static_cast<size_t>(g * num_levels + l)] =
          static_cast<float>(centroids[static_cast<size_t>(l)]);
    }
  }
}

// Inserts a fresh node at position `index` (shifting later nodes right) --
// transcribed from llm_int8_entry.cpp's own identical helper.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

}  // namespace

onnx::ModelProto ApplySqueezeLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, int64_t bits, double outlier_fraction,
    int64_t num_kmeans_iterations) {
  onnx::ModelProto out = model;

  // GatherND's own batch_dims support needs opset >= 12 -- the whole pass
  // declines older models, mirroring the reference.
  bool opset_ge_12 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 12) {
      opset_ge_12 = true;
      break;
    }
  }
  if (!opset_ge_12) {
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
    candidates.push_back({i, m->x_name, m->w_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  onnx::ModelProto probe_model = out;
  const std::unordered_map<std::string, std::vector<double>> sensitivity_map =
      ComputeActivationSensitivity(executor, probe_model, probe_names,
                                   calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention llm_int8_entry.cpp's own
  // identical block already establishes for this codebase.
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

  const int64_t num_levels = int64_t{1} << bits;
  int64_t net_insertions = 0;

  for (const auto& c : candidates) {
    auto sens_it = sensitivity_map.find(c.x_name);
    if (sens_it == sensitivity_map.end()) {
      continue;  // Never observed as a plain 2-D tensor; skip.
    }
    const std::vector<double>& sensitivity_k = sens_it->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (sensitivity_k.size() != static_cast<size_t>(k)) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }
    if (k % block_size != 0) {
      continue;
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    // w_nk, row-major [N, K] -- the SAME flat layout a `[num_groups,
    // block_size]` reshape would give directly (see WeightedKMeansQuantize
    // Groups' own comment): position p = n_idx * k + k_idx.
    std::vector<double> w_nk(static_cast<size_t>(n_rows * k));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        const float v = c.weight_transposed
                            ? flat[static_cast<size_t>(i * k + j)]
                            : flat[static_cast<size_t>(j * n_rows + i)];
        w_nk[static_cast<size_t>(i * k + j)] = static_cast<double>(v);
      }
    }

    const int64_t total = n_rows * k;
    std::vector<double> abs_sorted(static_cast<size_t>(total));
    for (int64_t p = 0; p < total; ++p) {
      abs_sorted[static_cast<size_t>(p)] =
          std::fabs(w_nk[static_cast<size_t>(p)]);
    }
    std::sort(abs_sorted.begin(), abs_sorted.end());
    const double threshold = Quantile(abs_sorted, 1.0 - outlier_fraction);

    // A weight element's own group-fit weight is its input channel's own
    // sensitivity (`sensitivity_k[k_idx]`, k_idx = p % k -- valid since a
    // group is exactly one contiguous block_size-wide run along K, so
    // every element of a flat position p shares the same k_idx modulo k
    // regardless of which output row n_idx it belongs to), zeroed at
    // outlier positions so they don't skew the codebook fit -- mirrors
    // `weights[outlier_mask_flat] = 0.0` exactly.
    std::vector<uint8_t> outlier_mask(static_cast<size_t>(total));
    std::vector<double> weights(static_cast<size_t>(total));
    for (int64_t p = 0; p < total; ++p) {
      const bool is_outlier =
          std::fabs(w_nk[static_cast<size_t>(p)]) > threshold;
      outlier_mask[static_cast<size_t>(p)] = is_outlier ? 1 : 0;
      weights[static_cast<size_t>(p)] =
          is_outlier ? 0.0 : sensitivity_k[static_cast<size_t>(p % k)];
    }

    const int64_t num_blocks = k / block_size;
    const int64_t num_groups = n_rows * num_blocks;

    std::vector<int64_t> codes;
    std::vector<float> codebook;
    WeightedKMeansQuantizeGroups(w_nk, weights, num_groups, block_size,
                                 num_levels, num_kmeans_iterations, codes,
                                 codebook);

    std::vector<float> sparse_diff(static_cast<size_t>(total), 0.0f);
    for (int64_t p = 0; p < total; ++p) {
      if (!outlier_mask[static_cast<size_t>(p)]) {
        continue;
      }
      const int64_t g = p / block_size;
      const int64_t code = codes[static_cast<size_t>(p)];
      const double dequant = static_cast<double>(
          codebook[static_cast<size_t>(g * num_levels + code)]);
      sparse_diff[static_cast<size_t>(p)] =
          static_cast<float>(w_nk[static_cast<size_t>(p)] - dequant);
    }

    // All four constant initializers are minted up front here, unlike
    // squeezellm.py's own interleaved order (codebook, codes, sparse_diff,
    // *then* gathered/gathernd_node, *then* shape, *then*
    // unblocked/reshape_node, ...) -- a pure code-structure
    // simplification with no effect on the resulting name strings for any
    // ordinary model (every suffix below is already a distinct string, so
    // `_unique_name`'s own base/base_1/base_2 bump only differs from the
    // reference if the ORIGINAL model already contained one of these
    // exact names, an adversarial edge case neither side is expected to
    // handle identically).
    const std::string prefix = c.w_name + "_squeezellm";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string codebook_name = add_const(
        "codebook", onnx::TensorProto::FLOAT, {num_groups, num_levels},
        codebook.data(), codebook.size() * sizeof(float), sizeof(float));
    const std::string codes_name = add_const(
        "codes", onnx::TensorProto::INT64, {num_groups, block_size, 1},
        codes.data(), codes.size() * sizeof(int64_t), sizeof(int64_t));
    const std::string sparse_diff_name = add_const(
        "sparse_diff", onnx::TensorProto::FLOAT, {n_rows, k},
        sparse_diff.data(), sparse_diff.size() * sizeof(float), sizeof(float));
    const std::vector<int64_t> shape_vals = {n_rows, k};
    const std::string shape_name =
        add_const("shape", onnx::TensorProto::INT64, {2}, shape_vals.data(),
                  shape_vals.size() * sizeof(int64_t), sizeof(int64_t));

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
      std::optional<int64_t> batch_dims;
      std::optional<std::vector<int64_t>> perm;
    };
    std::vector<NewNode> new_nodes;
    // `out_suffix` names the new output tensor (semantic, e.g.
    // "gathered"); `node_name_suffix` names the node itself, which
    // squeezellm.py mints as a DISTINCT, op-type-based suffix (e.g.
    // "gathernd_node", not "gathered_node") -- both suffixes must be
    // passed separately to match the reference's own two independent
    // `_unique_name` calls exactly.
    auto add_node = [&](const std::string& op_type,
                        const std::vector<std::string>& inputs,
                        const std::string& out_suffix,
                        const std::string& node_name_suffix) -> std::string& {
      NewNode n;
      n.op_type = op_type;
      n.inputs = inputs;
      n.output = unique_name(prefix + "_" + out_suffix);
      n.name = unique_name(prefix + "_" + node_name_suffix);
      new_nodes.push_back(std::move(n));
      return new_nodes.back().output;
    };

    const std::string gathered = add_node(
        "GatherND", {codebook_name, codes_name}, "gathered", "gathernd_node");
    new_nodes.back().batch_dims = 1;
    const std::string unblocked = add_node("Reshape", {gathered, shape_name},
                                           "unblocked", "reshape_node");
    const std::string corrected =
        add_node("Add", {unblocked, sparse_diff_name}, "corrected", "add_node");

    std::string final_name = corrected;
    if (!c.weight_transposed) {
      final_name =
          add_node("Transpose", {corrected}, "transposed", "transpose_node");
      new_nodes.back().perm = std::vector<int64_t>{1, 0};
    }

    const int live_index = c.node_index + static_cast<int>(net_insertions);
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
      if (spec.batch_dims.has_value()) {
        AddIntAttribute(n, "batch_dims", *spec.batch_dims);
      }
      if (spec.perm.has_value()) {
        AddIntsAttribute(n, "perm", *spec.perm);
      }
    }
    net_insertions += count;

    // The matched node itself is kept in place (now shifted right by
    // `count`), only its own weight input rewired -- mirrors the
    // reference's own `node.input[i] = final_name` exactly (no node is
    // replaced or destroyed here, unlike llm_int8_entry.cpp's own full
    // node-replacement rewrite).
    onnx::NodeProto* matched = graph->mutable_node(live_index + count);
    for (int i = 0; i < matched->input_size(); ++i) {
      if (matched->input(i) == c.w_name) {
        matched->set_input(i, final_name);
      }
    }
  }

  return out;
}
