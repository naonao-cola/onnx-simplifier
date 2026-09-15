// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See llm_int8_entry.h for the full rationale (including why this
// follows outlier_suppression_plus_entry.h's own protobuf-level,
// calibration-driven shape) and onnxsim/llm_int8.py for the technique
// this ports.

#include "llm_int8_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

// MatMulInteger's operands are uint8 (activation, offset by a zero-point
// of 128) and int8 (weight) -- see llm_int8.py's own module docstring.
// The worst-case per-term product magnitude is therefore 127 * 255, so
// past this many reduction terms the int32 accumulator could wrap around
// in the worst case. Mirrors llm_int8.py's own
// _MAX_SAFE_INT32_REDUCTION_DEPTH exactly.
constexpr int64_t kMaxSafeInt32ReductionDepth =
    (2147483647LL) / (127 * 255);  // 66353

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Transcribed from smoothquant_entry.cpp's own MatchMatMulLike, widened
// to exactly what onnxsim.llm_int8._match_matmul_like returns: the same
// MatMul / vanilla-Gemm acceptance, plus the bias input name (or empty
// when there is none) this pass threads into its combining Add.
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
// ReadFloatTensor transcribed from smoothquant_entry.cpp's own (FLOAT32
// only -- this pass, like its own Python reference onnxsim.llm_int8,
// never widens to FLOAT16/BFLOAT16). SetRawInitializer covers every
// constant encoding the rewrite emits (int64 indices, int8/uint8 codes,
// float32 scales), always as little-endian raw_data -- mirroring
// onnx.numpy_helper.from_array's own contiguous-C-order raw encoding via
// dlpack_bridge.h's kRawDataIsHostOrder/SwapElementBytes.

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

// Round-half-to-even (banker's rounding), matching numpy's own `round`
// (and the ONNX `Round` operator this rewrite emits, which the runtime
// will evaluate the same way): ties go to the even neighbor, unlike
// std::round's half-away-from-zero. Exact-.5 inputs arise here whenever a
// weight lands exactly halfway between two INT8 codes.
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

// --- Calibration: per-channel activation absmax ----------------------------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// smoothquant_entry.cpp's own ComputeChannelAbsmax, narrowed to exactly
// what onnxsim.llm_int8 needs: per-channel `max(|X_j|)` over every
// calibration batch (elementwise maximum across batches), FLOAT32 2-D
// tensors only.
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
            "ApplyLlmInt8: calibration batch is missing "
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
      const int64_t k = tp.dims(1);
      if (k <= 0) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      // First observation assigns (rather than max-ing against zero),
      // matching `m if name not in act_absmax else np.maximum(...)`: an
      // all-zero channel must still record 0.0 here so the caller's
      // threshold comparison -- not a spurious zero from an empty
      // accumulator -- decides whether it is an outlier.
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

// Inserts a fresh node at position `index` (shifting later nodes right).
// The caller re-fetches it via `mutable_node(index)`, since no NodeProto
// pointer is held across a mutation. (The generated RepeatedPtrField API
// exposes only Add, so the appended node is rotated left into place with
// content Swaps.)
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

onnx::ModelProto ApplyLlmInt8(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double outlier_threshold, double epsilon) {
  onnx::ModelProto out = model;

  // ReduceMax's axes-as-input form needs opset >= 18 -- the whole pass
  // declines older models, mirroring the reference.
  bool opset_ge_18 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 18) {
      opset_ge_18 = true;
      break;
    }
  }
  if (!opset_ge_18) {
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

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...) so freshly minted names match the Python
  // reference's own one-for-one. Candidates are processed in forward node
  // order for the same reason; each rewrite inserts its new nodes before
  // the old node and deletes the old node itself, so a candidate's live
  // index is its original index plus the net nodes already added by
  // earlier candidates (inserted count minus one deleted each) -- indices
  // are never held across mutations any other way (no NodeProto pointers
  // survive an insert/delete).
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
    const std::vector<double>& acts = acts_it->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (acts.size() != static_cast<size_t>(k)) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }

    // Outlier channels (strictly above threshold) vs. regular ones --
    // mirrors `np.where(acts > outlier_threshold)` /
    // `np.where(acts <= outlier_threshold)` exactly (both ascending).
    std::vector<int64_t> outlier_idx;
    std::vector<int64_t> regular_idx;
    for (int64_t j = 0; j < k; ++j) {
      if (acts[static_cast<size_t>(j)] > outlier_threshold) {
        outlier_idx.push_back(j);
      } else {
        regular_idx.push_back(j);
      }
    }
    if (outlier_idx.empty() || regular_idx.empty()) {
      continue;  // Nothing to decompose.
    }
    if (static_cast<int64_t>(regular_idx.size()) >
        kMaxSafeInt32ReductionDepth) {
      continue;  // MatMulInteger's int32 accumulator could overflow.
    }
    const int64_t no = static_cast<int64_t>(outlier_idx.size());
    const int64_t nr = static_cast<int64_t>(regular_idx.size());

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    // w_nk[i, j], row-major over [N, K]: the stored layout is [N, K]
    // itself when transposed, else [K, N] (so column j of w_nk is row j
    // of the stored tensor, strided by N == n_rows, not by K).
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // Outlier weight rows stay float32 ([no, N]); regular rows quantize
    // per output channel with an epsilon-floored absmax scale --
    // mirrors `col_scale = max(abs.max(axis=0), epsilon) / 127` and the
    // clipped banker's-rounded INT8 codes exactly.
    std::vector<float> w_outlier(static_cast<size_t>(no * n_rows));
    for (int64_t o = 0; o < no; ++o) {
      for (int64_t n = 0; n < n_rows; ++n) {
        w_outlier[static_cast<size_t>(o * n_rows + n)] =
            static_cast<float>(at_nk(n, outlier_idx[static_cast<size_t>(o)]));
      }
    }
    std::vector<double> col_scale(static_cast<size_t>(n_rows));
    for (int64_t n = 0; n < n_rows; ++n) {
      double wmax = 0.0;
      for (int64_t r = 0; r < nr; ++r) {
        const double v =
            std::abs(at_nk(n, regular_idx[static_cast<size_t>(r)]));
        if (v > wmax) {
          wmax = v;
        }
      }
      col_scale[static_cast<size_t>(n)] = std::max(wmax, epsilon) / 127.0;
    }
    std::vector<int8_t> wq_regular(static_cast<size_t>(nr * n_rows));
    for (int64_t r = 0; r < nr; ++r) {
      for (int64_t n = 0; n < n_rows; ++n) {
        const double q =
            RoundHalfToEven(at_nk(n, regular_idx[static_cast<size_t>(r)]) /
                            col_scale[static_cast<size_t>(n)]);
        const double clipped = std::min(127.0, std::max(-127.0, q));
        wq_regular[static_cast<size_t>(r * n_rows + n)] =
            static_cast<int8_t>(clipped);
      }
    }

    // Constants first, in the reference's own order (each minted via
    // _unique_name off `{x}_llm_int8_{suffix}`).
    const std::string prefix = c.x_name + "_llm_int8";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string outlier_idx_name = add_const(
        "outlier_idx", onnx::TensorProto::INT64, {no}, outlier_idx.data(),
        outlier_idx.size() * sizeof(int64_t), sizeof(int64_t));
    const std::string regular_idx_name = add_const(
        "regular_idx", onnx::TensorProto::INT64, {nr}, regular_idx.data(),
        regular_idx.size() * sizeof(int64_t), sizeof(int64_t));
    const std::string w_outlier_name = add_const(
        "w_outlier", onnx::TensorProto::FLOAT, {no, n_rows}, w_outlier.data(),
        w_outlier.size() * sizeof(float), sizeof(float));
    const std::string wq_regular_name = add_const(
        "wq_regular", onnx::TensorProto::INT8, {nr, n_rows}, wq_regular.data(),
        wq_regular.size() * sizeof(int8_t), sizeof(int8_t));
    std::vector<float> col_scale_f(col_scale.size());
    for (size_t i = 0; i < col_scale.size(); ++i) {
      col_scale_f[i] = static_cast<float>(col_scale[i]);
    }
    const std::string col_scale_name = add_const(
        "col_scale", onnx::TensorProto::FLOAT, {n_rows}, col_scale_f.data(),
        col_scale_f.size() * sizeof(float), sizeof(float));
    const int64_t one = 1;
    const std::string axis1_name =
        add_const("axis1", onnx::TensorProto::INT64, {1}, &one, sizeof(one),
                  sizeof(int64_t));
    const float eps_f = static_cast<float>(epsilon);
    const float c127_f = 127.0f;
    const float neg127_f = -127.0f;
    const float offset128_f = 128.0f;
    const uint8_t zp128_u8 = 128;
    const std::string eps_name =
        add_const("eps", onnx::TensorProto::FLOAT, {}, &eps_f, sizeof(eps_f),
                  sizeof(float));
    const std::string c127_name =
        add_const("c127", onnx::TensorProto::FLOAT, {}, &c127_f, sizeof(c127_f),
                  sizeof(float));
    const std::string neg127_name =
        add_const("neg127", onnx::TensorProto::FLOAT, {}, &neg127_f,
                  sizeof(neg127_f), sizeof(float));
    // pos127 duplicates c127's value under its own name (both feed
    // distinct Clip/Div inputs in the reference) -- emitted to match it
    // exactly.
    const std::string pos127_name =
        add_const("pos127", onnx::TensorProto::FLOAT, {}, &c127_f,
                  sizeof(c127_f), sizeof(float));
    const std::string offset128_name =
        add_const("offset128", onnx::TensorProto::FLOAT, {}, &offset128_f,
                  sizeof(offset128_f), sizeof(float));
    const std::string zp128_name =
        add_const("zp128", onnx::TensorProto::UINT8, {}, &zp128_u8,
                  sizeof(zp128_u8), sizeof(uint8_t));

    // Nodes next, each output minted before its node name -- mirrors the
    // reference's own `_new` helper exactly (including the biased node's
    // `{prefix}_bias_add_node` vs. plain `{prefix}_combine_node` tail).
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
    const std::string x_outlier = add_node(
        "Gather", {c.x_name, outlier_idx_name}, "x_outlier", {{"axis", 1}});
    const std::string x_regular = add_node(
        "Gather", {c.x_name, regular_idx_name}, "x_regular", {{"axis", 1}});
    const std::string y_outlier =
        add_node("MatMul", {x_outlier, w_outlier_name}, "y_outlier");
    const std::string x_abs = add_node("Abs", {x_regular}, "x_abs");
    const std::string row_max = add_node("ReduceMax", {x_abs, axis1_name},
                                         "row_max", {{"keepdims", 1}});
    const std::string row_max_safe =
        add_node("Max", {row_max, eps_name}, "row_max_safe");
    const std::string row_scale =
        add_node("Div", {row_max_safe, c127_name}, "row_scale");
    const std::string x_scaled =
        add_node("Div", {x_regular, row_scale}, "x_scaled");
    const std::string x_rounded = add_node("Round", {x_scaled}, "x_rounded");
    const std::string x_clipped =
        add_node("Clip", {x_rounded, neg127_name, pos127_name}, "x_clipped");
    const std::string x_shifted =
        add_node("Add", {x_clipped, offset128_name}, "x_shifted");
    const std::string xq =
        add_node("Cast", {x_shifted}, "xq", {{"to", onnx::TensorProto::UINT8}});
    const std::string y_int32 =
        add_node("MatMulInteger", {xq, wq_regular_name, zp128_name}, "y_int32");
    const std::string y_int32_f = add_node("Cast", {y_int32}, "y_int32_f",
                                           {{"to", onnx::TensorProto::FLOAT}});
    const std::string y_partial =
        add_node("Mul", {y_int32_f, row_scale}, "y_partial");
    const std::string y_regular =
        add_node("Mul", {y_partial, col_scale_name}, "y_regular");

    const std::string old_output =
        graph->node(c.node_index + static_cast<int>(net_insertions)).output(0);
    NewNode final_node;
    final_node.op_type = "Add";
    final_node.output = old_output;
    if (!c.bias_name.empty()) {
      const std::string combined =
          add_node("Add", {y_outlier, y_regular}, "combined");
      final_node.inputs = {combined, c.bias_name};
      final_node.name = unique_name(prefix + "_bias_add_node");
    } else {
      final_node.inputs = {y_outlier, y_regular};
      final_node.name = unique_name(prefix + "_combine_node");
    }
    new_nodes.push_back(std::move(final_node));

    // Splice the new nodes in before the old node, then delete the old
    // node itself -- mirrors insert-then-`del` exactly.
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
      for (const auto& [attr_name, attr_value] : spec.int_attrs) {
        AddIntAttribute(n, attr_name, attr_value);
      }
    }
    graph->mutable_node()->DeleteSubrange(live_index + count, 1);
    net_insertions += count - 1;
  }

  return out;
}
