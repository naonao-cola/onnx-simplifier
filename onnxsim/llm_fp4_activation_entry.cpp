// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See llm_fp4_activation_entry.h for the full rationale and
// onnxsim/llm_fp4.py for the technique this ports.

#include "llm_fp4_activation_entry.h"

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

// Calibration-fit MSE search over a whole flattened activation sample
// caps out at this many elements -- see this file's own header comment
// ("ACCEPTED, PERMANENT DIVERGENCE") for why this port takes the first
// this-many raveled elements rather than llm_fp4.py's own random
// subsample past the cap.
constexpr size_t kPerTensorFitMaxElements = size_t{1} << 18;

// --- MatMul/vanilla-Gemm matching, protobuf level ---------------------
//
// Transcribed from llm_fp4.py's own _match_matmul_like (billm_entry.cpp's
// own MatchMatMulLike is byte-identical; duplicated here per this
// codebase's established "no shared dependency between independently
// tested *_entry.cpp TUs" convention -- see billm_entry.h's own
// top-of-file comment).
struct MatMulLikeMatch {
  std::string x_name;
  std::string w_name;
};

std::optional<MatMulLikeMatch> MatchMatMulLike(const onnx::NodeProto& node) {
  if (node.op_type() == "MatMul") {
    if (node.input_size() != 2) {
      return std::nullopt;
    }
    return MatMulLikeMatch{node.input(0), node.input(1)};
  }
  if (node.op_type() == "Gemm") {
    const int num_inputs = node.input_size();
    if (num_inputs != 2 && num_inputs != 3) {
      return std::nullopt;
    }
    bool has_trans_a = false, has_alpha = false, has_beta = false;
    int64_t trans_a = 0;
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
    return MatMulLikeMatch{node.input(0), node.input(1)};
  }
  return std::nullopt;
}

// --- Recovering a layer's own already-baked LLM-FP4 codebook ----------
//
// Direct transcription of llm_fp4.py's own _find_llm_fp4_weight_codebook:
// walks backward through quantize_weight_only_llm_fp4[_cpp]'s own EXACT
// dequantization pattern (see passes/llm_fp4.h's own top-of-file
// comment for the node shape) and returns that layer's own Codebook
// initializer name, or empty if `w_name` isn't fed by exactly that
// pattern.
std::string FindLlmFp4WeightCodebook(
    const std::string& w_name, const onnx::GraphProto& graph,
    const std::unordered_map<std::string, int>& producer_index,
    const std::unordered_map<std::string, int>& init_index) {
  auto find_node = [&](const std::string& name) -> const onnx::NodeProto* {
    auto it = producer_index.find(name);
    if (it == producer_index.end()) {
      return nullptr;
    }
    return &graph.node(it->second);
  };

  const onnx::NodeProto* reshape3 = find_node(w_name);
  if (reshape3 == nullptr || reshape3->op_type() != "Reshape" ||
      reshape3->input_size() != 2) {
    return "";
  }
  const onnx::NodeProto* mul = find_node(reshape3->input(0));
  if (mul == nullptr || mul->op_type() != "Mul" || mul->input_size() != 2) {
    return "";
  }
  const onnx::NodeProto* reshape1 = find_node(mul->input(0));
  const onnx::NodeProto* reshape2 = find_node(mul->input(1));
  if (reshape1 == nullptr || reshape1->op_type() != "Reshape" ||
      reshape1->input_size() != 2) {
    return "";
  }
  if (reshape2 == nullptr || reshape2->op_type() != "Reshape" ||
      reshape2->input_size() != 2) {
    return "";
  }
  if (init_index.find(reshape2->input(0)) == init_index.end()) {
    return "";  // Ws: constant per-block scale.
  }
  const onnx::NodeProto* gather = find_node(reshape1->input(0));
  if (gather == nullptr || gather->op_type() != "Gather" ||
      gather->input_size() != 2) {
    return "";
  }
  const std::string codebook_name = gather->input(0);
  if (init_index.find(codebook_name) == init_index.end()) {
    return "";
  }
  const onnx::NodeProto* cast = find_node(gather->input(1));
  if (cast == nullptr || cast->op_type() != "Cast" || cast->input_size() != 1) {
    return "";
  }
  if (init_index.find(cast->input(0)) == init_index.end()) {
    return "";  // Wq: constant codebook indices.
  }
  return codebook_name;
}

// --- Tensor <-> flat buffers, protobuf level ---------------------------
//
// Transcribed from billm_entry.cpp's own ReadFloatTensor/
// SetRawInitializer (FLOAT32 only).

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

bool HasMinOpset(const onnx::ModelProto& model, int64_t min_version) {
  for (const auto& opset : model.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= min_version) {
      return true;
    }
  }
  return false;
}

// --- Shared candidate discovery ----------------------------------------

struct Candidate {
  onnx::NodeProto* node;
  std::string x_name;
  std::string w_name;
  std::string codebook_name;
};

std::vector<Candidate> FindCandidates(
    onnx::GraphProto* graph,
    const std::unordered_map<std::string, int>& init_index) {
  std::unordered_map<std::string, int> producer_index;
  for (int i = 0; i < graph->node_size(); ++i) {
    const auto& n = graph->node(i);
    for (const auto& out : n.output()) {
      producer_index[out] = i;
    }
  }

  std::vector<Candidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    auto m = MatchMatMulLike(graph->node(i));
    if (!m) {
      continue;
    }
    const std::string codebook_name =
        FindLlmFp4WeightCodebook(m->w_name, *graph, producer_index, init_index);
    if (codebook_name.empty()) {
      continue;
    }
    candidates.push_back(
        {graph->mutable_node(i), m->x_name, m->w_name, codebook_name});
  }
  return candidates;
}

// --- Shared unique-naming / node-insertion machinery --------------------
//
// Transcribed from billm_entry.cpp's own identical block (taken_names,
// unique_name, append_at -- see that file's own comment on why
// append_at's SwapElements-based insertion, not InsertEmptyNodeAt's
// content-Swap, is used: it preserves NodeProto* identity across a later
// candidate's own insertion).

struct GraphNamer {
  std::unordered_set<std::string> taken_names;

  explicit GraphNamer(const onnx::GraphProto& graph) {
    for (const auto& t : graph.initializer()) {
      taken_names.insert(t.name());
    }
    for (const auto& vi : graph.input()) {
      taken_names.insert(vi.name());
    }
    for (const auto& vi : graph.output()) {
      taken_names.insert(vi.name());
    }
    for (const auto& vi : graph.value_info()) {
      taken_names.insert(vi.name());
    }
    for (const auto& n : graph.node()) {
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
  }

  std::string UniqueName(const std::string& base) {
    std::string name = base;
    int i = 0;
    while (taken_names.count(name) != 0) {
      ++i;
      name = base + "_" + std::to_string(i);
    }
    taken_names.insert(name);
    return name;
  }
};

onnx::NodeProto* AppendAt(onnx::GraphProto* graph, const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::string& output, const std::string& name,
                          int target_index) {
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type(op_type);
  for (const auto& in : inputs) {
    node->add_input(in);
  }
  node->add_output(output);
  node->set_name(name);
  auto* nodes = graph->mutable_node();
  for (int i = nodes->size() - 1; i > target_index; --i) {
    nodes->SwapElements(i, i - 1);
  }
  return node;
}

int CurrentIndex(onnx::GraphProto* graph, onnx::NodeProto* node) {
  auto* nodes = graph->mutable_node();
  for (int i = 0; i < nodes->size(); ++i) {
    if (nodes->Mutable(i) == node) {
      return i;
    }
  }
  return -1;  // Unreachable: `node` is always still live in `graph`.
}

// --- Calibration: flat raveled activation values (NOT row/K-shaped) ----
//
// Unlike gptq_entry.cpp's/billm_entry.cpp's own AccumulateActivationRows
// (rank >= 2 required, K-width tracked), llm_fp4.py's own per-tensor fit
// ravels every captured floating-point value unconditionally -- see this
// file's own header comment.

std::unordered_map<std::string, std::vector<double>> AccumulateFlatActivations(
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
            "ApplyLlmFp4ActivationQuantizationPerTensor: calibration batch "
            "is missing required graph input '" +
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
      if (tp.data_type() != onnx::TensorProto::FLOAT) {
        continue;
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      std::vector<double>& acc = result[name];
      acc.reserve(acc.size() + data.size());
      for (float v : data) {
        acc.push_back(static_cast<double>(v));
      }
    }
  }
  return result;
}

// Nearest-codebook-value squared error for one candidate scale, summed
// over `values` -- the single-flat-group specialization of llm_fp4.py's
// own _search_fp4_clip_ratio (that function's `group_shape == ()` case).
double FlatGroupError(const std::vector<double>& values, double scale,
                      const std::vector<double>& codebook) {
  double err = 0.0;
  for (double v : values) {
    const double normalized = v / scale;
    double best_diff = std::numeric_limits<double>::infinity();
    double best_c = codebook[0];
    for (double c : codebook) {
      const double diff = std::fabs(normalized - c);
      if (diff < best_diff) {
        best_diff = diff;
        best_c = c;
      }
    }
    const double d = best_c - normalized;
    err += d * d;
  }
  return err * scale * scale;
}

// Fits a single real-valued scale for one flattened activation sample --
// the whole-tensor-as-one-group specialization of llm_fp4.py's own
// _search_fp4_clip_ratio, reusing its exact
// `scale = max_abs * ratio / max(codebook)` candidate construction.
double FitFlatClipScale(const std::vector<double>& values, double max_abs,
                        const std::vector<double>& codebook,
                        const std::vector<double>& ratios) {
  const double max_mag = codebook.back();
  double best_error = std::numeric_limits<double>::infinity();
  double best_scale = 0.0;
  for (double r : ratios) {
    const double scale = std::max(max_abs * r / max_mag, 1e-30);
    const double err = FlatGroupError(values, scale, codebook);
    if (err < best_error) {
      best_error = err;
      best_scale = scale;
    }
  }
  return best_scale;
}

}  // namespace

onnx::ModelProto ApplyLlmFp4ActivationQuantization(
    const onnx::ModelProto& model, double epsilon) {
  onnx::ModelProto out = model;
  if (!HasMinOpset(out, 18)) {
    return out;
  }
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  const std::vector<Candidate> candidates = FindCandidates(graph, init_index);
  if (candidates.empty()) {
    return out;
  }

  GraphNamer namer(*graph);

  const std::vector<int64_t> neg_one = {-1};
  const std::string axes_last_name = namer.UniqueName("llmfp4act_axes_last");
  SetRawInitializer(graph->add_initializer(), axes_last_name,
                    onnx::TensorProto::INT64, {1}, neg_one.data(),
                    sizeof(int64_t), sizeof(int64_t));
  const float eps_f = static_cast<float>(epsilon);
  const std::string eps_name = namer.UniqueName("llmfp4act_eps");
  SetRawInitializer(graph->add_initializer(), eps_name,
                    onnx::TensorProto::FLOAT, {}, &eps_f, sizeof(eps_f),
                    sizeof(float));

  std::unordered_map<std::string, std::string> codebook_max_names;
  for (const auto& c : candidates) {
    if (codebook_max_names.count(c.codebook_name) != 0) {
      continue;
    }
    const onnx::TensorProto& cb_init =
        graph->initializer(init_index[c.codebook_name]);
    const std::vector<float> cb = ReadFloatTensor(cb_init);
    float max_abs = 0.0f;
    for (float v : cb) {
      max_abs = std::max(max_abs, std::fabs(v));
    }
    const std::string max_name = namer.UniqueName(c.codebook_name + "_maxabs");
    SetRawInitializer(graph->add_initializer(), max_name,
                      onnx::TensorProto::FLOAT, {}, &max_abs, sizeof(max_abs),
                      sizeof(float));
    codebook_max_names.emplace(c.codebook_name, max_name);
  }

  for (const auto& c : candidates) {
    const std::string codebook_max_name = codebook_max_names[c.codebook_name];
    const std::string prefix = c.w_name + "_llmfp4act";

    int insertion_point = CurrentIndex(graph, c.node);

    auto add =
        [&](const std::string& op_type, const std::vector<std::string>& inputs,
            const std::string& suffix,
            const std::vector<std::pair<std::string, int64_t>>& attrs = {}) {
          const std::string output = namer.UniqueName(prefix + "_" + suffix);
          const std::string name =
              namer.UniqueName(prefix + "_" + suffix + "_node");
          onnx::NodeProto* node =
              AppendAt(graph, op_type, inputs, output, name, insertion_point);
          ++insertion_point;
          for (const auto& [attr_name, attr_value] : attrs) {
            AddIntAttribute(node, attr_name, attr_value);
          }
          return output;
        };

    const std::string x_abs = add("Abs", {c.x_name}, "x_abs");
    const std::string x_max =
        add("ReduceMax", {x_abs, axes_last_name}, "x_max", {{"keepdims", 1}});
    const std::string x_safe_max = add("Max", {x_max, eps_name}, "x_safe_max");
    const std::string x_scale =
        add("Div", {x_safe_max, codebook_max_name}, "x_scale");
    const std::string x_norm = add("Div", {c.x_name, x_scale}, "x_norm");
    const std::string x_norm_unsq =
        add("Unsqueeze", {x_norm, axes_last_name}, "x_norm_unsq");
    const std::string diff = add("Sub", {x_norm_unsq, c.codebook_name}, "diff");
    const std::string diff_abs = add("Abs", {diff}, "diff_abs");
    const std::string nearest_idx = add("ArgMin", {diff_abs}, "nearest_idx",
                                        {{"axis", -1}, {"keepdims", 0}});
    const std::string nearest_val = add(
        "Gather", {c.codebook_name, nearest_idx}, "nearest_val", {{"axis", 0}});
    const std::string x_dequant =
        add("Mul", {nearest_val, x_scale}, "x_dequant");

    c.node->set_input(0, x_dequant);
  }

  return out;
}

onnx::ModelProto ApplyLlmFp4ActivationQuantizationPerTensor(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    std::optional<std::vector<double>> clip_ratios) {
  onnx::ModelProto out = model;
  if (!HasMinOpset(out, 13)) {
    return out;
  }
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  const std::vector<Candidate> candidates = FindCandidates(graph, init_index);
  if (candidates.empty()) {
    return out;
  }

  std::vector<double> ratios;
  if (clip_ratios.has_value()) {
    ratios = *clip_ratios;
  } else {
    ratios.resize(17);
    for (int i = 0; i < 17; ++i) {
      ratios[static_cast<size_t>(i)] =
          0.5 + static_cast<double>(i) * 0.5 / 16.0;
    }
  }
  if (ratios.empty()) {
    throw std::invalid_argument(
        "ApplyLlmFp4ActivationQuantizationPerTensor: clip_ratios must be "
        "non-empty");
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  const std::unordered_map<std::string, std::vector<double>> activations =
      AccumulateFlatActivations(executor, out, probe_names, calibration_data);

  // Every scale is fit before the graph is touched at all -- mirrors
  // apply_llm_fp4_activation_quantization_per_tensor's own "Fit every
  // scale before touching the graph" comment exactly.
  struct Fitted {
    onnx::NodeProto* node;
    std::string x_name;
    std::string w_name;
    std::string codebook_name;
    float scale;
  };
  std::vector<Fitted> fitted;
  std::unordered_map<std::string, std::vector<double>> codebook_cache;
  for (const auto& c : candidates) {
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || ait->second.empty()) {
      continue;  // No calibration batch reached this activation.
    }
    std::vector<double> values;
    values.reserve(ait->second.size());
    for (double v : ait->second) {
      if (std::isfinite(v)) {
        values.push_back(v);
      }
    }
    if (values.empty()) {
      continue;
    }
    double max_abs = 0.0;
    for (double v : values) {
      max_abs = std::max(max_abs, std::fabs(v));
    }
    if (!(max_abs > 0.0)) {
      continue;  // All-zero activation has no meaningful scale.
    }

    std::vector<double>& codebook = codebook_cache[c.codebook_name];
    if (codebook.empty()) {
      const std::vector<float> cb =
          ReadFloatTensor(graph->initializer(init_index[c.codebook_name]));
      codebook.assign(cb.begin(), cb.end());
    }

    std::vector<double> fit_values = values;
    if (fit_values.size() > kPerTensorFitMaxElements) {
      // See this file's own header comment ("ACCEPTED, PERMANENT
      // DIVERGENCE") for why this deterministic truncation, not
      // llm_fp4.py's own RNG-based subsample, is used past the cap.
      fit_values.resize(kPerTensorFitMaxElements);
    }
    const double scale =
        FitFlatClipScale(fit_values, max_abs, codebook, ratios);
    if (!(scale > 0.0) || !std::isfinite(scale)) {
      continue;
    }
    fitted.push_back({c.node, c.x_name, c.w_name, c.codebook_name,
                      static_cast<float>(scale)});
  }
  if (fitted.empty()) {
    return out;
  }

  GraphNamer namer(*graph);
  const std::vector<int64_t> neg_one = {-1};
  const std::string axes_last_name = namer.UniqueName("llmfp4act_pt_axes_last");
  SetRawInitializer(graph->add_initializer(), axes_last_name,
                    onnx::TensorProto::INT64, {1}, neg_one.data(),
                    sizeof(int64_t), sizeof(int64_t));

  for (const auto& f : fitted) {
    const std::string prefix = f.w_name + "_llmfp4act_pt";
    const std::string scale_name = namer.UniqueName(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {}, &f.scale, sizeof(f.scale),
                      sizeof(float));

    int insertion_point = CurrentIndex(graph, f.node);
    auto add =
        [&](const std::string& op_type, const std::vector<std::string>& inputs,
            const std::string& suffix,
            const std::vector<std::pair<std::string, int64_t>>& attrs = {}) {
          const std::string output = namer.UniqueName(prefix + "_" + suffix);
          const std::string name =
              namer.UniqueName(prefix + "_" + suffix + "_node");
          onnx::NodeProto* node =
              AppendAt(graph, op_type, inputs, output, name, insertion_point);
          ++insertion_point;
          for (const auto& [attr_name, attr_value] : attrs) {
            AddIntAttribute(node, attr_name, attr_value);
          }
          return output;
        };

    const std::string x_norm = add("Div", {f.x_name, scale_name}, "x_norm");
    const std::string x_norm_unsq =
        add("Unsqueeze", {x_norm, axes_last_name}, "x_norm_unsq");
    const std::string diff = add("Sub", {x_norm_unsq, f.codebook_name}, "diff");
    const std::string diff_abs = add("Abs", {diff}, "diff_abs");
    const std::string nearest_idx = add("ArgMin", {diff_abs}, "nearest_idx",
                                        {{"axis", -1}, {"keepdims", 0}});
    const std::string nearest_val = add(
        "Gather", {f.codebook_name, nearest_idx}, "nearest_val", {{"axis", 0}});
    const std::string x_dequant =
        add("Mul", {nearest_val, scale_name}, "x_dequant");

    f.node->set_input(0, x_dequant);
  }

  return out;
}
