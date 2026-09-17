// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See bwa_ptq_entry.h for the full rationale (including why this follows
// billm_entry.h's own single-model, protobuf-level, calibration-driven
// shape) and onnxsim/bwa_ptq.py for the technique this ports.

#include "bwa_ptq_entry.h"

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

// --- MatMul/vanilla-Gemm matching, protobuf level ---------------------
//
// Transcribed from onnxsim.quip_sharp's own _match_matmul_like (which
// bwa_ptq.py itself imports and reuses) -- billm_entry.cpp's own
// MatchMatMulLike is byte-identical (bias, when present, is matched but
// never read or rewritten by either pass); duplicated here per this
// codebase's established "no shared dependency between independently
// tested *_entry.cpp TUs" convention.
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

// --- Tensor <-> flat float buffer, protobuf level -----------------------
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

// --- Calibration: concatenated activation rows -----------------------
//
// Transcribed from billm_entry.cpp's own AccumulateActivationRows.

struct ActivationRows {
  std::vector<double> data;  // [total_rows, K] row-major.
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
            "ApplyBwaPtq: calibration batch is missing "
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

// --- Hessian-weighted two-scale binary EM ------------------------------
//
// Direct transcription of bwa_ptq.py's own _em_two_scale_binary: median
// split seed, then alternating weighted-mean scale update (M-step) and
// nearest-scale reassignment (E-step) until the assignment stops
// changing (or max_iters is reached). `scale0 <= scale1` on return
// (canonicalized by swapping, mirroring the reference exactly).

struct EmResult {
  std::vector<int8_t> assign;  // 0/1, same length as abs_w.
  double scale0 = 0.0;
  double scale1 = 0.0;
};

// numpy's own np.median: average of the two middle elements for an even
// count, the single middle element for an odd one.
double Median(std::vector<double> values) {
  const size_t n = values.size();
  std::sort(values.begin(), values.end());
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

EmResult EmTwoScaleBinary(const std::vector<double>& abs_w,
                          const std::vector<double>& importance,
                          int64_t max_iters) {
  EmResult res;
  const size_t n = abs_w.size();
  res.assign.assign(n, 0);
  if (n == 0) {
    return res;
  }
  const double median = Median(abs_w);
  std::vector<int8_t> assign(n);
  for (size_t i = 0; i < n; ++i) {
    assign[i] = abs_w[i] >= median ? 1 : 0;
  }

  double scale0 = 0.0, scale1 = 0.0;
  for (int64_t iter = 0; iter < max_iters; ++iter) {
    double sum_w0 = 0.0, sum_w0x = 0.0, sum_w1 = 0.0, sum_w1x = 0.0;
    bool any0 = false, any1 = false;
    for (size_t i = 0; i < n; ++i) {
      if (assign[i] == 1) {
        any1 = true;
        sum_w1 += importance[i];
        sum_w1x += importance[i] * abs_w[i];
      } else {
        any0 = true;
        sum_w0 += importance[i];
        sum_w0x += importance[i] * abs_w[i];
      }
    }
    scale0 = any0 ? sum_w0x / std::max(sum_w0, 1e-12) : 0.0;
    scale1 = any1 ? sum_w1x / std::max(sum_w1, 1e-12) : 0.0;

    std::vector<int8_t> new_assign(n);
    bool changed = false;
    for (size_t i = 0; i < n; ++i) {
      const double d0 = abs_w[i] - scale0;
      const double d1 = abs_w[i] - scale1;
      const double err0 = importance[i] * d0 * d0;
      const double err1 = importance[i] * d1 * d1;
      new_assign[i] = (err1 < err0) ? 1 : 0;
      if (new_assign[i] != assign[i]) {
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    assign = new_assign;
  }

  if (scale0 > scale1) {
    for (size_t i = 0; i < n; ++i) {
      assign[i] = static_cast<int8_t>(1 - assign[i]);
    }
    std::swap(scale0, scale1);
  }
  res.assign = std::move(assign);
  res.scale0 = scale0;
  res.scale1 = scale1;
  return res;
}

// --- Per-group binarization over a whole [N, K] weight ------------------
//
// Direct transcription of bwa_ptq.py's own _bwa_quantize_weight: runs
// EmTwoScaleBinary over every group_size-column group of `w_nk` ([n, k],
// output channel first, row-major). A group's own two scales are shared
// by every output row; only sign and group-select vary per element.

struct BwaResult {
  std::vector<int8_t> sign_nk;          // n * k, row-major.
  std::vector<int8_t> group_select_nk;  // n * k, row-major.
  std::vector<double> scale0_g;         // num_groups.
  std::vector<double> scale1_g;         // num_groups.
};

BwaResult BwaQuantizeWeight(const std::vector<double>& w_nk, int64_t n,
                            int64_t k,
                            const std::vector<double>& hessian_diag_k,
                            int64_t group_size, int64_t max_iters) {
  const int64_t num_groups = (k + group_size - 1) / group_size;
  BwaResult res;
  res.sign_nk.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
  res.group_select_nk.assign(static_cast<size_t>(n) * static_cast<size_t>(k),
                             0);
  res.scale0_g.assign(static_cast<size_t>(num_groups), 0.0);
  res.scale1_g.assign(static_cast<size_t>(num_groups), 0.0);

  int64_t gi = 0;
  for (int64_t start = 0; start < k; start += group_size, ++gi) {
    const int64_t end = std::min(start + group_size, k);
    const int64_t gs = end - start;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(gs);
    std::vector<double> abs_flat(total);
    std::vector<double> importance_flat(total);
    std::vector<double> sign_flat(total);
    for (int64_t r = 0; r < n; ++r) {
      for (int64_t c = 0; c < gs; ++c) {
        const double v = w_nk[static_cast<size_t>(r) * static_cast<size_t>(k) +
                              static_cast<size_t>(start + c)];
        const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(gs) +
                           static_cast<size_t>(c);
        sign_flat[idx] = (v >= 0.0) ? 1.0 : -1.0;
        abs_flat[idx] = std::fabs(v);
        importance_flat[idx] = hessian_diag_k[static_cast<size_t>(start + c)];
      }
    }

    const EmResult em = EmTwoScaleBinary(abs_flat, importance_flat, max_iters);

    for (int64_t r = 0; r < n; ++r) {
      for (int64_t c = 0; c < gs; ++c) {
        const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(gs) +
                           static_cast<size_t>(c);
        const size_t out_idx = static_cast<size_t>(r) * static_cast<size_t>(k) +
                               static_cast<size_t>(start + c);
        res.sign_nk[out_idx] = static_cast<int8_t>(sign_flat[idx]);
        res.group_select_nk[out_idx] = em.assign[idx];
      }
    }
    res.scale0_g[static_cast<size_t>(gi)] = em.scale0;
    res.scale1_g[static_cast<size_t>(gi)] = em.scale1;
  }
  return res;
}

}  // namespace

onnx::ModelProto ApplyBwaPtq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t group_size, int64_t max_em_iters) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  std::unordered_map<std::string, int> init_index;
  for (int i = 0; i < graph->initializer_size(); ++i) {
    init_index.emplace(graph->initializer(i).name(), i);
  }

  struct Candidate {
    onnx::NodeProto* node;
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
    candidates.push_back(
        {graph->mutable_node(i), m->x_name, m->w_name, m->weight_transposed});
  }
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, model, probe_names,
                           calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly, the
  // same convention billm_entry.cpp's own identical block uses.
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

  for (const auto& c : candidates) {
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable activation (no feature axis); skip.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }
    const int64_t num_rows =
        k == 0 ? 0 : static_cast<int64_t>(rows.data.size()) / k;

    const std::vector<float> w_flat = ReadFloatTensor(w_init);
    std::vector<double> w_nk(static_cast<size_t>(n) * static_cast<size_t>(k));
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(i) * static_cast<size_t>(k) +
             static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(i) * static_cast<size_t>(k) +
                             static_cast<size_t>(j)]
                    : w_flat[static_cast<size_t>(j) * static_cast<size_t>(n) +
                             static_cast<size_t>(i)]);
      }
    }

    // Hessian DIAGONAL only: hessian_diag_k[j] = sum_r x[r, j]^2.
    std::vector<double> hessian_diag_k(static_cast<size_t>(k), 0.0);
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        const double v =
            rows.data[static_cast<size_t>(r) * static_cast<size_t>(k) +
                      static_cast<size_t>(j)];
        hessian_diag_k[static_cast<size_t>(j)] += v * v;
      }
    }

    const BwaResult res =
        BwaQuantizeWeight(w_nk, n, k, hessian_diag_k, group_size, max_em_iters);

    // Back to the stored [dim0, dim1] layout -- mirrors `sign_orig =
    // sign_nk if weight_transposed else sign_nk.T` exactly.
    std::vector<int8_t> sign_orig(static_cast<size_t>(dim0) *
                                  static_cast<size_t>(dim1));
    std::vector<int8_t> group_orig(sign_orig.size());
    if (c.weight_transposed) {
      sign_orig = res.sign_nk;
      group_orig = res.group_select_nk;
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          sign_orig[static_cast<size_t>(i) * static_cast<size_t>(dim1) +
                    static_cast<size_t>(j)] =
              res.sign_nk[static_cast<size_t>(j) * static_cast<size_t>(k) +
                          static_cast<size_t>(i)];
          group_orig[static_cast<size_t>(i) * static_cast<size_t>(dim1) +
                     static_cast<size_t>(j)] =
              res.group_select_nk[static_cast<size_t>(j) *
                                      static_cast<size_t>(k) +
                                  static_cast<size_t>(i)];
        }
      }
    }

    // scale{0,1}_full_k[c] = scale{0,1}_g[c / group_size] -- mirrors
    // `np.repeat(scale_g, group_size)[:k]` exactly (see this file's own
    // header comment).
    std::vector<float> scale0_f(static_cast<size_t>(k));
    std::vector<float> scale1_f(static_cast<size_t>(k));
    for (int64_t c = 0; c < k; ++c) {
      const size_t g = static_cast<size_t>(c / group_size);
      scale0_f[static_cast<size_t>(c)] = static_cast<float>(res.scale0_g[g]);
      scale1_f[static_cast<size_t>(c)] = static_cast<float>(res.scale1_g[g]);
    }
    const std::vector<int64_t> scale_dims = c.weight_transposed
                                                ? std::vector<int64_t>{k}
                                                : std::vector<int64_t>{k, 1};

    const std::string prefix = c.w_name + "_bwa";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string sign_name = add_const(
        "sign", onnx::TensorProto::INT8, {dim0, dim1}, sign_orig.data(),
        sign_orig.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string group_name = add_const(
        "group_select", onnx::TensorProto::INT8, {dim0, dim1},
        group_orig.data(), group_orig.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string scale0_name = add_const(
        "scale0", onnx::TensorProto::FLOAT, scale_dims, scale0_f.data(),
        scale0_f.size() * sizeof(float), sizeof(float));
    const std::string scale1_name = add_const(
        "scale1", onnx::TensorProto::FLOAT, scale_dims, scale1_f.data(),
        scale1_f.size() * sizeof(float), sizeof(float));

    const std::string sign_f_out = unique_name(prefix + "_sign_f");
    const std::string group_f_out = unique_name(prefix + "_group_f");
    const std::string diff_out = unique_name(prefix + "_scale_diff");
    const std::string sel_out = unique_name(prefix + "_scale_sel");
    const std::string scale_eff_out = unique_name(prefix + "_scale_eff");
    const std::string dq_out = unique_name(prefix + "_dq");

    // Re-finds c.node's own CURRENT index by pointer identity -- mirrors
    // billm_entry.cpp's own identical pattern (see that file's own
    // comment for why SwapElements-based insertion, not a content Swap,
    // preserves NodeProto* identity across an earlier candidate's own
    // insertion).
    auto* nodes = graph->mutable_node();
    int insertion_point = -1;
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == c.node) {
        insertion_point = i;
        break;
      }
    }
    auto append_at = [&](const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::string& output, const std::string& name,
                         int target_index) -> onnx::NodeProto* {
      onnx::NodeProto* node = graph->add_node();
      node->set_op_type(op_type);
      for (const auto& in : inputs) {
        node->add_input(in);
      }
      node->add_output(output);
      node->set_name(name);
      for (int i = nodes->size() - 1; i > target_index; --i) {
        nodes->SwapElements(i, i - 1);
      }
      return node;
    };

    onnx::NodeProto* sign_cast =
        append_at("Cast", {sign_name}, sign_f_out,
                  unique_name(prefix + "_sign_f_node"), insertion_point);
    AddIntAttribute(sign_cast, "to", onnx::TensorProto::FLOAT);
    onnx::NodeProto* group_cast =
        append_at("Cast", {group_name}, group_f_out,
                  unique_name(prefix + "_group_f_node"), insertion_point + 1);
    AddIntAttribute(group_cast, "to", onnx::TensorProto::FLOAT);
    append_at("Sub", {scale1_name, scale0_name}, diff_out,
              unique_name(prefix + "_scale_diff_node"), insertion_point + 2);
    append_at("Mul", {group_f_out, diff_out}, sel_out,
              unique_name(prefix + "_scale_sel_node"), insertion_point + 3);
    append_at("Add", {scale0_name, sel_out}, scale_eff_out,
              unique_name(prefix + "_scale_eff_node"), insertion_point + 4);
    append_at("Mul", {sign_f_out, scale_eff_out}, dq_out,
              unique_name(prefix + "_dequant_node"), insertion_point + 5);

    // Rewires every input equal to c.w_name (mirrors `for i, inp in
    // enumerate(node.input): if inp == w_name: node.input[i] = dq_out`
    // exactly).
    for (int i = 0; i < c.node->input_size(); ++i) {
      if (c.node->input(i) == c.w_name) {
        c.node->set_input(i, dq_out);
      }
    }
  }

  return out;
}
