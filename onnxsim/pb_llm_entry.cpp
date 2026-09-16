// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See pb_llm_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own protobuf-level, single-model calibration-driven
// shape) and onnxsim/pb_llm.py for the technique this ports.

#include "pb_llm_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
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
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike (itself
// transcribed from smoothquant_entry.cpp's own), mirroring
// onnxsim.quip_sharp._match_matmul_like exactly (pb_llm.py's own matcher).

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
// only -- this pass, like its own Python reference onnxsim.pb_llm, never
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

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// see llm_int8_entry.cpp's own identical helper for the full rationale
// (ties go to the even neighbor, unlike std::round's half-away-from-zero).
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

// Inserts a fresh (empty) node at position `index` (shifting later nodes
// right). Transcribed from llm_int8_entry.cpp's own identical helper.
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

// --- Calibration: per-channel Hessian diagonal (sum of squares) -----------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// gptq_entry.cpp's own AccumulateActivationRows, narrowed to exactly what
// pb_llm.py's own `diag_h = np.sum(x**2, axis=0)` needs: rather than
// keeping every captured row (gptq.py's own Hessian needs the raw rows for
// an outer-product accumulation; pb_llm.py's own diagonal-only score does
// not), this accumulates the per-column sum of squares directly as each
// batch is observed -- mathematically identical to summing the squares of
// gptq_entry.cpp's own concatenated [total_rows, K] buffer, without ever
// materializing it. Every observed 2-D-or-higher FLOAT32 activation is
// flattened to rows (`reshape(-1, K)`, exact, the same reasoning
// gptq_entry.cpp's own ActivationRows already documents); rank < 2
// contributes no rows at all.

struct DiagH {
  std::vector<double> sum_sq;  // [K]; empty when nothing was ever observed.
  int64_t k = -1;
  bool ok = false;
};

void AccumulateDiagH(
    std::unordered_map<std::string, DiagH>& acc, const ModelExecutor& executor,
    const onnx::ModelProto& model,
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
            "ApplyPbLlm: calibration batch is missing "
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
      DiagH& diag = acc[name];
      if (!diag.ok) {
        diag.k = kk;
        diag.sum_sq.assign(static_cast<size_t>(kk), 0.0);
        diag.ok = true;
      } else if (diag.k != kk) {
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (numpy would fail to concatenate).
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      for (int64_t flat = 0, total = static_cast<int64_t>(data.size());
           flat < total; ++flat) {
        const double v = static_cast<double>(data[static_cast<size_t>(flat)]);
        diag.sum_sq[static_cast<size_t>(flat % kk)] += v * v;
      }
    }
  }
}

}  // namespace

onnx::ModelProto ApplyPbLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double salient_ratio) {
  onnx::ModelProto out = model;
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
  std::unordered_map<std::string, DiagH> diag_h_by_name;
  AccumulateDiagH(diag_h_by_name, executor, probe_model, probe_names,
                  calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same established block every calibration-
  // driven *_entry.cpp in this repo re-derives locally.
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

  // Candidates are processed in forward node order, matching the
  // reference's own iteration order; each rewrite only INSERTS two new
  // nodes before the matched node (never deletes it, unlike
  // llm_int8_entry.cpp's own decompose-and-delete rewrite) so a
  // candidate's live index is simply its original index plus the net
  // insertions already made by earlier candidates (always +2 each, one
  // Cast plus one Mul).
  int64_t net_insertions = 0;
  for (const auto& c : candidates) {
    auto dit = diag_h_by_name.find(c.x_name);
    if (dit == diag_h_by_name.end() || !dit->second.ok) {
      continue;  // No usable activation (no feature axis); skip.
    }
    const DiagH& diag = dit->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (diag.k != k) {
      continue;  // Activation's feature dim doesn't match K; skip.
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    // w_nk[i, j], row-major over [N, K]: the stored layout is [N, K]
    // itself when transposed, else [K, N] (so column j of w_nk is row j
    // of the stored tensor, strided by N == n_rows, not by K) -- same
    // indexing convention llm_int8_entry.cpp's own at_nk already uses.
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // Salience = mean_n(|W[n, j]|) * diag_h[j] -- see pb_llm_entry.h's own
    // top-of-file comment.
    std::vector<double> col_mag(static_cast<size_t>(k), 0.0);
    for (int64_t j = 0; j < k; ++j) {
      double sum_abs = 0.0;
      for (int64_t n = 0; n < n_rows; ++n) {
        sum_abs += std::fabs(at_nk(n, j));
      }
      col_mag[static_cast<size_t>(j)] =
          n_rows > 0 ? sum_abs / static_cast<double>(n_rows) : 0.0;
    }
    std::vector<double> salience(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      salience[static_cast<size_t>(j)] =
          col_mag[static_cast<size_t>(j)] * diag.sum_sq[static_cast<size_t>(j)];
    }

    // order = argsort(-salience): descending by salience. std::stable_sort
    // keeps ties in their original column order -- not guaranteed
    // identical to numpy's own (unstable) quicksort tie-breaking, but this
    // only matters for an exact salience tie between two columns, a
    // measure-zero event for real floating-point calibration data.
    std::vector<int64_t> order(static_cast<size_t>(k));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      return salience[static_cast<size_t>(a)] >
             salience[static_cast<size_t>(b)];
    });

    int64_t num_salient = static_cast<int64_t>(
        RoundHalfToEven(salient_ratio * static_cast<double>(k)));
    num_salient = std::max<int64_t>(0, std::min(k, num_salient));

    std::vector<bool> is_salient(static_cast<size_t>(k), false);
    for (int64_t idx = 0; idx < num_salient; ++idx) {
      is_salient[static_cast<size_t>(order[static_cast<size_t>(idx)])] = true;
    }

    // Per-column code/scale -- salient columns get per-column symmetric
    // INT8; non-salient columns get a {-1, +1} sign code at a mean-abs
    // scale. Both reconstruct as `code * scale`, so one shared [dim0,
    // dim1]-oriented int8 Code tensor plus a per-column float32 Scale
    // tensor suffices.
    std::vector<int8_t> code_nk(static_cast<size_t>(n_rows * k));
    std::vector<double> scale_k(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      if (is_salient[static_cast<size_t>(j)]) {
        double wmax = 0.0;
        for (int64_t n = 0; n < n_rows; ++n) {
          wmax = std::max(wmax, std::fabs(at_nk(n, j)));
        }
        const double scale = std::max(wmax, 1e-12) / 127.0;
        scale_k[static_cast<size_t>(j)] = scale;
        for (int64_t n = 0; n < n_rows; ++n) {
          const double q = RoundHalfToEven(at_nk(n, j) / scale);
          const double clipped = std::min(127.0, std::max(-127.0, q));
          code_nk[static_cast<size_t>(n * k + j)] =
              static_cast<int8_t>(clipped);
        }
      } else {
        double sum_abs = 0.0;
        for (int64_t n = 0; n < n_rows; ++n) {
          sum_abs += std::fabs(at_nk(n, j));
        }
        const double scale =
            n_rows > 0 ? sum_abs / static_cast<double>(n_rows) : 0.0;
        scale_k[static_cast<size_t>(j)] = scale;
        for (int64_t n = 0; n < n_rows; ++n) {
          code_nk[static_cast<size_t>(n * k + j)] =
              (at_nk(n, j) >= 0.0) ? int8_t{1} : int8_t{-1};
        }
      }
    }

    // code_orig = code_nk if weight_transposed else code_nk.T -- write
    // back into the ORIGINAL [dim0, dim1] storage layout, matching the
    // reference's own `code_orig` exactly.
    std::vector<int8_t> code_orig(static_cast<size_t>(dim0 * dim1));
    if (c.weight_transposed) {
      code_orig = code_nk;  // Already [dim0, dim1] == [n_rows, k].
    } else {
      for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t j = 0; j < k; ++j) {
          code_orig[static_cast<size_t>(j * n_rows + n)] =
              code_nk[static_cast<size_t>(n * k + j)];
        }
      }
    }
    std::vector<float> scale_f(scale_k.size());
    for (size_t i = 0; i < scale_k.size(); ++i) {
      scale_f[i] = static_cast<float>(scale_k[i]);
    }
    // scale_orig = scale_k if weight_transposed else scale_k[:, newaxis]
    // -- see pb_llm_entry.h's own top-of-file comment for why: W stored
    // [N, K] (weight_transposed) broadcasts a length-K scale along its
    // own last axis directly; W stored [K, N] (not weight_transposed)
    // needs a trailing size-1 axis so the scale broadcasts along axis 0
    // instead.
    const std::vector<int64_t> scale_dims = c.weight_transposed
                                                ? std::vector<int64_t>{k}
                                                : std::vector<int64_t>{k, 1};

    const std::string prefix = c.w_name + "_pb_llm";
    const std::string code_name = unique_name(prefix + "_code");
    SetRawInitializer(graph->add_initializer(), code_name,
                      onnx::TensorProto::INT8, {dim0, dim1}, code_orig.data(),
                      code_orig.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, scale_dims, scale_f.data(),
                      scale_f.size() * sizeof(float), sizeof(float));

    const std::string cast_out = unique_name(prefix + "_code_f");
    const std::string cast_node_name = unique_name(prefix + "_code_f_node");
    const std::string dq_out = unique_name(prefix + "_dq");
    const std::string dequant_node_name = unique_name(prefix + "_dequant");

    // Insert Cast then Mul immediately before the matched node -- mirrors
    // the reference's own `graph.node.insert(insertion_point, new_node);
    // insertion_point += 1` loop exactly (Cast lands first, Mul second,
    // both before the original node).
    const int live_index = c.node_index + static_cast<int>(net_insertions);
    InsertEmptyNodeAt(graph, live_index);
    onnx::NodeProto* cast_node = graph->mutable_node(live_index);
    cast_node->set_op_type("Cast");
    cast_node->add_input(code_name);
    cast_node->add_output(cast_out);
    cast_node->set_name(cast_node_name);
    onnx::AttributeProto* to_attr = cast_node->add_attribute();
    to_attr->set_name("to");
    to_attr->set_type(onnx::AttributeProto::INT);
    to_attr->set_i(onnx::TensorProto::FLOAT);

    InsertEmptyNodeAt(graph, live_index + 1);
    onnx::NodeProto* mul_node = graph->mutable_node(live_index + 1);
    mul_node->set_op_type("Mul");
    mul_node->add_input(cast_out);
    mul_node->add_input(scale_name);
    mul_node->add_output(dq_out);
    mul_node->set_name(dequant_node_name);

    // The matched node itself is now two slots further along; rewire
    // every one of its own inputs equal to w_name (in practice exactly
    // one) to the freshly dequantized value, matching the reference's own
    // `for i, inp in enumerate(node.input): if inp == w_name: node.input[i]
    // = dq_out` loop exactly.
    onnx::NodeProto* matched_node = graph->mutable_node(live_index + 2);
    for (int i = 0; i < matched_node->input_size(); ++i) {
      if (matched_node->input(i) == c.w_name) {
        matched_node->set_input(i, dq_out);
      }
    }

    net_insertions += 2;
  }

  return out;
}
