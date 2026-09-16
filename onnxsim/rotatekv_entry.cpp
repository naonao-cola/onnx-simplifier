// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See rotatekv_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own single-model, calibration-driven, protobuf-level
// shape) and onnxsim/rotatekv.py for the technique this ports.
//
// EIGENDECOMPOSITION CHOICE: this port needs an orthonormal eigenvector
// basis of a small (head_dim x head_dim) real symmetric covariance
// matrix. No linear-algebra library is linked into this codebase (checked
// the same way low_rank_compensation_entry.cpp's own top-of-file comment
// already documents for its own SVD need), and unlike that file's own
// fallback -- an economy SVD of the raw (potentially tall) calibration
// data matrix, whose right singular vectors give the SAME eigenvectors a
// covariance's own eigendecomposition would, via X^T X == V S^2 V^T -- an
// ECONOMY SVD only returns min(num_samples, head_dim) basis vectors,
// which is NOT a complete head_dim x head_dim orthogonal matrix whenever
// the calibration sample count is smaller than head_dim (a real
// possibility this port cannot assume away, since rotatekv.py's own
// reference explicitly documents that np.linalg.eigh "always returns an
// orthonormal basis for any real symmetric matrix... exact and
// well-defined even for a rank-deficient (few-sample) covariance" -- a
// guarantee this port must preserve exactly). So this file instead
// hand-rolls a classical (cyclic) Jacobi eigenvalue algorithm applied
// DIRECTLY to the head_dim x head_dim covariance matrix itself (repeated
// sweeps of Givens rotations zeroing every off-diagonal pair in turn,
// the same well-known reference algorithm Numerical Recipes' own
// `jacobi()` routine implements), which is complete and well-defined
// for ANY real symmetric input regardless of how it was constructed.
//
// ACCEPTED, PERMANENT DIVERGENCE: see rotatekv_entry.h's own top-of-file
// comment -- this port's own Jacobi eigenvector basis is not expected to
// match numpy's own LAPACK-backed np.linalg.eigh sign-for-sign or
// column-order-for-column-order (eigenvectors are unique only up to a
// sign flip per vector, and up to rotation within a repeated eigenvalue's
// own subspace), but rotatekv.py's own exactness argument holds for ANY
// orthogonal R, so this does not affect correctness -- only which
// specific (equally valid) rotation gets baked into the graph.

#include "rotatekv_entry.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"

namespace {

// --- Tensor <-> flat buffers, protobuf level --------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helper (FLOAT32
// only -- this pass, like its own Python reference onnxsim.rotatekv,
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

// --- KV-cache Concat(past, new, axis=seq) matching, protobuf level --------
//
// Direct transcription of onnxsim.kv_cache_quantization's own
// _find_kv_cache_candidates (reused, unmodified, by rotatekv.py itself) --
// see onnxsim/passes/intactkv.h's own FindCandidate for the same
// structural match already established in this codebase's onnx-optimizer
// IR; this is the raw-protobuf-level re-derivation every calibration-
// driven *_entry.cpp in this repo uses instead.
struct KvCacheCandidate {
  std::string past_name;
  std::string present_name;  // == concat_node's own output(0).
  onnx::NodeProto* concat_node = nullptr;
  std::string new_name;  // the freshly-computed operand of Concat.
  bool new_is_first_input = false;
  int64_t seq_axis = 0;
  int64_t channel_axis = 0;
};

std::vector<KvCacheCandidate> FindKvCacheCandidates(onnx::GraphProto* graph) {
  std::unordered_set<std::string> output_names;
  for (const auto& o : graph->output()) {
    output_names.insert(o.name());
  }

  std::unordered_map<std::string, int64_t> float_input_rank;
  for (const auto& inp : graph->input()) {
    if (inp.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT) {
      continue;
    }
    float_input_rank[inp.name()] =
        static_cast<int64_t>(inp.type().tensor_type().shape().dim_size());
  }

  std::unordered_map<std::string, int64_t> consumer_count;
  for (const auto& node : graph->node()) {
    for (const auto& inp : node.input()) {
      consumer_count[inp] += 1;
    }
  }

  std::vector<KvCacheCandidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (node->op_type() != "Concat" || node->input_size() != 2) {
      continue;
    }
    if (node->output_size() != 1 || !output_names.count(node->output(0))) {
      continue;
    }
    const std::string& a = node->input(0);
    const std::string& b = node->input(1);
    std::string past_name, new_name;
    bool new_is_first = false;
    if (float_input_rank.count(a) && consumer_count[a] == 1) {
      past_name = a;
      new_name = b;
      new_is_first = false;
    } else if (float_input_rank.count(b) && consumer_count[b] == 1) {
      past_name = b;
      new_name = a;
      new_is_first = true;
    } else {
      continue;
    }

    bool has_axis = false;
    int64_t axis_val = 0;
    for (const auto& attr : node->attribute()) {
      if (attr.name() == "axis") {
        has_axis = true;
        axis_val = attr.i();
        break;
      }
    }
    if (!has_axis) {
      continue;
    }
    const int64_t rank = float_input_rank[past_name];
    const int64_t seq_axis = axis_val >= 0 ? axis_val : axis_val + rank;
    const int64_t channel_axis = rank - 1;
    if (seq_axis == channel_axis) {
      continue;  // No distinct channel axis left.
    }

    KvCacheCandidate c;
    c.past_name = past_name;
    c.present_name = node->output(0);
    c.concat_node = node;
    c.new_name = new_name;
    c.new_is_first_input = new_is_first;
    c.seq_axis = seq_axis;
    c.channel_axis = channel_axis;
    candidates.push_back(std::move(c));
  }
  return candidates;
}

// --- Decomposed attention subgraph matching, protobuf level ----------------
//
// Direct transcription of onnxsim.attention_quantization's own
// _find_matmul_producer/_find_attention_candidates (reused, unmodified, by
// rotatekv.py itself) -- see onnxsim/passes/attention_quantization.h's own
// FindQKMatMulProducer/MatchAttentionQuantization for the same structural
// match already established in this codebase's onnx-optimizer IR; this is
// the raw-protobuf-level re-derivation.
onnx::NodeProto* FindMatMulProducer(
    const std::string& name,
    const std::unordered_map<std::string, onnx::NodeProto*>& producer_by_output,
    int hops_left) {
  auto it = producer_by_output.find(name);
  if (it == producer_by_output.end()) {
    return nullptr;
  }
  onnx::NodeProto* node = it->second;
  if (node->op_type() == "MatMul") {
    return node;
  }
  if (hops_left <= 0 || (node->op_type() != "Mul" && node->op_type() != "Div" &&
                         node->op_type() != "Add")) {
    return nullptr;
  }
  if (node->input_size() == 0) {
    return nullptr;
  }
  return FindMatMulProducer(node->input(0), producer_by_output, hops_left - 1);
}

struct AttentionCandidate {
  onnx::NodeProto* qk_matmul = nullptr;
  onnx::NodeProto* softmax = nullptr;
  onnx::NodeProto* out_matmul = nullptr;
};

std::vector<AttentionCandidate> FindAttentionCandidates(
    onnx::GraphProto* graph) {
  std::unordered_map<std::string, onnx::NodeProto*> producer_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    for (const auto& out : n->output()) {
      producer_by_output[out] = n;
    }
  }
  std::unordered_map<std::string, std::vector<onnx::NodeProto*>>
      consumers_by_input;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    for (const auto& inp : n->input()) {
      consumers_by_input[inp].push_back(n);
    }
  }

  std::vector<AttentionCandidate> candidates;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* node = graph->mutable_node(i);
    if (node->op_type() != "Softmax" || node->input_size() < 1) {
      continue;
    }
    onnx::NodeProto* qk =
        FindMatMulProducer(node->input(0), producer_by_output, 2);
    if (qk == nullptr || node->output_size() < 1) {
      continue;
    }
    const std::string& softmax_out = node->output(0);
    onnx::NodeProto* out_matmul = nullptr;
    auto cit = consumers_by_input.find(softmax_out);
    if (cit != consumers_by_input.end()) {
      for (onnx::NodeProto* c : cit->second) {
        if (c->op_type() == "MatMul" && c->input_size() >= 1 &&
            c->input(0) == softmax_out) {
          out_matmul = c;
          break;
        }
      }
    }
    if (out_matmul == nullptr) {
      continue;
    }
    candidates.push_back({qk, node, out_matmul});
  }
  return candidates;
}

// Unwraps at most one Transpose hop: Kt = Transpose(X) resolves to X.
// Direct transcription of rotatekv.py's own _resolve_kt_source.
std::string ResolveKtSource(
    const std::string& kt_name,
    const std::unordered_map<std::string, onnx::NodeProto*>&
        producer_by_output) {
  auto it = producer_by_output.find(kt_name);
  if (it != producer_by_output.end() && it->second->op_type() == "Transpose" &&
      it->second->input_size() == 1) {
    return it->second->input(0);
  }
  return kt_name;
}

struct RotateKvTarget {
  KvCacheCandidate kv;
  std::string q_name;
  onnx::NodeProto* qk_matmul = nullptr;
};

// Direct transcription of rotatekv.py's own _find_rotatekv_targets:
// Key-style KV-cache candidates (present-output name without ".value" --
// rotatekv.py never exposes kv_cache_quantization.py's own
// value_output_names override, so this port doesn't either) whose own
// present-output name, resolved through at most one Transpose hop, is
// exactly some attention candidate's own Kt operand.
std::vector<RotateKvTarget> FindRotateKvTargets(onnx::GraphProto* graph) {
  std::vector<KvCacheCandidate> kv_all = FindKvCacheCandidates(graph);
  std::vector<KvCacheCandidate> kv_candidates;
  for (auto& c : kv_all) {
    if (c.present_name.find(".value") == std::string::npos) {
      kv_candidates.push_back(std::move(c));
    }
  }
  if (kv_candidates.empty()) {
    return {};
  }

  std::unordered_map<std::string, onnx::NodeProto*> producer_by_output;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    for (const auto& out : n->output()) {
      producer_by_output[out] = n;
    }
  }

  std::vector<AttentionCandidate> attn_all = FindAttentionCandidates(graph);
  std::unordered_set<onnx::NodeProto*> seen_qk;
  std::vector<AttentionCandidate> attention_candidates;
  for (auto& a : attn_all) {
    if (seen_qk.count(a.qk_matmul)) {
      continue;
    }
    seen_qk.insert(a.qk_matmul);
    attention_candidates.push_back(a);
  }

  std::unordered_map<onnx::NodeProto*, std::string> kt_source_by_matmul;
  for (auto& a : attention_candidates) {
    if (a.qk_matmul->input_size() < 2) {
      continue;
    }
    kt_source_by_matmul[a.qk_matmul] =
        ResolveKtSource(a.qk_matmul->input(1), producer_by_output);
  }

  std::vector<RotateKvTarget> targets;
  for (auto& c : kv_candidates) {
    for (auto& a : attention_candidates) {
      auto it = kt_source_by_matmul.find(a.qk_matmul);
      if (it != kt_source_by_matmul.end() && it->second == c.present_name) {
        if (a.qk_matmul->input_size() >= 1) {
          targets.push_back({c, a.qk_matmul->input(0), a.qk_matmul});
        }
        break;  // Mirrors Python's own next(...) -- first match only.
      }
    }
  }
  return targets;
}

// --- Calibration: per-stream covariance (sum of outer products) -----------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as every
// other calibration-driven *_entry.cpp in this repo, narrowed to exactly
// what apply_rotatekv needs: the sum of x*x^T over every observed row,
// plus the total row count, so cov = sum / rows can be formed once at the
// end (mathematically identical to concatenating every batch first, the
// same reasoning gptq_entry.cpp's own Hessian accumulation already
// documents). UNLIKE every other calibration-driven port in this repo
// (which requires rank >= 2, mirroring onnxsim.bias_correction.
// _activation_rows), rotatekv.py's own capture logic (`if arr.ndim == 0:
// continue`) also accepts a RANK-1 observed tensor (reshaped to a single
// [1, last_dim] "row"), so this port does too -- a plain "ndim != 0"
// check, not "ndim >= 2".
struct CalibAccum {
  std::vector<double> hxx;  // head_dim * head_dim, row-major.
  int64_t head_dim = -1;
  int64_t rows = 0;
  bool ok = false;
};

void AccumulateCovariance(
    std::unordered_map<std::string, CalibAccum>& acc,
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
            "ApplyRotateKv: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() == 0) {
        continue;  // FLOAT only; ndim == 0 (a genuine scalar) has no last
                   // axis to reshape against, mirroring `arr.ndim == 0`.
      }
      const int64_t last_dim = tp.dims(static_cast<int>(tp.dims_size() - 1));
      if (last_dim <= 0) {
        continue;
      }
      int64_t numel = 1;
      for (int64_t d : tp.dims()) {
        numel *= d;
      }
      if (numel <= 0) {
        continue;
      }
      const int64_t rows = numel / last_dim;

      CalibAccum& a = acc[name];
      if (!a.ok) {
        a.head_dim = last_dim;
        a.hxx.assign(static_cast<size_t>(last_dim * last_dim), 0.0);
        a.ok = true;
      } else if (a.head_dim != last_dim) {
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (numpy would fail to concatenate).
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      for (int64_t r = 0; r < rows; ++r) {
        const float* row = data.data() + r * last_dim;
        for (int64_t i = 0; i < last_dim; ++i) {
          const double vi = static_cast<double>(row[i]);
          double* dst = a.hxx.data() + i * last_dim;
          for (int64_t j = 0; j < last_dim; ++j) {
            dst[j] += vi * static_cast<double>(row[j]);
          }
        }
      }
      a.rows += rows;
    }
  }
}

// --- Classical (cyclic) Jacobi eigenvalue algorithm, real symmetric -------
//
// See this file's own top-of-file "EIGENDECOMPOSITION CHOICE" comment for
// why this is a from-scratch Jacobi eigensolver rather than an SVD of the
// raw calibration data. Returns V (n x n row-major, columns are
// eigenvectors) -- eigenvalues themselves are not needed by this port's
// own caller, so they are not returned.
struct EighResult {
  std::vector<double> v;
};

EighResult JacobiEigenSymmetric(const std::vector<double>& a_in, int64_t n) {
  std::vector<double> a = a_in;
  std::vector<double> v(static_cast<size_t>(n * n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    v[static_cast<size_t>(i * n + i)] = 1.0;
  }
  if (n <= 1) {
    return EighResult{std::move(v)};
  }

  constexpr double kEps = 1e-14;
  constexpr int kMaxSweeps = 100;
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    double off_norm_sq = 0.0;
    for (int64_t p = 0; p < n; ++p) {
      for (int64_t q = p + 1; q < n; ++q) {
        const double apq = a[static_cast<size_t>(p * n + q)];
        off_norm_sq += apq * apq;
      }
    }
    if (off_norm_sq < kEps) {
      break;
    }
    for (int64_t p = 0; p < n - 1; ++p) {
      for (int64_t q = p + 1; q < n; ++q) {
        const double apq = a[static_cast<size_t>(p * n + q)];
        if (std::fabs(apq) < 1e-300) {
          continue;
        }
        const double app = a[static_cast<size_t>(p * n + p)];
        const double aqq = a[static_cast<size_t>(q * n + q)];
        // Same numerically stable half-angle formula this codebase's own
        // one-sided Jacobi SVD (low_rank_compensation_entry.cpp/
        // kbvq_moe.h) already uses for its own rotations, applied here to
        // a symmetric eigenvalue rotation instead (the classical
        // Golub-Van-Loan / Numerical-Recipes formula).
        const double zeta = (aqq - app) / (2.0 * apq);
        const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = c * t;

        for (int64_t k = 0; k < n; ++k) {
          if (k == p || k == q) {
            continue;
          }
          const double akp = a[static_cast<size_t>(k * n + p)];
          const double akq = a[static_cast<size_t>(k * n + q)];
          const double new_akp = c * akp - s * akq;
          const double new_akq = s * akp + c * akq;
          a[static_cast<size_t>(k * n + p)] = new_akp;
          a[static_cast<size_t>(p * n + k)] = new_akp;
          a[static_cast<size_t>(k * n + q)] = new_akq;
          a[static_cast<size_t>(q * n + k)] = new_akq;
        }
        a[static_cast<size_t>(p * n + p)] = app - t * apq;
        a[static_cast<size_t>(q * n + q)] = aqq + t * apq;
        a[static_cast<size_t>(p * n + q)] = 0.0;
        a[static_cast<size_t>(q * n + p)] = 0.0;

        for (int64_t k = 0; k < n; ++k) {
          const double vkp = v[static_cast<size_t>(k * n + p)];
          const double vkq = v[static_cast<size_t>(k * n + q)];
          v[static_cast<size_t>(k * n + p)] = c * vkp - s * vkq;
          v[static_cast<size_t>(k * n + q)] = s * vkp + c * vkq;
        }
      }
    }
  }

  return EighResult{std::move(v)};
}

}  // namespace

onnx::ModelProto ApplyRotateKv(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  onnx::ModelProto out = model;
  onnx::GraphProto* graph = out.mutable_graph();

  const std::vector<RotateKvTarget> targets = FindRotateKvTargets(graph);
  if (targets.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& t : targets) {
    probe_names.insert(t.kv.new_name);
  }
  std::unordered_map<std::string, CalibAccum> accum;
  AccumulateCovariance(accum, executor, out, probe_names, calibration_data);

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention every calibration-driven
  // *_entry.cpp in this repo re-derives locally.
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

  // Appends a node at the very end, then walks it back down to exactly
  // `target_index` via adjacent pointer-slot swaps -- matches
  // low_rank_compensation_entry.cpp's/billm_entry.cpp's own identical
  // pattern: RepeatedPtrField::SwapElements swaps POINTER SLOTS, never
  // moving or freeing the pointed-to messages, so a NodeProto* captured
  // once at match time (concat_node/qk_matmul, held inside each target)
  // stays valid AND locatable by identity across every later
  // add_node()/SwapElements() call below, for every target processed.
  auto* nodes = graph->mutable_node();
  auto find_index = [&](onnx::NodeProto* target) {
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == target) {
        return i;
      }
    }
    return -1;
  };
  auto append_at = [&](const std::string& op_type,
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
    for (int i = nodes->size() - 1; i > target_index; --i) {
      nodes->SwapElements(i, i - 1);
    }
    return node;
  };

  for (const auto& t : targets) {
    auto ait = accum.find(t.kv.new_name);
    if (ait == accum.end() || !ait->second.ok || ait->second.rows <= 0) {
      continue;  // This stream's activation never appeared in any batch.
    }
    const CalibAccum& a = ait->second;
    const int64_t head_dim = a.head_dim;
    if (head_dim < 2) {
      continue;  // Nothing to rotate.
    }

    std::vector<double> cov(static_cast<size_t>(head_dim * head_dim));
    const double inv_rows = 1.0 / static_cast<double>(a.rows);
    for (size_t i = 0; i < cov.size(); ++i) {
      cov[i] = a.hxx[i] * inv_rows;
    }
    const EighResult eig = JacobiEigenSymmetric(cov, head_dim);

    const std::vector<float> r32(eig.v.begin(), eig.v.end());
    const std::string prefix = t.kv.present_name + "_rotatekv";
    const std::string r_name = unique_name(prefix + "_r");
    {
      onnx::TensorProto* rt = graph->add_initializer();
      rt->Clear();
      rt->set_name(r_name);
      rt->set_data_type(onnx::TensorProto::FLOAT);
      rt->add_dims(head_dim);
      rt->add_dims(head_dim);
      std::string raw(r32.size() * sizeof(float), '\0');
      std::memcpy(raw.data(), r32.data(), raw.size());
      if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
        onnxsim::dlpack::SwapElementBytes(
            reinterpret_cast<uint8_t*>(raw.data()), raw.size(), sizeof(float));
      }
      rt->set_raw_data(std::move(raw));
    }

    // Mirrors apply_rotatekv's own two separate node insertions, each via
    // a fresh identity-based re-lookup (its own `next(i for i, n in
    // enumerate(graph.node) if n is X)` idiom) -- neither index can be
    // cached across the two insertions below, since the first insertion
    // can shift the second target's own current position.
    const std::string new_key_rot_name =
        unique_name(t.kv.new_name + "_rotatekv");
    const int concat_index = find_index(t.kv.concat_node);
    append_at("MatMul", {t.kv.new_name, r_name}, new_key_rot_name,
              unique_name(prefix + "_new_key_node"), concat_index);
    if (t.kv.new_is_first_input) {
      t.kv.concat_node->set_input(0, new_key_rot_name);
    } else {
      t.kv.concat_node->set_input(1, new_key_rot_name);
    }

    const std::string q_rot_name = unique_name(t.q_name + "_rotatekv");
    const int qk_index = find_index(t.qk_matmul);
    append_at("MatMul", {t.q_name, r_name}, q_rot_name,
              unique_name(prefix + "_q_node"), qk_index);
    t.qk_matmul->set_input(0, q_rot_name);
  }

  return out;
}
