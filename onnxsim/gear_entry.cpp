// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See gear_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own single-model, protobuf-level, calibration-driven
// shape, and why the KV-cache candidate matcher and the Jacobi SVD below
// are each deliberate, file-local copies rather than shared dependencies)
// and onnxsim/gear.py for the technique this ports.

#include "gear_entry.h"

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

// --- KV-cache Concat(past, new, axis=seq) matching, protobuf level --------
//
// A fresh, protobuf-level transcription of
// onnxsim.kv_cache_quantization._find_kv_cache_candidates (gear.py itself
// imports and reuses that exact function) -- NOT a dependency on any C++
// port of kv_cache_quantization.py, which may or may not exist yet (see
// gear_entry.h's own top-of-file comment). onnxsim/passes/intactkv.h
// (already merged) reimplements the same structural shape against
// onnx-optimizer's own Node/Value IR for its own narrower purpose; this
// file needs the raw-protobuf equivalent, plus channel_axis (which
// IntactKV never resolves, since it has no per-channel quantization step).
struct KvCacheCandidate {
  std::string past_name;     // graph input name.
  std::string present_name;  // graph output name (Concat's own output).
  int concat_node_index;     // ORIGINAL index into graph.node().
  std::string new_name;      // the freshly-computed operand of Concat.
  bool new_is_first_input;
  int64_t seq_axis;
  int64_t channel_axis;
};

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  return axis >= 0 ? axis : axis + rank;
}

std::vector<KvCacheCandidate> FindKvCacheCandidates(
    const onnx::GraphProto& graph) {
  std::unordered_set<std::string> output_names;
  for (const auto& o : graph.output()) {
    output_names.insert(o.name());
  }
  std::unordered_map<std::string, int64_t> float_input_rank;
  for (const auto& inp : graph.input()) {
    if (inp.type().tensor_type().elem_type() != onnx::TensorProto::FLOAT) {
      continue;
    }
    float_input_rank[inp.name()] =
        static_cast<int64_t>(inp.type().tensor_type().shape().dim_size());
  }
  std::unordered_map<std::string, int64_t> consumer_count;
  for (const auto& node : graph.node()) {
    for (const auto& inp : node.input()) {
      consumer_count[inp] += 1;
    }
  }

  std::vector<KvCacheCandidate> candidates;
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& node = graph.node(i);
    if (node.op_type() != "Concat" || node.input_size() != 2) {
      continue;
    }
    if (node.output_size() != 1 || output_names.count(node.output(0)) == 0) {
      continue;
    }
    const std::string& a = node.input(0);
    const std::string& b = node.input(1);
    std::string past_name, new_name;
    bool new_is_first = false;
    auto a_it = float_input_rank.find(a);
    auto b_it = float_input_rank.find(b);
    if (a_it != float_input_rank.end() && consumer_count[a] == 1) {
      past_name = a;
      new_name = b;
      new_is_first = false;
    } else if (b_it != float_input_rank.end() && consumer_count[b] == 1) {
      past_name = b;
      new_name = a;
      new_is_first = true;
    } else {
      continue;
    }

    const onnx::AttributeProto* axis_attr = nullptr;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "axis") {
        axis_attr = &attr;
        break;
      }
    }
    if (axis_attr == nullptr) {
      continue;
    }
    const int64_t rank = float_input_rank[past_name];
    const int64_t seq_axis = ResolveAxis(axis_attr->i(), rank);
    const int64_t channel_axis = rank - 1;
    if (seq_axis == channel_axis) {
      continue;  // No distinct channel axis left to quantize per-channel.
    }
    candidates.push_back({past_name, node.output(0), i, new_name, new_is_first,
                          seq_axis, channel_axis});
  }
  return candidates;
}

// --- Tensor <-> flat buffers, protobuf level -------------------------------
//
// Transcribed from llm_int8_entry.cpp's own identical helpers (FLOAT32
// only -- this pass, like its own Python reference onnxsim.gear, never
// widens to FLOAT16/BFLOAT16).

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

// Round-half-to-even (banker's rounding), matching numpy's own `round` --
// see llm_int8_entry.cpp's own identical helper for the full rationale.
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

// --- Calibration: concatenated activation rows -----------------------------
//
// Same probe-injection/batch-iteration/DLPack-crossing shape as
// llm_int8_entry.cpp's own capture, generalized to any rank >= 1 (gear.py's
// own `arr.reshape(-1, arr.shape[-1])` skips only a genuine 0-d scalar) --
// every observed activation is flattened to `[rows, head_dim]` by
// collapsing every leading dimension, exact.
struct ActivationRows {
  std::vector<double> data;  // Concatenated [total_rows, head_dim], row-major.
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
            "ApplyGear: calibration batch is missing "
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
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() < 1) {
        continue;  // Mirrors `if arr.ndim == 0: continue`.
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
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (numpy would fail to concatenate).
      }
      int64_t numel = 1;
      for (int64_t d : tp.dims()) {
        numel *= d;
      }
      std::vector<float> flat(static_cast<size_t>(numel));
      if (tp.has_raw_data()) {
        std::memcpy(flat.data(), tp.raw_data().data(),
                    flat.size() * sizeof(float));
        if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
          onnxsim::dlpack::SwapElementBytes(
              reinterpret_cast<uint8_t*>(flat.data()),
              flat.size() * sizeof(float), sizeof(float));
        }
      } else {
        for (int64_t i = 0; i < numel; ++i) {
          flat[static_cast<size_t>(i)] = tp.float_data(static_cast<int>(i));
        }
      }
      rows.data.reserve(rows.data.size() + flat.size());
      for (float v : flat) {
        rows.data.push_back(static_cast<double>(v));
      }
    }
  }
}

// --- One-sided (Hestenes) Jacobi SVD -------------------------------------
//
// Deliberate, file-local TRANSCRIBED COPY of
// low_rank_compensation_entry.cpp's own identical JacobiSvdTall/EconomySvd
// (no linear-algebra library is linked into this codebase) -- mirrors the
// exact same "don't refactor an already-tested file as a side effect of an
// unrelated port" convention this session's own kbvq_moe.h/billm_entry.cpp
// already establish for their own local copies of, respectively, that same
// SVD and gptq_entry.cpp's own Cholesky machinery.

struct SvdResult {
  int64_t k = 0;          // = min(m, n): number of columns in u/v.
  std::vector<double> u;  // m x k, row-major.
  std::vector<double> s;  // k, descending.
  std::vector<double> v;  // n x k, row-major.
};

SvdResult JacobiSvdTall(int64_t m, int64_t n, const std::vector<double>& a) {
  std::vector<std::vector<double>> cols(
      static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m)));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      cols[static_cast<size_t>(j)][static_cast<size_t>(i)] =
          a[static_cast<size_t>(i * n + j)];
    }
  }
  std::vector<std::vector<double>> v(
      static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int64_t j = 0; j < n; ++j) {
    v[static_cast<size_t>(j)][static_cast<size_t>(j)] = 1.0;
  }

  constexpr double kEps = 1e-14;
  constexpr int kMaxSweeps = 60;
  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    double max_off = 0.0;
    for (int64_t p = 0; p < n - 1; ++p) {
      for (int64_t q = p + 1; q < n; ++q) {
        auto& cp = cols[static_cast<size_t>(p)];
        auto& cq = cols[static_cast<size_t>(q)];
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (int64_t i = 0; i < m; ++i) {
          alpha += cp[static_cast<size_t>(i)] * cp[static_cast<size_t>(i)];
          beta += cq[static_cast<size_t>(i)] * cq[static_cast<size_t>(i)];
          gamma += cp[static_cast<size_t>(i)] * cq[static_cast<size_t>(i)];
        }
        max_off = std::max(max_off, std::fabs(gamma));
        if (alpha <= 0.0 || beta <= 0.0 ||
            std::fabs(gamma) <= kEps * std::sqrt(alpha * beta)) {
          continue;
        }
        const double zeta = (beta - alpha) / (2.0 * gamma);
        const double t = (zeta >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = c * t;
        for (int64_t i = 0; i < m; ++i) {
          const double vp = cp[static_cast<size_t>(i)];
          const double vq = cq[static_cast<size_t>(i)];
          cp[static_cast<size_t>(i)] = c * vp - s * vq;
          cq[static_cast<size_t>(i)] = s * vp + c * vq;
        }
        auto& vp_col = v[static_cast<size_t>(p)];
        auto& vq_col = v[static_cast<size_t>(q)];
        for (int64_t i = 0; i < n; ++i) {
          const double vp = vp_col[static_cast<size_t>(i)];
          const double vq = vq_col[static_cast<size_t>(i)];
          vp_col[static_cast<size_t>(i)] = c * vp - s * vq;
          vq_col[static_cast<size_t>(i)] = s * vp + c * vq;
        }
      }
    }
    if (max_off < kEps) {
      break;
    }
  }

  std::vector<double> sv(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j) {
    double norm_sq = 0.0;
    for (int64_t i = 0; i < m; ++i) {
      const double x = cols[static_cast<size_t>(j)][static_cast<size_t>(i)];
      norm_sq += x * x;
    }
    sv[static_cast<size_t>(j)] = std::sqrt(norm_sq);
  }
  std::vector<int64_t> order(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j) {
    order[static_cast<size_t>(j)] = j;
  }
  std::sort(order.begin(), order.end(), [&](int64_t a_idx, int64_t b_idx) {
    return sv[static_cast<size_t>(a_idx)] > sv[static_cast<size_t>(b_idx)];
  });

  SvdResult res;
  res.k = n;
  res.s.resize(static_cast<size_t>(n));
  res.u.assign(static_cast<size_t>(m * n), 0.0);
  res.v.assign(static_cast<size_t>(n * n), 0.0);
  for (int64_t out_col = 0; out_col < n; ++out_col) {
    const int64_t src = order[static_cast<size_t>(out_col)];
    const double sigma = sv[static_cast<size_t>(src)];
    res.s[static_cast<size_t>(out_col)] = sigma;
    for (int64_t i = 0; i < m; ++i) {
      const double x = cols[static_cast<size_t>(src)][static_cast<size_t>(i)];
      res.u[static_cast<size_t>(i * n + out_col)] =
          (sigma > 1e-300) ? (x / sigma) : 0.0;
    }
    for (int64_t i = 0; i < n; ++i) {
      res.v[static_cast<size_t>(i * n + out_col)] =
          v[static_cast<size_t>(src)][static_cast<size_t>(i)];
    }
  }
  return res;
}

SvdResult EconomySvd(int64_t m, int64_t n, const std::vector<double>& a) {
  if (m >= n) {
    return JacobiSvdTall(m, n, a);
  }
  std::vector<double> at(static_cast<size_t>(n * m));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      at[static_cast<size_t>(j * m + i)] = a[static_cast<size_t>(i * n + j)];
    }
  }
  SvdResult sub = JacobiSvdTall(n, m, at);
  SvdResult res;
  res.k = sub.k;
  res.u = std::move(sub.v);
  res.v = std::move(sub.u);
  res.s = std::move(sub.s);
  return res;
}

// --- GEAR's own fit: per-channel scale, low-rank projector, sparse mask ---

struct GearFit {
  std::vector<float> scale;  // [head_dim].
  bool has_projector = false;
  std::vector<float> projector;  // [head_dim, head_dim] row-major.
  bool has_sparse_mask = false;
  std::vector<float> sparse_mask;  // [head_dim].
};

// Direct transcription of gear.py's own _fit_gear (one stream's worth --
// the caller loops over every probe name with observed activation rows).
GearFit FitGear(const std::vector<double>& x, int64_t num_rows,
                int64_t head_dim, int64_t rank, double outlier_fraction) {
  GearFit fit;
  fit.scale.assign(static_cast<size_t>(head_dim), 0.0f);

  std::vector<double> channel_absmax(static_cast<size_t>(head_dim), 0.0);
  for (int64_t r = 0; r < num_rows; ++r) {
    for (int64_t c = 0; c < head_dim; ++c) {
      const double v = std::fabs(x[static_cast<size_t>(r * head_dim + c)]);
      channel_absmax[static_cast<size_t>(c)] =
          std::max(channel_absmax[static_cast<size_t>(c)], v);
    }
  }
  std::vector<double> scale_d(static_cast<size_t>(head_dim));
  for (int64_t c = 0; c < head_dim; ++c) {
    scale_d[static_cast<size_t>(c)] =
        std::max(channel_absmax[static_cast<size_t>(c)], 1e-12) / 127.0;
    fit.scale[static_cast<size_t>(c)] =
        static_cast<float>(scale_d[static_cast<size_t>(c)]);
  }

  std::vector<double> residual(static_cast<size_t>(num_rows * head_dim));
  for (int64_t r = 0; r < num_rows; ++r) {
    for (int64_t c = 0; c < head_dim; ++c) {
      const size_t idx = static_cast<size_t>(r * head_dim + c);
      const double s = scale_d[static_cast<size_t>(c)];
      const double code =
          std::min(127.0, std::max(-128.0, RoundHalfToEven(x[idx] / s)));
      const double dequant = code * s;
      residual[idx] = x[idx] - dequant;
    }
  }

  const int64_t r = std::max<int64_t>(0, std::min({rank, head_dim, num_rows}));
  std::vector<double> remainder = residual;  // Default: no low-rank term.
  if (r > 0) {
    const SvdResult svd = EconomySvd(num_rows, head_dim, residual);
    // v_r = V[:, :r], the SAME [head_dim, k] row-major layout svd.v
    // already stores (v_r[d][j] = V[d, j] = svd.v[d * svd.k + j]) -- no
    // transpose needed, unlike kbvq_moe.h's own KltBasis (which stores its
    // own basis transposed, [r, dim_d]).
    std::vector<double> projector_d(static_cast<size_t>(head_dim * head_dim),
                                    0.0);
    for (int64_t i = 0; i < head_dim; ++i) {
      for (int64_t j = 0; j < head_dim; ++j) {
        double acc = 0.0;
        for (int64_t l = 0; l < r; ++l) {
          acc += svd.v[static_cast<size_t>(i * svd.k + l)] *
                 svd.v[static_cast<size_t>(j * svd.k + l)];
        }
        projector_d[static_cast<size_t>(i * head_dim + j)] = acc;
      }
    }
    fit.has_projector = true;
    fit.projector.assign(static_cast<size_t>(head_dim * head_dim), 0.0f);
    for (size_t i = 0; i < projector_d.size(); ++i) {
      fit.projector[i] = static_cast<float>(projector_d[i]);
    }

    // remainder = residual - residual @ projector.
    for (int64_t rr = 0; rr < num_rows; ++rr) {
      for (int64_t c = 0; c < head_dim; ++c) {
        double acc = 0.0;
        for (int64_t k2 = 0; k2 < head_dim; ++k2) {
          acc += residual[static_cast<size_t>(rr * head_dim + k2)] *
                 projector_d[static_cast<size_t>(k2 * head_dim + c)];
        }
        remainder[static_cast<size_t>(rr * head_dim + c)] =
            residual[static_cast<size_t>(rr * head_dim + c)] - acc;
      }
    }
  }

  const int64_t num_outliers = static_cast<int64_t>(
      RoundHalfToEven(outlier_fraction * static_cast<double>(head_dim)));
  if (num_outliers > 0) {
    std::vector<double> channel_score(static_cast<size_t>(head_dim), 0.0);
    for (int64_t rr = 0; rr < num_rows; ++rr) {
      for (int64_t c = 0; c < head_dim; ++c) {
        channel_score[static_cast<size_t>(c)] +=
            std::fabs(remainder[static_cast<size_t>(rr * head_dim + c)]);
      }
    }
    for (int64_t c = 0; c < head_dim; ++c) {
      channel_score[static_cast<size_t>(c)] /= static_cast<double>(num_rows);
    }
    // order = argsort(channel_score)[-num_outliers:]: the top num_outliers
    // channels by score. Uses a full deterministic sort (descending, ties
    // broken by ascending channel index) rather than reproducing numpy's
    // own argsort tie-handling -- see gear_entry.h's own top-of-file
    // "ACCEPTED, PERMANENT DIVERGENCE" note.
    std::vector<int64_t> order(static_cast<size_t>(head_dim));
    for (int64_t c = 0; c < head_dim; ++c) {
      order[static_cast<size_t>(c)] = c;
    }
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      const double sa = channel_score[static_cast<size_t>(a)];
      const double sb = channel_score[static_cast<size_t>(b)];
      if (sa != sb) {
        return sa > sb;
      }
      return a < b;
    });
    fit.has_sparse_mask = true;
    fit.sparse_mask.assign(static_cast<size_t>(head_dim), 0.0f);
    const int64_t take = std::min<int64_t>(num_outliers, head_dim);
    for (int64_t i = 0; i < take; ++i) {
      fit.sparse_mask[static_cast<size_t>(order[static_cast<size_t>(i)])] =
          1.0f;
    }
  }

  return fit;
}

// Inserts a fresh (empty) node at position `index` (shifting later nodes
// right). Transcribed from llm_int8_entry.cpp's own identical helper --
// safe here (a plain content Swap, not a pointer-identity requirement)
// since ApplyGear never DELETES a node, only inserts, tracking each
// candidate's own shifted index via a running `net_insertions` offset (the
// same convention squeezellm_entry.cpp's/pb_llm_entry.cpp's own identical
// insert-only rewrites already establish, as opposed to spqr_entry.cpp's/
// billm_entry.cpp's own insert-then-delete rewrites, which need the
// pointer-identity/SwapElements pattern instead).
void InsertEmptyNodeAt(onnx::GraphProto* graph, int index) {
  graph->add_node();
  int last = graph->node_size() - 1;
  for (int i = last; i > index; --i) {
    graph->mutable_node(i)->Swap(graph->mutable_node(i - 1));
  }
}

}  // namespace

onnx::ModelProto ApplyGear(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank, double outlier_fraction) {
  onnx::ModelProto out = model;

  // QuantizeLinear/DequantizeLinear's per-channel `axis` needs opset >= 13
  // -- mirrors apply_gear's own `_has_min_opset(model, 13)` gate exactly.
  bool opset_ge_13 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 13) {
      opset_ge_13 = true;
      break;
    }
  }
  if (!opset_ge_13) {
    return out;
  }

  onnx::GraphProto* graph = out.mutable_graph();
  const std::vector<KvCacheCandidate> candidates =
      FindKvCacheCandidates(*graph);
  if (candidates.empty()) {
    return out;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.new_name);
  }
  onnx::ModelProto probe_model = out;
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, probe_model, probe_names,
                           calibration_data);

  std::unordered_map<std::string, GearFit> fits;
  for (const auto& [name, rows] : activations) {
    if (!rows.ok || rows.data.empty()) {
      continue;
    }
    const int64_t head_dim = rows.k;
    const int64_t num_rows = static_cast<int64_t>(rows.data.size()) / head_dim;
    fits.emplace(
        name, FitGear(rows.data, num_rows, head_dim, rank, outlier_fraction));
  }
  if (fits.empty()) {
    return out;
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention every calibration-driven
  // *_entry.cpp in this repo already uses.
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
    auto fit_it = fits.find(c.new_name);
    if (fit_it == fits.end()) {
      continue;
    }
    const GearFit& fit = fit_it->second;
    const int64_t head_dim = static_cast<int64_t>(fit.scale.size());
    const std::string prefix = c.present_name + "_gear";

    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {head_dim}, fit.scale.data(),
                      fit.scale.size() * sizeof(float), sizeof(float));
    std::vector<int8_t> zp(static_cast<size_t>(head_dim), 0);
    const std::string zp_name = unique_name(prefix + "_zero_point");
    SetRawInitializer(graph->add_initializer(), zp_name,
                      onnx::TensorProto::INT8, {head_dim}, zp.data(),
                      zp.size() * sizeof(int8_t), sizeof(int8_t));

    // past: FLOAT -> INT8 (same shape) -- unchanged from
    // onnxsim.kv_cache_quantization's own Key-style rewrite.
    for (auto& inp : *graph->mutable_input()) {
      if (inp.name() == c.past_name) {
        inp.mutable_type()->mutable_tensor_type()->set_elem_type(
            onnx::TensorProto::INT8);
        break;
      }
    }
    for (auto& outp : *graph->mutable_output()) {
      if (outp.name() == c.present_name) {
        outp.mutable_type()->mutable_tensor_type()->set_elem_type(
            onnx::TensorProto::INT8);
        break;
      }
    }

    const int live_concat_index =
        c.concat_node_index + static_cast<int>(net_insertions);

    // QuantizeLinear(new, scale, zero_point, axis=channel_axis) -> new_q,
    // inserted directly before the Concat node.
    const std::string new_q_name = unique_name(prefix + "_new_q");
    InsertEmptyNodeAt(graph, live_concat_index);
    {
      onnx::NodeProto* n = graph->mutable_node(live_concat_index);
      n->set_op_type("QuantizeLinear");
      n->add_input(c.new_name);
      n->add_input(scale_name);
      n->add_input(zp_name);
      n->add_output(new_q_name);
      n->set_name(unique_name(prefix + "_new_q_node"));
      AddIntAttribute(n, "axis", c.channel_axis);
    }

    // The Concat node itself (now shifted to live_concat_index + 1): its
    // own `new`-operand input is rewired to the quantized value; its
    // `past`-operand input is left as-is (still float32-declared at the
    // node-input-name level -- only the graph INPUT's own declared dtype
    // changed above, matching apply_gear's own identical choice not to
    // touch the Concat node's own inputs beyond the `new` -> `new_q`
    // rewire).
    onnx::NodeProto* concat_node = graph->mutable_node(live_concat_index + 1);
    if (c.new_is_first_input) {
      concat_node->set_input(0, new_q_name);
    } else {
      concat_node->set_input(1, new_q_name);
    }

    struct NewNode {
      std::string op_type;
      std::vector<std::string> inputs;
      std::string output;
      std::string name;
      std::optional<int64_t> axis;
    };
    std::vector<NewNode> new_nodes;
    auto add_node = [&](const std::string& op_type,
                        const std::vector<std::string>& inputs,
                        const std::string& out_suffix) -> std::string& {
      NewNode n;
      n.op_type = op_type;
      n.inputs = inputs;
      n.output = unique_name(prefix + "_" + out_suffix);
      n.name = unique_name(prefix + "_" + out_suffix + "_node");
      new_nodes.push_back(std::move(n));
      return new_nodes.back().output;
    };

    const std::string past_dequant = add_node(
        "DequantizeLinear", {c.past_name, scale_name, zp_name}, "past_dequant");
    new_nodes.back().axis = c.channel_axis;
    const std::string new_dequant = add_node(
        "DequantizeLinear", {new_q_name, scale_name, zp_name}, "new_dequant");
    new_nodes.back().axis = c.channel_axis;
    const std::string residual =
        add_node("Sub", {c.new_name, new_dequant}, "new_residual");

    std::vector<std::string> correction_terms;
    std::string remainder = residual;
    if (fit.has_projector) {
      const std::string p_name = unique_name(prefix + "_p");
      SetRawInitializer(graph->add_initializer(), p_name,
                        onnx::TensorProto::FLOAT, {head_dim, head_dim},
                        fit.projector.data(),
                        fit.projector.size() * sizeof(float), sizeof(float));
      const std::string low_rank =
          add_node("MatMul", {residual, p_name}, "low_rank");
      correction_terms.push_back(low_rank);
      if (fit.has_sparse_mask) {
        remainder = add_node("Sub", {residual, low_rank}, "remainder");
      }
    }
    if (fit.has_sparse_mask) {
      const std::string mask_name = unique_name(prefix + "_sparse_mask");
      SetRawInitializer(graph->add_initializer(), mask_name,
                        onnx::TensorProto::FLOAT, {head_dim},
                        fit.sparse_mask.data(),
                        fit.sparse_mask.size() * sizeof(float), sizeof(float));
      const std::string sparse =
          add_node("Mul", {remainder, mask_name}, "sparse");
      correction_terms.push_back(sparse);
    }

    std::string new_corrected = new_dequant;
    for (const auto& term : correction_terms) {
      new_corrected = add_node("Add", {new_corrected, term}, "corrected");
    }

    std::vector<std::string> concat_inputs =
        c.new_is_first_input
            ? std::vector<std::string>{new_corrected, past_dequant}
            : std::vector<std::string>{past_dequant, new_corrected};
    const std::string present_corrected =
        add_node("Concat", concat_inputs, "present_corrected");
    new_nodes.back().axis = c.seq_axis;

    const int insert_at = live_concat_index + 2;
    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, insert_at + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* n = graph->mutable_node(insert_at + i);
      n->set_op_type(spec.op_type);
      for (const auto& s : spec.inputs) {
        n->add_input(s);
      }
      n->add_output(spec.output);
      n->set_name(spec.name);
      if (spec.axis.has_value()) {
        AddIntAttribute(n, "axis", *spec.axis);
      }
    }

    // Rewires every OTHER consumer of c.present_name (never the Concat
    // node itself, which still legitimately produces the raw INT8
    // `present` above) onto present_corrected -- mirrors apply_gear's own
    // `_rewire_consumers` exactly. Graph-level output DECLARATIONS are
    // left alone (present_name's own output entry keeps its name, only
    // its dtype changed above); only ordinary node inputs are redirected,
    // matching the Python reference, which never touches graph.output.
    for (int i = 0; i < graph->node_size(); ++i) {
      if (i == live_concat_index + 1) {
        continue;  // The Concat node itself.
      }
      onnx::NodeProto* n = graph->mutable_node(i);
      for (int j = 0; j < n->input_size(); ++j) {
        if (n->input(j) == c.present_name) {
          n->set_input(j, present_corrected);
        }
      }
    }

    net_insertions += 1 + count;  // 1 QuantizeLinear + `count` new nodes.
  }

  return out;
}
