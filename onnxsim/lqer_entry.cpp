// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See lqer_entry.h for the full rationale and onnxsim/lqer.py for the
// technique this ports. Candidate matching, tensor I/O helpers, and the
// Jacobi SVD below are transcribed from low_rank_compensation_entry.cpp
// (this port's own primary template -- see lqer_entry.h's own top-of-file
// comment for why they aren't shared via a common header: every other
// *_entry.cpp in this codebase duplicates its own small helpers rather
// than factoring them out, since each is privately parameterized by its
// own error-message text and there is no established shared-helpers
// header for this family of passes). Only the SVD's own INPUT (row-scaled
// by a calibration-measured activation RMS or not) and how `B` is
// un-scaled afterward differ from that file.

#include "lqer_entry.h"

#include <algorithm>
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

int64_t GetIntAttr(const onnx::NodeProto& node, const std::string& name,
                   int64_t fallback) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name && attr.type() == onnx::AttributeProto::INT) {
      return attr.i();
    }
  }
  return fallback;
}

// --- Tensor <-> flat double buffer -------------------------------------
// Transcribed verbatim from low_rank_compensation_entry.cpp's own
// identical helpers.

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

std::vector<double> ReadInt4Tensor(const onnx::TensorProto& t) {
  int64_t numel = 1;
  for (int64_t d : t.dims()) {
    numel *= d;
  }
  std::vector<double> out(static_cast<size_t>(numel));
  const std::string& raw = t.raw_data();
  for (int64_t i = 0; i < numel; ++i) {
    const auto byte = static_cast<uint8_t>(raw[static_cast<size_t>(i / 2)]);
    const uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    const int8_t signed_val = (nibble >= 8) ? static_cast<int8_t>(nibble - 16)
                                            : static_cast<int8_t>(nibble);
    out[static_cast<size_t>(i)] = static_cast<double>(signed_val);
  }
  return out;
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

std::vector<double> Dequantize(const std::vector<double>& codes, int64_t d0,
                               int64_t d1, const std::vector<double>& ws,
                               int64_t w1, int64_t axis, int64_t block_size) {
  std::vector<double> out(codes.size());
  for (int64_t i0 = 0; i0 < d0; ++i0) {
    for (int64_t i1 = 0; i1 < d1; ++i1) {
      const int64_t wi0 = (axis == 0) ? (i0 / block_size) : i0;
      const int64_t wi1 = (axis == 0) ? i1 : (i1 / block_size);
      const double scale = ws[static_cast<size_t>(wi0 * w1 + wi1)];
      const size_t idx = static_cast<size_t>(i0 * d1 + i1);
      out[idx] = codes[idx] * scale;
    }
  }
  return out;
}

// --- Candidate matching, at the raw protobuf level ----------------------
// Transcribed verbatim from low_rank_compensation_entry.cpp's own
// FindCandidates (itself reimplementing adaround.py's own
// _find_int4_matmul_candidates).

struct Candidate {
  std::string output_name;
  std::string x_name;
  const onnx::TensorProto* w_float;
  const onnx::TensorProto* wq;
  const onnx::TensorProto* ws;
  int64_t axis;
  int64_t block_size;
  bool weight_transposed;
};

std::vector<Candidate> FindCandidates(const onnx::ModelProto& float_model,
                                      const onnx::ModelProto& quantized_model) {
  const onnx::GraphProto& q_graph = quantized_model.graph();
  const onnx::GraphProto& f_graph = float_model.graph();

  std::unordered_map<std::string, int> q_init_index;
  for (int i = 0; i < q_graph.initializer_size(); ++i) {
    q_init_index[q_graph.initializer(i).name()] = i;
  }
  std::unordered_map<std::string, int> f_init_index;
  for (int i = 0; i < f_graph.initializer_size(); ++i) {
    f_init_index[f_graph.initializer(i).name()] = i;
  }

  std::vector<std::string> q_order;
  std::unordered_map<std::string, int> q_by_output;
  for (int i = 0; i < q_graph.node_size(); ++i) {
    const auto& n = q_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    if (!q_by_output.count(n.output(0))) {
      q_order.push_back(n.output(0));
    }
    q_by_output[n.output(0)] = i;
  }
  std::unordered_map<std::string, int> f_by_output;
  for (int i = 0; i < f_graph.node_size(); ++i) {
    const auto& n = f_graph.node(i);
    if (n.output_size() < 1) {
      continue;
    }
    f_by_output[n.output(0)] = i;
  }

  std::vector<Candidate> candidates;
  for (const auto& out_name : q_order) {
    const onnx::NodeProto& qn = q_graph.node(q_by_output[out_name]);
    if ((qn.op_type() != "MatMul" && qn.op_type() != "Gemm") ||
        qn.input_size() < 2) {
      continue;
    }
    auto fit = f_by_output.find(out_name);
    if (fit == f_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& fn = f_graph.node(fit->second);
    if (fn.op_type() != qn.op_type() || fn.input_size() < 2) {
      continue;
    }

    auto wfit = f_init_index.find(fn.input(1));
    if (wfit == f_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& w_float = f_graph.initializer(wfit->second);
    if (w_float.data_type() != onnx::TensorProto::FLOAT ||
        w_float.dims_size() != 2) {
      continue;
    }

    auto dqit = q_by_output.find(qn.input(1));
    if (dqit == q_by_output.end()) {
      continue;
    }
    const onnx::NodeProto& dq = q_graph.node(dqit->second);
    if (dq.op_type() != "DequantizeLinear" || dq.input_size() < 2) {
      continue;
    }
    auto wqit = q_init_index.find(dq.input(0));
    auto wsit = q_init_index.find(dq.input(1));
    if (wqit == q_init_index.end() || wsit == q_init_index.end()) {
      continue;
    }
    const onnx::TensorProto& wq = q_graph.initializer(wqit->second);
    const onnx::TensorProto& ws = q_graph.initializer(wsit->second);
    if (wq.data_type() != onnx::TensorProto::INT4 ||
        wq.dims_size() != w_float.dims_size()) {
      continue;
    }
    bool same_dims = true;
    for (int d = 0; d < wq.dims_size(); ++d) {
      if (wq.dims(d) != w_float.dims(d)) {
        same_dims = false;
        break;
      }
    }
    if (!same_dims || ws.data_type() != onnx::TensorProto::FLOAT ||
        ws.dims_size() != 2) {
      continue;
    }

    const int64_t block_size = GetIntAttr(dq, "block_size", 0);
    if (block_size <= 0) {
      continue;
    }
    const int64_t axis = GetIntAttr(dq, "axis", 1);
    if (axis != 0 && axis != 1) {
      continue;
    }

    const int64_t d0 = w_float.dims(0);
    const int64_t d1 = w_float.dims(1);
    const int64_t w0 = ws.dims(0);
    const int64_t w1 = ws.dims(1);
    if (axis == 0) {
      if (w1 != d1 || w0 * block_size < d0) {
        continue;
      }
    } else {
      if (w0 != d0 || w1 * block_size < d1) {
        continue;
      }
    }

    const bool weight_transposed =
        qn.op_type() == "Gemm" && GetIntAttr(qn, "transB", 0) != 0;

    candidates.push_back({out_name, fn.input(0), &w_float, &wq, &ws, axis,
                          block_size, weight_transposed});
  }
  return candidates;
}

// --- One-sided (Hestenes) Jacobi SVD -------------------------------------
// Transcribed verbatim from low_rank_compensation_entry.cpp's own
// JacobiSvdTall/EconomySvd -- see that file's own top-of-file comment for
// the full rationale (no linear-algebra library is linked into this
// codebase).

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

// --- Calibration: per-input-channel activation RMS -----------------------
//
// Mirrors lqer.py's own _per_channel_activation_rms exactly: probes
// `float_model` at every candidate's own activation input name, and --
// unlike gptq_entry.cpp/spqr_entry.cpp's own rank-agnostic
// AccumulateActivationRows (which flattens any rank >= 2 to rows) -- only
// accepts a STRICT 2-D ([rows, K]) observed tensor, matching lqer.py's own
// "skip non-2-D activations" convention (adaround's own convention, reused
// verbatim by lqer.py) rather than the rank-agnostic one gptq.py/spqr.py
// use. Accumulates sum-of-squares per column plus a row count; the RMS
// itself (`sqrt(mean(x_k^2))`) is finalized once, after every batch, by
// the caller.

struct ChannelRmsAccum {
  std::vector<double> sum_sq;  // [K]
  int64_t count = 0;
  bool ok = false;
};

void AccumulatePerChannelRms(
    std::unordered_map<std::string, ChannelRmsAccum>& acc,
    const ModelExecutor& executor, const onnx::ModelProto& float_model,
    const std::unordered_set<std::string>& probe_names,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data) {
  if (probe_names.empty()) {
    return;
  }

  onnx::ModelProto probe_model = float_model;
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
            "ApplyLqer: calibration batch is missing "
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
      // STRICT rank-2 only -- mirrors _per_channel_activation_rms's own
      // `if x.ndim != 2: continue`.
      if (tp.data_type() != onnx::TensorProto::FLOAT || tp.dims_size() != 2) {
        continue;
      }
      const int64_t rows = tp.dims(0);
      const int64_t k = tp.dims(1);
      if (k <= 0 || rows <= 0) {
        continue;
      }
      ChannelRmsAccum& a = acc[name];
      if (!a.ok) {
        a.sum_sq.assign(static_cast<size_t>(k), 0.0);
        a.count = 0;
        a.ok = true;
      } else if (static_cast<int64_t>(a.sum_sq.size()) != k) {
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (mirrors every other calibration-driven
                   // port's own identical guard).
      }
      const std::vector<double> data = ReadFloatTensor(tp);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < k; ++j) {
          const double v = data[static_cast<size_t>(r * k + j)];
          a.sum_sq[static_cast<size_t>(j)] += v * v;
        }
      }
      a.count += rows;
    }
  }
}

}  // namespace

onnx::ModelProto ApplyLqer(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank, double eps) {
  const std::vector<Candidate> candidates =
      FindCandidates(float_model, quantized_model);
  if (candidates.empty()) {
    return quantized_model;
  }

  std::unordered_set<std::string> probe_names;
  for (const auto& c : candidates) {
    probe_names.insert(c.x_name);
  }
  std::unordered_map<std::string, ChannelRmsAccum> rms_acc;
  AccumulatePerChannelRms(rms_acc, executor, float_model, probe_names,
                          calibration_data);

  onnx::ModelProto out = quantized_model;
  onnx::GraphProto* graph = out.mutable_graph();

  // Same NodeProto*-pointer-stability argument as
  // low_rank_compensation_entry.cpp's own identical comment.
  std::unordered_map<std::string, onnx::NodeProto*> q_by_output_ptr;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->output_size() > 0) {
      q_by_output_ptr[n->output(0)] = n;
    }
  }

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

  for (const Candidate& c : candidates) {
    const std::vector<double> codes = ReadInt4Tensor(*c.wq);
    const std::vector<double> ws = ReadFloatTensor(*c.ws);
    const std::vector<double> w_float = ReadFloatTensor(*c.w_float);
    const int64_t d0 = c.w_float->dims(0);
    const int64_t d1 = c.w_float->dims(1);
    const int64_t w1 = c.ws->dims(1);
    const std::vector<double> w_dequant =
        Dequantize(codes, d0, d1, ws, w1, c.axis, c.block_size);

    std::vector<double> residual(static_cast<size_t>(d0 * d1));
    for (size_t i = 0; i < residual.size(); ++i) {
      residual[i] = w_float[i] - w_dequant[i];
    }

    const int64_t k = c.weight_transposed ? d1 : d0;
    const int64_t n = c.weight_transposed ? d0 : d1;
    const int64_t r = std::min({rank, k, n});
    if (r <= 0) {
      continue;
    }

    std::vector<double> residual_kn(static_cast<size_t>(k * n));
    if (!c.weight_transposed) {
      residual_kn = residual;
    } else {
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i1 = 0; i1 < d1; ++i1) {
          residual_kn[static_cast<size_t>(i1 * d0 + i0)] =
              residual[static_cast<size_t>(i0 * d1 + i1)];
        }
      }
    }

    // Row-scale by the calibration-measured per-channel RMS (floored at
    // `eps`) before the SVD, exactly mirroring
    // weighted_low_rank_correction's own `weighted = residual_kn *
    // s[:, np.newaxis]` -- only when that layer's own activation was
    // observed as a strict 2-D tensor of matching width `k` (mirrors
    // apply_lqer's own `channel_weights = rms if rms is not None and
    // rms.shape[0] == k else None`).
    bool weighted = false;
    std::vector<double> s(static_cast<size_t>(k), 1.0);
    auto rms_it = rms_acc.find(c.x_name);
    if (rms_it != rms_acc.end() && rms_it->second.ok &&
        static_cast<int64_t>(rms_it->second.sum_sq.size()) == k &&
        rms_it->second.count > 0) {
      weighted = true;
      for (int64_t i = 0; i < k; ++i) {
        const double mean_sq = rms_it->second.sum_sq[static_cast<size_t>(i)] /
                               static_cast<double>(rms_it->second.count);
        s[static_cast<size_t>(i)] = std::max(std::sqrt(mean_sq), eps);
      }
    }

    std::vector<double> svd_input = residual_kn;
    if (weighted) {
      for (int64_t i = 0; i < k; ++i) {
        const double si = s[static_cast<size_t>(i)];
        for (int64_t j = 0; j < n; ++j) {
          svd_input[static_cast<size_t>(i * n + j)] *= si;
        }
      }
    }

    const SvdResult svd = EconomySvd(k, n, svd_input);

    std::vector<double> b_kn(static_cast<size_t>(k * r));
    for (int64_t i = 0; i < k; ++i) {
      const double divisor = weighted ? s[static_cast<size_t>(i)] : 1.0;
      for (int64_t j = 0; j < r; ++j) {
        b_kn[static_cast<size_t>(i * r + j)] =
            svd.u[static_cast<size_t>(i * svd.k + j)] *
            svd.s[static_cast<size_t>(j)] / divisor;
      }
    }
    std::vector<double> a_rn(static_cast<size_t>(r * n));
    for (int64_t i = 0; i < r; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        a_rn[static_cast<size_t>(i * n + j)] =
            svd.v[static_cast<size_t>(j * svd.k + i)];
      }
    }

    const std::string prefix = c.output_name + "_lqer";
    const std::string b_name = unique_name(prefix + "_b");
    SetFloatInitializer(graph->add_initializer(), b_name, {k, r}, b_kn);
    const std::string a_name = unique_name(prefix + "_a");
    SetFloatInitializer(graph->add_initializer(), a_name, {r, n}, a_rn);

    onnx::NodeProto* qn = q_by_output_ptr.at(c.output_name);
    const std::string old_output = qn->output(0);
    const std::string base_name = unique_name(prefix + "_base");
    qn->set_output(0, base_name);

    auto* nodes = graph->mutable_node();
    int qn_index = -1;
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == qn) {
        qn_index = i;
        break;
      }
    }
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
    };

    const std::string tmp_name = unique_name(prefix + "_tmp");
    append_at("MatMul", {c.x_name, b_name}, tmp_name,
              unique_name(prefix + "_matmul1_node"), qn_index + 1);

    const std::string lowrank_name = unique_name(prefix + "_lowrank");
    append_at("MatMul", {tmp_name, a_name}, lowrank_name,
              unique_name(prefix + "_matmul2_node"), qn_index + 2);

    append_at("Add", {base_name, lowrank_name}, old_output,
              unique_name(prefix + "_add_node"), qn_index + 3);
  }

  return out;
}
