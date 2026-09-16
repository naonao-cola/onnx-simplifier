// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See low_rank_compensation_entry.h for the full rationale and
// onnxsim/low_rank_compensation.py for the technique this ports.
//
// SVD CHOICE: no linear-algebra library (Eigen or otherwise) is linked
// into this codebase (checked: no existing SVD/eigendecomposition usage
// anywhere under onnxsim/*.cpp or onnxsim/*_entry.cpp, and CMakeLists.txt
// links nothing like it), so this file hand-rolls a one-sided (Hestenes)
// Jacobi SVD -- a small, well-known, numerically robust algorithm for a
// dense matrix of the modest sizes this port's own weight-error matrices
// are (it repeatedly applies Givens rotations to pairs of columns until
// every pair is orthogonal; the column norms become the singular values,
// the normalized columns become U, and the accumulated rotations become
// V). This needs no external dependency and, unlike a truncated
// power-iteration scheme, is a *complete* SVD (computes every singular
// value/vector, truncated to the top `r` only after sorting by
// magnitude) rather than an approximate top-r subspace estimate.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own Jacobi SVD is not
// LAPACK's own Golub-Kahan/bidiagonal-QR algorithm (what numpy's
// np.linalg.svd calls into), so individual singular vectors/values are
// not expected to agree sign-for-sign or bit-for-bit with the Python
// reference -- SVD itself is only unique up to a sign flip per singular
// vector (and up to an orthogonal rotation within any repeated/
// near-repeated singular value's own subspace). What *is* expected to
// agree closely is the reconstructed rank-r correction matrix B @ A
// itself (basis- and sign-invariant, and unique by the Eckart-Young
// theorem whenever the r-th and (r+1)-th singular values are well
// separated) -- tests compare that, not raw U/S/V, and size tolerances
// accordingly. See low_rank_compensation_entry.h's own note.

#include "low_rank_compensation_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"

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

// Same low-nibble-first unpacking test_adaround_cpp.py's own _int4_codes
// helper (and adaround_entry.cpp's own PackInt4, its exact inverse)
// establish: byte i packs codes[2i] in its low nibble, codes[2i+1] in its
// high nibble, each a twos-complement nibble in [-8, 7]. INT4 tensors are
// always raw_data (the ONNX spec has no int32_data-style fallback for
// sub-byte types), so unlike ReadFloatTensor there is no float_data path.
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

// codes/out: flat [d0, d1] row-major. ws: flat [w0, w1] row-major scale.
// Mirrors `codes * np.repeat(ws, block_size, axis)[..., :codes.shape[axis]]`
// -- each element's scale is the ws entry at floor(index_along_axis /
// block_size) along `axis`, unchanged along the other axis.
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
//
// Reimplements adaround.py's own _find_int4_matmul_candidates (also
// reimplemented, identically, by adaround_entry.cpp's own
// FindInt4MatmulCandidates -- not reused directly since it is private to
// that translation unit's own anonymous namespace) directly on raw
// onnx::NodeProto/TensorProto: matches a MatMul/Gemm node in
// `quantized_model` to its counterpart in `float_model` by output name,
// requires the quantized side's weight input to come from a
// DequantizeLinear(Wq: INT4, Ws: FLOAT, axis=.., block_size=..) whose Wq
// has the same shape as the float side's own FLOAT 2-D weight.

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

    // Not part of adaround.py's own matching (numpy's own np.repeat/slice
    // would silently tolerate a malformed scale shape; raw protobuf reads
    // need an explicit bound check instead of risking an out-of-range
    // access in Dequantize below).
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
//
// Economy SVD of an m x n matrix (m >= n), given as a flat row-major
// buffer: sweeps of Givens rotations applied to pairs of columns until
// every pair is orthogonal (off-diagonal of A^T A below `eps` times the
// pair's own norms, for every pair, in one full sweep). Singular values
// are the converged columns' own norms; U's columns are those columns
// normalized; V is the accumulated product of every rotation applied
// (started from the identity). Output is sorted by singular value,
// descending -- Jacobi's own sweep order does not do this.

struct SvdResult {
  int64_t k = 0;          // = min(m, n): number of columns in u/v.
  std::vector<double> u;  // m x k, row-major.
  std::vector<double> s;  // k, descending.
  std::vector<double> v;  // n x k, row-major.
};

SvdResult JacobiSvdTall(int64_t m, int64_t n, const std::vector<double>& a) {
  // Column-major working copies: cols[j][i] = a[i * n + j].
  std::vector<std::vector<double>> cols(
      static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m)));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      cols[static_cast<size_t>(j)][static_cast<size_t>(i)] =
          a[static_cast<size_t>(i * n + j)];
    }
  }
  // V accumulates the same rotations, starting from the n x n identity,
  // stored the same column-major way.
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

// Economy SVD of an arbitrary m x n matrix: transposes to the tall
// orientation first when m < n (JacobiSvdTall above requires m >= n),
// then swaps U/V back (A = (A^T)^T = (V' S U'^T)^T = U' S V'^T when A^T's
// own SVD is U' S V'^T).
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

}  // namespace

onnx::ModelProto ApplyLowRankCompensation(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, int64_t rank) {
  const std::vector<Candidate> candidates =
      FindCandidates(float_model, quantized_model);
  if (candidates.empty()) {
    return quantized_model;
  }

  onnx::ModelProto out = quantized_model;
  onnx::GraphProto* graph = out.mutable_graph();

  // Mirrors low_rank_compensation.py's own `q_by_output = _node_outputs(graph)`
  // (rebuilt on the *copy*, not `quantized_model`): NodeProto* pointers into
  // `graph` stay valid across every later add_node()/SwapElements() call in
  // this function (protobuf's RepeatedPtrField swaps pointer slots, never
  // moves or frees the pointed-to messages), so this map -- built once, up
  // front -- can be safely reused by every candidate below even after an
  // earlier candidate has inserted nodes and renamed another node's output,
  // exactly mirroring Python's own dict-of-object-references.
  std::unordered_map<std::string, onnx::NodeProto*> q_by_output_ptr;
  for (int i = 0; i < graph->node_size(); ++i) {
    onnx::NodeProto* n = graph->mutable_node(i);
    if (n->output_size() > 0) {
      q_by_output_ptr[n->output(0)] = n;
    }
  }

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly
  // (base, base_1, base_2, ...) -- computed once, up front, exactly like
  // daq_entry.cpp's own identical block.
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
      // Matches low_rank_compensation.py's own `if r <= 0: continue` --
      // this layer's clamped rank vanished, so it is left untouched
      // entirely (no renamed output, no new nodes).
      continue;
    }

    std::vector<double> residual_kn(static_cast<size_t>(k * n));
    if (!c.weight_transposed) {
      residual_kn = residual;  // Already [d0, d1] == [k, n].
    } else {
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i1 = 0; i1 < d1; ++i1) {
          residual_kn[static_cast<size_t>(i1 * d0 + i0)] =
              residual[static_cast<size_t>(i0 * d1 + i1)];
        }
      }
    }

    const SvdResult svd = EconomySvd(k, n, residual_kn);

    std::vector<double> b_kn(static_cast<size_t>(k * r));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < r; ++j) {
        b_kn[static_cast<size_t>(i * r + j)] =
            svd.u[static_cast<size_t>(i * svd.k + j)] *
            svd.s[static_cast<size_t>(j)];
      }
    }
    std::vector<double> a_rn(static_cast<size_t>(r * n));
    for (int64_t i = 0; i < r; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        a_rn[static_cast<size_t>(i * n + j)] =
            svd.v[static_cast<size_t>(j * svd.k + i)];
      }
    }

    const std::string prefix = c.output_name + "_lorc";
    const std::string b_name = unique_name(prefix + "_b");
    SetFloatInitializer(graph->add_initializer(), b_name, {k, r}, b_kn);
    const std::string a_name = unique_name(prefix + "_a");
    SetFloatInitializer(graph->add_initializer(), a_name, {r, n}, a_rn);

    onnx::NodeProto* qn = q_by_output_ptr.at(c.output_name);
    const std::string old_output = qn->output(0);
    const std::string base_name = unique_name(prefix + "_base");
    qn->set_output(0, base_name);

    // Re-finds qn's *current* node-list index by pointer identity, exactly
    // mirroring low_rank_compensation.py's own
    // `next(i for i, nd in enumerate(graph.node) if nd is qn)` -- earlier
    // candidates' own insertions can have shifted it since q_by_output_ptr
    // was built, so this cannot be cached across candidates.
    auto* nodes = graph->mutable_node();
    int qn_index = -1;
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == qn) {
        qn_index = i;
        break;
      }
    }
    // Appends a node at the very end, then walks it back down to exactly
    // `target_index` via adjacent swaps -- matches bias_correction_entry.cpp's
    // own identical pattern, itself matching low_rank_compensation.py's own
    // `graph.node.insert(target_index, ...)` placement (ONNX doesn't require
    // topological node order, but onnxsim always leaves the graph in one).
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
