// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See svdquant_entry.h for the full rationale (including why this composes
// with smoothquant_entry.h's own ApplySmoothQuant rather than reimplementing
// it, and the accepted Jacobi-SVD divergence) and onnxsim/svdquant.py for
// the technique this ports.

#include "svdquant_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
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
// Transcribed from spinquant_entry.cpp's own MatchMatMulLike, the exact
// matcher onnxsim.svdquant._match_matmul_like (== onnxsim.quip_sharp's own)
// checks.
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

// Round-half-to-even (banker's rounding) -- matches numpy's own `np.round`
// (used by omniquant.py's own _quantize_blockwise_int4_with_clip, which
// svdquant.py reuses). Transcribed from spinquant_entry.cpp's own identical
// helper.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4.
std::string PackInt4(const std::vector<double>& codes_flat) {
  std::string packed;
  packed.resize((codes_flat.size() + 1) / 2);
  for (size_t i = 0; i * 2 < codes_flat.size(); ++i) {
    const auto lo =
        static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i]));
    uint8_t hi = 0;
    if (2 * i + 1 < codes_flat.size()) {
      hi = static_cast<uint8_t>(static_cast<int64_t>(codes_flat[2 * i + 1]));
    }
    packed[i] = static_cast<char>((lo & 0xF) | ((hi & 0xF) << 4));
  }
  return packed;
}

// --- Block-wise INT4 quantization with clip --------------------------------
//
// Transcribed from spinquant_entry.cpp's own identical
// QuantizeBlockwiseInt4WithClip (itself transcribed from omniquant.py's own
// _quantize_blockwise_int4_with_clip, which svdquant.py reuses directly,
// always with clip_ratio=1.0).
struct BlockwiseInt4 {
  std::vector<double> codes;         // [n, k]
  std::vector<double> scale_blocks;  // [n, num_blocks]
};

BlockwiseInt4 QuantizeBlockwiseInt4WithClip(const std::vector<double>& w_nk,
                                            int64_t n, int64_t k,
                                            int64_t block_size,
                                            double clip_ratio) {
  const int64_t num_blocks = k / block_size;
  BlockwiseInt4 out;
  out.codes.resize(static_cast<size_t>(n * k));
  out.scale_blocks.resize(static_cast<size_t>(n * num_blocks));
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double max_abs = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const int64_t c = b * block_size + j;
        max_abs =
            std::max(max_abs, std::abs(w_nk[static_cast<size_t>(r * k + c)]));
      }
      const double scale = std::max(max_abs * clip_ratio, 1e-12) / 7.0;
      out.scale_blocks[static_cast<size_t>(r * num_blocks + b)] = scale;
      for (int64_t j = 0; j < block_size; ++j) {
        const int64_t c = b * block_size + j;
        const double q = std::min(
            7.0,
            std::max(-7.0, RoundHalfToEven(
                               w_nk[static_cast<size_t>(r * k + c)] / scale)));
        out.codes[static_cast<size_t>(r * k + c)] = q;
      }
    }
  }
  return out;
}

// --- One-sided (Hestenes) Jacobi SVD ----------------------------------------
//
// Transcribed verbatim (algorithm-for-algorithm) from
// low_rank_compensation_entry.cpp's own JacobiSvdTall/EconomySvd -- not
// reused directly since they are private to that translation unit's own
// anonymous namespace. See that file's own top-of-file "SVD CHOICE"/
// "ACCEPTED, PERMANENT DIVERGENCE" comments (also mirrored in this port's
// own svdquant_entry.h) for the full rationale and numerical caveat.

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

onnx::ModelProto ApplySvdquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank, int64_t block_size, std::optional<double> smooth_alpha) {
  onnx::ModelProto out = model;

  bool opset_ge_21 = false;
  for (const auto& opset : out.opset_import()) {
    if ((opset.domain().empty() || opset.domain() == "ai.onnx") &&
        opset.version() >= 21) {
      opset_ge_21 = true;
      break;
    }
  }
  if (!opset_ge_21) {
    return out;
  }

  if (smooth_alpha.has_value()) {
    // Mirrors apply_svdquant's own `apply_smoothquant(model, ...,
    // alpha=smooth_alpha, ...)` call exactly, including SmoothQuant's own
    // fixed epsilon=1e-5 default (apply_svdquant exposes no override).
    out =
        ApplySmoothQuant(out, executor, calibration_data, *smooth_alpha, 1e-5);
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
    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    // [K, N], matching X @ W -- mirrors `w_kn = w.T if weight_transposed
    // else w`.
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    const int64_t n_cols = c.weight_transposed ? dim0 : dim1;
    if (k % block_size != 0) {
      continue;
    }
    const int64_t r = std::min({rank, k, n_cols});
    if (r <= 0) {
      continue;
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    // w_kn[i, j], row-major over [K, N]: the stored layout is [N, K] when
    // transposed (so column j of w_kn is row j of the stored tensor,
    // strided by K), else already [K, N] directly.
    auto at_kn = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(j * k + i)]
                          : flat[static_cast<size_t>(i * n_cols + j)];
      return static_cast<double>(v);
    };
    std::vector<double> w_kn(static_cast<size_t>(k * n_cols));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < n_cols; ++j) {
        w_kn[static_cast<size_t>(i * n_cols + j)] = at_kn(i, j);
      }
    }

    const SvdResult svd = EconomySvd(k, n_cols, w_kn);

    // l1_kr = u[:, :r] * s[:r], [K, r]; l2_rn = vt[:r, :], [r, N] -- mirrors
    // `u[:, :r] * s[np.newaxis, :r]` / `vt[:r, :]` exactly (svd.v is stored
    // as V, [N, k_svd] row-major, i.e. v[j, c] == V[j, c] == Vt[c, j]).
    std::vector<double> l1_kr(static_cast<size_t>(k * r));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < r; ++j) {
        l1_kr[static_cast<size_t>(i * r + j)] =
            svd.u[static_cast<size_t>(i * svd.k + j)] *
            svd.s[static_cast<size_t>(j)];
      }
    }
    std::vector<double> l2_rn(static_cast<size_t>(r * n_cols));
    for (int64_t i = 0; i < r; ++i) {
      for (int64_t j = 0; j < n_cols; ++j) {
        l2_rn[static_cast<size_t>(i * n_cols + j)] =
            svd.v[static_cast<size_t>(j * svd.k + i)];
      }
    }

    // residual_kn = w_kn - l1_kr @ l2_rn, [K, N].
    std::vector<double> residual_kn(static_cast<size_t>(k * n_cols), 0.0);
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t rr = 0; rr < r; ++rr) {
        const double lv = l1_kr[static_cast<size_t>(i * r + rr)];
        if (lv == 0.0) {
          continue;
        }
        for (int64_t j = 0; j < n_cols; ++j) {
          residual_kn[static_cast<size_t>(i * n_cols + j)] +=
              lv * l2_rn[static_cast<size_t>(rr * n_cols + j)];
        }
      }
    }
    for (size_t idx = 0; idx < residual_kn.size(); ++idx) {
      residual_kn[idx] = w_kn[idx] - residual_kn[idx];
    }

    // residual_nk = residual_kn.T, [N, K] -- output channel first, the
    // layout QuantizeBlockwiseInt4WithClip needs.
    std::vector<double> residual_nk(static_cast<size_t>(n_cols * k));
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < n_cols; ++j) {
        residual_nk[static_cast<size_t>(j * k + i)] =
            residual_kn[static_cast<size_t>(i * n_cols + j)];
      }
    }

    const BlockwiseInt4 quant =
        QuantizeBlockwiseInt4WithClip(residual_nk, n_cols, k, block_size, 1.0);
    const int64_t num_blocks = k / block_size;

    // Back to [K, N] layout, ready for a plain MatMul -- mirrors
    // `codes_kn = codes_nk.T` / `scale_kn = scale_blocks.T` exactly.
    std::vector<double> codes_kn(static_cast<size_t>(k * n_cols));
    for (int64_t nn = 0; nn < n_cols; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        codes_kn[static_cast<size_t>(j * n_cols + nn)] =
            quant.codes[static_cast<size_t>(nn * k + j)];
      }
    }
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * n_cols));
    for (int64_t nn = 0; nn < n_cols; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        scale_kn[static_cast<size_t>(b * n_cols + nn)] = static_cast<float>(
            quant.scale_blocks[static_cast<size_t>(nn * num_blocks + b)]);
      }
    }

    const std::string prefix = c.w_name + "_svdquant";

    const std::string codes_name = unique_name(prefix + "_codes");
    {
      onnx::TensorProto* t = graph->add_initializer();
      t->Clear();
      t->set_name(codes_name);
      t->set_data_type(onnx::TensorProto::INT4);
      t->add_dims(k);
      t->add_dims(n_cols);
      t->set_raw_data(PackInt4(codes_kn));
    }
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_blocks, n_cols},
                      scale_kn.data(), scale_kn.size() * sizeof(float),
                      sizeof(float));

    std::vector<float> l1_float(l1_kr.begin(), l1_kr.end());
    const std::string l1_name = unique_name(prefix + "_l1");
    SetRawInitializer(graph->add_initializer(), l1_name,
                      onnx::TensorProto::FLOAT, {k, r}, l1_float.data(),
                      l1_float.size() * sizeof(float), sizeof(float));
    std::vector<float> l2_float(l2_rn.begin(), l2_rn.end());
    const std::string l2_name = unique_name(prefix + "_l2");
    SetRawInitializer(graph->add_initializer(), l2_name,
                      onnx::TensorProto::FLOAT, {r, n_cols}, l2_float.data(),
                      l2_float.size() * sizeof(float), sizeof(float));

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

    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string base = add_node("MatMul", {c.x_name, w_dequant}, "base");
    const std::string lowrank_tmp =
        add_node("MatMul", {c.x_name, l1_name}, "lowrank_tmp");
    const std::string lowrank =
        add_node("MatMul", {lowrank_tmp, l2_name}, "lowrank");
    const std::string summed = add_node("Add", {base, lowrank}, "sum");

    const std::string old_output =
        graph->node(c.node_index + static_cast<int>(net_insertions)).output(0);
    NewNode final_node;
    final_node.output = old_output;
    if (!c.bias_name.empty()) {
      final_node.op_type = "Add";
      final_node.inputs = {summed, c.bias_name};
      final_node.name = unique_name(prefix + "_bias_add_node");
    } else {
      final_node.op_type = "Identity";
      final_node.inputs = {summed};
      final_node.name = unique_name(prefix + "_identity_node");
    }
    new_nodes.push_back(std::move(final_node));

    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const int count = static_cast<int>(new_nodes.size());
    for (int i = 0; i < count; ++i) {
      InsertEmptyNodeAt(graph, live_index + i);
    }
    for (int i = 0; i < count; ++i) {
      const NewNode& spec = new_nodes[static_cast<size_t>(i)];
      onnx::NodeProto* nd = graph->mutable_node(live_index + i);
      nd->set_op_type(spec.op_type);
      for (const auto& s : spec.inputs) {
        nd->add_input(s);
      }
      nd->add_output(spec.output);
      nd->set_name(spec.name);
      for (const auto& [attr_name, attr_value] : spec.int_attrs) {
        AddIntAttribute(nd, attr_name, attr_value);
      }
    }
    graph->mutable_node()->DeleteSubrange(live_index + count, 1);
    net_insertions += count - 1;
  }

  return out;
}
