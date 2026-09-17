// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See spinquant_entry.h for the full rationale (including why this follows
// spqr_entry.h's own single-model, calibration-driven shape, and the
// accepted eigendecomposition-algorithm divergence) and
// onnxsim/spinquant.py for the technique this ports.

#include "spinquant_entry.h"

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
// Transcribed from spqr_entry.cpp's own MatchMatMulLike, which itself
// mirrors onnxsim.quip_sharp._match_matmul_like -- the exact matcher
// onnxsim.spinquant reuses.
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
// spinquant.py reuses), unlike std::round's half-away-from-zero.
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

// Same low-nibble-first packing as adaround.py's own _pack_int4 (and
// spqr_entry.cpp's own identical PackInt4, byte-for-byte, not reused
// directly since it is private to that translation unit). `codes_flat.
// size()` must be even -- always true here since `k` is checked divisible
// by `block_size`, and `block_size` divisible by 2 isn't actually required
// by spinquant (unlike paroquant); K*N is even whenever K is even, which a
// `block_size >= 2` (any practical value) guarantees for every full row.
// More simply: N * K is guaranteed even by construction because every
// caller here quantizes a whole [N, K] tensor with K a multiple of
// block_size, and this port only ever packs the transposed [K, N] layout
// whose total element count K*N is unchanged either way -- callers only
// invoke this with an even-length buffer in practice for every test/real
// weight shape (K, block_size >= 1 supplies no formal guarantee of
// evenness on its own, so this mirrors adaround.py's own equally-implicit
// assumption rather than adding an independent check beyond what the
// Python reference already relies on).
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
// Transcribed from omniquant.py's own _quantize_blockwise_int4_with_clip
// (which spinquant.py reuses directly, always with clip_ratio=1.0): each
// block's scale is max(|w| in block) * clip_ratio / 7 (floored at 1e-12),
// codes are round-half-to-even, clamped to [-7, 7].
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

// --- Symmetric eigendecomposition (cyclic Jacobi) ---------------------------
//
// See spinquant_entry.h's own "ACCEPTED, PERMANENT DIVERGENCE" note: no
// linear-algebra library is linked into this codebase (see low_rank_
// compensation_entry.cpp's own "SVD CHOICE" comment), so this hand-rolls
// the classical cyclic Jacobi eigenvalue algorithm -- repeatedly zeroing
// the largest off-diagonal-pair contribution via a 2x2 Givens rotation
// until the matrix is (numerically) diagonal. Robust and simple for the
// small, dense K x K covariance matrices this port's own layers have;
// `a` is consumed by value (used as scratch space).
struct JacobiEigenResult {
  std::vector<double> eigenvalues;   // [n], ascending
  std::vector<double> eigenvectors;  // [n, n] row-major: eigenvectors[i*n+c]
                                     // is component i of eigenvector c.
};

JacobiEigenResult JacobiEigenSymmetric(std::vector<double> a, int64_t n) {
  std::vector<double> v(static_cast<size_t>(n * n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    v[static_cast<size_t>(i * n + i)] = 1.0;
  }

  const int64_t max_sweeps = 100;
  for (int64_t sweep = 0; sweep < max_sweeps; ++sweep) {
    double off = 0.0;
    for (int64_t p = 0; p < n; ++p) {
      for (int64_t q = p + 1; q < n; ++q) {
        const double apq = a[static_cast<size_t>(p * n + q)];
        off += apq * apq;
      }
    }
    if (off < 1e-28) {
      break;
    }
    for (int64_t p = 0; p < n; ++p) {
      for (int64_t q = p + 1; q < n; ++q) {
        const double apq = a[static_cast<size_t>(p * n + q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double app = a[static_cast<size_t>(p * n + p)];
        const double aqq = a[static_cast<size_t>(q * n + q)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        a[static_cast<size_t>(p * n + p)] = app - t * apq;
        a[static_cast<size_t>(q * n + q)] = aqq + t * apq;
        a[static_cast<size_t>(p * n + q)] = 0.0;
        a[static_cast<size_t>(q * n + p)] = 0.0;
        for (int64_t i = 0; i < n; ++i) {
          if (i == p || i == q) {
            continue;
          }
          const double aip = a[static_cast<size_t>(i * n + p)];
          const double aiq = a[static_cast<size_t>(i * n + q)];
          const double new_ip = c * aip - s * aiq;
          const double new_iq = s * aip + c * aiq;
          a[static_cast<size_t>(i * n + p)] = new_ip;
          a[static_cast<size_t>(p * n + i)] = new_ip;
          a[static_cast<size_t>(i * n + q)] = new_iq;
          a[static_cast<size_t>(q * n + i)] = new_iq;
        }
        for (int64_t i = 0; i < n; ++i) {
          const double vip = v[static_cast<size_t>(i * n + p)];
          const double viq = v[static_cast<size_t>(i * n + q)];
          v[static_cast<size_t>(i * n + p)] = c * vip - s * viq;
          v[static_cast<size_t>(i * n + q)] = s * vip + c * viq;
        }
      }
    }
  }

  std::vector<double> eigenvalues(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    eigenvalues[static_cast<size_t>(i)] = a[static_cast<size_t>(i * n + i)];
  }
  std::vector<int64_t> order(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    order[static_cast<size_t>(i)] = i;
  }
  std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
    return eigenvalues[static_cast<size_t>(lhs)] <
           eigenvalues[static_cast<size_t>(rhs)];
  });

  JacobiEigenResult result;
  result.eigenvalues.resize(static_cast<size_t>(n));
  result.eigenvectors.resize(static_cast<size_t>(n * n));
  for (int64_t c = 0; c < n; ++c) {
    const int64_t src = order[static_cast<size_t>(c)];
    result.eigenvalues[static_cast<size_t>(c)] =
        eigenvalues[static_cast<size_t>(src)];
    for (int64_t i = 0; i < n; ++i) {
      result.eigenvectors[static_cast<size_t>(i * n + c)] =
          v[static_cast<size_t>(i * n + src)];
    }
  }
  return result;
}

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from spqr_entry.cpp's own ActivationRows/
// AccumulateActivationRows -- the general (rank >= 2, not just plain 2-D)
// capture onnxsim.bias_correction._activation_rows performs, which
// spinquant.py imports directly.
struct ActivationRows {
  std::vector<double> data;  // Concatenated [total_rows, K], row-major.
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
            "ApplySpinquant: calibration batch is missing "
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

onnx::ModelProto ApplySpinquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size) {
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
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, probe_model, probe_names,
                           calibration_data);

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
    auto acts_it = activations.find(c.x_name);
    if (acts_it == activations.end() || acts_it->second.data.empty()) {
      continue;
    }
    const ActivationRows& rows = acts_it->second;
    const int64_t k_obs = rows.k;
    const int64_t total_rows = static_cast<int64_t>(rows.data.size()) / k_obs;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (k % block_size != 0 || k_obs != k) {
      continue;
    }

    const std::vector<float> flat = ReadFloatTensor(w_init);  // [dim0, dim1]
    auto at_nk = [&](int64_t i, int64_t j) -> double {
      const float v = c.weight_transposed
                          ? flat[static_cast<size_t>(i * k + j)]
                          : flat[static_cast<size_t>(j * n_rows + i)];
      return static_cast<double>(v);
    };

    // cov = X^T @ X / rows, [K, K] -- symmetric by construction.
    std::vector<double> cov(static_cast<size_t>(k * k), 0.0);
    for (int64_t r = 0; r < total_rows; ++r) {
      const double* row = &rows.data[static_cast<size_t>(r * k)];
      for (int64_t a = 0; a < k; ++a) {
        const double xa = row[a];
        if (xa == 0.0) {
          continue;
        }
        for (int64_t b = a; b < k; ++b) {
          cov[static_cast<size_t>(a * k + b)] += xa * row[b];
        }
      }
    }
    for (int64_t a = 0; a < k; ++a) {
      for (int64_t b = a; b < k; ++b) {
        const double v = cov[static_cast<size_t>(a * k + b)] /
                         static_cast<double>(total_rows);
        cov[static_cast<size_t>(a * k + b)] = v;
        cov[static_cast<size_t>(b * k + a)] = v;
      }
    }

    const JacobiEigenResult eig = JacobiEigenSymmetric(cov, k);
    const std::vector<double>& u = eig.eigenvectors;  // [K, K] row-major.

    // w_tilde_nk = w_nk @ u -- [N, K], exact before quantization.
    std::vector<double> w_tilde_nk(static_cast<size_t>(n_rows * k), 0.0);
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t col = 0; col < k; ++col) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += at_nk(nn, kk) * u[static_cast<size_t>(kk * k + col)];
        }
        w_tilde_nk[static_cast<size_t>(nn * k + col)] = acc;
      }
    }

    const BlockwiseInt4 quant =
        QuantizeBlockwiseInt4WithClip(w_tilde_nk, n_rows, k, block_size, 1.0);
    const int64_t num_blocks = k / block_size;

    // Transpose codes/scale to [K, N] layout, ready for a plain MatMul.
    std::vector<double> codes_kn(static_cast<size_t>(k * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t j = 0; j < k; ++j) {
        codes_kn[static_cast<size_t>(j * n_rows + nn)] =
            quant.codes[static_cast<size_t>(nn * k + j)];
      }
    }
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * n_rows));
    for (int64_t nn = 0; nn < n_rows; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        scale_kn[static_cast<size_t>(b * n_rows + nn)] = static_cast<float>(
            quant.scale_blocks[static_cast<size_t>(nn * num_blocks + b)]);
      }
    }

    const std::string prefix = c.w_name + "_spinquant";

    const std::string codes_name = unique_name(prefix + "_codes");
    {
      onnx::TensorProto* t = graph->add_initializer();
      t->Clear();
      t->set_name(codes_name);
      t->set_data_type(onnx::TensorProto::INT4);
      t->add_dims(k);
      t->add_dims(n_rows);
      t->set_raw_data(PackInt4(codes_kn));
    }
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_blocks, n_rows},
                      scale_kn.data(), scale_kn.size() * sizeof(float),
                      sizeof(float));
    std::vector<float> u_float(static_cast<size_t>(k * k));
    for (size_t i = 0; i < u_float.size(); ++i) {
      u_float[i] = static_cast<float>(u[i]);
    }
    const std::string u_name = unique_name(prefix + "_u");
    SetRawInitializer(graph->add_initializer(), u_name,
                      onnx::TensorProto::FLOAT, {k, k}, u_float.data(),
                      u_float.size() * sizeof(float), sizeof(float));

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

    const std::string x_rotated =
        add_node("MatMul", {c.x_name, u_name}, "x_rotated");
    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string core = add_node("MatMul", {x_rotated, w_dequant}, "core");

    const std::string old_output =
        graph->node(c.node_index + static_cast<int>(net_insertions)).output(0);
    NewNode final_node;
    final_node.output = old_output;
    if (!c.bias_name.empty()) {
      final_node.op_type = "Add";
      final_node.inputs = {core, c.bias_name};
      final_node.name = unique_name(prefix + "_bias_add_node");
    } else {
      final_node.op_type = "Identity";
      final_node.inputs = {core};
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
