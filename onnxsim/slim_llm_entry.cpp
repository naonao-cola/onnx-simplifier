// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See slim_llm_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own single-model, node-inserting rewrite shape rather
// than gptq_entry.h's own two-model in-place-rewrite shape) and
// onnxsim/slim_llm.py for the technique this ports.

#include "slim_llm_entry.h"

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
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike (itself from
// onnxsim.quip_sharp's own _match_matmul_like, which slim_llm.py's own
// candidate loop imports and reuses directly).
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

// --- Tensor <-> flat float buffer -------------------------------------------
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

// Round-half-to-even (banker's rounding), matching numpy's own `round`.
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

// --- Small dense double-precision linear algebra ----------------------------
//
// Transcribed from gptq_entry.cpp's own CholeskyLower/InverseSPD/
// InverseHessianCholesky -- this codebase's established one-copy-per-TU
// convention rather than a shared header (see gptq_entry.cpp's own comment
// on why: small, once-per-layer kernels, no BLAS/LAPACK dependency).
using Matrix = std::vector<std::vector<double>>;

Matrix CholeskyLower(const Matrix& a) {
  const size_t n = a.size();
  Matrix l(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double s = a[i][j];
      for (size_t k = 0; k < j; ++k) {
        s -= l[i][k] * l[j][k];
      }
      if (i == j) {
        l[i][j] = std::sqrt(std::max(s, 1e-24));
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return l;
}

Matrix InverseSPD(const Matrix& a) {
  const size_t n = a.size();
  const Matrix l = CholeskyLower(a);
  Matrix inv(n, std::vector<double>(n, 0.0));
  std::vector<double> y(n), x(n);
  for (size_t c = 0; c < n; ++c) {
    for (size_t i = 0; i < n; ++i) {
      double s = (i == c) ? 1.0 : 0.0;
      for (size_t k = 0; k < i; ++k) {
        s -= l[i][k] * y[k];
      }
      y[i] = s / l[i][i];
    }
    for (size_t i = n; i-- > 0;) {
      double s = y[i];
      for (size_t k = i + 1; k < n; ++k) {
        s -= l[k][i] * x[k];
      }
      x[i] = s / l[i][i];
    }
    for (size_t i = 0; i < n; ++i) {
      inv[i][c] = x[i];
    }
  }
  return inv;
}

// Returns the upper-triangular Cholesky factor U of h's inverse
// (inverse(h) == U^T @ U) -- mirrors gptq.py's own
// _inverse_hessian_cholesky exactly.
Matrix InverseHessianCholesky(const Matrix& h, double percdamp) {
  const size_t k = h.size();
  Matrix damped = h;
  double diag_sum = 0.0;
  for (size_t i = 0; i < k; ++i) {
    if (damped[i][i] == 0.0) {
      damped[i][i] = 1.0;
    }
    diag_sum += damped[i][i];
  }
  const double damp =
      std::max(percdamp * diag_sum / static_cast<double>(k), 1e-8);
  for (size_t i = 0; i < k; ++i) {
    damped[i][i] += damp;
  }
  const Matrix h_inv = InverseSPD(damped);
  const Matrix l = CholeskyLower(h_inv);
  Matrix u(k, std::vector<double>(k, 0.0));
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = i; j < k; ++j) {
      u[i][j] = l[j][i];
    }
  }
  return u;
}

// --- Calibration: concatenated activation rows -----------------------------
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
            "ApplySlimLlm: calibration batch is missing "
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

// Round-to-nearest block-wise symmetric quantization of w_nk ([N, K],
// output channel first) to `bits`-bit codes in
// [-(2^(bits-1) - 1), 2^(bits-1) - 1] -- mirrors slim_llm.py's own
// _quantize_blockwise_nbit exactly. Returns (codes_nk, scale_blocks
// [N, K/block_size]).
struct BlockwiseQuant {
  Matrix codes_nk;
  Matrix scale_blocks;
};

BlockwiseQuant QuantizeBlockwiseNbit(const Matrix& w_nk, int64_t block_size,
                                     int64_t bits) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const size_t num_blocks = k / static_cast<size_t>(block_size);
  const double qmax = static_cast<double>((int64_t{1} << (bits - 1)) - 1);

  Matrix scale_blocks(n, std::vector<double>(num_blocks));
  for (size_t r = 0; r < n; ++r) {
    for (size_t b = 0; b < num_blocks; ++b) {
      double amax = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const double v = std::fabs(w_nk[r][b * static_cast<size_t>(block_size) +
                                           static_cast<size_t>(j)]);
        if (v > amax) {
          amax = v;
        }
      }
      scale_blocks[r][b] = std::max(amax, 1e-12) / qmax;
    }
  }

  Matrix codes_nk(n, std::vector<double>(k));
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < k; ++c) {
      const size_t blk = c / static_cast<size_t>(block_size);
      const double s = scale_blocks[r][blk];
      const double code = RoundHalfToEven(w_nk[r][c] / s);
      codes_nk[r][c] = std::clamp(code, -qmax, qmax);
    }
  }
  return {codes_nk, scale_blocks};
}

}  // namespace

onnx::ModelProto ApplySlimLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double target_bits, int64_t low_bits, int64_t high_bits, int64_t group_size,
    double percdamp) {
  if (low_bits < 2 || high_bits <= low_bits) {
    throw std::invalid_argument(
        "ApplySlimLlm: require 2 <= low_bits < high_bits");
  }

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
    const int64_t k = m->weight_transposed ? w_init.dims(1) : w_init.dims(0);
    if (group_size == 0 || k % group_size != 0) {
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
  std::unordered_map<std::string, ActivationRows> activations;
  AccumulateActivationRows(activations, executor, out, probe_names,
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

  const double target_bits_clamped =
      std::min(std::max(target_bits, static_cast<double>(low_bits)),
               static_cast<double>(high_bits));
  const double fraction_high =
      (target_bits_clamped - static_cast<double>(low_bits)) /
      static_cast<double>(high_bits - low_bits);

  int64_t net_insertions = 0;
  for (const auto& c : candidates) {
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable calibration activation observed -- skip.
    }
    const ActivationRows& rows = ait->second;

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = c.weight_transposed ? dim0 : dim1;
    const int64_t k = c.weight_transposed ? dim1 : dim0;
    if (rows.k != k) {
      continue;  // Activation's feature dim doesn't match K -- skip.
    }
    const int64_t num_samples =
        static_cast<int64_t>(rows.data.size()) / (k == 0 ? 1 : k);

    const std::vector<float> w_flat = ReadFloatTensor(w_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
                std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n_rows; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(i * k + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + i)]);
      }
    }

    Matrix x(static_cast<size_t>(num_samples),
             std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < num_samples; ++r) {
      for (int64_t c2 = 0; c2 < k; ++c2) {
        x[static_cast<size_t>(r)][static_cast<size_t>(c2)] =
            rows.data[static_cast<size_t>(r * k + c2)];
      }
    }

    Matrix h(static_cast<size_t>(k),
             std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_samples; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi = x[static_cast<size_t>(r)][static_cast<size_t>(i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * x[static_cast<size_t>(r)][static_cast<size_t>(j)];
        }
      }
    }
    const Matrix u = InverseHessianCholesky(h, percdamp);  // inv(h)=u^T@u
    std::vector<double> h_inv_diag(static_cast<size_t>(k), 0.0);
    for (int64_t j = 0; j < k; ++j) {
      double sq = 0.0;
      for (int64_t i = 0; i < k; ++i) {
        const double uv = u[static_cast<size_t>(i)][static_cast<size_t>(j)];
        sq += uv * uv;
      }
      h_inv_diag[static_cast<size_t>(j)] = std::max(sq, 1e-12);
    }

    const BlockwiseQuant low =
        QuantizeBlockwiseNbit(w_nk, group_size, low_bits);
    std::vector<double> col_error(static_cast<size_t>(k), 0.0);
    for (int64_t j = 0; j < k; ++j) {
      const size_t blk =
          static_cast<size_t>(j) / static_cast<size_t>(group_size);
      double sq_sum = 0.0;
      for (int64_t r = 0; r < n_rows; ++r) {
        const double s = low.scale_blocks[static_cast<size_t>(r)][blk];
        const double recon =
            low.codes_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] * s;
        const double diff =
            w_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] - recon;
        sq_sum += diff * diff;
      }
      col_error[static_cast<size_t>(j)] = sq_sum / static_cast<double>(n_rows);
    }

    std::vector<double> sensitivity(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      sensitivity[static_cast<size_t>(j)] = col_error[static_cast<size_t>(j)] /
                                            h_inv_diag[static_cast<size_t>(j)];
    }

    const int64_t num_groups = k / group_size;
    std::vector<double> group_salience(static_cast<size_t>(num_groups), 0.0);
    for (int64_t g = 0; g < num_groups; ++g) {
      double sum = 0.0;
      for (int64_t j = 0; j < group_size; ++j) {
        sum += sensitivity[static_cast<size_t>(g * group_size + j)];
      }
      group_salience[static_cast<size_t>(g)] =
          sum / static_cast<double>(group_size);
    }

    const int64_t num_high = std::clamp<int64_t>(
        static_cast<int64_t>(
            RoundHalfToEven(fraction_high * static_cast<double>(num_groups))),
        0, num_groups);
    std::vector<int64_t> order(static_cast<size_t>(num_groups));
    for (int64_t g = 0; g < num_groups; ++g) {
      order[static_cast<size_t>(g)] = g;
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      return group_salience[static_cast<size_t>(a)] >
             group_salience[static_cast<size_t>(b)];
    });
    std::unordered_set<int64_t> high_groups;
    for (int64_t i = 0; i < num_high; ++i) {
      high_groups.insert(order[static_cast<size_t>(i)]);
    }

    const BlockwiseQuant high =
        QuantizeBlockwiseNbit(w_nk, group_size, high_bits);

    Matrix codes_nk = low.codes_nk;
    Matrix scale_blocks_nk = low.scale_blocks;
    std::vector<int64_t> group_bits(static_cast<size_t>(num_groups), low_bits);
    for (int64_t g = 0; g < num_groups; ++g) {
      if (!high_groups.count(g)) {
        continue;
      }
      for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t j = 0; j < group_size; ++j) {
          const int64_t col = g * group_size + j;
          codes_nk[static_cast<size_t>(r)][static_cast<size_t>(col)] =
              high.codes_nk[static_cast<size_t>(r)][static_cast<size_t>(col)];
        }
        scale_blocks_nk[static_cast<size_t>(r)][static_cast<size_t>(g)] =
            high.scale_blocks[static_cast<size_t>(r)][static_cast<size_t>(g)];
      }
      group_bits[static_cast<size_t>(g)] = high_bits;
    }

    // codes_kn [K, N], scale_kn [num_groups, N] -- ready for
    // DequantizeLinear(axis=0, block_size=group_size).
    std::vector<int8_t> codes_kn(static_cast<size_t>(k * n_rows));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        codes_kn[static_cast<size_t>(j * n_rows + r)] =
            static_cast<int8_t>(static_cast<int64_t>(
                codes_nk[static_cast<size_t>(r)][static_cast<size_t>(j)]));
      }
    }
    std::vector<float> scale_kn(static_cast<size_t>(num_groups * n_rows));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t g = 0; g < num_groups; ++g) {
        scale_kn[static_cast<size_t>(g * n_rows + r)] = static_cast<float>(
            scale_blocks_nk[static_cast<size_t>(r)][static_cast<size_t>(g)]);
      }
    }

    const std::string prefix = c.w_name + "_slimllm";
    const std::string codes_name = unique_name(prefix + "_codes");
    SetRawInitializer(graph->add_initializer(), codes_name,
                      onnx::TensorProto::INT8, {k, n_rows}, codes_kn.data(),
                      codes_kn.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string scale_name = unique_name(prefix + "_scale");
    SetRawInitializer(graph->add_initializer(), scale_name,
                      onnx::TensorProto::FLOAT, {num_groups, n_rows},
                      scale_kn.data(), scale_kn.size() * sizeof(float),
                      sizeof(float));
    const std::string bits_name = unique_name(prefix + "_group_bits");
    SetRawInitializer(graph->add_initializer(), bits_name,
                      onnx::TensorProto::INT64, {num_groups}, group_bits.data(),
                      group_bits.size() * sizeof(int64_t), sizeof(int64_t));

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
                 {{"axis", 0}, {"block_size", group_size}});
    const std::string core = add_node("MatMul", {c.x_name, w_dequant}, "core");

    const int live_index = c.node_index + static_cast<int>(net_insertions);
    const std::string old_output = graph->node(live_index).output(0);
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
