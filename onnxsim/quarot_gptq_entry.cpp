// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See quarot_gptq_entry.h for the full rationale (including why this
// combines gptq_entry.h's own calibration/Hessian machinery with
// passes/quarot.h's own rotation/node-emission shape) and
// onnxsim/quarot.py's apply_quarot_gptq for the technique this ports.

#include "quarot_gptq_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_bridge.h"
#include "onnxsim.h"
#include "passes/random_orthogonal.h"

namespace {

// --- MatMul/vanilla-Gemm matching, protobuf level --------------------------
//
// Transcribed from llm_int8_entry.cpp's own MatchMatMulLike (itself
// transcribed from smoothquant_entry.cpp's): the same MatMul / vanilla-Gemm
// acceptance, plus the bias input name (or empty when there is none).
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
//
// Transcribed from gptq_entry.cpp's own ReadFloatTensor (FLOAT32 only,
// mirroring apply_quarot_gptq's own FLOAT-only weight/activation scope).
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

// --- Small dense double-precision linear algebra ----------------------------
//
// Transcribed verbatim from gptq_entry.cpp's own kernels -- see that file's
// own comment for why scalar kernels rather than LAPACK, and the accepted
// numerical scope that follows from it.
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

// GPTQ's sequential, Hessian-compensated integer-code search -- transcribed
// verbatim from gptq_entry.cpp's own GptqQuantizeColumns. w_nk/scale_blocks
// are [N, K] / [N, K/block_size]; returns the [N, K] code matrix (float64,
// integral values in [-7, 7]).
Matrix GptqQuantizeColumns(const Matrix& w_nk, const Matrix& scale_blocks,
                           int64_t block_size, const Matrix& h, double percdamp,
                           int64_t proc_block_size) {
  const size_t n = w_nk.size();
  const size_t k = w_nk[0].size();
  const Matrix hinv = InverseHessianCholesky(h, percdamp);

  Matrix codes(n, std::vector<double>(k, 0.0));
  Matrix w_work = w_nk;

  for (int64_t block_start = 0; block_start < static_cast<int64_t>(k);
       block_start += proc_block_size) {
    const int64_t block_end =
        std::min(block_start + proc_block_size, static_cast<int64_t>(k));
    const size_t bs = static_cast<size_t>(block_end - block_start);
    Matrix w1(n, std::vector<double>(bs));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < bs; ++j) {
        w1[i][j] = w_work[i][static_cast<size_t>(block_start) + j];
      }
    }
    Matrix err1(n, std::vector<double>(bs, 0.0));

    for (size_t i = 0; i < bs; ++i) {
      const int64_t k_abs = block_start + static_cast<int64_t>(i);
      const size_t group = static_cast<size_t>(k_abs / block_size);
      for (size_t r = 0; r < n; ++r) {
        const double s = scale_blocks[r][group];
        const double w_col = w1[r][i];
        double code = RoundHalfToEven(w_col / s);
        code = std::min(7.0, std::max(-7.0, code));
        codes[r][static_cast<size_t>(k_abs)] = code;
        const double d = hinv[static_cast<size_t>(block_start) + i]
                             [static_cast<size_t>(block_start) + i];
        const double err = (w_col - code * s) / d;
        err1[r][i] = err;
        for (size_t j = i + 1; j < bs; ++j) {
          w1[r][j] -= err * hinv[static_cast<size_t>(block_start) + i]
                                [static_cast<size_t>(block_start) + j];
        }
      }
    }

    if (block_end < static_cast<int64_t>(k)) {
      for (size_t r = 0; r < n; ++r) {
        for (int64_t j = block_end; j < static_cast<int64_t>(k); ++j) {
          double acc = 0.0;
          for (size_t i = 0; i < bs; ++i) {
            acc += err1[r][i] * hinv[static_cast<size_t>(block_start) + i]
                                    [static_cast<size_t>(j)];
          }
          w_work[r][static_cast<size_t>(j)] -= acc;
        }
      }
    }
  }
  return codes;
}

// Low-nibble-first INT4 packing straight into [K, N] layout, ceil-dividing
// the byte count -- transcribed from passes/quarot.h's own inline packing
// loop (not gptq_entry.cpp's PackInt4, which assumes an even element count:
// this pass's K * N need not be even, since block_size need not divide N).
std::string PackInt4Kn(const std::vector<int8_t>& codes_kn) {
  const int64_t numel = static_cast<int64_t>(codes_kn.size());
  std::string packed(static_cast<size_t>((numel + 1) / 2), '\0');
  for (int64_t i = 0; i < numel; ++i) {
    const uint8_t nibble =
        static_cast<uint8_t>(codes_kn[static_cast<size_t>(i)]) & 0x0F;
    uint8_t& byte =
        reinterpret_cast<uint8_t&>(packed[static_cast<size_t>(i / 2)]);
    if (i % 2 == 0) {
      byte = nibble;
    } else {
      byte = static_cast<uint8_t>(byte | (nibble << 4));
    }
  }
  return packed;
}

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own ActivationRows/
// AccumulateActivationRows: every observed 2-D-or-higher FLOAT32 activation
// flattened to rows (`reshape(-1, K)`, exact) and concatenated across
// batches. Rank < 2 resolves to no rows at all.
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
            "ApplyQuarotGptq: calibration batch is missing "
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

// Inserts a fresh node at position `index` (shifting later nodes right) --
// transcribed from llm_int8_entry.cpp's own InsertEmptyNodeAt.
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

onnx::ModelProto ApplyQuarotGptq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    uint64_t seed, int64_t block_size, double percdamp, int64_t proc_block_size,
    float epsilon) {
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
    int64_t n;
    int64_t k;
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
    const int64_t dim0 = w_init.dims(0);
    const int64_t dim1 = w_init.dims(1);
    const int64_t n_rows = m->weight_transposed ? dim0 : dim1;
    const int64_t k = m->weight_transposed ? dim1 : dim0;
    if (k % block_size != 0) {
      continue;  // Rotation-eligibility check, same as quarot.py's own
                 // "Pass 1" loop -- a candidate failing this never draws a
                 // rotation at all.
    }
    candidates.push_back({i, m->x_name, m->w_name, m->bias_name,
                          m->weight_transposed, n_rows, k});
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
    auto ait = activations.find(c.x_name);
    if (ait == activations.end() || !ait->second.ok) {
      continue;  // No usable calibration activation -- leave untouched.
    }
    const ActivationRows& rows = ait->second;
    if (rows.k != c.k) {
      continue;  // Activation's feature dim doesn't match K -- leave
                 // untouched.
    }
    const int64_t num_rows =
        static_cast<int64_t>(rows.data.size()) / (c.k == 0 ? 1 : c.k);
    const int64_t n_rows = c.n;
    const int64_t k = c.k;
    const int64_t num_blocks = k / block_size;

    // ACCEPTED, PERMANENT DIVERGENCE from quarot.py's own RNG derivation
    // (see quarot_gptq_entry.h and passes/quarot.h's own identical note):
    // a fresh std::mt19937_64 reseeded per matched node, rather than a
    // single numpy.random.Generator sequenced across matches.
    std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ULL *
                                (static_cast<uint64_t>(c.node_index) + 1)));
    const std::vector<float> u =
        onnx::optimization::onnxsim_passes::RandomOrthogonalMatrix(
            k, rng);  // [K, K], row-major.

    const onnx::TensorProto& w_init = graph->initializer(init_index[c.w_name]);
    const std::vector<float> w_flat = ReadFloatTensor(w_init);
    Matrix w_nk(static_cast<size_t>(n_rows),
                std::vector<double>(static_cast<size_t>(k)));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t j = 0; j < k; ++j) {
        w_nk[static_cast<size_t>(r)][static_cast<size_t>(j)] =
            static_cast<double>(
                c.weight_transposed
                    ? w_flat[static_cast<size_t>(r * k + j)]
                    : w_flat[static_cast<size_t>(j * n_rows + r)]);
      }
    }

    // w_tilde_nk = w_nk @ u -- [N, K], exact before quantization.
    Matrix w_tilde_nk(static_cast<size_t>(n_rows),
                      std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t col = 0; col < k; ++col) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += w_nk[static_cast<size_t>(r)][static_cast<size_t>(kk)] *
                 static_cast<double>(u[static_cast<size_t>(kk * k + col)]);
        }
        w_tilde_nk[static_cast<size_t>(r)][static_cast<size_t>(col)] = acc;
      }
    }

    // x_rotated = rows @ u -- [S, K], then H = x_rotated^T @ x_rotated.
    std::vector<double> x_rotated(static_cast<size_t>(num_rows * k), 0.0);
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t col = 0; col < k; ++col) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += rows.data[static_cast<size_t>(r * k + kk)] *
                 static_cast<double>(u[static_cast<size_t>(kk * k + col)]);
        }
        x_rotated[static_cast<size_t>(r * k + col)] = acc;
      }
    }
    Matrix h(static_cast<size_t>(k),
             std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi = x_rotated[static_cast<size_t>(r * k + i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * x_rotated[static_cast<size_t>(r * k + j)];
        }
      }
    }

    // Same per-(output channel, block) scale onnxsim.omniquant's own
    // _quantize_blockwise_int4_with_clip computes at clip_ratio == 1.0:
    // max(max(|w| in block), 1e-12) / 7 -- only the scale is reused here
    // (GPTQ's own column algorithm decides which integer each element
    // rounds to).
    Matrix scale_blocks_nk(
        static_cast<size_t>(n_rows),
        std::vector<double>(static_cast<size_t>(num_blocks)));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t blk = 0; blk < num_blocks; ++blk) {
        double max_abs = 0.0;
        for (int64_t j = 0; j < block_size; ++j) {
          const int64_t col = blk * block_size + j;
          max_abs = std::max(max_abs, std::fabs(w_tilde_nk[static_cast<size_t>(
                                          r)][static_cast<size_t>(col)]));
        }
        scale_blocks_nk[static_cast<size_t>(r)][static_cast<size_t>(blk)] =
            std::max(max_abs, 1e-12) / 7.0;
      }
    }

    const Matrix codes_nk = GptqQuantizeColumns(
        w_tilde_nk, scale_blocks_nk, block_size, h, percdamp, proc_block_size);

    std::vector<int8_t> codes_kn(static_cast<size_t>(k * n_rows));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t col = 0; col < k; ++col) {
        codes_kn[static_cast<size_t>(col * n_rows + r)] = static_cast<int8_t>(
            codes_nk[static_cast<size_t>(r)][static_cast<size_t>(col)]);
      }
    }
    std::vector<float> scale_kn(static_cast<size_t>(num_blocks * n_rows));
    for (int64_t r = 0; r < n_rows; ++r) {
      for (int64_t blk = 0; blk < num_blocks; ++blk) {
        scale_kn[static_cast<size_t>(blk * n_rows + r)] = static_cast<float>(
            scale_blocks_nk[static_cast<size_t>(r)][static_cast<size_t>(blk)]);
      }
    }
    const std::string packed = PackInt4Kn(codes_kn);

    const std::string prefix = c.w_name + "_quarot_gptq";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string codes_name =
        add_const("codes", onnx::TensorProto::INT4, {k, n_rows}, packed.data(),
                  packed.size(), 1);
    const std::string scale_name = add_const(
        "scale", onnx::TensorProto::FLOAT, {num_blocks, n_rows},
        scale_kn.data(), scale_kn.size() * sizeof(float), sizeof(float));
    std::vector<float> u_f32(u.begin(), u.end());
    const std::string u_name =
        add_const("u", onnx::TensorProto::FLOAT, {k, k}, u_f32.data(),
                  u_f32.size() * sizeof(float), sizeof(float));
    const float eps_f = epsilon;
    const float seven_f = 7.0f;
    const float clip_min_f = -7.0f;
    const float clip_max_f = 7.0f;
    const int64_t neg_one = -1;
    const std::string eps_name =
        add_const("eps", onnx::TensorProto::FLOAT, {}, &eps_f, sizeof(eps_f),
                  sizeof(float));
    const std::string seven_name =
        add_const("seven", onnx::TensorProto::FLOAT, {}, &seven_f,
                  sizeof(seven_f), sizeof(float));
    const std::string clip_min_name =
        add_const("clip_min", onnx::TensorProto::FLOAT, {}, &clip_min_f,
                  sizeof(clip_min_f), sizeof(float));
    const std::string clip_max_name =
        add_const("clip_max", onnx::TensorProto::FLOAT, {}, &clip_max_f,
                  sizeof(clip_max_f), sizeof(float));
    const std::string axes_name =
        add_const("reduce_axes", onnx::TensorProto::INT64, {1}, &neg_one,
                  sizeof(neg_one), sizeof(int64_t));

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

    const std::string x_rotated_name =
        add_node("MatMul", {c.x_name, u_name}, "x_rotated");
    const std::string x_abs = add_node("Abs", {x_rotated_name}, "x_abs");
    const std::string x_max =
        add_node("ReduceMax", {x_abs, axes_name}, "x_max", {{"keepdims", 1}});
    const std::string x_safe_max =
        add_node("Clip", {x_max, eps_name}, "x_safe_max");
    const std::string x_scale =
        add_node("Div", {x_safe_max, seven_name}, "x_scale");
    const std::string x_scaled =
        add_node("Div", {x_rotated_name, x_scale}, "x_scaled");
    const std::string x_rounded = add_node("Round", {x_scaled}, "x_rounded");
    const std::string x_clipped = add_node(
        "Clip", {x_rounded, clip_min_name, clip_max_name}, "x_clipped");
    const std::string x_dequant =
        add_node("Mul", {x_clipped, x_scale}, "x_dequant");
    const std::string w_dequant =
        add_node("DequantizeLinear", {codes_name, scale_name}, "w_dequant",
                 {{"axis", 0}, {"block_size", block_size}});
    const std::string core = add_node("MatMul", {x_dequant, w_dequant}, "core");

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
