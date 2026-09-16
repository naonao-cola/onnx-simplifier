// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// See billm_entry.h for the full rationale (including why this follows
// llm_int8_entry.h's own single-model, protobuf-level, calibration-driven
// shape, and why the dense Hessian/Cholesky machinery below is a
// deliberate local copy of gptq_entry.cpp's own, not a shared dependency)
// and onnxsim/billm.py for the technique this ports.

#include "billm_entry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
// Transcribed from onnxsim.quip_sharp's own _match_matmul_like (which
// billm.py itself imports and reuses): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1. The bias itself,
// when present, is never read or rewritten by this pass -- mirrors
// billm.py's own candidate tuple, which discards the bias name entirely.
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

// --- Tensor <-> flat float buffer, protobuf level ---------------------------
//
// Transcribed from llm_int8_entry.cpp's own ReadFloatTensor/
// SetRawInitializer (FLOAT32 only -- this pass, like its own Python
// reference onnxsim.billm, never widens to FLOAT16/BFLOAT16).

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

// --- Small dense double-precision linear algebra ----------------------------
//
// A deliberate, header-file-local TRANSCRIBED COPY of gptq_entry.cpp's
// own CholeskyLower/InverseSPD/InverseHessianCholesky (see billm_entry.h's
// own top-of-file comment for why this is a local copy rather than a
// shared dependency between the two translation units).

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

// --- Calibration: concatenated activation rows -----------------------------
//
// Transcribed from gptq_entry.cpp's own AccumulateActivationRows, on
// `model` directly (single-model shape) rather than a separate
// `float_model`.

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
            "ApplyBillm: calibration batch is missing "
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
      ActivationRows& rows = acc[name];
      if (!rows.ok) {
        rows.k = kk;
        rows.ok = true;
      } else if (rows.k != kk) {
        continue;  // Feature width changed mid-calibration; keep the
                   // first width (numpy would fail to concatenate).
      }
      const std::vector<float> data = ReadFloatTensor(tp);
      rows.data.reserve(rows.data.size() + data.size());
      for (float v : data) {
        rows.data.push_back(static_cast<double>(v));
      }
    }
  }
}

// --- BiLLM's own binary()/select_salient_columns()/block quantization ------
//
// Direct transcription of billm.py's own _sign/_binary/
// _select_salient_columns/_billm_quantize_block. Every "matrix" below is a
// flat, row-major std::vector<double> plus explicit row/col counts (no
// small-matrix class in this codebase), the same convention
// gptq_entry.cpp's own GptqQuantizeColumns already uses for its own [N, K]
// buffers.

struct Binary {
  std::vector<double> sign;  // rows * cols, row-major.
  double scale = 0.0;
};

// The paper's own binary() primitive (Algorithm 2, Equation 4): a single
// scalar scale (mean(|w|), the L2-optimal closed-form solution to
// argmin_scale ||w - scale*sign(w)||^2) for the WHOLE of `w`, plus its
// elementwise sign -- `sign(x) = 1 if x >= 0 else -1` (billm.py's own
// _sign, NOT np.sign, which maps exactly 0 to 0).
Binary ComputeBinary(const std::vector<double>& w, int64_t rows, int64_t cols) {
  Binary b;
  const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  b.sign.assign(total, 0.0);
  if (total == 0) {
    b.scale = 0.0;
    return b;
  }
  double sum_abs = 0.0;
  for (size_t i = 0; i < total; ++i) {
    const double v = w[i];
    b.sign[i] = (v >= 0.0) ? 1.0 : -1.0;
    sum_abs += std::fabs(v);
  }
  b.scale = sum_abs / static_cast<double>(total);
  return b;
}

// Gathers columns `cols` (local indices into `mat`'s own `k2` columns) out
// of a `rows` x `k2` row-major matrix into a new `rows` x `cols.size()`
// row-major matrix.
std::vector<double> GatherCols(const std::vector<double>& mat, int64_t rows,
                               int64_t k2, const std::vector<int64_t>& cols) {
  std::vector<double> out(static_cast<size_t>(rows) * cols.size());
  for (int64_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols.size(); ++c) {
      out[static_cast<size_t>(r) * cols.size() + c] =
          mat[static_cast<size_t>(r) * static_cast<size_t>(k2) +
              static_cast<size_t>(cols[c])];
    }
  }
  return out;
}

// Returns the column indices (local to `w_block`'s own `bs` columns) of
// the salient columns for one block, per the paper's Algorithm 2
// salient(): rank columns by Hessian-based sensitivity
// s_i = w_i^2 / [H_c]_ii^2 (summed per column, epsilon-floored diagonal),
// then search a bounded number of leading (most-salient) columns for the
// count that minimizes plain-binary reconstruction error of the whole
// block. Direct transcription of billm.py's own _select_salient_columns.
//
// Uses std::stable_sort where the reference uses np.argsort (not
// guaranteed stable) -- see billm_entry.h's own "Accepted numerical
// scope" note for the (vanishingly unlikely, tie-only) divergence this
// can cause.
std::vector<int64_t> SelectSalientColumns(const std::vector<double>& w_block,
                                          int64_t n, int64_t bs,
                                          const Matrix& hc_block,
                                          int64_t max_search) {
  if (bs == 0) {
    return {};
  }
  std::vector<double> col_salience(static_cast<size_t>(bs), 0.0);
  for (int64_t c = 0; c < bs; ++c) {
    double diag = hc_block[static_cast<size_t>(c)][static_cast<size_t>(c)];
    if (std::fabs(diag) < 1e-12) {
      diag = 1e-12;
    }
    double sum = 0.0;
    for (int64_t r = 0; r < n; ++r) {
      const double w =
          w_block[static_cast<size_t>(r) * static_cast<size_t>(bs) +
                  static_cast<size_t>(c)];
      const double sens = (w * w) / (diag * diag);
      sum += std::fabs(sens);
    }
    col_salience[static_cast<size_t>(c)] = sum;
  }
  std::vector<int64_t> order(static_cast<size_t>(bs));
  for (int64_t i = 0; i < bs; ++i) {
    order[static_cast<size_t>(i)] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return col_salience[static_cast<size_t>(a)] >
           col_salience[static_cast<size_t>(b)];
  });

  const int64_t upper = std::min(max_search, bs - 1);
  if (upper < 1) {
    return {};
  }

  double best_err = std::numeric_limits<double>::infinity();
  int64_t best_n = 0;
  for (int64_t i = 1; i <= upper; ++i) {
    std::vector<int64_t> sal(order.begin(), order.begin() + i);
    std::vector<int64_t> nonsal(order.begin() + i, order.end());
    const std::vector<double> w_sal = GatherCols(w_block, n, bs, sal);
    const std::vector<double> w_ns = GatherCols(w_block, n, bs, nonsal);
    const Binary bsal =
        ComputeBinary(w_sal, n, static_cast<int64_t>(sal.size()));
    const Binary bns =
        ComputeBinary(w_ns, n, static_cast<int64_t>(nonsal.size()));
    double err = 0.0;
    for (int64_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < sal.size(); ++c) {
        const double w =
            w_block[static_cast<size_t>(r) * static_cast<size_t>(bs) +
                    static_cast<size_t>(sal[c])];
        const double recon =
            bsal.sign[static_cast<size_t>(r) * sal.size() + c] * bsal.scale;
        const double d = w - recon;
        err += d * d;
      }
      for (size_t c = 0; c < nonsal.size(); ++c) {
        const double w =
            w_block[static_cast<size_t>(r) * static_cast<size_t>(bs) +
                    static_cast<size_t>(nonsal[c])];
        const double recon =
            bns.sign[static_cast<size_t>(r) * nonsal.size() + c] * bns.scale;
        const double d = w - recon;
        err += d * d;
      }
    }
    if (err < best_err) {
      best_err = err;
      best_n = i;
    }
  }
  return std::vector<int64_t>(order.begin(), order.begin() + best_n);
}

struct BillmResult {
  std::vector<int8_t> code1;  // n * k, row-major [n][k].
  std::vector<int8_t> code2;
  std::vector<double> scale1;  // k.
  std::vector<double> scale2;  // k.
};

// Runs BiLLM's block-wise salient selection, binary residual
// approximation, plain non-salient binarization, and OBC-style error
// compensation over `w_nk` ([n, k], output channel first, row-major).
// Direct transcription of billm.py's own _billm_quantize_block.
BillmResult BillmQuantizeBlock(const std::vector<double>& w_nk, int64_t n,
                               int64_t k, const Matrix& h, int64_t block_size,
                               double percdamp, int64_t max_salient_search) {
  const Matrix hc = InverseHessianCholesky(h, percdamp);

  BillmResult res;
  res.code1.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 1);
  res.code2.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
  res.scale1.assign(static_cast<size_t>(k), 0.0);
  res.scale2.assign(static_cast<size_t>(k), 0.0);

  std::vector<double> w_work = w_nk;

  for (int64_t block_start = 0; block_start < k; block_start += block_size) {
    const int64_t block_end = std::min(block_start + block_size, k);
    const int64_t bs = block_end - block_start;

    std::vector<double> w_b(static_cast<size_t>(n) * static_cast<size_t>(bs));
    for (int64_t r = 0; r < n; ++r) {
      for (int64_t c = 0; c < bs; ++c) {
        w_b[static_cast<size_t>(r) * static_cast<size_t>(bs) +
            static_cast<size_t>(c)] =
            w_work[static_cast<size_t>(r) * static_cast<size_t>(k) +
                   static_cast<size_t>(block_start + c)];
      }
    }
    Matrix hc_b(static_cast<size_t>(bs),
                std::vector<double>(static_cast<size_t>(bs)));
    for (int64_t i = 0; i < bs; ++i) {
      for (int64_t j = 0; j < bs; ++j) {
        hc_b[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            hc[static_cast<size_t>(block_start + i)]
              [static_cast<size_t>(block_start + j)];
      }
    }

    const std::vector<int64_t> sal_local =
        SelectSalientColumns(w_b, n, bs, hc_b, max_salient_search);
    std::vector<bool> sal_mask(static_cast<size_t>(bs), false);
    for (int64_t s : sal_local) {
      sal_mask[static_cast<size_t>(s)] = true;
    }
    std::vector<int64_t> nonsal_local;
    for (int64_t c = 0; c < bs; ++c) {
      if (!sal_mask[static_cast<size_t>(c)]) {
        nonsal_local.push_back(c);
      }
    }

    std::vector<double> b_block(
        static_cast<size_t>(n) * static_cast<size_t>(bs), 0.0);

    if (!sal_local.empty()) {
      const std::vector<double> w_sal = GatherCols(w_b, n, bs, sal_local);
      const Binary b1 =
          ComputeBinary(w_sal, n, static_cast<int64_t>(sal_local.size()));
      std::vector<double> r(w_sal.size());
      for (size_t i = 0; i < r.size(); ++i) {
        r[i] = w_sal[i] - b1.sign[i] * b1.scale;
      }
      const Binary b2 =
          ComputeBinary(r, n, static_cast<int64_t>(sal_local.size()));
      for (int64_t rr = 0; rr < n; ++rr) {
        for (size_t c = 0; c < sal_local.size(); ++c) {
          const size_t li = static_cast<size_t>(rr) * sal_local.size() + c;
          const double val = b1.sign[li] * b1.scale + b2.sign[li] * b2.scale;
          b_block[static_cast<size_t>(rr) * static_cast<size_t>(bs) +
                  static_cast<size_t>(sal_local[c])] = val;
        }
      }
      for (size_t c = 0; c < sal_local.size(); ++c) {
        const int64_t col_abs = block_start + sal_local[c];
        for (int64_t rr = 0; rr < n; ++rr) {
          const size_t li = static_cast<size_t>(rr) * sal_local.size() + c;
          res.code1[static_cast<size_t>(rr) * static_cast<size_t>(k) +
                    static_cast<size_t>(col_abs)] =
              static_cast<int8_t>(b1.sign[li]);
          res.code2[static_cast<size_t>(rr) * static_cast<size_t>(k) +
                    static_cast<size_t>(col_abs)] =
              static_cast<int8_t>(b2.sign[li]);
        }
        res.scale1[static_cast<size_t>(col_abs)] = b1.scale;
        res.scale2[static_cast<size_t>(col_abs)] = b2.scale;
      }
    }

    if (!nonsal_local.empty()) {
      const std::vector<double> w_ns = GatherCols(w_b, n, bs, nonsal_local);
      const Binary bns =
          ComputeBinary(w_ns, n, static_cast<int64_t>(nonsal_local.size()));
      for (int64_t rr = 0; rr < n; ++rr) {
        for (size_t c = 0; c < nonsal_local.size(); ++c) {
          const size_t li = static_cast<size_t>(rr) * nonsal_local.size() + c;
          b_block[static_cast<size_t>(rr) * static_cast<size_t>(bs) +
                  static_cast<size_t>(nonsal_local[c])] =
              bns.sign[li] * bns.scale;
        }
      }
      for (size_t c = 0; c < nonsal_local.size(); ++c) {
        const int64_t col_abs = block_start + nonsal_local[c];
        for (int64_t rr = 0; rr < n; ++rr) {
          const size_t li = static_cast<size_t>(rr) * nonsal_local.size() + c;
          res.code1[static_cast<size_t>(rr) * static_cast<size_t>(k) +
                    static_cast<size_t>(col_abs)] =
              static_cast<int8_t>(bns.sign[li]);
          res.code2[static_cast<size_t>(rr) * static_cast<size_t>(k) +
                    static_cast<size_t>(col_abs)] = 0;
        }
        res.scale1[static_cast<size_t>(col_abs)] = bns.scale;
        res.scale2[static_cast<size_t>(col_abs)] = 0.0;
      }
    }

    if (block_end < k && !nonsal_local.empty()) {
      // Forward error compensation (paper Algorithm 1 lines 12-13, the
      // same OBC/GPTQ mechanism onnxsim.gptq already implements),
      // deliberately restricted to non-salient columns -- see billm.py's
      // own extensive comment on why a salient column's own (near-
      // singular by construction) Hessian diagonal makes this unsafe to
      // compute for it.
      const int64_t future_cols = k - block_end;
      std::vector<double> diag_hc(nonsal_local.size());
      for (size_t i = 0; i < nonsal_local.size(); ++i) {
        double d = hc_b[static_cast<size_t>(nonsal_local[i])]
                       [static_cast<size_t>(nonsal_local[i])];
        if (std::fabs(d) < 1e-8) {
          d = 1e-8;
        }
        diag_hc[i] = d;
      }
      std::vector<double> err1(static_cast<size_t>(n) * nonsal_local.size());
      for (int64_t rr = 0; rr < n; ++rr) {
        for (size_t c = 0; c < nonsal_local.size(); ++c) {
          const int64_t local_col = nonsal_local[c];
          const double w =
              w_b[static_cast<size_t>(rr) * static_cast<size_t>(bs) +
                  static_cast<size_t>(local_col)];
          const double bb =
              b_block[static_cast<size_t>(rr) * static_cast<size_t>(bs) +
                      static_cast<size_t>(local_col)];
          err1[static_cast<size_t>(rr) * nonsal_local.size() + c] =
              (w - bb) / diag_hc[c];
        }
      }
      for (int64_t rr = 0; rr < n; ++rr) {
        for (int64_t f = 0; f < future_cols; ++f) {
          double acc = 0.0;
          for (size_t c = 0; c < nonsal_local.size(); ++c) {
            const int64_t abs_row = block_start + nonsal_local[c];
            acc += err1[static_cast<size_t>(rr) * nonsal_local.size() + c] *
                   hc[static_cast<size_t>(abs_row)]
                     [static_cast<size_t>(block_end + f)];
          }
          w_work[static_cast<size_t>(rr) * static_cast<size_t>(k) +
                 static_cast<size_t>(block_end + f)] -= acc;
        }
      }
    }
  }

  return res;
}

}  // namespace

onnx::ModelProto ApplyBillm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size, double percdamp, int64_t max_salient_search) {
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

  // Mirrors onnxsim.bias_correction._all_names/_unique_name exactly (base,
  // base_1, base_2, ...), the same convention llm_int8_entry.cpp's own
  // identical block uses.
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
    // [N, K], output channels first -- mirrors `w_nk = w if
    // weight_transposed else w.T`.
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

    // Hessian over every concatenated row: H = X^T X.
    Matrix h(static_cast<size_t>(k),
             std::vector<double>(static_cast<size_t>(k), 0.0));
    for (int64_t r = 0; r < num_rows; ++r) {
      for (int64_t i = 0; i < k; ++i) {
        const double vi =
            rows.data[static_cast<size_t>(r) * static_cast<size_t>(k) +
                      static_cast<size_t>(i)];
        for (int64_t j = 0; j < k; ++j) {
          h[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
              vi * rows.data[static_cast<size_t>(r) * static_cast<size_t>(k) +
                             static_cast<size_t>(j)];
        }
      }
    }

    const BillmResult res = BillmQuantizeBlock(w_nk, n, k, h, block_size,
                                               percdamp, max_salient_search);

    // Back to the stored [dim0, dim1] layout -- mirrors `code1_orig =
    // code1_nk if weight_transposed else code1_nk.T` exactly.
    std::vector<int8_t> code1_orig(static_cast<size_t>(dim0) *
                                   static_cast<size_t>(dim1));
    std::vector<int8_t> code2_orig(code1_orig.size());
    if (c.weight_transposed) {
      code1_orig = res.code1;
      code2_orig = res.code2;
    } else {
      for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
          code1_orig[static_cast<size_t>(i) * static_cast<size_t>(dim1) +
                     static_cast<size_t>(j)] =
              res.code1[static_cast<size_t>(j) * static_cast<size_t>(k) +
                        static_cast<size_t>(i)];
          code2_orig[static_cast<size_t>(i) * static_cast<size_t>(dim1) +
                     static_cast<size_t>(j)] =
              res.code2[static_cast<size_t>(j) * static_cast<size_t>(k) +
                        static_cast<size_t>(i)];
        }
      }
    }

    std::vector<float> scale1_f(res.scale1.begin(), res.scale1.end());
    std::vector<float> scale2_f(res.scale2.begin(), res.scale2.end());
    // scale1_k/scale2_k are indexed along K (the reduction dim). When
    // weight_transposed (W is [N, K], K last), a plain length-K vector
    // broadcasts against W as-is; otherwise (W is [K, N], K first) it
    // needs a trailing size-1 axis to broadcast against axis 0 instead of
    // axis -1 -- mirrors quantize_weight_only_billm's own
    // scale1_k[:, np.newaxis] exactly.
    const std::vector<int64_t> scale_dims = c.weight_transposed
                                                ? std::vector<int64_t>{k}
                                                : std::vector<int64_t>{k, 1};

    const std::string prefix = c.w_name + "_billm";
    auto add_const = [&](const std::string& suffix, int32_t data_type,
                         const std::vector<int64_t>& dims, const void* data,
                         size_t bytes, size_t elem_size) {
      const std::string name = unique_name(prefix + "_" + suffix);
      SetRawInitializer(graph->add_initializer(), name, data_type, dims, data,
                        bytes, elem_size);
      return name;
    };
    const std::string code1_name = add_const(
        "code1", onnx::TensorProto::INT8, {dim0, dim1}, code1_orig.data(),
        code1_orig.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string code2_name = add_const(
        "code2", onnx::TensorProto::INT8, {dim0, dim1}, code2_orig.data(),
        code2_orig.size() * sizeof(int8_t), sizeof(int8_t));
    const std::string scale1_name = add_const(
        "scale1", onnx::TensorProto::FLOAT, scale_dims, scale1_f.data(),
        scale1_f.size() * sizeof(float), sizeof(float));
    const std::string scale2_name = add_const(
        "scale2", onnx::TensorProto::FLOAT, scale_dims, scale2_f.data(),
        scale2_f.size() * sizeof(float), sizeof(float));

    const std::string cast1_out = unique_name(prefix + "_code1_f");
    const std::string cast2_out = unique_name(prefix + "_code2_f");
    const std::string term1_out = unique_name(prefix + "_term1");
    const std::string term2_out = unique_name(prefix + "_term2");
    const std::string dq_out = unique_name(prefix + "_dq");

    // Re-finds c.node's own CURRENT index by pointer identity -- earlier
    // candidates' own insertions can have shifted it since `candidates`
    // was built, so this cannot be cached across candidates. Mirrors
    // low_rank_compensation_entry.cpp's own identical pattern:
    // RepeatedPtrField::SwapElements (used by append_at below) swaps
    // POINTER SLOTS, never moving or freeing the pointed-to messages, so
    // a NodeProto* stays valid (and locatable by identity) across a later
    // SwapElements call -- unlike Message::Swap (a CONTENT swap between
    // two objects' own fields, llm_int8_entry.cpp's/gptq_entry.cpp's own
    // InsertEmptyNodeAt helper), which this file deliberately does NOT
    // use, since it would silently break this exact pointer-identity
    // re-lookup.
    auto* nodes = graph->mutable_node();
    int insertion_point = -1;
    for (int i = 0; i < nodes->size(); ++i) {
      if (nodes->Mutable(i) == c.node) {
        insertion_point = i;
        break;
      }
    }

    // Appends a node at the very end, then walks it back down to exactly
    // `target_index` via adjacent pointer-slot swaps -- matches
    // low_rank_compensation_entry.cpp's own identical pattern, itself
    // matching quantize_weight_only_billm's own `graph.node.insert(
    // insertion_point, new_node); insertion_point += 1` placement (ONNX
    // doesn't require topological node order, but onnxsim always leaves
    // the graph in one).
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

    // Inserts 5 new nodes directly before c.node, in the reference's own
    // insertion order (cast1, cast2, mul1, mul2, add).
    onnx::NodeProto* cast1_node =
        append_at("Cast", {code1_name}, cast1_out,
                  unique_name(prefix + "_code1_f_node"), insertion_point);
    AddIntAttribute(cast1_node, "to", onnx::TensorProto::FLOAT);

    onnx::NodeProto* cast2_node =
        append_at("Cast", {code2_name}, cast2_out,
                  unique_name(prefix + "_code2_f_node"), insertion_point + 1);
    AddIntAttribute(cast2_node, "to", onnx::TensorProto::FLOAT);

    append_at("Mul", {cast1_out, scale1_name}, term1_out,
              unique_name(prefix + "_term1_node"), insertion_point + 2);
    append_at("Mul", {cast2_out, scale2_name}, term2_out,
              unique_name(prefix + "_term2_node"), insertion_point + 3);
    append_at("Add", {term1_out, term2_out}, dq_out,
              unique_name(prefix + "_dequant_node"), insertion_point + 4);

    // Rewires c.node's own weight input (only, matching `for i, inp in
    // enumerate(node.input): if inp == w_name: node.input[i] = dq_out`)
    // to the dequantized reconstruction.
    for (int i = 0; i < c.node->input_size(); ++i) {
      if (c.node->input(i) == c.w_name) {
        c.node->set_input(i, dq_out);
      }
    }
  }

  return out;
}
