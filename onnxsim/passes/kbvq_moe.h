// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// KBVQ-MoE (Xu et al., 2026, ICLR 2026, "KBVQ-MoE: KLT-guided SVD with
// Bias-Corrected Vector Quantization for MoE Large Language Models",
// https://arxiv.org/abs/2602.11184) -- C++ port of kbvq_moe.py's own
// apply_kbvq_moe. See that module's own docstring for the full rationale:
// unlike every other MoE-aware or vector-quantization scheme already in
// onnxsim, KBVQ-MoE fits a **shared** basis across a whole router group's
// own ``E`` experts (their cross-expert redundancy), then vector-quantizes
// only each expert's own genuinely idiosyncratic residual against that
// shared basis:
//
//   1. Flatten a router group's ``fc1_experts_weights``/
//      ``fc2_experts_weights`` (handled independently) into ``[E, D]``
//      (``D`` = one expert's own flattened element count). Take the KLT
//      (Karhunen-Loeve Transform, i.e. PCA) of that stack: center it, then
//      take the top-``rank`` right singular vectors of an economy SVD of
//      the centered ``[E, D]`` matrix (numerically the same as the ``E``
//      experts' own ``D x D`` covariance matrix's top eigenvectors, but
//      without ever forming that far larger matrix -- the same closed-form
//      -SVD style onnxsim.low_rank_compensation/this repo's own
//      low_rank_compensation_entry.cpp already use).
//   2. Every expert's own projection onto that shared basis
//      (``shared_e = mean + coeff_e @ basis``) is its "free" reconstruction
//      -- paid for once per router group, not once per expert.
//   3. Each expert's own residual (``w_e - shared_e``) is then
//      vector-quantized with an ordinary per-expert k-means codebook,
//      reusing kmeans_quantization.h's own QuantizeDequantizeKMeans
//      directly (kbvq_moe.py's own default ``bits=4``/``kmeans_iters=20``
//      match kmeans_quantization_detail::kNumCodes(16)/kIters(20) exactly,
//      so this port calls that existing, already-tested function rather
//      than re-implementing 1-D Lloyd's k-means a third time in this
//      repo).
//
// Before (illustrated for a `com.microsoft::MoE` node; router_probs and
// the node's other inputs/attributes are left completely untouched):
//   Y = MoE(X, RouterProbs, W1, [B1], W2, [B2])
//       W1: fc1_experts_weights [E, inter_size, hidden_size], float
//       W2: fc2_experts_weights [E, hidden_size, inter_size], float
// After:
//   Y = MoE(X, RouterProbs, W1', [B1], W2', [B2])
//       W1'/W2': same shape/dtype as W1/W2, every expert's own slice
//                replaced by shared-KLT-basis-reconstruction plus
//                dequantized-residual (simulated/"fake" quantization --
//                the node itself, its dtype, and every other input are
//                left completely unchanged, matching kbvq_moe.py's own
//                scope exactly).
//
// SCOPE NARROWING (matching kbvq_moe.py's own documented scope exactly,
// not an additional one this port adds): only FLOAT32
// fc1_experts_weights/fc2_experts_weights are quantized (a matched node
// whose fc1/fc2 is FLOAT16/BFLOAT16 has that tensor left untouched, same
// as kbvq_moe.py's own `if init.data_type != onnx.TensorProto.FLOAT:
// continue`); the node-matching conditions themselves (activation_type,
// fc3 absence, shape/single-consumer checks) mirror pruning.py's own
// `_match_moe_producer`/`_find_moe_chains` (reused, unmodified, by
// kbvq_moe.py itself) exactly, since those functions gate on
// `_is_supported_float_dtype` (FLOAT/FLOAT16/BFLOAT16), not FLOAT32 alone
// -- see this file's own patternMatchPredicate/runTransform for the exact
// condition list, each annotated with which piece of `_match_moe_producer`
// it mirrors. The paper's own channel-wise affine output bias correction
// is out of scope here too, exactly as it is in kbvq_moe.py itself (see
// that module's own docstring, last paragraph) -- a natural follow-up, not
// something this port silently drops relative to its own Python
// counterpart.
//
// This port also hardcodes kbvq_moe.py's own defaults (rank=4, bits=4,
// kmeans_iters=20) rather than exposing them as parameters -- several
// other *_cpp ports in this repo already establish that a C++ port need
// not mirror every optional knob its Python counterpart has.
//
// SVD IMPLEMENTATION NOTE: this file's own JacobiSvdTall/EconomySvd below
// is a deliberate, header-only COPY of onnxsim/low_rank_compensation_entry
// .cpp's own identical hand-rolled one-sided (Hestenes) Jacobi SVD (no
// linear-algebra library is linked into this codebase -- see that file's
// own top-of-file comment for why), not a shared/refactored-out dependency
// between the two: low_rank_compensation_entry.cpp is a .cpp translation
// unit (compiled once, already shipped and tested as part of PR #1534),
// while this is a header-only PredicateBasedPass registered via
// custom_optimizer_passes.cpp's RegisterOrReplace -- extracting a shared
// header here would mean refactoring that already-tested file's own
// internals as a side effect of this unrelated port, a risk judged not
// worth taking for one shared ~120-line algorithm. If a future port needs
// this a third time, that is the point to actually extract a shared
// header.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM kbvq_moe.py: none beyond what
// kmeans_quantization.h's own top-of-file comment already documents for
// QuantizeDequantizeKMeans (percentile-init k-means with a deterministic,
// repeat-largest-centroid padding fallback in the rare too-few-distinct-
// percentiles case, instead of kbvq_moe.py's own seeded-random-sample
// fallback, inherited unchanged from kmeans_quantization.py's own
// _kmeans_1d) -- reused here, not re-derived. The KLT basis fit itself has
// no RNG at all (an ordinary, deterministic SVD), so it is expected to
// track the Python reference's own float64 numpy implementation closely,
// up to ordinary floating-point rounding/summation-order differences (the
// same caveat every other closed-form-SVD port in this repo, e.g.
// low_rank_compensation, already carries) -- individual singular
// vectors/values are not expected to match sign-for-sign, but the
// reconstructed shared component is (Eckart-Young uniqueness, the same
// argument low_rank_compensation_entry.cpp's own top-of-file comment
// makes).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/kmeans_quantization.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace kbvq_moe_detail {

constexpr int64_t kRank = 4;  // kbvq_moe.py's own default `rank`.
// kbvq_moe.py's own defaults `bits=4`/`kmeans_iters=20` match
// kmeans_quantization_detail::kNumCodes(16)/kIters(20) exactly (see this
// file's own top-of-file comment) -- no separate constants needed here.

// --- One-sided (Hestenes) Jacobi SVD -------------------------------------
// Header-only copy of onnxsim/low_rank_compensation_entry.cpp's own
// identical JacobiSvdTall/EconomySvd -- see this file's own top-of-file
// "SVD IMPLEMENTATION NOTE" for why this isn't a shared dependency.
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

inline SvdResult JacobiSvdTall(int64_t m, int64_t n,
                               const std::vector<double>& a) {
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
inline SvdResult EconomySvd(int64_t m, int64_t n,
                            const std::vector<double>& a) {
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

// Reads `t` (a constant float32 tensor of any rank) into a flat row-major
// host-order double buffer of `numel` elements -- ReadFloatMatrix's own
// convention (quantize_matmul_common.h), generalized to arbitrary rank
// since fc1/fc2_experts_weights are rank 3.
inline std::vector<double> ReadFloatFlat(const Tensor& t, int64_t numel) {
  std::vector<float> raw;
  if (t.is_raw_data()) {
    raw = ReadRawDataHostOrder<float>(t.data<float>(), numel);
  } else {
    raw = t.floats();
  }
  return std::vector<double>(raw.begin(), raw.end());
}

// The KLT (Karhunen-Loeve Transform, i.e. PCA) basis shared across the `E`
// rows of `stack_ed` ([E, D] flat row-major): the top-`kRank` eigenvectors
// of the `E` experts' own [D, D] covariance matrix, computed via one
// economy SVD of the centered [E, D] stack (see this file's own
// top-of-file comment). `basis_out` is `r x D` flat row-major,
// `r = clamp(kRank, 0, min(E, D))` (`r_out`); `r == 0` leaves `basis_out`
// empty -- every expert's own "shared" reconstruction is then just the
// group mean. Mirrors kbvq_moe.py's own _klt_basis exactly.
inline void KltBasis(const std::vector<double>& stack_ed, int64_t num_e,
                     int64_t dim_d, std::vector<double>& mean_out,
                     std::vector<double>& basis_out, int64_t& r_out) {
  mean_out.assign(static_cast<size_t>(dim_d), 0.0);
  for (int64_t e = 0; e < num_e; ++e) {
    for (int64_t d = 0; d < dim_d; ++d) {
      mean_out[static_cast<size_t>(d)] +=
          stack_ed[static_cast<size_t>(e * dim_d + d)];
    }
  }
  for (int64_t d = 0; d < dim_d; ++d) {
    mean_out[static_cast<size_t>(d)] /= static_cast<double>(num_e);
  }

  const int64_t r = std::max<int64_t>(0, std::min({kRank, num_e, dim_d}));
  r_out = r;
  basis_out.assign(static_cast<size_t>(r * dim_d), 0.0);
  if (r == 0) {
    return;
  }

  std::vector<double> centered(static_cast<size_t>(num_e * dim_d));
  for (int64_t e = 0; e < num_e; ++e) {
    for (int64_t d = 0; d < dim_d; ++d) {
      centered[static_cast<size_t>(e * dim_d + d)] =
          stack_ed[static_cast<size_t>(e * dim_d + d)] -
          mean_out[static_cast<size_t>(d)];
    }
  }

  const SvdResult svd = EconomySvd(num_e, dim_d, centered);
  // svd.v is [dim_d, svd.k] row-major, svd.k == min(num_e, dim_d) -- since
  // r <= min(num_e, dim_d) always (the clamp above), indexing svd.v's own
  // columns [0, r) below is always in-bounds regardless of which of
  // EconomySvd's two branches ran. basis row j (the j-th right singular
  // vector, length dim_d) is svd.v's own column j.
  for (int64_t j = 0; j < r; ++j) {
    for (int64_t d = 0; d < dim_d; ++d) {
      basis_out[static_cast<size_t>(j * dim_d + d)] =
          svd.v[static_cast<size_t>(d * svd.k + j)];
    }
  }
}

// KBVQ-MoE's own shared-basis-plus-per-expert-residual-codebook
// reconstruction of one fc1/fc2 expert-weight tensor, flattened to
// [num_e, dim_d] in place. Mirrors kbvq_moe.py's own _kbvq_reconstruct
// exactly, reusing kmeans_quantization_detail::QuantizeDequantizeKMeans
// (see this file's own top-of-file comment for why that's a safe direct
// reuse rather than a re-implementation).
inline void KbvqReconstructInPlace(std::vector<double>& w_ed, int64_t num_e,
                                   int64_t dim_d) {
  std::vector<double> mean;
  std::vector<double> basis;
  int64_t r = 0;
  KltBasis(w_ed, num_e, dim_d, mean, basis, r);

  std::vector<double> shared(static_cast<size_t>(num_e * dim_d));
  std::vector<double> coeff(static_cast<size_t>(r));
  for (int64_t e = 0; e < num_e; ++e) {
    for (int64_t j = 0; j < r; ++j) {
      double acc = 0.0;
      for (int64_t d = 0; d < dim_d; ++d) {
        acc += (w_ed[static_cast<size_t>(e * dim_d + d)] -
                mean[static_cast<size_t>(d)]) *
               basis[static_cast<size_t>(j * dim_d + d)];
      }
      coeff[static_cast<size_t>(j)] = acc;
    }
    for (int64_t d = 0; d < dim_d; ++d) {
      double acc = mean[static_cast<size_t>(d)];
      for (int64_t j = 0; j < r; ++j) {
        acc += coeff[static_cast<size_t>(j)] *
               basis[static_cast<size_t>(j * dim_d + d)];
      }
      shared[static_cast<size_t>(e * dim_d + d)] = acc;
    }
  }

  std::vector<double> residual(static_cast<size_t>(dim_d));
  for (int64_t e = 0; e < num_e; ++e) {
    for (int64_t d = 0; d < dim_d; ++d) {
      residual[static_cast<size_t>(d)] =
          w_ed[static_cast<size_t>(e * dim_d + d)] -
          shared[static_cast<size_t>(e * dim_d + d)];
    }
    kmeans_quantization_detail::QuantizeDequantizeKMeans(residual.data(),
                                                         dim_d);
    for (int64_t d = 0; d < dim_d; ++d) {
      w_ed[static_cast<size_t>(e * dim_d + d)] =
          shared[static_cast<size_t>(e * dim_d + d)] +
          residual[static_cast<size_t>(d)];
    }
  }
}

}  // namespace kbvq_moe_detail

// KBVQ-MoE -- matches a `com.microsoft::MoE` node the same structural way
// pruning.py's own `_match_moe_producer` does (reused, unmodified, by
// kbvq_moe.py itself), then reconstructs each of fc1_experts_weights/
// fc2_experts_weights (independently, FLOAT32 only) via the shared-KLT-
// basis-plus-per-expert-residual-codebook scheme above.
struct KbvqMoe final : public PredicateBasedPass {
  explicit KbvqMoe()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "kbvq_moe"; }

  // Mirrors _match_moe_producer's own structural conditions (see that
  // function's own docstring in pruning.py, reused unmodified by
  // kbvq_moe.py, for the exact safety argument each check below draws
  // from) -- NOT gated on FLOAT32 specifically here (kbvq_moe.py's own
  // FLOAT32-only restriction is a per-tensor decision inside
  // apply_kbvq_moe's own loop, mirrored in runTransform below instead).
  static bool MatchMoeNode(Node* n, Tensor const*& fc1_t,
                           Tensor const*& fc2_t) {
    if (n->kind() != Symbol("MoE") || !n->has_domain() ||
        n->domain() != "com.microsoft") {
      return false;
    }
    std::string activation = n->hasAttribute(Symbol("activation_type"))
                                 ? n->s(Symbol("activation_type"))
                                 : "relu";
    const int64_t swiglu_fusion = n->hasAttribute(Symbol("swiglu_fusion"))
                                      ? n->i(Symbol("swiglu_fusion"))
                                      : 0;
    if ((activation != "relu" && activation != "identity" &&
         activation != "silu" && activation != "gelu") ||
        swiglu_fusion != 0) {
      return false;
    }
    // fc3_experts_weights (input 6), if provided at all (even as an
    // explicit non-empty name), is out of scope -- no CPU MoE kernel in
    // this environment implements it (see pruning.py's own comment).
    if (n->inputs().size() > 6 && n->input(6)->node()->kind() != kUndefined) {
      return false;
    }
    if (n->inputs().size() < 5) {
      return false;
    }
    Value* fc1_w = n->input(2);
    Value* fc2_w = n->input(4);
    if (fc1_w->node()->kind() == kUndefined ||
        fc2_w->node()->kind() == kUndefined) {
      return false;
    }
    const Tensor* fc1 = FetchConstantTensor(fc1_w);
    const Tensor* fc2 = FetchConstantTensor(fc2_w);
    if (fc1 == nullptr || fc2 == nullptr || fc1_w->uses().size() != 1 ||
        fc2_w->uses().size() != 1) {
      return false;
    }
    if (!IsSupportedFloatDtype(fc1->elem_type()) ||
        !IsSupportedFloatDtype(fc2->elem_type()) || fc1->sizes().size() != 3 ||
        fc2->sizes().size() != 3) {
      return false;
    }
    const int64_t num_experts = fc1->sizes()[0];
    const int64_t inter_size = fc1->sizes()[1];
    const int64_t hidden_size = fc1->sizes()[2];
    if (fc2->sizes()[0] != num_experts || fc2->sizes()[1] != hidden_size ||
        fc2->sizes()[2] != inter_size) {
      return false;
    }
    // Optional fc1_experts_bias (input 3) / fc2_experts_bias (input 5), if
    // present, must be constant, a supported float dtype, single-consumer,
    // and exactly [num_experts, inter_size]/[num_experts, hidden_size] --
    // matched here (to confirm a well-formed MoE node) but never touched
    // by this pass.
    if (!ValidOptionalBias(n, 3, num_experts, inter_size) ||
        !ValidOptionalBias(n, 5, num_experts, hidden_size)) {
      return false;
    }
    fc1_t = fc1;
    fc2_t = fc2;
    return true;
  }

  static bool IsSupportedFloatDtype(int32_t dt) {
    return dt == TensorProto_DataType_FLOAT ||
           dt == TensorProto_DataType_FLOAT16 ||
           dt == TensorProto_DataType_BFLOAT16;
  }

  static bool ValidOptionalBias(Node* n, size_t idx, int64_t dim0,
                                int64_t dim1) {
    if (n->inputs().size() <= idx ||
        n->input(idx)->node()->kind() == kUndefined) {
      return true;  // absent -- fine, nothing to check.
    }
    Value* b = n->input(idx);
    const Tensor* bt = FetchConstantTensor(b);
    if (bt == nullptr || !IsSupportedFloatDtype(bt->elem_type()) ||
        b->uses().size() != 1 || bt->sizes().size() != 2 ||
        bt->sizes()[0] != dim0 || bt->sizes()[1] != dim1) {
      return false;
    }
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    const Tensor* fc1 = nullptr;
    const Tensor* fc2 = nullptr;
    return MatchMoeNode(n, fc1, fc2);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const Tensor* fc1 = nullptr;
    const Tensor* fc2 = nullptr;
    if (!MatchMoeNode(n, fc1, fc2)) {
      return false;
    }

    bool changed = false;
    // input index 2 = fc1_experts_weights, 4 = fc2_experts_weights.
    for (const auto& pair : {std::pair<size_t, const Tensor*>{2, fc1},
                             std::pair<size_t, const Tensor*>{4, fc2}}) {
      const size_t input_idx = pair.first;
      const Tensor* w_t = pair.second;
      if (w_t->elem_type() != TensorProto_DataType_FLOAT) {
        continue;  // FLOAT16/BFLOAT16 out of scope -- see top-of-file note.
      }
      const auto& sizes = w_t->sizes();
      const int64_t num_e = sizes[0];
      const int64_t dim_d = sizes[1] * sizes[2];
      const int64_t numel = num_e * dim_d;

      std::vector<double> w_ed = kbvq_moe_detail::ReadFloatFlat(*w_t, numel);
      kbvq_moe_detail::KbvqReconstructInPlace(w_ed, num_e, dim_d);

      Tensor w_out;
      w_out.elem_type() = TensorProto_DataType_FLOAT;
      w_out.sizes() = sizes;
      std::vector<float> out_float(w_ed.begin(), w_ed.end());
      w_out.floats() = std::move(out_float);

      Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
      n->replaceInput(input_idx, w_out_v);
      changed = true;
    }
    return changed;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
