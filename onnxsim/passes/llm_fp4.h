// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// LLM-FP4 (Liu, Yuan, Yang, Cheng, Yang, Liu, Zhu and Xu, 2023, EMNLP,
// "LLM-FP4: 4-Bit Floating-Point Quantized Transformers") -- C++ port of
// llm_fp4.py's own quantize_weight_only_llm_fp4 (weight-only half only;
// that module's own apply_llm_fp4_activation_quantization[_per_tensor]
// functions are separate, unrelated activation-quantization passes, out of
// scope here). See that module's own docstring for the full rationale:
// unlike weight_only_quantize_mxfp4_matmul.h's own fixed E2M1 codebook and
// power-of-two scale, LLM-FP4 is a standard sign/exponent/mantissa FP4
// format whose per-block scale is an ordinary real-valued float and whose
// exponent/mantissa bit split is itself searched (E1M2, E2M1, E3M0 -- every
// way to divide FP4's 3 non-sign bits) per tensor, picking whichever
// (format, per-block scale) combination minimizes reconstruction MSE.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Codebook = <float32, [16]>       -- winning format's 16 fixed values
//   Codes    = <uint8, W's own shape>  -- codebook index per element
//   Scale    = <float32, per-(output-channel, block)>  -- real-valued,
//              NOT restricted to a power of two (unlike MXFP4's own scale)
//   What_hat = Reshape(Mul(Reshape(Gather(Codebook, Cast(Codes, INT64)),
//              <blocked shape>), Reshape(Scale, <matching blocked shape,
//              block dim singleton>)), W's own original shape)
//   Y = MatMul(X, What_hat) [+ bias]
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor whose reduction dimension K is evenly divisible by
// kBlockSize, and whose activation (input 0) is float32. ONNX has no
// native FP4 tensor type, so -- following the exact same approach
// weight_only_quantize_mxfp4_matmul.h/nf4.h already use for their own
// codebook formats -- this pass builds the dequantization out of ordinary
// opset-11+ ops (Gather + Mul), matching llm_fp4.py's own graph rewrite
// exactly rather than folding to a single initializer (the codebook/codes
// split IS the point of this family of formats, the same reasoning
// squeezellm_entry.cpp's own GatherND-based rewrite already documents for
// itself).
//
// SCOPE NARROWING: this port hardcodes llm_fp4.py's own defaults
// (block_size=32, formats=(e1m2, e2m1, e3m0), num_scale_candidates=17,
// min_clip_ratio=0.5) rather than exposing them as parameters -- several
// other *_cpp ports in this repo already establish that a C++ port need
// not mirror every optional knob its Python counterpart has. This port
// also omits llm_fp4.py's own `skip_names` parameter, for the same reason.
// Unlike a per-match-only codebook (this pass, like
// weight_only_quantize_mxfp4_matmul.h's own per-match codebook, creates a
// fresh codebook initializer for every matched layer rather than sharing
// one across the whole model, even when two layers pick the same winning
// format) -- see that file's own top-of-file comment for why (this
// codebase's own established convention for a PredicateBasedPass, whose
// instance is reused via RegisterOrReplace's std::call_once across future
// calls and so must never cache a Value*/Node* pointer into a specific
// graph across matches).
//
// ACCEPTED NUMERICAL SCOPE: this is a closed-form grid search with no RNG
// and no iterative fitting algorithm -- every candidate (format, clip
// ratio) pair is evaluated by direct reconstruction MSE, the same
// deterministic objective on both sides. This port is therefore expected
// to track llm_fp4.py's own float64 numpy implementation closely, up to
// ordinary floating-point summation-order differences and an exact tie in
// the argmin/best-error comparisons (an exact tie is broken by keeping the
// first-encountered candidate on both sides -- numpy's own np.argmin and
// this port's own strict `<` comparisons agree on that convention). No
// ACCEPTED, PERMANENT DIVERGENCE note applies here, unlike this repo's
// k-means/rotation-family ports.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace llm_fp4_detail {

constexpr int64_t kBlockSize = 32;
constexpr int64_t kNumScaleCandidates = 17;
constexpr double kMinClipRatio = 0.5;

struct FormatSpec {
  int64_t e_bits;
  int64_t m_bits;
};
// llm_fp4.py's own FP4_FORMATS default iteration order (e1m2, e2m1, e3m0)
// -- the tie-break order for the format search below (the first-
// encountered format wins a strict `<` total-error comparison, matching
// `if total_error < best_total_error`'s own first-occurrence convention).
constexpr FormatSpec kFormats[3] = {{1, 2}, {2, 1}, {3, 0}};

// The 8 non-negative magnitudes an e_bits-exponent/m_bits-mantissa 4-bit
// float evaluates to, per the standard IEEE-754-style definition (bias
// 2^(e_bits-1) - 1, exponent field 0 == subnormal). Ascending, starting at
// 0.0. Direct transcription of llm_fp4.py's own _fp4_magnitudes
// (`e_bits + m_bits` is always 3, FP4's 3 non-sign bits).
inline std::vector<double> Fp4Magnitudes(int64_t e_bits, int64_t m_bits) {
  const int64_t bias = e_bits > 0 ? (int64_t{1} << (e_bits - 1)) - 1 : 0;
  std::set<double> magnitudes;
  const int64_t exp_count = int64_t{1} << e_bits;
  const int64_t mant_count = int64_t{1} << m_bits;
  for (int64_t exp_field = 0; exp_field < exp_count; ++exp_field) {
    for (int64_t mant_field = 0; mant_field < mant_count; ++mant_field) {
      const double frac =
          static_cast<double>(mant_field) / static_cast<double>(mant_count);
      double value;
      if (exp_field == 0) {
        value = frac * std::pow(2.0, static_cast<double>(1 - bias));
      } else {
        value =
            (1.0 + frac) * std::pow(2.0, static_cast<double>(exp_field - bias));
      }
      magnitudes.insert(value);
    }
  }
  return std::vector<double>(magnitudes.begin(), magnitudes.end());
}

// The full 16 signed codes for an e_bits/m_bits FP4 format: [-max, ...,
// -0.0, 0.0, ..., max] -- the same negatives-then-positives-with-a-
// duplicate-zero layout weight_only_quantize_mxfp4_matmul.h's own
// MXFP4Codebook uses, so the nearest-code search below indexes it the same
// way. Direct transcription of llm_fp4.py's own _fp4_codebook.
inline std::vector<double> Fp4Codebook(int64_t e_bits, int64_t m_bits) {
  const std::vector<double> magnitudes = Fp4Magnitudes(e_bits, m_bits);
  std::vector<double> codebook;
  codebook.reserve(magnitudes.size() * 2);
  for (auto it = magnitudes.rbegin(); it != magnitudes.rend(); ++it) {
    codebook.push_back(-*it);
  }
  for (double m : magnitudes) {
    codebook.push_back(m);
  }
  return codebook;
}

// Grid search over kFormats x kNumScaleCandidates clip ratios (evenly
// spaced over [kMinClipRatio, 1.0]) minimizing total reconstruction MSE --
// direct transcription of llm_fp4.py's own _search_fp4_clip_ratio fused
// with _search_llm_fp4_blockwise (this port hardcodes the format list
// rather than accepting one at runtime, so both loops collapse into one
// pass over `w_nk`). `w_nk` is [N, K], output-channel-first, flat
// row-major. Each (output-channel, block) GROUP independently picks
// whichever clip ratio minimizes its own reconstruction error (mirrors
// `_search_fp4_clip_ratio`'s own per-group `improved = error < best_error`
// exactly); the format with the lowest TOTAL error summed over every group
// wins (mirrors `_search_llm_fp4_blockwise`'s own
// `if total_error < best_total_error` exactly). Fills `codes_nk` ([N, K],
// codebook indices in [0, 15]) and `scale_blocks` ([N, num_blocks],
// real-valued per-group scale) for the winning format, and returns that
// format's own 16-value codebook.
inline std::vector<double> SearchLlmFp4Blockwise(
    const std::vector<double>& w_nk, int64_t n, int64_t k, int64_t block_size,
    std::vector<uint8_t>& codes_nk, std::vector<double>& scale_blocks) {
  const int64_t num_blocks = k / block_size;
  const int64_t num_groups = n * num_blocks;

  std::vector<double> max_abs(static_cast<size_t>(num_groups), 0.0);
  for (int64_t nn = 0; nn < n; ++nn) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double m = 0.0;
      for (int64_t i = 0; i < block_size; ++i) {
        m = std::max(
            m,
            std::fabs(w_nk[static_cast<size_t>(nn * k + b * block_size + i)]));
      }
      max_abs[static_cast<size_t>(nn * num_blocks + b)] = std::max(m, 1e-30);
    }
  }

  std::vector<double> clip_ratios(static_cast<size_t>(kNumScaleCandidates));
  for (int64_t i = 0; i < kNumScaleCandidates; ++i) {
    clip_ratios[static_cast<size_t>(i)] =
        kNumScaleCandidates > 1
            ? kMinClipRatio + static_cast<double>(i) * (1.0 - kMinClipRatio) /
                                  static_cast<double>(kNumScaleCandidates - 1)
            : kMinClipRatio;
  }

  codes_nk.assign(static_cast<size_t>(n * k), 0);
  scale_blocks.assign(static_cast<size_t>(num_groups), 0.0);
  std::vector<double> best_codebook;
  double best_total_error = std::numeric_limits<double>::infinity();

  std::vector<uint8_t> fmt_codes(static_cast<size_t>(n * k));
  std::vector<double> fmt_scale(static_cast<size_t>(num_groups));
  std::vector<uint8_t> group_codes(static_cast<size_t>(block_size));
  std::vector<uint8_t> best_group_codes(static_cast<size_t>(block_size));

  for (const auto& fmt : kFormats) {
    const std::vector<double> codebook = Fp4Codebook(fmt.e_bits, fmt.m_bits);
    const double max_mag = codebook.back();
    double total_error = 0.0;

    for (int64_t nn = 0; nn < n; ++nn) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        const int64_t group = nn * num_blocks + b;
        double best_group_err = std::numeric_limits<double>::infinity();
        double best_group_scale = 0.0;

        for (double r : clip_ratios) {
          const double scale = std::max(
              max_abs[static_cast<size_t>(group)] * r / max_mag, 1e-30);
          double err = 0.0;
          for (int64_t i = 0; i < block_size; ++i) {
            const double v =
                w_nk[static_cast<size_t>(nn * k + b * block_size + i)];
            const double normalized = v / scale;
            size_t best_c = 0;
            double best_diff = std::fabs(normalized - codebook[0]);
            for (size_t c = 1; c < codebook.size(); ++c) {
              const double diff = std::fabs(normalized - codebook[c]);
              if (diff < best_diff) {
                best_diff = diff;
                best_c = c;
              }
            }
            group_codes[static_cast<size_t>(i)] = static_cast<uint8_t>(best_c);
            const double dequant_normalized = codebook[best_c];
            const double d = dequant_normalized - normalized;
            err += d * d;
          }
          err *= scale * scale;
          if (err < best_group_err) {
            best_group_err = err;
            best_group_scale = scale;
            best_group_codes = group_codes;
          }
        }

        total_error += best_group_err;
        fmt_scale[static_cast<size_t>(group)] = best_group_scale;
        for (int64_t i = 0; i < block_size; ++i) {
          fmt_codes[static_cast<size_t>(nn * k + b * block_size + i)] =
              best_group_codes[static_cast<size_t>(i)];
        }
      }
    }

    if (total_error < best_total_error) {
      best_total_error = total_error;
      best_codebook = codebook;
      codes_nk = fmt_codes;
      scale_blocks = fmt_scale;
    }
  }

  return best_codebook;
}

}  // namespace llm_fp4_detail

// LLM-FP4 weight-only quantization -- matches MatMul/vanilla-Gemm the same
// way weight_only_quantize_mxfp4_matmul.h does, then quantizes the
// constant weight via a searched (format, per-block scale) FP4 encoding
// (see this header's own top-of-file comment).
struct LlmFp4 final : public PredicateBasedPass {
  static constexpr int64_t kBlockSize = llm_fp4_detail::kBlockSize;

  explicit LlmFp4()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "llm_fp4"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    return k % kBlockSize == 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    const int64_t dim0 = w_t->sizes()[0];
    const int64_t dim1 = w_t->sizes()[1];
    // [N, K], output channel first -- mirrors llm_fp4.py's own
    // `w_nk = w if weight_transposed else w.T`.
    const int64_t n_dim = info.weight_transposed ? dim0 : dim1;
    const int64_t k_dim = info.weight_transposed ? dim1 : dim0;
    if (k_dim % kBlockSize != 0) {
      return false;
    }
    const int64_t num_blocks = k_dim / kBlockSize;

    const std::vector<float> flat = ReadFloatMatrix(*w_t);
    std::vector<double> w_nk(static_cast<size_t>(n_dim * k_dim));
    for (int64_t nn = 0; nn < n_dim; ++nn) {
      for (int64_t kk = 0; kk < k_dim; ++kk) {
        const float v = info.weight_transposed
                            ? flat[static_cast<size_t>(nn * k_dim + kk)]
                            : flat[static_cast<size_t>(kk * dim1 + nn)];
        w_nk[static_cast<size_t>(nn * k_dim + kk)] = static_cast<double>(v);
      }
    }

    std::vector<uint8_t> codes_nk;
    std::vector<double> scale_blocks;
    const std::vector<double> codebook_d =
        llm_fp4_detail::SearchLlmFp4Blockwise(w_nk, n_dim, k_dim, kBlockSize,
                                              codes_nk, scale_blocks);

    // codes_orig / scale_orig -- back to the ORIGINAL [dim0, dim1] storage
    // layout, mirroring llm_fp4.py's own `codes_orig = codes_nk if
    // weight_transposed else codes_nk.T` / `scale_orig = scale_blocks if
    // weight_transposed else scale_blocks.T` exactly.
    Tensor codes_t;
    codes_t.elem_type() = TensorProto_DataType_UINT8;
    codes_t.sizes() = {dim0, dim1};
    std::vector<uint8_t> codes_orig(static_cast<size_t>(dim0 * dim1));
    if (info.weight_transposed) {
      codes_orig = codes_nk;  // Already [dim0, dim1] == [n_dim, k_dim].
    } else {
      for (int64_t nn = 0; nn < n_dim; ++nn) {
        for (int64_t kk = 0; kk < k_dim; ++kk) {
          codes_orig[static_cast<size_t>(kk * dim1 + nn)] =
              codes_nk[static_cast<size_t>(nn * k_dim + kk)];
        }
      }
    }
    codes_t.set_raw_data(WriteRawDataLittleEndian(codes_orig));

    Tensor scale_t;
    scale_t.elem_type() = TensorProto_DataType_FLOAT;
    std::vector<float> scale_orig(static_cast<size_t>(n_dim * num_blocks));
    if (info.weight_transposed) {
      scale_t.sizes() = {n_dim, num_blocks};
      for (size_t i = 0; i < scale_blocks.size(); ++i) {
        scale_orig[i] = static_cast<float>(scale_blocks[i]);
      }
    } else {
      scale_t.sizes() = {num_blocks, n_dim};
      for (int64_t nn = 0; nn < n_dim; ++nn) {
        for (int64_t b = 0; b < num_blocks; ++b) {
          scale_orig[static_cast<size_t>(b * n_dim + nn)] = static_cast<float>(
              scale_blocks[static_cast<size_t>(nn * num_blocks + b)]);
        }
      }
    }
    scale_t.floats() = std::move(scale_orig);

    Tensor codebook_t;
    codebook_t.elem_type() = TensorProto_DataType_FLOAT;
    codebook_t.sizes() = {static_cast<int64_t>(codebook_d.size())};
    std::vector<float> codebook_f(codebook_d.begin(), codebook_d.end());
    codebook_t.floats() = std::move(codebook_f);

    Value* codebook_v = graph.addInitializerAndCreateValue(codebook_t);
    Value* codes_v = graph.addInitializerAndCreateValue(codes_t);
    Value* scale_v = graph.addInitializerAndCreateValue(scale_t);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(codes_v);
    cast->i_(kto, TensorProto_DataType_INT64);
    cast->insertBefore(n);
    cast->output()->setElemType(TensorProto_DataType_INT64);
    cast->output()->setSizes(codes_v->sizes());

    Node* gather = graph.create(Symbol("Gather"), 1);
    gather->addInput(codebook_v);
    gather->addInput(cast->output());
    gather->i_(kaxis, 0);
    gather->insertBefore(n);
    gather->output()->setElemType(TensorProto_DataType_FLOAT);
    gather->output()->setSizes(codes_v->sizes());

    // Reshape the gathered (elementwise) values and the per-block scale
    // into matching 3-D shapes that expose the block dimension explicitly,
    // so a plain Mul can broadcast each block's single scale across its
    // own block_size elements -- same reasoning
    // weight_only_quantize_mxfp4_matmul.h's own identical reshape pair
    // uses.
    std::vector<int64_t> blocked_shape;
    std::vector<int64_t> scale_shape;
    if (info.weight_transposed) {
      blocked_shape = {n_dim, num_blocks, kBlockSize};
      scale_shape = {n_dim, num_blocks, 1};
    } else {
      blocked_shape = {num_blocks, kBlockSize, n_dim};
      scale_shape = {num_blocks, 1, n_dim};
    }

    auto make_shape_initializer = [&](const std::vector<int64_t>& shape) {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = {static_cast<int64_t>(shape.size())};
      t.int64s() = shape;
      return graph.addInitializerAndCreateValue(t);
    };

    Node* reshape1 = graph.create(kReshape, 1);
    reshape1->addInput(gather->output());
    reshape1->addInput(make_shape_initializer(blocked_shape));
    reshape1->insertBefore(n);
    reshape1->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* reshape2 = graph.create(kReshape, 1);
    reshape2->addInput(scale_v);
    reshape2->addInput(make_shape_initializer(scale_shape));
    reshape2->insertBefore(n);
    reshape2->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* mul = graph.create(kMul, 1);
    mul->addInput(reshape1->output());
    mul->addInput(reshape2->output());
    mul->insertBefore(n);
    mul->output()->setElemType(TensorProto_DataType_FLOAT);

    Node* reshape3 = graph.create(kReshape, 1);
    reshape3->addInput(mul->output());
    reshape3->addInput(make_shape_initializer({dim0, dim1}));
    reshape3->insertBefore(n);
    reshape3->output()->setElemType(TensorProto_DataType_FLOAT);
    if (info.w->sizes().size() > 0) {
      reshape3->output()->setSizes(info.w->sizes());
    }

    // The MatMul/Gemm node and its activation input are left untouched;
    // only the weight input changes, to its dequantized counterpart.
    n->replaceInput(1, reshape3->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
