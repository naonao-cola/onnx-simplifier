// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// D2Quant's Dual-Scale Quantizer (DSQ) (Yan, Bao, Li, Zhang, Zhang, Xie,
// Sun and Zhang, 2026, "D2Quant: Accurate Low-bit Post-Training Weight
// Quantization for LLMs") -- C++ port of d2quant.py's own apply_dsq. See
// that module's own docstring for the full rationale: a weight-side fix
// targeted specifically at down-projection matrices in a SwiGLU/GLU-style
// MLP block (``down(silu(gate(x)) * up(x))``) -- the gated activation
// feeding a down-projection has an unusually heavy-tailed distribution, a
// documented quantization bottleneck. DSQ derives a per-input-channel
// auxiliary scale for the down-projection (fit by alternating: quantize
// W/s with an ordinary blockwise INT4 quantizer, then re-solve s in closed
// form as a per-column weighted least squares against the quantized
// reconstruction) and folds its reciprocal into the paired up-projection's
// own raw weight -- "absorbable," no new runtime op beyond the
// down-projection's own DequantizeLinear.
//
// Match (mirrors apply_dsq's own matching exactly): a MatMul/vanilla-Gemm
// node (the down-projection) whose activation input is produced by an
// elementwise Mul with exactly two operands, with exactly one consumer
// itself and not a graph output; at least one of that Mul's own two
// operands must be *directly* the output of another MatMul/vanilla-Gemm
// node (the up-projection) with exactly one consumer, itself not a graph
// output, whose own weight likewise has exactly one consumer (only a
// fully-owned up-projection's weight is safe to rescale) -- the shape any
// SwiGLU MLP or plain bilinear GLU takes, regardless of which Mul operand
// carries the nonlinearity (only the *unactivated* operand's producer, the
// up-projection, is rescaled; scaling before a nonlinearity would not
// commute with it). Both weights must be constant 2-D float32 tensors
// whose shared dimension (down's own reduction axis == up's own output
// axis) is divisible by block_size.
//
// Before (illustrated for MatMul; a "vanilla" Gemm is handled the same
// way):
//   up_out = MatMul(x, Wup)                    -- Wup constant, [K, H]
//   gated  = Mul(NonlinearFn(...), up_out)      -- or Mul(up_out, ...);
//                                                  operand order doesn't
//                                                  matter
//   y      = MatMul(gated, Wdown) [+ bias]      -- Wdown constant, [H, N]
// After (opset 21+ only -- INT4 tensors and DequantizeLinear's own
// block_size attribute both need it):
//   Wup'   = Wup, rescaled per output column h by 1/s[h]   -- SAME
//            initializer values, no new node
//   Wq     = <int4, per-(block, output-channel) symmetric, fit to
//             Wdown / s rather than Wdown directly>
//   Ws     = <float32, per-(block, output-channel) scale>
//   Wdq    = DequantizeLinear(Wq, Ws, axis=<reduction axis>,
//            block_size=block_size)
//   up_out = MatMul(x, Wup')
//   gated  = Mul(NonlinearFn(...), up_out)      -- now pre-scaled by s
//   y      = MatMul(gated, Wdq) [+ bias]        -- s cancels exactly (up
//            to quantization error in Wdq itself)
//
// SCOPE NARROWING: this port hardcodes d2quant.py's own defaults
// (block_size=32, num_iterations=15) rather than exposing them as
// parameters -- several other *_cpp ports in this repo already establish
// that a C++ port need not mirror every optional knob its Python
// counterpart has.
//
// ACCEPTED, PERMANENT DIVERGENCE: none -- DSQ's own alternating scale fit
// is a closed-form, deterministic procedure with no RNG anywhere (a fixed
// number of alternating least-squares/re-quantization steps from a fixed
// s=1 initialization), so this port is expected to track d2quant.py's own
// float64 numpy implementation closely, up to ordinary floating-point
// summation-order differences.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace d2quant_dsq_detail {

constexpr int64_t kBlockSize = 32;
constexpr int64_t kNumIterations = 15;

inline bool IsGraphOutput(Graph& graph, Value* v) {
  for (Value* out : graph.outputs()) {
    if (out == v) {
      return true;
    }
  }
  return false;
}

// Round-half-to-even (banker's rounding), matching numpy's own `np.round`
// -- d2quant.py's own _quantize_int4_blockwise_symmetric uses np.round,
// unlike std::round's half-away-from-zero.
inline double RoundHalfToEven(double v) {
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

// Ordinary symmetric INT4 block quantization -- one absmax-derived scale
// per (output row, block_size-wide slice of K) -- direct transcription of
// d2quant.py's own _quantize_int4_blockwise_symmetric, over a flat
// row-major [n, k] buffer. codes_out/scale_out are resized in place.
inline void QuantizeInt4BlockwiseSymmetric(const std::vector<double>& w_nk,
                                           int64_t n, int64_t k,
                                           int64_t block_size,
                                           std::vector<double>& codes_out,
                                           std::vector<double>& scale_out) {
  const int64_t num_blocks = k / block_size;
  codes_out.assign(static_cast<size_t>(n * k), 0.0);
  scale_out.assign(static_cast<size_t>(n * num_blocks), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      double amax = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const double v = w_nk[static_cast<size_t>(i * k + b * block_size + j)];
        amax = std::max(amax, std::fabs(v));
      }
      const double scale = std::max(amax / 7.0, 1e-12);
      scale_out[static_cast<size_t>(i * num_blocks + b)] = scale;
      for (int64_t j = 0; j < block_size; ++j) {
        const size_t idx = static_cast<size_t>(i * k + b * block_size + j);
        const double q = RoundHalfToEven(w_nk[idx] / scale);
        codes_out[idx] = std::min(7.0, std::max(-7.0, q));
      }
    }
  }
}

// Solves `min_s ||W - Q(W / s) * s||_F^2` for a per-column scale `s` (one
// scalar per reduction-dimension column, shared by every output row/block)
// by alternating: (a) freeze s, re-quantize W/s with the ordinary block
// quantizer above; (b) freeze the resulting integer codes' dequantized
// values, re-solve s in closed form (per-column weighted least squares:
// `s[h] = sum_row(W[:,h] * dequant[:,h]) / sum_row(dequant[:,h] ** 2)`).
// Direct transcription of d2quant.py's own _dsq_optimize. Returns the
// [n, k] codes / [n, num_blocks] scale from the FINAL iteration's own W/s
// quantization, plus the fitted s itself ([k]).
inline void DsqOptimize(const std::vector<double>& w_nk, int64_t n, int64_t k,
                        int64_t block_size, int64_t num_iterations,
                        std::vector<double>& s_out,
                        std::vector<double>& codes_out,
                        std::vector<double>& scale_out) {
  s_out.assign(static_cast<size_t>(k), 1.0);
  std::vector<double> w_norm(static_cast<size_t>(n * k));
  std::vector<double> dequant_norm(static_cast<size_t>(n * k));
  const int64_t num_blocks = k / block_size;

  for (int64_t iter = 0; iter < std::max<int64_t>(num_iterations, 0); ++iter) {
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        w_norm[static_cast<size_t>(i * k + j)] =
            w_nk[static_cast<size_t>(i * k + j)] /
            s_out[static_cast<size_t>(j)];
      }
    }
    QuantizeInt4BlockwiseSymmetric(w_norm, n, k, block_size, codes_out,
                                   scale_out);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t b = 0; b < num_blocks; ++b) {
        const double scale = scale_out[static_cast<size_t>(i * num_blocks + b)];
        for (int64_t j = 0; j < block_size; ++j) {
          const size_t idx = static_cast<size_t>(i * k + b * block_size + j);
          dequant_norm[idx] = codes_out[idx] * scale;
        }
      }
    }
    std::vector<double> num(static_cast<size_t>(k), 0.0);
    std::vector<double> den(static_cast<size_t>(k), 0.0);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        const size_t idx = static_cast<size_t>(i * k + j);
        num[static_cast<size_t>(j)] += w_nk[idx] * dequant_norm[idx];
        den[static_cast<size_t>(j)] += dequant_norm[idx] * dequant_norm[idx];
      }
    }
    for (int64_t j = 0; j < k; ++j) {
      if (den[static_cast<size_t>(j)] > 1e-20) {
        s_out[static_cast<size_t>(j)] =
            num[static_cast<size_t>(j)] / den[static_cast<size_t>(j)];
      }
    }
  }

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      w_norm[static_cast<size_t>(i * k + j)] =
          w_nk[static_cast<size_t>(i * k + j)] / s_out[static_cast<size_t>(j)];
    }
  }
  QuantizeInt4BlockwiseSymmetric(w_norm, n, k, block_size, codes_out,
                                 scale_out);
}

// Two's-complement nibble packing, low-nibble-first (ONNX's documented
// INT4 raw_data layout) -- same convention
// TryQuantizeWeightBlockwiseInt4InPlace (quantize_matmul_common.h) already
// establishes for this codebase.
inline std::string PackInt4(const std::vector<double>& codes_flat) {
  std::string packed(static_cast<size_t>((codes_flat.size() + 1) / 2), '\0');
  for (size_t i = 0; i < codes_flat.size(); ++i) {
    const uint8_t nibble =
        static_cast<uint8_t>(static_cast<int8_t>(codes_flat[i])) & 0x0F;
    uint8_t& byte = reinterpret_cast<uint8_t&>(packed[i / 2]);
    if (i % 2 == 0) {
      byte = nibble;
    } else {
      byte = static_cast<uint8_t>(byte | (nibble << 4));
    }
  }
  return packed;
}

struct DsqMatch {
  Node* down_node = nullptr;
  MatMulLikeInfo down_info;
  Node* up_node = nullptr;
  MatMulLikeInfo up_info;
};

// Mirrors apply_dsq's own matching loop exactly -- see this header's own
// top-of-file comment for the exact conditions.
inline bool MatchDsq(Node* n, DsqMatch& m) {
  // getOpsetVersion is a static PredicateBasedPass method -- MatchDsq is a
  // free function (shared by patternMatchPredicate/runTransform below), so
  // it must be called qualified here rather than relying on member lookup.
  const int opset = PredicateBasedPass::getOpsetVersion(*n->owningGraph());
  if (opset != 0 && opset < 21) {
    return false;
  }
  MatMulLikeInfo down_info;
  if (!MatchMatMulLike(n, down_info)) {
    return false;
  }
  Graph& graph = *n->owningGraph();
  Value* down_x = down_info.x;
  Node* gate_mul = down_x->node();
  if (gate_mul->kind() != kMul || gate_mul->inputs().size() != 2) {
    return false;
  }
  if (down_x->uses().size() != 1 || IsGraphOutput(graph, down_x)) {
    return false;
  }

  for (Value* operand : gate_mul->inputs()) {
    Node* up_node = operand->node();
    MatMulLikeInfo up_info;
    if (!MatchMatMulLike(up_node, up_info)) {
      continue;
    }
    if (operand->uses().size() != 1 || IsGraphOutput(graph, operand)) {
      continue;
    }
    if (up_info.w->uses().size() != 1) {
      continue;
    }
    const Tensor* down_w_t = FetchConstantTensor(down_info.w);
    const Tensor* up_w_t = FetchConstantTensor(up_info.w);
    if (down_w_t == nullptr || up_w_t == nullptr ||
        down_w_t->elem_type() != TensorProto_DataType_FLOAT ||
        up_w_t->elem_type() != TensorProto_DataType_FLOAT ||
        down_w_t->sizes().size() != 2 || up_w_t->sizes().size() != 2) {
      continue;
    }
    const int64_t down_k = down_info.weight_transposed ? down_w_t->sizes()[1]
                                                       : down_w_t->sizes()[0];
    const int64_t up_n =
        up_info.weight_transposed ? up_w_t->sizes()[0] : up_w_t->sizes()[1];
    if (down_k != up_n || down_k % kBlockSize != 0) {
      continue;
    }
    m.down_node = n;
    m.down_info = down_info;
    m.up_node = up_node;
    m.up_info = up_info;
    return true;
  }
  return false;
}

}  // namespace d2quant_dsq_detail

// D2Quant's Dual-Scale Quantizer -- matches a down-projection MatMul/Gemm
// whose gated activation input traces back to a paired up-projection (see
// this header's own top-of-file comment), then quantizes the down-
// projection to INT4 with an auxiliary per-column scale absorbed into the
// up-projection's own raw weight.
struct D2QuantDsq final : public PredicateBasedPass {
  explicit D2QuantDsq()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "d2quant_dsq"; }

  bool patternMatchPredicate(Node* n) override {
    d2quant_dsq_detail::DsqMatch m;
    return d2quant_dsq_detail::MatchDsq(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // This pass only ever replaces n's own weight input (and rescales the
    // paired up-projection's own weight in place, as a new initializer) --
    // n itself is never destroyed.
    destroy_current = NodeDestroyType::DestroyZero;
    d2quant_dsq_detail::DsqMatch m;
    if (!d2quant_dsq_detail::MatchDsq(n, m)) {
      return false;
    }

    const Tensor* down_w_t = FetchConstantTensor(m.down_info.w);
    const Tensor* up_w_t = FetchConstantTensor(m.up_info.w);
    const auto& down_sizes = down_w_t->sizes();
    const int64_t down_dim0 = down_sizes[0];
    const int64_t down_dim1 = down_sizes[1];
    const bool down_transposed = m.down_info.weight_transposed;
    const int64_t down_n = down_transposed ? down_dim0 : down_dim1;  // N=Dout
    const int64_t down_k = down_transposed ? down_dim1 : down_dim0;  // K=H

    const std::vector<float> down_flat = ReadFloatMatrix(*down_w_t);
    std::vector<double> w_nk(static_cast<size_t>(down_n * down_k));
    for (int64_t i = 0; i < down_n; ++i) {
      for (int64_t j = 0; j < down_k; ++j) {
        const float v = down_transposed
                            ? down_flat[static_cast<size_t>(i * down_dim1 + j)]
                            : down_flat[static_cast<size_t>(j * down_dim1 + i)];
        w_nk[static_cast<size_t>(i * down_k + j)] = static_cast<double>(v);
      }
    }

    std::vector<double> s_c, codes_nk, scale_nb;
    d2quant_dsq_detail::DsqOptimize(
        w_nk, down_n, down_k, d2quant_dsq_detail::kBlockSize,
        d2quant_dsq_detail::kNumIterations, s_c, codes_nk, scale_nb);

    // Rescale the up-projection's raw weight -- a NEW initializer,
    // rewiring up_node's own weight input, this repo's own established
    // convention rather than mutating the existing Tensor's own storage.
    // up_nk: [N=H, K] (up's own [output_channel, reduction] canonical
    // view) -- mirrors `up_nk = up_w if up_transposed else up_w.T`. Row h
    // of up_nk is up's own output channel h, exactly down's own reduction
    // channel h -- s_c[h] scales that whole row.
    const auto& up_sizes = up_w_t->sizes();
    const int64_t up_dim0 = up_sizes[0];
    const int64_t up_dim1 = up_sizes[1];
    const bool up_transposed = m.up_info.weight_transposed;
    const int64_t up_n2 = up_transposed ? up_dim0 : up_dim1;
    const int64_t up_k2 = up_transposed ? up_dim1 : up_dim0;
    ONNX_ASSERT(up_n2 == down_k);
    const std::vector<float> up_flat = ReadFloatMatrix(*up_w_t);
    std::vector<float> up_new_flat(static_cast<size_t>(up_dim0 * up_dim1));
    for (int64_t h = 0; h < up_n2; ++h) {
      for (int64_t j = 0; j < up_k2; ++j) {
        const float v = up_transposed
                            ? up_flat[static_cast<size_t>(h * up_dim1 + j)]
                            : up_flat[static_cast<size_t>(j * up_dim1 + h)];
        const float scaled = static_cast<float>(static_cast<double>(v) *
                                                s_c[static_cast<size_t>(h)]);
        if (up_transposed) {
          up_new_flat[static_cast<size_t>(h * up_dim1 + j)] = scaled;
        } else {
          up_new_flat[static_cast<size_t>(j * up_dim1 + h)] = scaled;
        }
      }
    }
    Tensor up_new_t;
    up_new_t.elem_type() = TensorProto_DataType_FLOAT;
    up_new_t.sizes() = {up_dim0, up_dim1};
    up_new_t.floats() = std::move(up_new_flat);
    Value* up_new_v = graph.addInitializerAndCreateValue(up_new_t);
    m.up_node->replaceInput(1, up_new_v);

    // codes_orig/scale_orig: back to down_w_t's own ORIGINAL [dim0, dim1]
    // storage layout -- mirrors `codes_orig = codes_nk if down_transposed
    // else codes_nk.T` / `scale_orig = scale_nb if down_transposed else
    // scale_nb.T` exactly.
    const int64_t num_blocks = down_k / d2quant_dsq_detail::kBlockSize;
    std::vector<double> codes_orig;
    std::vector<double> scale_orig;
    if (down_transposed) {
      codes_orig = codes_nk;  // Already [down_n, down_k] == [dim0, dim1].
      scale_orig = scale_nb;  // Already [down_n, num_blocks].
    } else {
      codes_orig.assign(static_cast<size_t>(down_dim0 * down_dim1), 0.0);
      scale_orig.assign(static_cast<size_t>(num_blocks * down_n), 0.0);
      for (int64_t i = 0; i < down_n; ++i) {
        for (int64_t j = 0; j < down_k; ++j) {
          codes_orig[static_cast<size_t>(j * down_dim1 + i)] =
              codes_nk[static_cast<size_t>(i * down_k + j)];
        }
        for (int64_t b = 0; b < num_blocks; ++b) {
          scale_orig[static_cast<size_t>(b * down_dim1 + i)] =
              scale_nb[static_cast<size_t>(i * num_blocks + b)];
        }
      }
    }

    Tensor wq;
    wq.elem_type() = TensorProto_DataType_INT4;
    wq.sizes() = {down_dim0, down_dim1};
    wq.set_raw_data(d2quant_dsq_detail::PackInt4(codes_orig));
    Value* wq_v = graph.addInitializerAndCreateValue(wq);

    Tensor ws;
    ws.elem_type() = TensorProto_DataType_FLOAT;
    ws.sizes() = down_transposed ? std::vector<int64_t>{down_n, num_blocks}
                                 : std::vector<int64_t>{num_blocks, down_n};
    ws.floats() = std::vector<float>(scale_orig.begin(), scale_orig.end());
    Value* ws_v = graph.addInitializerAndCreateValue(ws);

    Node* dq_node = graph.create(Symbol("DequantizeLinear"), 1);
    dq_node->addInput(wq_v);
    dq_node->addInput(ws_v);
    dq_node->i_(kaxis, down_transposed ? 1 : 0);
    dq_node->i_(Symbol("block_size"), d2quant_dsq_detail::kBlockSize);
    dq_node->insertBefore(n);
    dq_node->output()->setElemType(TensorProto_DataType_FLOAT);
    if (m.down_info.w->sizes().size() > 0) {
      dq_node->output()->setSizes(m.down_info.w->sizes());
    }

    n->replaceInput(1, dq_node->output());
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
