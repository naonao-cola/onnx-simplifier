// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuses a "hand-written" multi-head self-attention block -- the pattern a
// typical from-scratch eager attention implementation (e.g. a
// HuggingFace-style `nn.Linear` Q/K/V projection + reshape/transpose +
// scaled dot-product + softmax + `nn.Linear` output projection, *not*
// PyTorch's built-in `nn.MultiheadAttention`, whose legacy TorchScript
// export emits a very different ~70-node trace full of dynamic-shape
// bookkeeping this pass does not attempt to recognize) -- into a single
// ONNX Runtime "com.microsoft" contrib op, `Attention`.
//
// Before (Q/K/V bias optional; each projection may be a single Gemm(x, W,
// B[, transB=1]) node or a bare MatMul(x, W) optionally followed by a
// separate Add(B, .) node -- see MatchProjection):
//   Q = Reshape(Linear(X, Wq[, Bq]), [B,S,H,D]).Transpose([0,2,1,3])
//   K = Reshape(Linear(X, Wk[, Bk]), [B,S,H,D]).Transpose([0,2,3,1])
//   V = Reshape(Linear(X, Wv[, Bv]), [B,S,H,D]).Transpose([0,2,1,3])
//   scores = (Q @ K) / sqrt(head_size)      -- or `* (1/sqrt(head_size))`
//   attn   = Softmax(scores, axis=-1)
//   ctx    = (attn @ V).Transpose([0,2,1,3]).Reshape([B,S,H*D])
//   Y      = Linear(ctx, Wout[, Bout])
// After:
//   Wqkv = Concat(Wq, Wk, Wv, axis=1)      -- computed once, here
//   Bqkv = Concat(Bq, Bk, Bv)              -- only when Q/K/V all have bias
//   Y = Attention(X, Wqkv[, Bqkv], num_heads=H, scale=1/sqrt(head_size),
//                 qkv_hidden_sizes=[Nq, Nk, Nv])
//   Y = Linear(Y, Wout[, Bout])            -- unchanged, not part of the fuse
//
// This collapses roughly a dozen shape/matmul/softmax nodes per attention
// block into one `Attention` node (the output projection is left as its own
// Linear -- `Attention`'s own schema only covers the QKV side). Unlike
// `fuse_rms_norm.h`'s target (`RMSNormalization`, standard ONNX), `Attention`
// is a "com.microsoft" contrib op, so this pass adds that domain (version 1)
// to the model's opset imports the first time it fires, if not already
// present, and the result needs a "com.microsoft"-aware runtime to execute.
//
// K's head-split transpose is required to land directly at permutation
// [0,2,3,1] (X.Transpose([0,2,1,3]) is the more common overall convention,
// but the exact spelling of a *second*, standalone "swap the last two axes"
// transpose that some exporters emit for K specifically is not matched --
// only the single-transpose-straight-to-[0,2,3,1] form is; see MatchHeadSplit).
// Only self-attention (Q, K, V all reading the *same* source activation) with
// no attention mask, no past/present KV cache, and a Softmax over the last
// axis (`axis=-1` or the input's last dimension index -- both are safe here
// regardless of the pre-/post-opset-13 Softmax axis-semantics split, since
// the matched score tensor is always exactly rank 4 by construction, and
// "flatten everything before the last axis" degenerates to the same
// per-row reduction as "reduce the last axis in place" for a fixed rank)
// is handled. Q and K must have the same per-head size (required for the
// Q@K^T dot product to be well-formed); V's hidden size may differ, per
// `Attention`'s own documented `qkv_hidden_sizes` semantics.

#include <cstdint>
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

// A single Q/K/V Linear-style projection matched feeding `out`. `out` is
// always required to be single-use: every node this struct's `chain` records
// is destroyed once the fusion fires, so nothing else may still depend on it.
struct AttentionProjectionInfo {
  Value* input = nullptr;
  Value* weight = nullptr;  // constant 2-D float32
  Value* bias = nullptr;    // constant 1-D float32, or nullptr
  bool weight_transposed = false;
  // Node(s) strictly between (input, weight[, bias]) and `out`, ordered
  // innermost (closest to `out`'s consumer) first: just the standalone
  // MatMul node when `out` is a separate bias-Add's own output (the Add
  // node itself IS `out->node()`, so it is included too, ahead of the
  // MatMul it consumes); a single entry (`out->node()` itself) when `out`
  // is already the Gemm/MatMul.
  std::vector<Node*> chain;
};

// Matches a Linear-style projection whose result is `out`: a vanilla Gemm(x,
// W[, B]) or bare MatMul(x, W) node directly (`MatchMatMulLike`), or a
// MatMul(x, W) followed by a separate Add(B, .) bias node (the shape a plain
// `nn.Linear` exported via `MatMul` rather than `Gemm` produces). `weight`
// and, if present, `bias` must be constant float32 tensors. `out` must be
// single-use -- the caller is always going to tear down its producer chain.
inline bool MatchAttentionProjection(Value* out,
                                     AttentionProjectionInfo& info) {
  if (out->uses().size() != 1) {
    return false;
  }
  Node* node = out->node();
  if (CheckKind(node, kAdd) && node->inputs().size() == 2) {
    for (int i = 0; i < 2; ++i) {
      Value* mm_out = node->input(i);
      Value* bias = node->input(1 - i);
      if (mm_out->uses().size() != 1 || !CheckKind(mm_out, kMatMul)) {
        continue;
      }
      MatMulLikeInfo mm_info;
      if (!MatchMatMulLike(mm_out->node(), mm_info)) {
        continue;
      }
      const Tensor* bias_t = FetchConstantTensor(bias);
      if (bias_t == nullptr ||
          bias_t->elem_type() != TensorProto_DataType_FLOAT ||
          bias_t->sizes().size() != 1) {
        continue;
      }
      const Tensor* w_t = FetchConstantTensor(mm_info.w);
      if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
          w_t->sizes().size() != 2) {
        continue;
      }
      info.input = mm_info.x;
      info.weight = mm_info.w;
      info.bias = bias;
      info.weight_transposed = false;  // kMatMul only, never transposed.
      info.chain = {node, mm_out->node()};
      return true;
    }
    return false;
  }
  MatMulLikeInfo mm_info;
  if (!MatchMatMulLike(node, mm_info)) {
    return false;
  }
  const Tensor* w_t = FetchConstantTensor(mm_info.w);
  if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
      w_t->sizes().size() != 2) {
    return false;
  }
  if (mm_info.bias != nullptr) {
    const Tensor* bias_t = FetchConstantTensor(mm_info.bias);
    if (bias_t == nullptr ||
        bias_t->elem_type() != TensorProto_DataType_FLOAT ||
        bias_t->sizes().size() != 1) {
      return false;
    }
  }
  info.input = mm_info.x;
  info.weight = mm_info.w;
  info.bias = mm_info.bias;
  info.weight_transposed = mm_info.weight_transposed;
  info.chain = {node};
  return true;
}

// Reads a scalar constant as a double regardless of whether it is stored as
// float32 or float64 -- FetchSoleValueOfTensor<T> requires an exact
// elem_type match (see quantize_matmul_common.h/pass_util.h), and the
// scaling constant a real export emits (e.g. `1/sqrt(head_size)`) is
// ordinary float32, not double, so asking for T=double alone would always
// silently fail. Mirrors fuse_rms_norm.h's own FetchScalarAsFloat.
inline bool FetchScalarAsDouble(Value* v, double& out) {
  if (FetchSoleValueOfTensor(v, out)) {
    return true;
  }
  float f;
  if (FetchSoleValueOfTensor(v, f)) {
    out = static_cast<double>(f);
    return true;
  }
  return false;
}

// Matches `transposed = Transpose(Reshape(proj_out, [B, S, num_heads,
// head_size]), want_perm)`, both single-use, extracting num_heads/head_size
// from the Reshape's constant target-shape input. Appends [transpose,
// reshape] to `chain` (innermost/closest-to-consumer first).
inline bool MatchAttentionHeadSplit(Value* transposed, int64_t& num_heads,
                                    int64_t& head_size, Value*& proj_out,
                                    const std::vector<int64_t>& want_perm,
                                    std::vector<Node*>& chain) {
  if (!CheckKind(transposed, kTranspose) || transposed->uses().size() != 1) {
    return false;
  }
  Node* tr = transposed->node();
  std::vector<int64_t> perm;
  if (!GetValueFromAttr(tr, kperm, perm) || perm != want_perm) {
    return false;
  }
  Value* reshaped = tr->input(0);
  if (!CheckKind(reshaped, kReshape) || reshaped->uses().size() != 1) {
    return false;
  }
  Node* rs = reshaped->node();
  if (rs->inputs().size() != 2) {
    return false;
  }
  std::vector<int64_t> shape;
  if (!GetValueFromInput(rs->input(1), shape) || shape.size() != 4) {
    return false;
  }
  num_heads = shape[2];
  head_size = shape[3];
  if (num_heads <= 0 || head_size <= 0) {
    return false;
  }
  proj_out = rs->input(0);
  chain.push_back(tr);
  chain.push_back(rs);
  return true;
}

struct AttentionMatch {
  AttentionProjectionInfo q, k, v;
  int64_t num_heads = 0;
  double scale = 0.0;
  // Every node strictly upstream of `n` (the ctx-producing Reshape the pass
  // anchors on) down to, but not including, the QKV inputs/weights,
  // ordered innermost (closest to `n`) first, so destroying them in this
  // order always has a zero-use output to destroy. Does not include `n`
  // itself -- `n` is fully replaced by the fused Attention node's output
  // and destroyed by the pass driver instead (see FuseAttention::
  // runTransform), the same idiom fuse_rms_norm.h uses for its own anchor.
  std::vector<Node*> dead_chain;
};

// Anchors on `n` = the Reshape that produces `ctx` (the attention context,
// [B,S,H*D]) -- i.e. Reshape(Transpose(V_mm_out, [0,2,1,3]), [B,S,H*D]) --
// rather than on the (untouched, never matched or torn down) output
// projection that consumes `ctx`. This keeps the output projection, whatever
// shape it takes, entirely out of scope: only `n`'s own two inputs and
// everything transitively upstream of the Transpose are examined.
inline bool MatchAttention(Node* n, AttentionMatch& m) {
  if (!CheckKind(n, kReshape) || n->inputs().size() != 2 ||
      n->outputs().size() != 1) {
    return false;
  }
  Value* transposed_back = n->input(0);
  if (!CheckKind(transposed_back, kTranspose) ||
      transposed_back->uses().size() != 1) {
    return false;
  }
  Node* tr_back = transposed_back->node();
  std::vector<int64_t> perm_back;
  if (!GetValueFromAttr(tr_back, kperm, perm_back) ||
      perm_back != std::vector<int64_t>{0, 2, 1, 3}) {
    return false;
  }
  Value* v_matmul_out = tr_back->input(0);
  if (!CheckKind(v_matmul_out, kMatMul) || v_matmul_out->uses().size() != 1) {
    return false;
  }
  Node* v_mm = v_matmul_out->node();
  if (v_mm->inputs().size() != 2) {
    return false;
  }
  Value* softmax_out = nullptr;
  Value* v_t = nullptr;
  for (int i = 0; i < 2; ++i) {
    if (CheckKind(v_mm->input(i), kSoftmax)) {
      softmax_out = v_mm->input(i);
      v_t = v_mm->input(1 - i);
    }
  }
  if (softmax_out == nullptr || softmax_out->uses().size() != 1) {
    return false;
  }
  Node* softmax = softmax_out->node();
  if (softmax->inputs().size() != 1) {
    return false;
  }
  const int64_t sm_axis =
      GetValueFromAttrWithDefault(softmax, kaxis, int64_t(-1));
  if (sm_axis != -1 && sm_axis != 3) {
    return false;  // Must reduce the last of the 4 (B,H,S,S) axes.
  }
  Value* scores = softmax->input(0);
  if (scores->uses().size() != 1) {
    return false;
  }

  std::vector<Node*> v_head_chain;
  Value* v_proj_out = nullptr;
  int64_t v_heads = 0, v_head_size = 0;
  if (!MatchAttentionHeadSplit(v_t, v_heads, v_head_size, v_proj_out,
                               {0, 2, 1, 3}, v_head_chain) ||
      !MatchAttentionProjection(v_proj_out, m.v)) {
    return false;
  }

  Node* scores_node = scores->node();
  Value* qk = nullptr;
  if (CheckKind(scores_node, kDiv) && scores_node->inputs().size() == 2) {
    double divisor = 0.0;
    if (!FetchScalarAsDouble(scores_node->input(1), divisor) ||
        divisor == 0.0) {
      return false;
    }
    qk = scores_node->input(0);
    m.scale = 1.0 / divisor;
  } else if (CheckKind(scores_node, kMul) &&
             scores_node->inputs().size() == 2) {
    for (int i = 0; i < 2; ++i) {
      double mult = 0.0;
      if (FetchScalarAsDouble(scores_node->input(1 - i), mult)) {
        qk = scores_node->input(i);
        m.scale = mult;
        break;
      }
    }
    if (qk == nullptr) {
      return false;
    }
  } else {
    return false;
  }
  if (qk->uses().size() != 1 || !CheckKind(qk, kMatMul)) {
    return false;
  }
  Node* qk_mm = qk->node();
  if (qk_mm->inputs().size() != 2) {
    return false;
  }

  std::vector<Node*> q_head_chain;
  Value* q_proj_out = nullptr;
  int64_t q_heads = 0, q_head_size = 0;
  if (!MatchAttentionHeadSplit(qk_mm->input(0), q_heads, q_head_size,
                               q_proj_out, {0, 2, 1, 3}, q_head_chain) ||
      !MatchAttentionProjection(q_proj_out, m.q)) {
    return false;
  }

  std::vector<Node*> k_head_chain;
  Value* k_proj_out = nullptr;
  int64_t k_heads = 0, k_head_size = 0;
  if (!MatchAttentionHeadSplit(qk_mm->input(1), k_heads, k_head_size,
                               k_proj_out, {0, 2, 3, 1}, k_head_chain) ||
      !MatchAttentionProjection(k_proj_out, m.k)) {
    return false;
  }

  if (q_heads != k_heads || q_heads != v_heads || q_head_size != k_head_size) {
    return false;
  }
  m.num_heads = q_heads;

  if (m.q.input != m.k.input || m.q.input != m.v.input) {
    return false;  // Self-attention only: Q/K/V share the same source.
  }

  const bool any_bias =
      m.q.bias != nullptr || m.k.bias != nullptr || m.v.bias != nullptr;
  const bool all_bias =
      m.q.bias != nullptr && m.k.bias != nullptr && m.v.bias != nullptr;
  if (any_bias && !all_bias) {
    return false;  // Attention has one merged bias input -- all or none.
  }

  m.dead_chain.push_back(tr_back);
  m.dead_chain.push_back(v_mm);
  m.dead_chain.push_back(softmax);
  for (Node* nd : v_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.v.chain) m.dead_chain.push_back(nd);
  m.dead_chain.push_back(scores_node);
  m.dead_chain.push_back(qk_mm);
  for (Node* nd : q_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.q.chain) m.dead_chain.push_back(nd);
  for (Node* nd : k_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.k.chain) m.dead_chain.push_back(nd);
  return true;
}

// Reads `w_t` (a constant 2-D float32 tensor, [N, K] when `transposed` else
// [K, N]) into a flat, row-major [K, N] host-byte-order buffer regardless of
// storage layout -- `Attention`'s merged weight has no transpose attribute
// of its own, so a Gemm(transB=1)-sourced weight must be physically
// transposed here, unlike quantize_matmul_common.h's
// QuantizeWeightPerChannelInPlace (which deliberately keeps the source
// layout for a different rewrite).
inline std::vector<float> ReadAttentionWeightAsKN(const Tensor& w_t,
                                                  bool transposed, int64_t& K,
                                                  int64_t& N) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  K = transposed ? dim1 : dim0;
  N = transposed ? dim0 : dim1;
  const std::vector<float> data = ReadFloatMatrix(w_t);
  std::vector<float> out(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      out[static_cast<size_t>(k * N + n)] =
          transposed ? data[static_cast<size_t>(n * K + k)]
                     : data[static_cast<size_t>(k * N + n)];
    }
  }
  return out;
}

// Reads a constant 1-D float32 tensor into a flat host-byte-order buffer.
inline std::vector<float> ReadAttentionBiasVector(const Tensor& t) {
  const int64_t numel = t.sizes()[0];
  if (t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(t.data<float>(), numel);
  }
  return t.floats();
}

struct FuseAttention final : public PredicateBasedPass {
  explicit FuseAttention()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_attention"; }

  bool patternMatchPredicate(Node* n) override {
    AttentionMatch m;
    return MatchAttention(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    AttentionMatch m;
    if (!MatchAttention(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    const Tensor* wq_t = FetchConstantTensor(m.q.weight);
    const Tensor* wk_t = FetchConstantTensor(m.k.weight);
    const Tensor* wv_t = FetchConstantTensor(m.v.weight);
    int64_t kq = 0, nq = 0, kk = 0, nk = 0, kv = 0, nv = 0;
    const std::vector<float> wq =
        ReadAttentionWeightAsKN(*wq_t, m.q.weight_transposed, kq, nq);
    const std::vector<float> wk =
        ReadAttentionWeightAsKN(*wk_t, m.k.weight_transposed, kk, nk);
    const std::vector<float> wv =
        ReadAttentionWeightAsKN(*wv_t, m.v.weight_transposed, kv, nv);
    if (kq != kk || kq != kv) {
      return false;  // Q/K/V must all read the same input_hidden_size.
    }

    Tensor merged_w;
    merged_w.elem_type() = TensorProto_DataType_FLOAT;
    merged_w.sizes() = {kq, nq + nk + nv};
    merged_w.floats().resize(static_cast<size_t>(kq * (nq + nk + nv)));
    for (int64_t k = 0; k < kq; ++k) {
      float* row = merged_w.floats().data() + k * (nq + nk + nv);
      std::copy(wq.begin() + k * nq, wq.begin() + (k + 1) * nq, row);
      std::copy(wk.begin() + k * nk, wk.begin() + (k + 1) * nk, row + nq);
      std::copy(wv.begin() + k * nv, wv.begin() + (k + 1) * nv, row + nq + nk);
    }
    Value* weights_v = graph.addInitializerAndCreateValue(merged_w);

    // Always synthesize Attention's (schema-optional) bias input, even when
    // Q/K/V have none, rather than omitting it: at least one real ONNX
    // Runtime CPU build (1.29.0, verified directly against a minimal
    // hand-built model, independent of this pass) segfaults executing its
    // own `Attention` kernel with only 2 inputs -- a zero-filled bias is a
    // no-op numerically and sidesteps that kernel path entirely.
    Tensor merged_b;
    merged_b.elem_type() = TensorProto_DataType_FLOAT;
    merged_b.sizes() = {nq + nk + nv};
    merged_b.floats().assign(static_cast<size_t>(nq + nk + nv), 0.0f);
    if (m.q.bias != nullptr) {
      const Tensor* bq_t = FetchConstantTensor(m.q.bias);
      const Tensor* bk_t = FetchConstantTensor(m.k.bias);
      const Tensor* bv_t = FetchConstantTensor(m.v.bias);
      const std::vector<float> bq = ReadAttentionBiasVector(*bq_t);
      const std::vector<float> bk = ReadAttentionBiasVector(*bk_t);
      const std::vector<float> bv = ReadAttentionBiasVector(*bv_t);
      if (static_cast<int64_t>(bq.size()) != nq ||
          static_cast<int64_t>(bk.size()) != nk ||
          static_cast<int64_t>(bv.size()) != nv) {
        return false;
      }
      std::copy(bq.begin(), bq.end(), merged_b.floats().begin());
      std::copy(bk.begin(), bk.end(), merged_b.floats().begin() + nq);
      std::copy(bv.begin(), bv.end(), merged_b.floats().begin() + nq + nk);
    }
    Value* bias_v = graph.addInitializerAndCreateValue(merged_b);

    Node* attn = graph.create(Symbol("Attention"), 1);
    attn->addInput(m.q.input);
    attn->addInput(weights_v);
    attn->addInput(bias_v);
    attn->i_(Symbol("num_heads"), m.num_heads);
    attn->f_(Symbol("scale"), static_cast<float>(m.scale));
    attn->is_(Symbol("qkv_hidden_sizes"), std::vector<int64_t>{nq, nk, nv});
    attn->setDomain("com.microsoft");
    attn->insertBefore(n);
    attn->output()->copyMetadata(n->output());

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    // `n` (the ctx-producing Reshape) is fully replaced by `attn`'s output
    // -- whatever consumed `ctx` (the untouched output projection) now
    // reads straight from `attn` -- and destroyed by the pass driver below
    // (destroy_current = DestroyOne), the same idiom fuse_rms_norm.h and
    // fuse_add_bias_into_conv.h use for their own anchors.
    if (!tryReplacingAllUsesWith(n, attn)) {
      return false;
    }

    // `n` still holds its own edge into the dead chain (its input(0), the
    // Transpose's output) until we detach it here; only then does the dead
    // chain's own destroy loop below see every node's output at zero uses
    // exactly when it is destroyed, innermost (closest to `n`) first.
    n->removeInput(0);
    for (Node* dead : m.dead_chain) {
      dead->destroy();
    }

    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
