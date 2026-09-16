#pragma once

// Calibration-driven RotateKV (Su, Wan, Zhu, Yang, Kang, Chen, Peng, Shao,
// He, Chen, Sui, Ma, Yang, 2025, "RotateKV: Accurate and Robust 2-Bit KV
// Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations") entry
// point exposed to Python -- C++ port of onnxsim.rotatekv's own
// apply_rotatekv (see onnxsim/rotatekv.py's module docstring for the full
// technique: conjugating a KV-cache stream's own fresh Key activation (and
// compensating the matching Query) by an orthogonal rotation matrix fit
// per matched stream from that stream's own calibration-activation
// covariance, so a later INT-quantization pass sees outlier mass spread
// evenly across every channel instead of concentrated in a handful of
// persistent-outlier ones).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another SINGLE-model, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather than
// through onnxoptimizer's Node/Value IR: threading a live ModelExecutor
// plus calibration batches through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase
// (every PredicateBasedPass in onnxsim/passes/ is data-free by
// construction). Unlike every other calibration-driven port in this
// codebase so far, this one matches no MatMul/Gemm WEIGHT layer at all --
// it matches a KV-cache Concat plus a downstream decomposed-attention
// Softmax subgraph, both purely activation-side (the same two structural
// patterns onnxsim/passes/intactkv.h's own FindCandidate and
// onnxsim/passes/attention_quantization.h's own MatchAttentionQuantization
// already establish in this codebase's onnx-optimizer IR -- re-derived
// here directly against raw onnx::GraphProto/NodeProto instead, the
// convention every calibration-driven *_entry.cpp in this repo uses,
// since kv_cache_quantization.py/attention_quantization.py's own Python
// matchers are not exposed as reusable C++ symbols outside their own
// translation units).
//
// Match, per candidate (mirrors rotatekv.py's own _find_rotatekv_targets
// exactly): a Key-style KV-cache stream (rotatekv.py's own
// _find_kv_cache_candidates, filtered to exclude any stream whose
// present-output name contains ".value" -- rotatekv.py never exposes
// kv_cache_quantization.py's own value_output_names override, so this
// port doesn't either) whose own present-output name, optionally unwrapped
// through exactly one Transpose hop, is exactly the Kt operand (input 1)
// of some decomposed attention subgraph's QK^T MatMul (rotatekv.py's own
// _find_attention_candidates, which itself walks back through at most 2
// optional Mul/Div/Add hops from a Softmax's own input to find that
// MatMul -- attention_quantization.h's own FindQKMatMulProducer already
// establishes the identical hop-walking algorithm in this codebase, just
// against the optimizer IR rather than raw protobuf). A KV-cache stream
// with no such attention consumer is left completely untouched: rotating
// Key alone with no way to compensate Query would silently change the
// attention scores, and this pass never does that.
//
// Before:
//   new_key: float32 [..., seq_new, head_dim]   -- this step's fresh Key
//   present_key = Concat(past_key, new_key, axis=seq)   -- feeds the cache
//   Kt = present_key, or Transpose(present_key)          -- feeds QK^T
//   scores = MatMul(Q, Kt)
// After:
//   R: initializer, float32 [head_dim, head_dim]   -- the fitted rotation
//   new_key_rotated = MatMul(new_key, R)
//   present_key = Concat(past_key, new_key_rotated, axis=seq)  -- Concat
//                 node itself unchanged; now carries rotated data for
//                 every token cached through this same (modified) graph
//   Q_rotated = MatMul(Q, R)
//   scores = MatMul(Q_rotated, Kt)     -- Kt still reads present_key
//                 exactly as before, now transparently rotated
//
// Exactness: for any orthogonal R (R^T @ R == I), (Q @ R) @ (X @ R)^T ==
// Q @ (R @ R^T) @ X^T == Q @ X^T, so this migration is provably exact (up
// to floating-point rounding) -- this pass returns a float-equivalent
// model, no quantization happens here at all, matching rotatekv.py's own
// intended pipeline position (immediately before quantize_kv_cache).
//
// Rotation fit: the classical, closed-form eigenvector basis of the
// calibration activation's own covariance -- rotatekv.py's own substitute
// (matching onnxsim.spinquant's own identical substitution, for the same
// reason) for the paper's own more involved, non-closed-form outlier-aware
// optimization. This port computes that eigenbasis with a from-scratch
// classical (cyclic) Jacobi eigenvalue algorithm for real symmetric
// matrices, applied directly to the `head_dim x head_dim` covariance
// matrix -- NOT (see rotatekv_entry.cpp's own top-of-file comment for why)
// an economy SVD of the raw calibration data matrix the way
// low_rank_compensation_entry.cpp's/kbvq_moe.h's own "no linear-algebra
// library is linked into this codebase" fallback usually reaches for.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a `calibration_data` batch
// missing one of `model`'s own graph inputs throws `std::invalid_argument`.
// NOT subgraph-aware, matching every one of those passes' own scope
// decision. A matched stream whose own calibration activation never
// appeared in any batch, or whose own head_dim is less than 2 (nothing to
// rotate), is left untouched -- mirrors apply_rotatekv's own per-stream
// skip conditions exactly.
//
// SCOPE NARROWING (already rotatekv.py's own, not an additional one this
// port adds -- see that module's own docstring, "Scope/simplification
// relative to the paper"): this fits and applies one rotation per matched
// KV-cache stream, not one rotation per attention head the way the paper's
// own reference does (splitting a fused/flat QKV projection into per-head
// blocks structurally, for an arbitrary ONNX graph and an unknown
// num_heads, is out of scope) -- a sound, exact, purely-additive
// "one-rotation-per-Key-tensor" contribution on its own, per that same
// docstring section's own argument.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own Jacobi eigenvalue
// algorithm is not LAPACK's own eigh (what numpy's np.linalg.eigh calls
// into), so the exact eigenvector basis it returns is not expected to
// match the Python reference's own sign-for-sign or column-order-for-
// column-order -- an eigendecomposition's own eigenvectors are unique only
// up to a sign flip per vector (and up to an orthogonal rotation within
// any repeated/near-repeated eigenvalue's own subspace), and
// rotatekv.py's own exactness argument above holds for ANY orthogonal R,
// not merely the specific one LAPACK happens to return. Reconstruction
// quality (a rotation's whole point: spreading outlier mass evenly across
// channels before a later INT-quantization pass) is expected to track the
// Python reference closely since both sides fit from the identical
// closed-form covariance statistic; only the exact basis differs. This
// mirrors the same "independently correct, non-interchangeable,
// comparable-not-identical" contract this repo's own SVD-based ports
// (low_rank_compensation, kbvq_moe) already establish for their own basis
// non-uniqueness.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies RotateKV-style outlier-aware rotation preprocessing to every
// matched Key-style KV-cache stream whose own present-output name also
// feeds some attention subgraph's QK^T MatMul -- see this header's own
// top-of-file comment for the exact match/rewrite and the rotation fit's
// own documented divergence from the Python reference.
onnx::ModelProto ApplyRotateKv(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data);
