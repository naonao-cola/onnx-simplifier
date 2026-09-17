#pragma once

// Calibration-driven, provably-lossless SmoothAttention scale-migration
// entry point exposed to Python -- C++ port of onnxsim.qoq's own
// apply_smooth_attention (see onnxsim/qoq.py's module docstring, point
// "2. SmoothAttention", for the full technique: for channel `j` of the
// shared head-dim axis, `(K_j / s_j) . (Q_j * s_j) == K_j . Q_j`, so
// dividing Key's channel j by a calibrated s_j and multiplying Query's
// matching channel by the same s_j leaves the attention scores exactly
// (up to floating-point rounding) unchanged while flattening Key's own
// per-channel range -- meant to run immediately before
// onnxsim.quantize_kv_cache in a pipeline).
//
// Single-model, calibration-driven, protobuf-level shape as
// llm_int8_entry.h's own ApplyLlmInt8/daq_entry.h's own ApplyDaq (probe
// real activations through `executor`, then insert new Mul/Div nodes) --
// the same "provably lossless calibrated scale migration" graph-rewrite
// shape smoothquant_entry.h's own ApplySmoothQuant and rptq_entry.h's own
// ApplyRptq already establish, applied here to the attention QK^T pattern
// specifically instead of a weight-bearing MatMul/Gemm.
//
// Candidate matching REUSES the exact decomposed-attention matcher
// `MatchAttentionQuantization`/`AttentionQuantizationMatch` already
// established in passes/attention_quantization.h for
// apply_attention_quantization (`MatMul(Q,Kt) -> [Mul/Div] -> [Add] ->
// Softmax -> MatMul(_,V)`, anchored on the Softmax node, walking backward
// through at most 2 optional Mul/Div/Add hops to find the score MatMul
// and forward through the Softmax output's own uses to find the
// out-projection MatMul) -- but that matcher operates on onnxoptimizer's
// Node/Value IR (`onnx::Graph`), and every calibration-driven pass in
// this codebase (which all need a live `ModelExecutor` over raw
// `onnx::GraphProto`) operates at the protobuf level instead; no existing
// *_entry.cpp combines the two. So this file TRANSCRIBES an equivalent
// protobuf-level matcher (`FindQkMatmulProducer`/`FindOutMatmulConsumer`,
// named to mirror `FindQKMatMulProducer`/`FindOutMatMulConsumer` in
// attention_quantization.h exactly, node-output-name-keyed instead of
// Value-pointer-keyed) rather than re-deriving different matching logic
// -- the same self-containment convention every other *_entry.cpp already
// follows for a matcher it reuses from elsewhere in this codebase (see
// e.g. llm_int8_entry.cpp's own MatchMatMulLike, transcribed from
// smoothquant_entry.cpp's). This is also an exact transcription of
// qoq.py's own `_find_attention_candidates`/`_find_matmul_producer`
// (which `apply_smooth_attention` imports and reuses directly in Python),
// so this port's matching logic is identical to both its Python
// reference and attention_quantization.h's own C++ pass by construction.
//
// Unlike ApplyAttentionQuantization (a data-free, per-token dynamic
// round trip needing no calibration), this pass measures Key's own
// per-head-dim-channel max-abs value from `calibration_data` first,
// mirroring rptq_entry.cpp's own ComputeChannelAbsmax shape but reducing
// over every axis except the second-to-last (head_dim) one -- Kt has
// shape `[..., head_dim, seq_k]`, so this is NOT the plain last-axis
// channel reduction llm_int8_entry.cpp's/rptq_entry.cpp's own
// ComputeChannelAbsmax already implement (that reduces over every axis
// except the LAST one); a new reduction is implemented here rather than
// reused.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. NOT
// subgraph-aware, matching every one of those passes' own scope decision.
//
// A matched subgraph whose Key tensor never appeared as a plain-enough
// (rank >= 2) probe is left untouched for that subgraph -- mirrors
// apply_smooth_attention's own per-candidate skip condition exactly. A
// model with no matching subgraph at all is returned unchanged (a plain
// copy). Duplicate Softmax matches sharing the same score MatMul are
// deduplicated to one candidate, mirroring apply_smooth_attention's own
// `seen_matmuls`/`id(c.qk_matmul)` deduplication exactly (transcribed
// here by score-MatMul node index instead of Python object id, since
// protobuf offers no stable object identity across a copy).
//
// `epsilon` mirrors apply_smooth_attention's own parameter of the same
// name exactly (floors every per-channel Key max-abs value before
// dividing by it).
//
// FLOAT32-only throughout, mirroring every other calibration-driven pass
// in this codebase's own FLOAT32-only scope: this pass's own opset floor
// (>= 18, matching AttentionQuantization's own -- ReduceMax's axes-as-
// input form is not needed here, but this pass shares
// attention_quantization.h's own matcher and its own opset gate) is
// intentionally NOT enforced here, since apply_smooth_attention itself
// (the Python reference) has no opset gate at all -- this is a
// deliberate, documented scope difference: the C++ matcher transcription
// above omits attention_quantization.h's own opset >= 18 predicate,
// matching the Python reference's own unconditional matching instead.
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond ordinary floating-point
// summation-order differences from numpy -- this is a closed-form
// diagonal rescaling with no RNG and no accumulation step beyond an
// ordinary per-channel max reduction, so this port is expected to track
// the Python reference closely. See tests/test_smooth_attention_cpp.py
// for the measured agreement.

#include <onnx/onnx_pb.h>

#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplySmoothAttention(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double epsilon = 1e-5);
