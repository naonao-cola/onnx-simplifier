#pragma once

// Calibration-driven PB-LLM (Shang, Yuan, Wu, Dong, 2024, ICLR 2024,
// "PB-LLM: Partially Binarized Large Language Models") entry point exposed
// to Python -- C++ port of onnxsim.pb_llm's own quantize_weight_only_pb_llm
// (see onnxsim/pb_llm.py's module docstring for the full technique: a
// structured mixed-precision binarizer -- per matched layer, the
// `salient_ratio` fraction of input channels with the highest
// Hessian-diagonal-weighted magnitude stay INT8, every other channel is
// binarized to ~1 bit/element, and both live in the same Code/Scale
// initializer pair).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another SINGLE-model, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather than
// through onnxoptimizer's Node/Value IR: threading a live ModelExecutor
// plus calibration batches through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase
// (every PredicateBasedPass in onnxsim/passes/ is data-free by
// construction).
//
// Unlike BiLLM/GPTQ/OWQ, PB-LLM's own salience score needs no Hessian
// *inversion* at all -- just the Hessian's own diagonal, `diag(H)_j =
// sum_samples(X[:, j]^2)`, a plain per-column sum of squares over
// calibration activations. So this port carries none of GPTQ's/BiLLM's
// own "accepted numerical scope" caveat about a scalar Cholesky solve
// diverging from LAPACK in the last ulp or two: every computation here
// (the sum-of-squares statistic, the per-column salience ranking, the
// per-column INT8/binary quantization) is closed-form and expected to
// track onnxsim.pb_llm's own float64 numpy implementation closely, up to
// ordinary floating-point summation-order differences.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Partially binarizes every MatMul/vanilla-Gemm layer with a constant 2-D
// FLOAT32 weight present in `model`: for each layer, computes a per-input-
// channel salience score `mean_n(|W[n, j]|) * diag(H)_j` (`H = X^T X` from
// real activations captured by running `calibration_data` through
// `executor`), keeps the top `salient_ratio` fraction of channels (by
// count, rounded to the nearest column) at per-column symmetric INT8
// (`scale_j = max_n(|W[n, j]|) / 127`), and binarizes every other channel
// to `{-1, +1}` at `scale_j = mean_n(|W[n, j]|)` (the paper's own
// "Naive" binarization mode, not its residual-refined "Optimal" one --
// see onnxsim/pb_llm.py's own docstring, point 4, for that documented
// scope narrowing, already present in the Python reference itself, not
// added by this port). Both quantization schemes reconstruct the same way
// (`value = code * scale`), so a matched layer's weight input is rewired
// to a single `Mul(Cast(Code, float), Scale)` feeding the original node --
// no new MatMul/Gemm node, the original one stays in place with only its
// weight input changed.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// exactly (a MatMul, or a Gemm with transA=0, alpha=1, and -- when it has
// a bias -- beta=1): a constant 2-D FLOAT32 weight is required; the bias,
// when present, is left completely untouched (this pass never reads or
// rewrites it).
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a `calibration_data` batch
// missing one of `model`'s own graph inputs throws `std::invalid_argument`.
// NOT subgraph-aware, matching every one of those passes' own scope
// decision. A captured activation is flattened to `[rows, K]` regardless
// of its own rank (`[batch, seq, K]` collapses to `[batch * seq, K]`,
// exact, the same reasoning gptq_entry.cpp's own ActivationRows already
// documents) -- a layer whose activation was never observed with a
// feature axis at all (rank < 2), or whose feature dim does not match the
// weight's reduction dimension `K`, is left untouched, mirroring
// quantize_weight_only_pb_llm's own per-layer skip conditions exactly.
//
// `salient_ratio` is pb_llm.py's own parameter of the same name (default
// 0.15); `0.0` binarizes every column (no INT8 columns at all) and `1.0`
// quantizes every column to INT8 (no binarization at all) -- see that
// module's own docstring for both limits.
//
// SCOPE NARROWING (matching pb_llm.py's own documented scope exactly, not
// an additional one this port adds): the paper's own "Optimal" residual-
// refinement mode for non-salient columns is out of scope here, exactly
// as it is in quantize_weight_only_pb_llm itself (see that function's own
// docstring, point 4) -- this is not something this port silently drops
// relative to its own Python counterpart. This port also omits
// pb_llm.py's own `skip_names` parameter -- several other *_cpp ports in
// this repo already establish that a C++ port need not mirror every
// optional knob its Python counterpart has.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// quantize_weight_only_pb_llm's own FLOAT-only weight requirement: a
// FLOAT16/BFLOAT16 activation is never observed (skipped like an
// unobserved one), never converted.
onnx::ModelProto ApplyPbLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double salient_ratio = 0.15);
