#pragma once

// Calibration-driven QuantEase (Behdin, Acharya, Gupta, Song, Zhu and
// Keerthi, 2023) entry point exposed to Python -- C++ port of
// onnxsim.quantease's own apply_quantease (see onnxsim/quantease.py's
// module docstring for the full technique and its own derivation: a
// fourth lever on quantize_weight_only_int4's own block-wise symmetric
// INT4 scheme, alongside onnxsim.adaround/onnxsim.gptq/onnxsim.awq --
// plain cyclic COORDINATE DESCENT over the same per-row reconstruction
// objective GPTQ minimizes via a single greedy Hessian-compensated sweep,
// solved instead by repeatedly sweeping every column and moving it to the
// closed-form unconstrained optimum of the objective itself, rounded to
// the nearest grid point, for a small fixed number of full sweeps
// ("epochs")).
//
// Like gptq_entry.h's own ApplyGptq (the closest existing precedent --
// another two-model, calibration-driven, per-column INT4-code-optimizing
// port at the protobuf level, sharing this module's own candidate
// matcher, adaround.py's _find_int4_matmul_candidates, and needing only
// the layer's own Hessian `H = X^T @ X`, never its inverse), this
// operates directly on onnx::GraphProto rather than through
// onnxoptimizer's Node/Value IR -- see gptq_entry.h's own identical note
// for why.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes QuantEase-style cyclic coordinate-descent rounding for every
// quantize_weight_only_int4-quantized MatMul/Gemm layer present (by node
// output name) in both `float_model` and `quantized_model`, using real
// activations captured from `float_model` through `executor`. Rewrites
// each matched layer's INT4 weight initializer in `quantized_model` to its
// QuantEase-optimized codes (same shape, dtype, and scale -- only which
// integer each element rounds to changes).
//
// Candidate matching mirrors adaround.py's own
// _find_int4_matmul_candidates exactly -- see gptq_entry.h's own identical
// note (a MatMul/Gemm node, same op type on both sides, sharing its output
// tensor name across the two models, whose float weight is a constant 2-D
// FLOAT tensor and whose quantized weight arrives through a
// DequantizeLinear -- with a block_size attribute -- from a same-shaped
// INT4 initializer).
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `float_model`'s own graph inputs) shape as every
// other calibration-driven pass in this codebase -- a `calibration_data`
// batch missing one of `float_model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware, matching every one of those
// passes' own scope decision.
//
// A layer whose activation was never observed with a feature axis (rank <
// 2 after the reference's own leading-axis flattening), or whose
// activation feature dim does not match the weight's reduction dimension
// `K`, is left untouched -- mirrors apply_quantease's own per-layer skip
// conditions exactly.
//
// `num_epochs` is the number of full cyclic column sweeps -- mirrors
// apply_quantease's own parameter of the same name exactly (more epochs
// never hurt accuracy, only cost, since each sweep only ever decreases the
// shared reconstruction objective).
//
// Per-column update, exactly _quantease_quantize_columns's own closed-form
// coordinate-descent step (see onnxsim/quantease.py's module docstring for
// the full derivation): starting from a fresh plain round-to-nearest INT4
// quantization of the float weight (using the SAME formula
// quantize_weight_only_int4/apply_awq's own `_quantize_blockwise_int4`
// uses -- `scale = max(|w| in block, 1e-12) / 7`, codes clamped to
// `[-7, 7]` -- but note that `quantized_model`'s own EXISTING scale, not
// this fresh quantization's own internal one, is what's actually reused
// for `w_hat`/every later step, mirroring the Python reference's own
// `codes_nk * scale_full` line exactly, where `scale_full` is built from
// the caller-supplied `scale_blocks`, not `_quantize_blockwise_int4`'s own
// return value), then for `num_epochs` full sweeps over every column `k`:
//   delta = (r @ H[:, k]) / max(H[k, k], 1e-12)   -- [N], vectorized rows
//   new_code = clip(round(w_hat[:, k] + delta), -7, 7)
//   r[:, k] -= new_code * s - w_hat[:, k]; w_hat[:, k] = new_code * s
// where `r` is the running per-row residual (`w_nk - w_hat`, updated in
// place) and `s` is that column's own existing per-(output channel,
// group) scale. `H` itself is used directly, never inverted/factorized
// (unlike ApplyGptq's own Cholesky-factored inverse Hessian).
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_quantease's own `FLOAT`-only float-weight requirement and
// ApplyGptq's own FLOAT32-only calibration scope: a FLOAT16/BFLOAT16
// activation is never observed (skipped like an unobserved one), never
// converted.
//
// No accepted numerical-algorithm divergence here (unlike ApplyGptq's own
// Cholesky-inverse note): this port needs no matrix inversion or
// factorization at all, just `H` itself and elementary column updates, so
// there is no linear-algebra routine whose choice of algorithm could
// diverge from numpy's own -- floating-point summation order differences
// (double-precision accumulation here throughout, matching numpy's own
// float64 Hessian and residual) are the only source of any residual
// disagreement, well within ordinary rounding tolerance. See
// tests/test_quantease_cpp.py for the measured agreement.
onnx::ModelProto ApplyQuantease(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_epochs = 4);
