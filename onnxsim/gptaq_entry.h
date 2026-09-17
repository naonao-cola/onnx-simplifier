// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// GPTAQ (Li, Yin, Lee, Xiao, Panda, 2025, "GPTAQ: Efficient Finetuning-Free
// Quantization for Asymmetric Calibration", https://arxiv.org/abs/2504.02692)
// -- C++ port of gptaq.py's own apply_gptaq (see that module's own
// docstring for the full first-principles derivation this port follows
// exactly).
//
// GPTAQ is a small, closed-form correction layered on top of GPTQ's own
// per-column sequential procedure (gptq_entry.h's own ApplyGptq, which this
// port reuses almost verbatim for candidate matching and the Hessian/
// Cholesky/column-processing machinery): real GPTQ calibrates each layer
// against activations that already reflect every earlier layer's own
// (already-quantized) weights, so it implicitly targets reconstructing
// that *corrupted* signal rather than what the original float network
// would have produced. GPTAQ captures BOTH the true activation X̃ (from
// `float_model`) and the corrupted activation X (from `quantized_model`,
// already quantized/corrected by whatever ran on it so far) at the same
// probe point, and folds `δX = X̃ - X`'s effect into GPTQ's own objective
// via one small, exact pre-computation: GPTQ's own per-column algorithm,
// applied UNCHANGED, to the weight matrix `W + Shift` instead of `W`,
// where `Shift = (H^{-1} C)^T`, `H = X^T X` (GPTQ's own Hessian, from the
// corrupted activation), and `C = X^T (δX W^T)` (a fixed, precomputable
// linear-term coefficient). When a layer has no upstream quantization yet
// (`X == X̃`), `Shift` is numerically zero and this reduces to plain GPTQ
// exactly.
//
// Same protobuf-level, two-model shape as gptq_entry.h's own ApplyGptq,
// EXCEPT that (unlike ApplyGptq, which only ever probes `float_model`)
// this ALSO probes `quantized_model` at each candidate's own quantized-
// side node input (mirrors norm_tweaking_entry.h's own ApplyNormTweaking
// dual-model-probing shape, not ApplyGptq's own single-model one).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes GPTAQ-style (asymmetric-calibration) sequential, Hessian-
// compensated rounding for every quantize_weight_only_int4-quantized
// MatMul/Gemm layer present (by node output name) in both `float_model`
// and `quantized_model`, using real activations captured from BOTH models
// through `executor`. Rewrites each matched layer's INT4 weight
// initializer in `quantized_model` to its GPTAQ-optimized codes (same
// shape, dtype, and scale -- only which integer each element rounds to
// changes).
//
// Candidate matching mirrors gptq_entry.h's own ApplyGptq (and
// adaround.py's own _find_int4_matmul_candidates, which both this port
// and apply_gptaq itself reuse) exactly: a MatMul/Gemm node (same op type
// on both sides) sharing its output tensor name across the two models,
// whose float weight is a constant 2-D FLOAT tensor and whose quantized
// weight arrives through a DequantizeLinear (with a block_size attribute)
// from a same-shaped INT4 initializer.
//
// For each candidate, this probes `float_model` at the candidate node's
// own float-side input[0] (the true activation X̃) and `quantized_model`
// at the MATCHED quantized-side node's own input[0] (the corrupted
// activation X -- generally the SAME tensor name as the float side, but
// not assumed to be: `quantized_model` may have already been rewired by
// an earlier correction pass). Both are flattened to rows exactly like
// gptq_entry.h's own ApplyGptq (`reshape(-1, K)`, exact for any leading
// batch/sequence axes). Mirrors apply_gptaq's own per-batch, per-model
// independent rank/shape filtering: a batch whose probed tensor has no
// feature axis at all (rank < 2) on EITHER model contributes no row to
// EITHER model's accumulated rows for that candidate (dropped, not
// zero-filled); if after filtering the float- and quantized-side row
// counts, or any batch's own row count/feature width, still disagree, the
// whole layer is left untouched (mirrors apply_gptaq's own
// `if not x_true_parts or len(x_true_parts) != len(x_quant_parts):
// continue` plus its own per-part shape-equality check). ACCEPTED
// DIVERGENCE from a literal transcription: this port additionally
// requires every accumulated part (across all batches, not just the
// float/quantized pair at one batch) to share the same feature width K --
// apply_gptaq's own `np.concatenate` would raise on a real width
// mismatch between batches rather than silently proceeding, so this is
// strictly more defensive, not a behavior change on any input the
// reference actually accepts.
//
// A layer whose activation was never observed on either model with a
// feature axis, or whose feature dim does not match the weight's
// reduction dimension `K`, is left untouched -- mirrors apply_gptaq's own
// per-layer skip conditions exactly.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to BOTH `float_model`'s AND `quantized_model`'s own
// graph inputs -- both models are actually run) shape as every other
// calibration-driven pass in this codebase -- a `calibration_data` batch
// missing a graph input either model needs throws `std::invalid_argument`.
// NOT subgraph-aware, matching every one of those passes' own scope
// decision.
//
// `percdamp` and `proc_block_size` are passed through unchanged to the
// same GPTQ column-processing/Hessian-inversion machinery gptq_entry.h's
// own ApplyGptq uses -- mirrors apply_gptaq's own parameters of the same
// names exactly (both forwarded verbatim to
// :func:`onnxsim.gptq`'s own `_gptq_quantize_columns`/
// `_inverse_hessian_cholesky` in the Python reference).
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// ApplyGptq's own FLOAT32-only scope exactly: a FLOAT16/BFLOAT16
// activation on either model is never observed (skipped like an
// unobserved one), never converted.
//
// ACCEPTED, PERMANENT DIVERGENCE (shared with gptq_entry.h's own
// ApplyGptq): the dense inverse and Cholesky factorization at this
// algorithm's heart -- both for the Hessian inversion GPTQ's own column
// algorithm needs AND for this module's own additional `Shift`
// pre-computation -- are computed with this TU's own scalar
// double-precision kernels rather than LAPACK, so results can differ from
// numpy's in the last ulp or two on some inputs. The sequential rounding
// then agrees with the reference on the overwhelming majority of codes
// (residual flips concentrate on exact rounding ties); reconstruction
// error tracks the reference's to well within quantization noise. See
// tests/test_gptaq_cpp.py for the measured agreement, including the
// degenerate case (`quantized_model == float_model`, i.e. `δX ≈ 0`) that
// this module's own docstring calls out as reducing to plain GPTQ.
onnx::ModelProto ApplyGptaq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp = 0.01, int64_t proc_block_size = 128);
