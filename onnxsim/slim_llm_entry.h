#pragma once

// Calibration-driven SliM-LLM entry point exposed to Python -- C++ port of
// onnxsim.slim_llm's own apply_slim_llm (see that module's own docstring
// for the full technique: Huang, Shao, Dong, Luo, Qiao et al., 2024's
// salience-driven mixed-precision quantization, picking a bit-width per
// *group within a layer* rather than per whole layer the way
// onnxsim.mixed_precision does).
//
// Single-model, calibration-driven, unlike gptq_entry.h's own two-model
// ApplyGptq (there is no separate "already quantized" model here -- this
// pass builds and inserts its own DequantizeLinear-based dequantization
// subgraph from scratch, the same single-model rewrite shape as
// llm_int8_entry.h's own ApplyLlmInt8). Reuses gptq_entry.cpp's own
// Hessian/Cholesky machinery (this file's own independent transcription of
// it, matching this codebase's established one-copy-per-TU convention) for
// the per-column OWQ-style salience score slim_llm.py's own docstring
// derives; the per-*group* bit-width assignment on top of that score is
// this pass's own distinguishing contribution.
//
// Candidates (every MatMul/"vanilla" Gemm -- transA=0, alpha=1, and, when
// biased, beta=1, exactly onnxsim.quip_sharp's own _match_matmul_like,
// which slim_llm.py's own candidate loop reuses) are processed
// independently, so `executor` is invoked exactly once, up front.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Quantizes every MatMul/"vanilla" Gemm layer with a constant 2-D FLOAT32
// weight whose reduction dimension K is divisible by `group_size` to a
// per-group mix of `low_bits`/`high_bits` INT8-stored integer codes, chosen
// from a calibration-driven per-group salience score so each layer's own
// average bits/weight lands at `target_bits`. See
// onnxsim/slim_llm.py's own module docstring for the full technique
// (salience score, bit assignment, and storage format).
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. A layer whose
// activation was never observed, or whose feature dimension doesn't match
// the weight's own K, is left completely untouched -- mirrors
// apply_slim_llm's own per-layer skip conditions.
//
// `target_bits`/`low_bits`/`high_bits`/`group_size`/`percdamp` mirror
// apply_slim_llm's own parameters of the same names exactly, including
// `target_bits`'s own clamp into `[low_bits, high_bits]`. Throws
// `std::invalid_argument` when `low_bits < 2` or `high_bits <= low_bits`,
// matching apply_slim_llm's own `ValueError` precondition.
//
// A model with no matching layer, or an opset older than 21
// (`DequantizeLinear`'s `block_size` attribute needs opset 21), is
// returned unchanged.
//
// FLOAT32-only throughout, mirroring ApplyGptq's/ApplyAdaround's own
// FLOAT32-only calibration scope.
//
// ACCEPTED, PERMANENT DIVERGENCE (same class as ApplyGptq's own): the
// dense inverse/Cholesky factorization behind this pass's own per-column
// salience score is computed with this TU's own scalar double-precision
// kernels rather than LAPACK, so the salience ranking -- and therefore
// which groups land in the top `fraction_high` -- can differ from the
// reference on a near-tied salience boundary. See
// tests/test_slim_llm_cpp.py for the measured agreement.
onnx::ModelProto ApplySlimLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double target_bits = 3.0, int64_t low_bits = 2, int64_t high_bits = 4,
    int64_t group_size = 32, double percdamp = 0.01);
