#pragma once

// Calibration-driven sensitivity-based mixed-precision weight
// quantization entry point exposed to Python -- C++ port of
// onnxsim.mixed_precision's own apply_mixed_precision_quantization (see
// onnxsim/mixed_precision.py's module docstring for the full technique:
// every matched MatMul/vanilla-Gemm layer is quantized to either
// block-wise INT8 or block-wise INT4, chosen per layer from a
// calibration-driven Hessian-diagonal (or full-Hessian) sensitivity
// score, budgeted so the top `high_bits_fraction` most-sensitive layers
// get INT8).
//
// Single-model, calibration-driven, protobuf-level shape as
// llm_int8_entry.h's own ApplyLlmInt8 (the closest existing precedent --
// match candidates, probe real activations through `executor`, then
// rewrite each matched layer's own weight-consuming node into a new
// DequantizeLinear+MatMul[+Add] chain) rather than gptq_entry.h's own
// two-model shape: this technique needs no separately-quantized
// counterpart model, only the original float weights and a calibration-
// driven sensitivity ranking.
//
// Candidate matching mirrors mixed_precision.py's own top-level loop
// exactly: a MatMul/vanilla-Gemm node (via quip_sharp.py's own
// `_match_matmul_like`, transcribed the same way llm_int8_entry.cpp's own
// MatchMatMulLike already transcribes it) whose weight is a constant 2-D
// FLOAT32 initializer with a reduction dimension `K` divisible by
// `block_size`.
//
// The Hessian-diagonal accumulation reuses gptq_entry.cpp's own
// AccumulateActivationRows/`H = X^T X` machinery (see that file's own
// comments for the probe-injection/DLPack-crossing shape) -- this file
// transcribes an equivalent accumulator rather than sharing one via a
// common header, matching this codebase's established per-*_entry.cpp
// self-containment convention. `sensitivity_metric="full_hessian"` only
// additionally accumulates the full [K, K] Gram matrix per candidate
// activation name (mirrors mixed_precision.py's own `need_full_hessian`
// gate exactly -- the diagonal is always accumulated, the full matrix
// only when actually needed).
//
// The INT4 candidate scoring path reuses omniquant_entry.cpp's own
// QuantizeBlockwiseInt4WithClip (clip_ratio=1.0, i.e. plain min/max
// block-wise INT4 RTN -- transcribed here rather than shared, same
// self-containment convention) to build the reconstruction error `E`
// every candidate layer's sensitivity score is computed from, whether or
// not that layer ultimately receives the INT4 or INT8 tier.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. NOT
// subgraph-aware, matching every one of those passes' own scope decision.
//
// A layer whose activation was never observed with a matching feature
// axis is left completely ineligible (no bit-width tier assigned, weight
// untouched) -- mirrors apply_mixed_precision_quantization's own
// `sensitivities.append(None)` / `eligible_idx` skip exactly. Layers with
// a non-constant, non-2-D, or non-block-divisible weight are never even
// considered a candidate, also mirroring the reference.
//
// `high_bits_fraction`, `block_size`, `sensitivity_metric` mirror
// apply_mixed_precision_quantization's own parameters of the same names
// exactly; `sensitivity_metric` must be "hessian_diag" (the default) or
// "full_hessian", else `std::invalid_argument`. `num_high_bits =
// round(high_bits_fraction * eligible_count)` uses round-half-to-even,
// matching Python's own `round()` builtin exactly; ties in the
// sensitivity ranking keep their original candidate order (a stable
// descending sort), matching Python's own `sorted(..., reverse=True)`
// stability exactly.
//
// FLOAT32-only throughout, mirroring every other calibration-driven pass
// in this codebase's own FLOAT32-only scope. Models below opset 21 are
// returned unchanged (INT4's tensor type and DequantizeLinear's
// block_size attribute both need opset 21), also mirroring the reference.
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond ordinary floating-point
// summation-order/accumulation differences from numpy -- the Hessian
// accumulation is a plain reduction (no Cholesky/SVD/eigendecomposition),
// and both quantization tiers are bounded, deterministic round-to-nearest
// with no RNG, so this port is expected to track the Python reference
// closely, including exact-tie behavior at the sensitivity-ranking
// boundary. See tests/test_mixed_precision_cpp.py for the measured
// agreement.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyMixedPrecisionQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double high_bits_fraction = 0.2, int64_t block_size = 32,
    const std::string& sensitivity_metric = "hessian_diag");
