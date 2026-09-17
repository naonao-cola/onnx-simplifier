// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// D2Quant's Deviation-Aware Correction (DAC) (Yan, Bao, Li, Zhang, Zhang,
// Xie, Sun and Zhang, 2026, "D2Quant: Accurate Low-bit Post-Training
// Weight Quantization for LLMs", https://arxiv.org/abs/2602.02546) -- C++
// port of d2quant.py's own apply_dac (see that module's own module
// docstring and apply_dac's own docstring for the full technique;
// d2quant.py's own apply_dsq -- a SEPARATE technique in the same paper --
// is already ported elsewhere, under quantize_entry.h/.cpp's own
// ApplyDsq; this header is scoped to apply_dac only).
//
// Two-model, calibration-driven, protobuf-level pass with the exact same
// overall shape as norm_tweaking_entry.h's own ApplyNormTweaking (a real
// ModelExecutor runs real calibration data through BOTH models, since
// what needs measuring is precisely the quantized model's own CURRENT
// deviation from the float model) -- apply_dac's own docstring explicitly
// says it "mirrors onnxsim.correct_bias's own measurement machinery", and
// bias_correction_entry.h/.cpp is NOT a usable template here (it is a
// browser/JS-split design where JS runs both models, not a
// ModelExecutor-based one -- see this repo's own established
// bias_correction_entry.h top-of-file note); gptq_entry.h/spqr_entry.h are
// the real ModelExecutor-based precedent this follows instead, adapted to
// probe a second model the same way norm_tweaking_entry.h's own
// ApplyNormTweaking does.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Measures, per channel and per `LayerNormalization` node present (by
// output tensor name) in both models, how much of that channel's own
// quantization-induced output deviation (`float_output - quantized_output`,
// captured by running BOTH models through `executor` on
// `calibration_data`) is a consistent, directional MEAN shift rather than
// unstructured noise, and folds that shift directly into the SAME
// `LayerNormalization`'s own bias for every channel where the shift
// dominates -- see d2quant.py's own apply_dac docstring for the full
// derivation (`LayerNormalization` already computes
// `normalize(x) * scale + bias`, so adding a per-channel constant `mu` to
// its output is exactly equivalent to using `bias + mu`; a
// `LayerNormalization` with no bias input at all gets one added -- a
// plain new initializer, still no new graph node).
//
// A channel is corrected only when its own deviation's
// `mu^2 / (mu^2 + sigma^2 + 1e-12)` (the paper's own closed-form expected
// squared-error reduction from correcting a pure mean shift) reaches
// `min_expected_error_reduction`; a `LayerNormalization` whose largest
// per-channel correction (after that gate) never exceeds
// `correction_threshold` in absolute value is left untouched entirely --
// mirrors apply_dac's own two skip conditions exactly. A `gamma`
// (`scale`, `input[1]`) that is not a 1-D FLOAT initializer of the
// correction's own channel width is also left untouched (mirrors
// apply_dac's own `_apply_ln_bias_correction` gating on `gamma_init`).
//
// Per-channel mean/variance-of-the-deviation are accumulated the same
// channels-last `reshape(-1, channels)` convention as
// norm_tweaking_entry.h's own ApplyNormTweaking (running sum/sum-of-
// squares in double precision, not concatenation), gated identically: a
// batch's probed tensor is only accumulated for a candidate when its
// float- and quantized-side shapes agree AND its rank is >= 1.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as ApplyNormTweaking's own -- a batch missing one
// of `float_model`'s OR `quantized_model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware.
//
// FLOAT32-only throughout, mirroring ApplyNormTweaking's own FLOAT32-only
// scope.
//
// ACCEPTED NUMERICAL SCOPE: a closed-form per-channel statistic with no
// iterative solver or RNG -- this port's own double-precision running
// sum/sum-of-squares accumulation is expected to track apply_dac's own
// numpy accumulation closely (summation-order floating-point differences
// only), not necessarily bit-for-bit. See tests/test_dac_cpp.py for the
// measured agreement.
onnx::ModelProto ApplyDac(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double min_expected_error_reduction = 0.5,
    double correction_threshold = 1e-12);
