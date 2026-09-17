// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Norm Tweaking (Li, Xu, Ni, Chen, Ye, Sun, 2023, "Norm Tweaking:
// High-performance Low-bit Quantization of Large Language Models",
// https://arxiv.org/abs/2309.02784) -- C++ port of norm_tweaking.py's own
// apply_norm_tweaking (see that module's own docstring for the full
// technique and closed-form derivation).
//
// Two-model, calibration-driven, protobuf-level pass -- same overall
// shape as gptq_entry.h's own ApplyGptq (the closest existing precedent:
// a real ModelExecutor runs real calibration data through both models),
// but simpler in two ways that matter for this header's own scope: (1)
// candidate matching is a single-op-type (LayerNormalization) output-name
// join, with no weight/DequantizeLinear structure to match at all; (2) the
// correction is a closed-form per-channel affine fit (first/second moment
// matching) requiring BOTH models to be probed and run through `executor`
// for every calibration batch (unlike ApplyGptq, which only ever probes
// `float_model`) -- the quantized model's own current LayerNorm output
// distribution is exactly what needs correcting, so it has to be
// observed, not derived.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Recalibrates every matched `LayerNormalization` node's own `scale`/
// `bias` initializers in `quantized_model` so its output distribution's
// per-channel mean and standard deviation, measured on `calibration_data`
// by running BOTH models through `executor`, matches `float_model`'s own
// -- see norm_tweaking.py's own module docstring for the closed-form
// derivation (`alpha = sigma_float / (sigma_quantized + eps)`,
// `beta = mu_float - alpha * mu_quantized`, folded directly into a new
// `scale`/`bias` pair for the same node).
//
// Candidate matching mirrors apply_norm_tweaking's own exactly: a
// `LayerNormalization` node present (by output tensor name) in both
// models, whose quantized-side node has an own `scale` input (`input[1]`)
// resolving to a 1-D initializer. A `LayerNormalization` with a
// multi-axis or non-1-D `scale`, or with no matching counterpart, is left
// untouched.
//
// Per-channel mean/variance are accumulated the same channels-last
// `reshape(-1, channels)` convention apply_norm_tweaking's own Python
// implementation uses (the LAST axis is always the channel axis for a
// fused `LayerNormalization`, regardless of the input's own rank), summed
// incrementally across batches in double precision (running sum/sum-of-
// squares, not concatenation) -- a calibration batch's probed tensor is
// only accumulated for a candidate when its float- and quantized-side
// shapes agree AND its rank is >= 1 (mirrors apply_norm_tweaking's own
// `if f.shape != q.shape or f.ndim == 0: continue`).
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a batch missing one of `float_model`'s OR `quantized_model`'s
// own graph inputs throws `std::invalid_argument`. NOT subgraph-aware.
//
// FLOAT32-only throughout, mirroring ApplyGptq's own FLOAT32-only scope.
// A candidate never observed with a matching float/quantized shape pair
// on any batch is left untouched (matches apply_norm_tweaking's own
// `if name not in counts: continue`).
//
// ACCEPTED NUMERICAL SCOPE: a closed-form moment-matching fit with no
// iterative solver or RNG -- this port's own double-precision running
// sum/sum-of-squares accumulation is expected to track
// apply_norm_tweaking's own numpy accumulation closely (summation-order
// floating-point differences only), not necessarily bit-for-bit. See
// tests/test_norm_tweaking_cpp.py for the measured agreement.
onnx::ModelProto ApplyNormTweaking(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double eps = 1e-6);
