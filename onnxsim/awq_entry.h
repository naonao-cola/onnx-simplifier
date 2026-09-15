#pragma once

// Calibration-driven AWQ (Lin et al., 2023) entry point exposed to
// Python -- C++ port of onnxsim.awq's own apply_awq (see
// onnxsim/awq.py's module docstring for the full technique:
// grid-searching a per-channel rescaling exponent that minimizes each
// quantize_weight_only_int4-quantized layer's own reconstruction error,
// re-quantizing from scratch at every grid point).
//
// Like gptq_entry.h's own ApplyGptq (the closest existing precedent --
// the same adaround-style candidate join across a float model and its
// INT4-quantized counterpart, the same protobuf-level
// calibration-driven shape), this operates directly on onnx::GraphProto
// rather than through onnxoptimizer's Node/Value IR: threading a live
// ModelExecutor plus calibration batches through OptimizeFixed's
// single-node-match PredicateBasedPass model has no established path in
// this codebase (every PredicateBasedPass in onnxsim/passes/ is data-free
// by construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies AWQ-style activation-aware per-channel weight rescaling to
// every quantize_weight_only_int4-quantized MatMul/Gemm layer present (by
// node output name) in both `float_model` and `quantized_model`, using
// real activations captured from `float_model` through `executor`.
// Returns `quantized_model` with every measurably improved layer
// rewritten (INT4 weight and scale initializers replaced with the
// rescaled-and-requantized versions, plus a new `Mul` before it applying
// the compensating inverse channel scale); a layer the search cannot
// improve (plain round-to-nearest best) is left completely untouched.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `float_model`'s own graph inputs) shape as
// every other calibration-driven pass in this codebase -- a
// `calibration_data` batch missing one of `float_model`'s own graph
// inputs throws `std::invalid_argument`. NOT subgraph-aware, matching
// every one of those passes' own scope decision.
//
// A layer whose activation was never observed with a feature axis
// (rank < 2 after the reference's own leading-axis flattening), or whose
// feature dim does not match the weight's reduction dimension `K`, is
// left untouched -- mirrors apply_awq's own per-layer skip conditions
// exactly.
//
// `num_alpha_steps` is the grid density over [0, 1] inclusive (alpha = 0
// always measured first, so a layer that cannot improve keeps its
// original quantization with no inserted node); higher values search
// more finely at proportionally more cost -- mirrors apply_awq's own
// parameter of the same name exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_awq's own `FLOAT`-only float-weight requirement and ApplyGptq's
// own FLOAT32-only calibration scope: a FLOAT16/BFLOAT16 activation is
// never observed (skipped like an unobserved one), never converted.
//
// Accepted numerical scope (like ApplyGptq's): every grid point's error
// is measured with scalar double-precision kernels rather than BLAS, so
// near-tied grid points can resolve differently from numpy in the last
// ulp or two on some inputs; the winning alpha and final codes agree
// with the reference on everything measured (see
// tests/test_awq_cpp.py). Deliberately faithful rather than fast: one
// full re-quantization plus reconstruction-error measurement per grid
// point per layer, exactly like the reference.
onnx::ModelProto ApplyAwq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_alpha_steps = 20);
