#pragma once

// Calibration-driven GPTQ (Frantar et al., 2022) entry point exposed to
// Python -- C++ port of onnxsim.gptq's own apply_gptq (see
// onnxsim/gptq.py's module docstring for the full technique: sequential,
// Hessian-compensated rounding of every quantize_weight_only_int4-
// quantized MatMul/Gemm layer, reusing that scheme's own per-block
// scales and changing only which integer each element rounds to).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another weight-rewriting, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather
// than through onnxoptimizer's Node/Value IR: threading a live
// ModelExecutor plus calibration batches through OptimizeFixed's
// single-node-match PredicateBasedPass model has no established path in
// this codebase (every PredicateBasedPass in onnxsim/passes/ is data-free
// by construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes GPTQ-style sequential, Hessian-compensated rounding for
// every quantize_weight_only_int4-quantized MatMul/Gemm layer present (by
// node output name) in both `float_model` and `quantized_model`, using
// real activations captured from `float_model` through `executor`.
// Rewrites each matched layer's INT4 weight initializer in
// `quantized_model` to its GPTQ-optimized codes (same shape, dtype, and
// scale -- only which integer each element rounds to changes).
//
// Candidate matching mirrors adaround.py's own
// _find_int4_matmul_candidates exactly: a MatMul/Gemm node (same op type
// on both sides) sharing its output tensor name across the two models,
// whose float weight is a constant 2-D FLOAT tensor and whose quantized
// weight arrives through a DequantizeLinear (with a block_size
// attribute) from a same-shaped INT4 initializer.
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
// left untouched -- mirrors apply_gptq's own per-layer skip conditions
// exactly.
//
// `percdamp` is the Hessian damping factor (fraction of the mean
// diagonal added before inversion); `proc_block_size` is GPTQ's own
// column-processing block size (not the quantization scale's block size,
// which is always reused unchanged from `quantized_model`) -- mirrors
// apply_gptq's own parameters of the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_gptq's own `FLOAT`-only float-weight requirement and
// ApplyLlmInt8's own FLOAT32-only calibration scope: a FLOAT16/BFLOAT16
// activation is never observed (skipped like an unobserved one), never
// converted.
//
// Accepted numerical scope (unlike the bit-exact migration ports): the
// dense inverse and Cholesky factorization at this algorithm's heart are
// computed with this TU's own scalar double-precision kernels rather
// than LAPACK, so the inverse-Hessian factor can differ from numpy's in
// the last ulp or two on some inputs. The sequential rounding then
// agrees with the reference on the overwhelming majority of codes (any
// residual flips concentrate on exact rounding ties); reconstruction
// error tracks the reference's to well within quantization noise. See
// tests/test_gptq_cpp.py for the measured agreement.
onnx::ModelProto ApplyGptq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp = 0.01, int64_t proc_block_size = 128);
