#pragma once

// Calibration-driven FOEM entry point exposed to Python -- C++ port of
// onnxsim.foem's own apply_foem (see that module's own module docstring
// for the full technique: "First-Order Error Matters" -- a sequential,
// Hessian-*and*-first-order-drift-compensated extension of GPTQ's own
// per-column rounding).
//
// Read gptq_entry.h first -- this extends it directly, in the sense the
// paper itself frames its own contribution: an add-on correction to
// GPTQ's own compensation, not a replacement for it. Same protobuf-level,
// two-model shape as gptq_entry.h's own ApplyGptq (candidates are
// processed independently, so `executor` is invoked exactly once, up
// front); every candidate-matching/tensor-conversion/Hessian/Cholesky
// helper is transcribed near-verbatim from gptq_entry.cpp (this TU cannot
// include that anonymous-namespace machinery, so it is duplicated here
// rather than shared -- the same choice gptaq_entry.cpp's own near-verbatim
// duplication of gptq_entry.cpp already makes).
//
// GPTQ's own OBS-style correction, applied at each column `i`, treats the
// quantization error at that column as *the* error to propagate forward,
// implicitly assuming the column it just quantized is still (to first
// order) `W`'s own untouched original column. That assumption is false by
// construction partway through the pass: every earlier column's own
// forward-propagation step already nudged column `i`'s own
// pre-quantization value away from `W`'s own true original column -- a
// first-order deviation that accumulates column by column. FOEM's own fix:
// alongside the fresh rounding error GPTQ already compensates, also charge
// forward a (damped) fraction of that accumulated deviation -- how far the
// column's current pre-quantization value has already drifted from `W`'s
// own untouched original column -- so later columns' own compensation
// accounts for both sources of error, not just the newest one. At column
// `i`, alongside GPTQ's own `second_order_err = (w_col - code_col * s) /
// d`, this also computes `drift = (w_col_before_quantizing - w_orig_col) /
// d` and propagates `second_order_err - foem_beta * drift` forward instead
// of `second_order_err` alone -- `foem_beta` the paper's own damping
// factor. `foem_beta == 0.0` recovers plain GPTQ exactly (byte-identical
// codes, since the drift term is then multiplied by zero before it can
// affect anything).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes FOEM-style sequential, Hessian-*and*-first-order-drift-
// compensated rounding for every quantize_weight_only_int4-quantized
// MatMul/Gemm layer present (by node output name) in both `float_model`
// and `quantized_model`, using real activations captured from
// `float_model` through `executor`. See onnxsim/foem.py's own module
// docstring for the full technique and its relationship to
// onnxsim.apply_gptq.
//
// Candidate matching, calibration probing, and Hessian construction are
// identical to gptq_entry.h's own ApplyGptq -- see that header for the
// exact scope (float weight a constant 2-D FLOAT tensor, quantized weight
// via a DequantizeLinear with a block_size attribute, a `calibration_data`
// batch missing a required graph input throws `std::invalid_argument`,
// NOT subgraph-aware, FLOAT32-only throughout). `percdamp`/
// `proc_block_size` mirror apply_foem's own parameters of the same names
// and defaults (identical to apply_gptq's own).
//
// `foem_beta` is FOEM's own addition: the damping factor on the
// first-order drift term (see this header's own docstring above for the
// exact formula), mirroring apply_foem's own parameter and default
// exactly.
//
// ACCEPTED, PERMANENT DIVERGENCE (shared with gptq_entry.h's own
// ApplyGptq): the dense inverse and Cholesky factorization at this
// algorithm's heart are computed with this TU's own scalar
// double-precision kernels rather than LAPACK, so the inverse-Hessian
// factor can differ from numpy's in the last ulp or two on some inputs.
// The sequential rounding then agrees with the reference on the
// overwhelming majority of codes (residual flips concentrate on exact
// rounding ties); reconstruction error tracks the reference's to well
// within quantization noise. See tests/test_foem_cpp.py for the measured
// agreement, including the `foem_beta == 0.0` degenerate case (exact
// parity with tests/test_gptq_cpp.py's own GPTQ agreement, since the two
// algorithms become byte-identical there).
onnx::ModelProto ApplyFoem(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp = 0.01, int64_t proc_block_size = 128,
    double foem_beta = 0.005);
