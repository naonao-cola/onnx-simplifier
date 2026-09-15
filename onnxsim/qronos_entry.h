#pragma once

// Calibration-driven Qronos entry point exposed to Python -- C++ port of
// onnxsim.qronos's own apply_qronos (see that module's docstring for the
// full technique: a sequential, whole-model generalization of GPTQ that
// additionally accounts for the error already baked into a layer's
// activations because upstream layers were quantized first, not just
// this layer's own rounding error).
//
// Same two-model (float model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as gptq_entry.h's
// own ApplyGptq (the closest existing precedent -- this reduces to
// ApplyGptq exactly when a layer has no already-quantized upstream layer
// feeding it). Unlike ApplyGptq, which computes every layer's Hessian
// from the untouched float model independently and in any order, this
// pass processes layers strictly in float-model forward-execution order,
// re-probing the progressively-corrected quantized model before each
// subsequent layer -- so, unlike every other calibration-driven pass in
// this codebase, `executor` is invoked once per matched layer (not once
// up front for all of them).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Qronos-corrects every quantize_weight_only_int4-quantized MatMul/Gemm
// layer present (by node output name) in both `float_model` and
// `quantized_model`, processing layers in `float_model`'s own node order
// (a valid forward-execution order for a topologically-sorted ONNX
// graph) so each layer's correction accounts for every upstream layer's
// own already-applied quantization error, not just its own rounding.
//
// For each matched layer: `x_float` (that layer's activation captured
// once, up front, from the untouched `float_model` -- same as ApplyGptq's
// own single probing pass) and `x_quant` (the same activation re-probed,
// via `executor`, from the *current*, progressively-corrected working
// copy of `quantized_model` -- freshly captured for every layer) are
// compared: `dx = x_quant - x_float`. Writing `H = x_quant^T @ x_quant`
// and reusing the same damped-Cholesky-of-`H^{-1}` reformulation
// ApplyGptq's own kernels already implement, the float weight is shifted
// by `W_opt = W - W @ dx^T @ x_quant @ H^{-1}` before being handed,
// unchanged in every other respect, to the identical column-quantization
// search ApplyGptq itself uses (same `H`, same per-block scale reused
// from `quantized_model`). See onnxsim/qronos.py's own module docstring
// for the full derivation and why this is an exact GPTQ reduction when
// `dx` is zero (the very first layer, or any layer with no
// already-quantized upstream predecessor).
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape, `percdamp`
// (Hessian damping factor) and `proc_block_size` (GPTQ's own
// column-processing block size) parameters as ApplyGptq's own, with the
// same meaning. A `calibration_data` batch missing one of `float_model`'s
// own graph inputs throws `std::invalid_argument`. NOT subgraph-aware.
//
// A layer whose float activation was never observed with a feature axis,
// whose re-probed quantized-side activation likewise has none or has a
// different total row count (keeping `x_float`/`x_quant` sample-aligned
// for `dx`, mirroring apply_qronos's own alignment guard), or whose
// feature dimension doesn't match the weight's own K, is left completely
// untouched -- mirrors apply_qronos's own per-layer skip conditions.
//
// FLOAT32-only throughout, mirroring ApplyGptq's own FLOAT32-only
// calibration scope. Accepted numerical scope (like ApplyGptq's own): the
// dense inverse/Cholesky kernels are this TU's own scalar double-precision
// ones, not LAPACK -- see gptq_entry.h's own identical note.
onnx::ModelProto ApplyQronos(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double percdamp = 0.01, int64_t proc_block_size = 128);
