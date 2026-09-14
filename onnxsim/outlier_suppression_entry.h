#pragma once

// Calibration-driven Outlier Suppression "Gamma Migration" (Wei et al.,
// 2022) entry point exposed to Python -- C++ port of
// onnxsim.outlier_suppression's own apply_outlier_suppression (see
// onnxsim/outlier_suppression.py's module docstring for the full
// technique: folding a SmoothQuant-style per-channel migration scale
// directly into a LayerNormalization's own gamma/beta -- adding zero
// runtime nodes -- compensated by scaling every downstream MatMul/Gemm
// consumer's weight rows by the same scale).
//
// Like smoothquant_entry.h's own ApplySmoothQuant (the closest existing
// precedent -- the same closed-form alpha-parameterized scale, the same
// protobuf-level calibration-driven shape, the same float-model-out
// contract), this operates directly on onnx::GraphProto rather than
// through onnxoptimizer's Node/Value IR: threading a live ModelExecutor
// plus calibration batches through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase
// (every PredicateBasedPass in onnxsim/passes/ is data-free by
// construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see smoothquant_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies Gamma Migration to every LayerNormalization node whose output
// feeds exclusively into one or more plain MatMul/vanilla-Gemm layers (as
// their activation input) and is not itself a graph output, using real
// calibration activations run through `executor`: divides the
// LayerNormalization's gamma (and bias, when present) by the migration
// scale `s` in place and multiplies every compensated consumer's weight
// rows by the same `s` in place. Returns a float model -- feed the result
// to a W8A8 quantizer (e.g. QuantizeStatic) to actually quantize it.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (smoothquant_entry.h's own ApplySmoothQuant,
// imatrix_quant_entry.h's ApplyImatrixQuantization) -- a
// `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument`. NOT subgraph-aware, matching every one
// of those passes' own scope decision: calibration_data batches are keyed
// to the top-level graph's own inputs only.
//
// A LayerNormalization with any consumer other than a plain
// MatMul/vanilla-Gemm activation input (a residual Add, a weight/bias
// slot, a graph output, ...), with a non-constant/non-1-D-FLOAT gamma, a
// malformed/non-constant bias, or a consumer weight whose reduction
// dimension does not match the gamma dim, is left completely untouched.
// So is a candidate whose output was never observed as a FLOAT32 tensor
// (any rank >= 1, reduced over every leading axis) of matching channel
// count -- mirrors apply_outlier_suppression's own per-layer skip
// conditions exactly.
//
// `alpha` is the migration strength (0.5 splits difficulty evenly on a log
// scale); `epsilon` floors every per-channel activation/weight max-abs
// value before computing `s` (and floors `s` itself) -- mirrors
// apply_outlier_suppression's own parameters of the same names exactly.
//
// FLOAT32-only throughout (gamma, bias, weights, and activations alike),
// mirroring apply_outlier_suppression's own FLOAT-only gamma/weight
// requirement and ApplySmoothQuant's own FLOAT32-only calibration scope: a
// FLOAT16/BFLOAT16 activation is never observed (skipped like an
// unobserved one), never converted. (The Python reference does not check
// the bias dtype explicitly; this port requires FLOAT there too -- a
// non-FLOAT bias under a FLOAT gamma is pathological, and converting it
// would change the model's own dtype contract. See the .cpp's own note.)
onnx::ModelProto ApplyOutlierSuppression(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha = 0.5, double epsilon = 1e-5);
