#pragma once

// Calibration-driven Outlier Suppression+ (Wei et al., 2023) entry point
// exposed to Python -- C++ port of onnxsim.outlier_suppression_plus's own
// apply_outlier_suppression_plus (see
// onnxsim/outlier_suppression_plus.py's module docstring for the full
// technique: a per-channel shift recentering each activation channel
// around zero, ahead of SmoothQuant's own per-channel scale, with the
// shift's constant contribution folded back in through a new output
// `Add`).
//
// Like smoothquant_entry.h's own ApplySmoothQuant (the closest existing
// precedent -- the same closed-form alpha-parameterized scale, the same
// protobuf-level calibration-driven shape, the same float-model-out
// contract, plus node insertion), this operates directly on
// onnx::GraphProto rather than through onnxoptimizer's Node/Value IR:
// threading a live ModelExecutor plus calibration batches through
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see smoothquant_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies Outlier Suppression+'s channel-wise shifting and scaling to
// every matched MatMul/vanilla-Gemm node with a constant 2-D FLOAT32
// weight and a plain 2-D FLOAT32 activation input, using real calibration
// activations run through `executor`: rescales the weight's
// reduction-dimension columns by `s` in place, inserts a `Sub`+`Mul` pair
// applying the shift `z` and scale `1/s` to the activation right before
// it, and inserts an `Add` right after it restoring the shift's constant
// contribution (`z @ W`) to the output. Returns a float model -- feed the
// result to a W8A8 quantizer (e.g. QuantizeStatic) to actually quantize
// it.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (smoothquant_entry.h's own ApplySmoothQuant,
// outlier_suppression_entry.h's ApplyOutlierSuppression) -- a
// `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument`. NOT subgraph-aware, matching every one
// of those passes' own scope decision: calibration_data batches are keyed
// to the top-level graph's own inputs only.
//
// A candidate whose activation was never observed as a plain 2-D FLOAT32
// tensor across `calibration_data`, or whose activation feature dimension
// does not match the weight's reduction dimension `K`, is left untouched
// -- mirrors apply_outlier_suppression_plus's own per-layer skip
// conditions exactly (including the strict 2-D requirement: a rank-3+
// activation is skipped, never reduced).
//
// `alpha` is the scaling step's migration strength (applied to the
// *shifted* activation's range); `epsilon` floors every per-channel
// activation/weight max-abs value before computing `s` (and floors `s`
// itself) -- mirrors apply_outlier_suppression_plus's own parameters of
// the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_outlier_suppression_plus's own `FLOAT`-only weight requirement
// and ApplySmoothQuant's own FLOAT32-only calibration scope: a FLOAT16/
// BFLOAT16 activation is never observed (skipped like an unobserved one),
// never converted.
onnx::ModelProto ApplyOutlierSuppressionPlus(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha = 0.5, double epsilon = 1e-5);
