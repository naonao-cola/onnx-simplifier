#pragma once

// Calibration-driven SmoothQuant (Xiao et al., 2022) migration entry point
// exposed to Python -- C++ port of onnxsim.smoothquant's own
// apply_smoothquant (see onnxsim/smoothquant.py's module docstring for the
// full technique: per-channel `s_j = max(|X_j|)**alpha /
// max(|W_j|)**(1-alpha)` migrates activation quantization difficulty into
// the weight, a lossless pre-conditioning transform ahead of a separate
// W8A8 quantizer, never a quantization scheme itself).
//
// Like imatrix_quant_entry.h's own ApplyImatrixQuantization (the closest
// existing precedent -- another calibration-driven MatMul/Gemm rewrite at
// the protobuf level), this operates directly on onnx::GraphProto rather
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

// Forward declaration only -- see imatrix_quant_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Migrates activation quantization difficulty into the weight for every
// matched MatMul/vanilla-Gemm node with a constant 2-D FLOAT32 weight,
// using real calibration activations run through `executor`: rescales the
// weight's reduction-dimension columns by `s` in place and inserts a new
// `Mul` node dividing that layer's activation input by the same `s` right
// before it. Returns a float model -- feed the result to a W8A8 quantizer
// (e.g. QuantizeStatic) to actually quantize it.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (imatrix_quant_entry.h's own ApplyImatrixQuantization,
// structured_pruning_entry.h's ApplyWandaPruning/ApplySparseGptPruning) --
// a `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument`. NOT subgraph-aware, matching every one of
// those passes' own scope decision: calibration_data batches are keyed to
// the top-level graph's own inputs only.
//
// A candidate whose activation was never observed as a plain 2-D FLOAT32
// tensor across `calibration_data`, or whose activation feature dimension
// does not match the weight's reduction dimension `K`, is left untouched --
// mirrors apply_smoothquant's own per-layer skip conditions exactly
// (including the strict 2-D requirement: a rank-3+ activation is skipped,
// never reduced).
//
// `alpha` is the migration strength (0.5 splits difficulty evenly on a log
// scale); `epsilon` floors every per-channel activation/weight max-abs
// value before computing `s` (and floors `s` itself), avoiding a
// divide-by-zero on an all-zero channel -- mirrors apply_smoothquant's own
// parameters of the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_smoothquant's own `FLOAT`-only weight requirement and
// ApplyImatrixQuantization's own FLOAT32-only calibration scope: a FLOAT16/
// BFLOAT16 activation is never observed (skipped like an unobserved one),
// never converted.
onnx::ModelProto ApplySmoothQuant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha = 0.5, double epsilon = 1e-5);
