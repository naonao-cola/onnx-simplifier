#pragma once

// Calibration-driven SpQR (Dettmers et al., 2023) entry point exposed to
// Python -- C++ port of onnxsim.spqr's own quantize_weight_only_spqr (see
// onnxsim/spqr.py's module docstring for the full technique: block-wise
// INT4 quantization where a small fraction of per-element outliers are
// excluded from each block's own scale computation and stored instead as
// an exact sparse correction).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another single-model, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather
// than through onnxoptimizer's Node/Value IR: threading a live
// ModelExecutor plus calibration batches through OptimizeFixed's
// single-node-match PredicateBasedPass model has no established path in
// this codebase (every PredicateBasedPass in onnxsim/passes/ is data-free
// by construction).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Wq    = <int4, per-(block, output-channel) symmetric, outlier
//            positions excluded from each block's own scale>
//   Ws    = <float32, [K/block_size, N]>
//   Wdq   = DequantizeLinear(Wq, Ws, axis=0, block_size=block_size)
//   zeros = ConstantOfShape([K, N], value=0.0)
//   correction = ScatterND(zeros, outlier_indices, outlier_values)
//   Wreconstructed = Wdq + correction
//   Y = MatMul(X, Wreconstructed) [+ bias]
// (when no outlier is selected at all -- outlier_fraction so small that
// round(outlier_fraction * N * K) rounds to 0 -- the ConstantOfShape/
// ScatterND/Add trio is omitted entirely and Wreconstructed is just Wdq,
// mirroring quantize_weight_only_spqr's own identical `else` branch.)

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies SpQR-style outlier-aware block-wise INT4 quantization to every
// MatMul/vanilla-Gemm layer with a constant 2-D FLOAT32 weight whose
// reduction dimension `K` is divisible by `block_size`, using real
// activations captured from `model` through `executor` to compute each
// weight element's sensitivity score.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (reused, unmodified, by spqr.py itself): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1.
//
// Sensitivity score: the classical diagonal-Hessian OBQ approximation,
// `w_k^2 * mean(X[:, k]^2)` -- see spqr.py's own module docstring for the
// closed-form derivation from the full-Hessian objective. The
// `outlier_fraction` (by count, rounded half-to-even like Python's own
// `round()`) elements with the largest score across the WHOLE layer
// (not per-block/per-row) are excluded from their own block's scale
// computation and become the sparse correction.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `model`'s own graph inputs) shape as every
// other calibration-driven pass in this codebase -- a `calibration_data`
// batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument`. Activations are captured the same
// rank-agnostic way onnxsim.bias_correction._activation_rows does (any
// observed tensor of rank >= 2 is flattened to `[rows, K]` by collapsing
// every leading dimension, not just a plain 2-D one), matching spqr.py's
// own import of that helper -- NOT subgraph-aware, matching every other
// calibration-driven pass's own scope decision.
//
// A layer whose activation was never observed, or whose feature dim does
// not match the weight's reduction dimension `K`, or whose `K` is not
// divisible by `block_size`, is left untouched -- mirrors
// quantize_weight_only_spqr's own per-layer skip conditions exactly. A
// model with no matching layer, or older than opset 21 (INT4's tensor
// type and DequantizeLinear's block_size attribute both need it), is
// returned unchanged.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// quantize_weight_only_spqr's own FLOAT-only weight requirement.
//
// ACCEPTED, PERMANENT DIVERGENCE: outlier selection here uses a full
// deterministic sort by sensitivity score (descending, ties broken by
// ascending flat index) rather than reproducing numpy's own
// `np.argpartition`, whose internal tie-handling/order among equal-rank
// candidates is an unspecified implementation detail of its introselect
// algorithm -- not something worth reproducing bit-for-bit (the same
// judgment this repo's own k-means-family ports already make for their
// own RNG divergences, applied here to a selection-order divergence
// instead). This changes nothing about WHICH positions are selected as
// outliers (both sides select the exact top-`num_outliers` positions by
// sensitivity score; only a genuine tie exactly at the selection boundary
// -- vanishingly unlikely for real float calibration data -- could ever
// make the SET differ), and the reconstructed model is numerically
// identical regardless of storage order (ScatterND with disjoint indices
// is order-independent) -- only the outlier_indices/outlier_values
// initializers' own row order is not guaranteed to match the Python
// reference byte-for-byte. See tests/test_spqr_cpp.py for how this is
// verified (by outlier position SET and by reconstructed numeric output,
// not raw initializer byte equality).
onnx::ModelProto ApplySpqr(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size = 16, double outlier_fraction = 0.01);
