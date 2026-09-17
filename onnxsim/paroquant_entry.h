#pragma once

// Calibration-driven ParoQuant (Liang et al., 2025) entry point exposed to
// Python -- C++ port of onnxsim.paroquant's own apply_paroquant (see
// onnxsim/paroquant.py's module docstring for the full technique, and its
// own explicit contrast with onnxsim.spinquant: instead of SpinQuant's one
// dense `[K, K]` rotation, fit many independent, cheap 2x2 Givens
// (pairwise) rotations on fixed adjacent-channel pairs `(0, 1), (2, 3),
// ...` within each quantization block -- each block-diagonal, each angle
// grid-searched independently against its own block's INT4 round-to-
// -nearest reconstruction error -- combined with a SmoothQuant-style
// per-channel scale migration, then block-wise INT4 quantization).
//
// Like spqr_entry.h's own ApplySpqr (the closest existing precedent --
// another single-model, calibration-driven, block-wise-INT4-quantizing
// MatMul/Gemm rewrite at the protobuf level, sharing this module's own
// matcher, onnxsim.quip_sharp._match_matmul_like), this operates directly
// on onnx::GraphProto rather than through onnxoptimizer's Node/Value IR:
// threading a live ModelExecutor plus calibration batches through
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C carried through unchanged):
//   Y = MatMul(X, W) [+ bias]                  W constant, [K, N], float32
// After:
//   inv_s: initializer, float32 [K]            -- 1 / SmoothQuant-style scale
//   R: initializer, float32 [K, K]              -- block-diagonal, 2x2 Givens
//                                                  blocks on (0,1),(2,3),...
//   Xscaled = Mul(X, inv_s)
//   Xrot = MatMul(Xscaled, R)
//   Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size)
//                                                -- INT4 codes, [K, N]
//   Y = MatMul(Xrot, Wtilde_hat) [+ bias]

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see spqr_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies ParoQuant-style channel-wise SmoothQuant scaling plus pairwise
// (Givens) rotation preprocessing followed by block-wise INT4 quantization
// to every MatMul/vanilla-Gemm layer with a constant 2-D FLOAT32 weight
// whose reduction dimension `K` is divisible by `block_size` (which must
// itself be even), using real activations captured from `model` through
// `executor` to compute each layer's own per-channel scale.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (reused, unmodified, by paroquant.py itself): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1.
//
// Rotation: within each `block_size`-wide quantization block, an
// independent 2x2 Givens rotation on each fixed adjacent-channel pair
// `(0, 1), (2, 3), ...`, angle grid-searched over `num_angle_steps` points
// in `[-pi/4, pi/4]` (`0` always included when `num_angle_steps` is odd)
// to minimize that pair's own contribution to its block's mean-squared
// INT4 round-to-nearest reconstruction error, evaluated on the block's
// already-partially-rotated state (pairs within a block are optimized
// left-to-right, each seeing every earlier pair's own effect). `R`, the
// resulting `[K, K]` matrix, is block-diagonal (2x2 orthogonal Givens
// blocks on the chosen pairs, identity elsewhere) and therefore itself
// orthogonal -- `Wtilde_nk = (W_nk * s) @ R` is exact before quantization,
// same "provably exact migration, then quantize" contract every rotate-
// -then-quantize scheme in this codebase already uses.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own graph
// inputs throws `std::invalid_argument`. Activation capture is 2-D-only
// (mirrors `if x.ndim != 2: continue` exactly -- the per-channel abs-max
// this scale needs, NOT the rank-agnostic `_activation_rows` flattening
// onnxsim.spinquant's own entry point uses), matching apply_paroquant's own
// scope. NOT subgraph-aware, matching every other calibration-driven
// pass's own scope decision.
//
// A layer whose activation was never observed as a plain 2-D FLOAT32
// tensor, or whose feature dim does not match the weight's reduction
// dimension `K`, or whose `K` is not divisible by `block_size`, is left
// untouched -- mirrors apply_paroquant's own per-layer skip conditions
// exactly. A model with no matching layer, an odd `block_size`, or an
// opset older than 21 (INT4's tensor type and DequantizeLinear's
// block_size attribute both need it), is returned unchanged.
//
// `alpha` is the SmoothQuant-style migration strength; `epsilon` floors
// every per-channel activation/weight max-abs value (and the resulting
// scale) before use, avoiding a divide-by-zero on an all-zero channel --
// mirrors apply_paroquant's own parameters of the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_paroquant's own FLOAT-only weight requirement.
//
// ACCEPTED SCOPE: the per-pair angle search here evaluates the SAME
// `num_angle_steps`-point grid, in the same left-to-right, block-then-pair
// order, against the same mean-squared-reconstruction-error objective as
// apply_paroquant's own _fit_paroquant_pairwise_rotation -- unlike this
// codebase's k-means/eigendecomposition ports, there is no RNG or LAPACK-
// -equivalent algorithm choice here to diverge on, so this port's own
// fitted angles are expected to agree closely (grid search over identical
// candidate angles against identical, deterministic double-precision
// arithmetic), subject only to ordinary floating-point summation-order
// noise -- see tests/test_paroquant_cpp.py for the measured agreement.
onnx::ModelProto ApplyParoquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size = 32, double alpha = 0.5, int64_t num_angle_steps = 9,
    double epsilon = 1e-5);
