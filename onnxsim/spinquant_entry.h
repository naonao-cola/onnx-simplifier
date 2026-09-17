#pragma once

// Calibration-driven SpinQuant (Liu et al., 2024) "R1-only" entry point
// exposed to Python -- C++ port of onnxsim.spinquant's own apply_spinquant
// (see onnxsim/spinquant.py's module docstring for the full technique: fit
// a single dense `[K, K]` rotation per matched MatMul/vanilla-Gemm layer as
// the eigenvector basis of that layer's own calibration-activation
// covariance -- a closed-form substitute for SpinQuant's own learned,
// Cayley-manifold-optimized rotation -- then conjugate the weight by it
// before block-wise INT4 quantization).
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
//   U: initializer, float32 [K, K]             -- the fitted rotation
//   Xrot = MatMul(X, U)
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

// Applies SpinQuant-style ("R1-only", closed-form) learned-rotation
// preprocessing plus block-wise INT4 quantization to every MatMul/vanilla-
// -Gemm layer with a constant 2-D FLOAT32 weight whose reduction dimension
// `K` is divisible by `block_size`, using real activations captured from
// `model` through `executor` to fit each layer's own rotation.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (reused, unmodified, by spinquant.py itself): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1.
//
// Rotation: `U`, the eigenvector basis of `cov = X^T @ X / rows` (`X`:
// every captured calibration-activation row for this layer, rank-agnostic
// -ly flattened to `[rows, K]` -- see below), computed by an ordinary
// symmetric (cyclic Jacobi) eigendecomposition. `Wtilde_nk = W_nk @ U` is
// exact before quantization for any orthogonal `U` -- see
// onnxsim/spinquant.py's own module docstring for why the eigenvector
// basis specifically is the classical, closed-form answer to "which
// rotation makes this layer's own activation second-moment structure as
// close to isotropic as possible".
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own graph
// inputs throws `std::invalid_argument`. Activations are captured the same
// rank-agnostic way onnxsim.bias_correction._activation_rows does (any
// observed tensor of rank >= 2 is flattened to `[rows, K]` by collapsing
// every leading dimension, not just a plain 2-D one), matching
// spinquant.py's own import of that helper -- NOT subgraph-aware, matching
// every other calibration-driven pass's own scope decision.
//
// A layer whose activation was never observed, or whose feature dim does
// not match the weight's reduction dimension `K`, or whose `K` is not
// divisible by `block_size`, is left untouched -- mirrors apply_spinquant's
// own per-layer skip conditions exactly. A model with no matching layer, or
// older than opset 21 (INT4's tensor type and DequantizeLinear's
// block_size attribute both need it), is returned unchanged.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_spinquant's own FLOAT-only weight requirement.
//
// ACCEPTED, PERMANENT DIVERGENCE: the eigendecomposition here is a
// hand-rolled cyclic Jacobi eigenvalue algorithm, not LAPACK's own
// symmetric tridiagonal QR/divide-and-conquer routine (what numpy's
// numpy.linalg.eigh calls into) -- see low_rank_compensation_entry.cpp's
// own "SVD CHOICE" top-of-file comment for why no linear-algebra library is
// linked into this codebase at all, the same rationale applies here. Like
// any eigendecomposition/SVD, the eigenvector basis is only unique up to a
// sign flip per eigenvector (and up to an arbitrary orthogonal rotation
// within any repeated/near-repeated eigenvalue's own subspace), so
// individual columns of `U` are not expected to agree sign-for-sign, order-
// -for-order, or bit-for-bit with the Python reference. What IS
// guaranteed, by construction, regardless of which valid eigenbasis either
// side's algorithm lands on: `U` is orthogonal (`U @ U.T == I`, checked
// directly), and the rotate-then-quantize composition is exact before
// quantization for ANY orthogonal `U` -- the reconstructed model's own
// numerical behavior is comparably close to the Python reference's own
// (INT4 quantization noise dominates either way), not bit-identical. See
// tests/test_spinquant_cpp.py for how this is verified (by orthogonality
// and reconstruction-error bounds, not raw U column equality).
onnx::ModelProto ApplySpinquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size = 32);
