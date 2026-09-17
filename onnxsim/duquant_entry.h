#pragma once

// Calibration-driven DuQuant (Lin et al., 2024) entry point exposed to
// Python -- C++ port of onnxsim.duquant's own apply_duquant (see
// onnxsim/duquant.py's module docstring for the full technique: a
// calibration-ranked permutation that redistributes each layer's worst
// outlier input channels one-per-block across the quantization grouping,
// composed with an independent Haar-random orthogonal rotation applied
// *within* each block, then INT4 round-to-nearest quantization of BOTH
// the weight (offline, block-wise) and the activation (data-free,
// per-token, at graph-run time)).
//
// Like spinquant_entry.h's own ApplySpinquant (the closest existing
// precedent -- another single-model, calibration-driven, block-wise-INT4
// -quantizing MatMul/Gemm rewrite at the protobuf level that also fits a
// per-layer `[K, K]` rotation, sharing this module's own matcher,
// onnxsim.quip_sharp._match_matmul_like), this operates directly on
// onnx::GraphProto rather than through onnxoptimizer's Node/Value IR:
// threading a live ModelExecutor plus calibration batches through
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C carried through unchanged):
//   Y = MatMul(X, W) [+ bias]                  W constant, [K, N], float32
// After (X's own INT4 round-to-nearest quantization is simulated data-free,
// via an immediate dequantize -- X itself is never constant, unlike the
// weight):
//   U: initializer, float32 [K, K]     -- permutation @ block-local rotation
//   Xrot = MatMul(X, U)
//   Xq_hat = <per-token round-to-nearest-INT4 dequantize of Xrot>
//   Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size)
//                                                -- INT4 codes, [K, N]
//   Y = MatMul(Xq_hat, Wtilde_hat) [+ bias]
//
// Forward declaration only -- see spinquant_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in onnxsim.h,
// which includes this header back).
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelExecutor;

// Applies DuQuant-style calibrated permutation + block-local random rotation
// plus INT4 round-to-nearest quantization of BOTH the weight and the
// activation to every MatMul/vanilla-Gemm layer with a constant 2-D
// FLOAT32 weight whose reduction dimension `K` is divisible by
// `block_size`, using real activations captured from `model` through
// `executor` to rank each layer's own input channels by outlier magnitude.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (reused, unmodified, by duquant.py itself) -- see spinquant_entry.h's own
// identical note.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own graph
// inputs throws `std::invalid_argument`. Activation capture is 2-D-only
// (mirrors `if x.ndim != 2: continue` exactly -- NOT the rank-agnostic
// `_activation_rows` flattening spqr_entry.h/gptq_entry.h use), matching
// apply_duquant's own per-channel abs-max scope (transcribed from
// smoothquant_entry.cpp's/rptq_entry.cpp's own identical ComputeChannelAbsmax).
// NOT subgraph-aware, matching every other calibration-driven pass's own
// scope decision.
//
// A layer whose activation was never observed as a plain 2-D FLOAT32
// tensor, or whose feature dim does not match the weight's reduction
// dimension `K`, or whose `K` is not divisible by `block_size`, is left
// untouched -- mirrors apply_duquant's own per-layer skip conditions
// exactly. A model with no matching layer, or older than opset 21 (INT4's
// tensor type and DequantizeLinear's block_size attribute both need it),
// is returned unchanged.
//
// Rotation construction `U = P @ R` (mirrors duquant.py's own
// `_build_duquant_rotation` exactly at the algorithm level):
//   1. `P`: a genuine permutation matrix. The `outlier_fraction * K`
//      (rounded, floored at 1, capped at `K`) largest-abs-max channels are
//      greedily assigned, largest first, one at a time, to whichever
//      quantization block currently holds the least total assigned outlier
//      magnitude; every block's remaining slots are filled with the
//      non-outlier channels in their original relative magnitude order.
//   2. `R`: block-diagonal, each `block_size x block_size` diagonal block
//      an independent Haar-random orthogonal matrix (RandomOrthogonalMatrix,
//      passes/random_orthogonal.h -- the same generator quarot_gptq_entry.cpp
//      already uses).
// `P @ R` is orthogonal by construction (a permutation composed with a
// block-diagonal-orthogonal matrix), so `Wtilde_nk = W_nk @ U` is exact
// before quantization for any layer this reaches -- see this header's own
// "ACCEPTED, PERMANENT DIVERGENCE" note below for what that does and does
// not guarantee relative to the Python reference.
//
// `epsilon` floors a token's own max-abs activation value before it is used
// as a per-token INT4 activation scale, avoiding a divide-by-zero on an
// all-zero token -- mirrors apply_duquant's own parameter of the same name
// exactly. `seed` seeds the block-local rotation draws.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_duquant's own FLOAT-only weight requirement.
//
// ACCEPTED, PERMANENT DIVERGENCE: like quarot_gptq_entry.cpp's own RNG
// derivation (see that file's own identical note, and
// passes/random_orthogonal.h's own top-of-file investigation), this port
// seeds a FRESH std::mt19937_64 per matched node (from `seed` combined with
// that node's own index), rather than reproducing duquant.py's own single,
// sequentially-advancing `numpy.random.Generator` thread across every layer
// and every block. Each block's own rotation is independently drawn from
// that per-node RNG, in block order, mirroring `_build_duquant_rotation`'s
// own per-block loop shape (just re-scoped per node rather than globally
// sequential) -- and this port's own RandomOrthogonalMatrix is a
// Gram-Schmidt construction, not `_random_orthogonal_matrix`'s sign-
// -corrected QR (see random_orthogonal.h's own top-of-file comment for why
// both are independently Haar-uniform and neither reproduces the other
// bit-for-bit). Consequently, individual columns of `U` -- and which exact
// channels a near-tied outlier ranking assigns to which block -- are not
// expected to agree between the two ports for the same `seed`. What IS
// guaranteed, by construction, regardless of which valid rotation/
// permutation either side's algorithm lands on: `U` is orthogonal (`U @
// U.T == I`, checked directly), and the permute-rotate-then-quantize
// composition is exact before quantization for ANY orthogonal `U` -- the
// reconstructed model's own numerical behavior is comparably close to the
// Python reference's own (INT4 quantization noise dominates either way),
// not bit-identical. See tests/test_duquant_cpp.py for how this is
// verified (by orthogonality and reconstruction-error bounds, not raw U
// column equality).
onnx::ModelProto ApplyDuquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t seed = 0, int64_t block_size = 32, double outlier_fraction = 0.05,
    double epsilon = 1e-12);
