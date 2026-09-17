#pragma once

// SVDQuant / Nunchaku (Li, Lin, Zhang, et al., 2024) entry point exposed to
// Python -- C++ port of onnxsim.svdquant's own apply_svdquant (see
// onnxsim/svdquant.py's module docstring for the full technique: an
// optional SmoothQuant-style migration, then for every matched MatMul/
// vanilla-Gemm layer with a constant 2-D FLOAT32 weight, a low-rank/
// residual split of the weight -- `W' = L1 @ L2 + R` via truncated SVD,
// keeping the rank-`r` dominant/outlier structure in a full-precision
// low-rank branch and block-wise INT4-quantizing only the (now much more
// uniform) residual `R`).
//
// This is single-model and calibration-driven ONLY through the optional
// SmoothQuant preprocessing step -- the low-rank/residual split itself is a
// static decomposition of the weight, needing no calibration data at all
// (mirrors apply_svdquant's own docstring: "the low-rank/residual split
// itself needs no calibration data"). Composes directly with
// smoothquant_entry.h's own ApplySmoothQuant (declared there, included via
// onnxsim.h, which this header is itself included from -- see that header's
// own include list) rather than reimplementing the migration step, the same
// way apply_svdquant itself calls apply_smoothquant directly rather than
// duplicating it.
//
// Like low_rank_compensation_entry.h's own ApplyLowRankCompensation (the
// closest existing precedent -- another SVD-based, low-rank-branch-plus-
// -residual port at the protobuf level), this operates directly on
// onnx::GraphProto rather than through onnxoptimizer's Node/Value IR: a
// two-branch (low-rank correction plus quantized residual) rewrite summed
// into a layer's own output has no established PredicateBasedPass shape in
// this codebase (every PredicateBasedPass in onnxsim/passes/ is data-free
// AND single-branch by construction).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C carried through unchanged):
//   Y = MatMul(X, W) [+ bias]                  W constant, [K, N], float32
// After:
//   L1: initializer, float32 [K, r]      -- low-rank branch, first factor
//   L2: initializer, float32 [r, N]      -- low-rank branch, second factor
//   Rq: DequantizeLinear(R_codes, R_scale, axis=0, block_size) -- [K, N]
//   base    = MatMul(X, Rq)
//   lowrank = MatMul(MatMul(X, L1), L2)
//   Y = Add(base, lowrank) [+ bias]
//
// Forward declaration only -- see low_rank_compensation_entry.h's own
// identical forward declaration for why (the full ModelExecutor interface
// lives in onnxsim.h, which includes this header back).
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelExecutor;

// Applies SVDQuant-style low-rank-branch-plus-residual-quantization to
// every MatMul/vanilla-Gemm layer with a constant 2-D FLOAT32 weight whose
// reduction dimension `K` is divisible by `block_size`. See this header's
// own top-of-file comment for the technique and onnxsim/svdquant.py's
// module docstring for its full scope and deliberately-not-ported pieces
// (activation quantization, Nunchaku's own kernel fusion, iterative
// low-rank refinement, GPTQ-for-residual).
//
// When `smooth_alpha` has a value, `model` is first migrated in place by
// ApplySmoothQuant(model, executor, calibration_data, *smooth_alpha, 1e-5)
// -- the exact preprocessing step apply_svdquant's own
// `apply_smoothquant(model, ..., alpha=smooth_alpha, ...)` call performs,
// including SmoothQuant's own fixed `epsilon=1e-5` default (apply_svdquant
// exposes no separate epsilon knob of its own to override it). Passing
// `std::nullopt` skips migration entirely and decomposes the raw weight
// instead -- mirrors apply_svdquant's own `smooth_alpha=None` sentinel.
// `calibration_data` (one `{graph input name: TensorProto}` map per batch)
// is forwarded to that step unchanged, and only matters when `smooth_alpha`
// has a value: a batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument` (ApplySmoothQuant's own contract), but only when
// migration actually runs.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (reused, unmodified, by svdquant.py itself) -- see
// low_rank_compensation_entry.h's own top-of-file comment for why this
// port, unlike that one, matches directly against `model`'s own weight
// rather than a separately-supplied quantized counterpart (svdquant has no
// pre-existing INT4 model to correct; it produces its own quantization from
// scratch, like spinquant_entry.h's ApplySpinquant).
//
// `rank` is the low-rank branch's own rank `r`, clamped to `min(rank, K,
// N)` per layer (a layer whose clamped rank is `<= 0` is left untouched,
// mirroring apply_svdquant's own `if r <= 0: continue`); `block_size` is
// the residual's own quantization block size along `K`, matching
// `quantize_weight_only_int4`'s own granularity -- mirrors apply_svdquant's
// own parameters of the same names exactly. A layer whose reduction
// dimension `K` is not divisible by `block_size` is left untouched. A model
// with no matching layer, or older than opset 21 (INT4's tensor type and
// DequantizeLinear's `block_size` attribute both need it), is returned
// unchanged.
//
// FLOAT32-only throughout, mirroring apply_svdquant's own FLOAT-only weight
// requirement.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own SVD is a hand-rolled
// one-sided (Hestenes) Jacobi SVD (transcribed from
// low_rank_compensation_entry.cpp's own EconomySvd/JacobiSvdTall -- not
// reused directly since they are private to that translation unit's own
// anonymous namespace), not LAPACK's own Golub-Kahan/bidiagonal-QR
// algorithm (what numpy's np.linalg.svd calls into) -- see that file's own
// top-of-file "SVD CHOICE"/"ACCEPTED, PERMANENT DIVERGENCE" comments for
// the full rationale (no linear-algebra library is linked into this
// codebase) and numerical caveat this choice implies: individual singular
// vectors/values are not expected to agree sign-for-sign or bit-for-bit
// with the Python reference, but the reconstructed rank-`r` low-rank branch
// `L1 @ L2` itself (basis- and sign-invariant, unique by the Eckart-Young
// theorem whenever the r-th and (r+1)-th singular values are well
// separated) is expected to agree closely. See tests/test_svdquant_cpp.py
// for how this is verified.
onnx::ModelProto ApplySvdquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank = 32, int64_t block_size = 32,
    std::optional<double> smooth_alpha = 0.5);
