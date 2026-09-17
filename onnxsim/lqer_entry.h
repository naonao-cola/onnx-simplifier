// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// LQER (Zhang et al., 2024, "LQER: Low-Rank Quantization Error
// Reconstruction for LLMs", https://arxiv.org/abs/2402.02446) -- C++ port
// of lqer.py's own apply_lqer (see that module's own docstring for the
// full technique).
//
// Same graph rewrite as low_rank_compensation_entry.h's own
// ApplyLowRankCompensation (candidate matching -- reimplemented here
// directly on raw onnx::GraphProto/NodeProto/TensorProto, identically to
// that file's own FindCandidates -- and the two-chained-MatMul-plus-Add
// correction shape); LQER's own contribution is entirely in HOW the
// correction's SVD input is built: instead of factoring the raw
// dequantization error matrix, this row-scales it by a calibration-
// measured per-input-channel activation RMS before the SVD (and un-scales
// the result back afterward), so the low-rank fit is optimal in an
// activation-weighted sense rather than a plain (unweighted) Frobenius
// sense -- exactly AWQ's own "some channels matter more" idea, applied to
// error compensation instead of the quantization grid itself.
//
// Unlike ApplyLowRankCompensation (data-free), this needs real calibration
// activations to measure that RMS, so -- like gptq_entry.h's own ApplyGptq
// (the closest calibration-driven precedent for a two-model, protobuf-
// level pass) -- it takes a ModelExecutor plus calibration_data and probes
// `float_model` directly rather than going through onnxoptimizer's
// Node/Value IR.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Adds an activation-weighted rank-`rank` low-rank correction to every
// INT4-quantized MatMul/Gemm layer matched between `float_model` and
// `quantized_model` (by output name -- see adaround.py's own
// _find_int4_matmul_candidates, reimplemented here directly on raw
// protobuf, identically to low_rank_compensation_entry.h's own
// FindCandidates), using real activations captured from `float_model`
// through `executor` to measure each layer's own per-input-channel
// activation RMS.
//
// For each matched layer: the plain (unweighted) dequantization error
// matrix `residual` ([K, N]) is row-scaled by `max(rms_k, eps)` (`rms_k`
// the calibration-measured RMS of that layer's own activation input's
// k-th column -- ONLY when that activation was observed as a plain 2-D
// ([rows, K]) tensor, matching lqer.py's own "skip non-2-D activations"
// convention exactly; a layer whose activation was never observed with a
// matching feature width falls back to an unweighted SVD, i.e. exactly
// ApplyLowRankCompensation's own correction), factored via this TU's own
// hand-rolled Jacobi SVD (see low_rank_compensation_entry.cpp's own
// top-of-file comment for why -- no linear-algebra library is linked into
// this codebase), truncated to the top `rank` components, and un-scaled
// back. The correction is injected the same way ApplyLowRankCompensation
// does: two new chained MatMul nodes plus an Add summed into the layer's
// output (`Y = X @ Wq_dequant + (X @ B) @ A`), the layer's own existing
// INT4 weight/scale left completely untouched. `rank` is clamped to
// `min(rank, K, N)` per layer (a layer whose clamped rank is <= 0 is left
// untouched).
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `float_model`'s own graph inputs) shape as every
// other calibration-driven pass in this codebase -- a `calibration_data`
// batch missing one of `float_model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware, matching every other
// calibration-driven pass's own scope decision. An empty
// `calibration_data` (or one with no matched layer at all) is a strict,
// exact reduction to ApplyLowRankCompensation's own unweighted output
// (every candidate's channel_weights resolves to std::nullopt, taking the
// identical unweighted-SVD code path).
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// ApplyGptq's own FLOAT32-only scope.
//
// ACCEPTED, PERMANENT DIVERGENCE (identical to low_rank_compensation_entry.h's
// own note): this port's own SVD is a hand-rolled one-sided (Hestenes)
// Jacobi SVD, not LAPACK's own Golub-Kahan algorithm -- individual singular
// vectors/values are not expected to agree sign-for-sign or bit-for-bit
// with lqer.py's own numpy-SVD-based reference. What IS expected to agree
// closely is the reconstructed rank-`rank` correction matrix `B @ A`
// itself, unique (by the same weighted Eckart-Young-style argument
// lqer.py's own docstring makes) whenever the matched layer's r-th and
// (r+1)-th weighted singular values are well separated. See
// tests/test_lqer_cpp.py for how this is verified.
onnx::ModelProto ApplyLqer(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank = 8, double eps = 1e-6);
