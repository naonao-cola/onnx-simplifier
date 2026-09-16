// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Low-Rank Compensation (LoRC), from ZeroQuant-V2 (Yao et al., 2023) --
// C++ port of low_rank_compensation.py's own apply_low_rank_compensation.
//
// Like daq_entry.h (see that header's own top-of-file comment for the
// general rationale), this is a data-free, two-model port: it takes a
// float model and an already-INT4-quantized model (produced separately,
// e.g. by quantize_weight_only_int4) and needs no ModelExecutor/
// calibration data at all -- so it follows daq_entry.h's own plain
// top-level-function shape (raw onnx::GraphProto/NodeProto/TensorProto
// manipulation, not onnx-optimizer's Node/Graph IR, which is unavailable
// outside the single-graph PredicateBasedPass registry mechanism this
// two-model shape doesn't fit).
//
// Unlike every other data-free *_cpp port in this repo (which folds its
// correction into a single replacement weight initializer), LoRC's own
// correction is *additive to a matched layer's output*, not a weight
// replacement: for every MatMul/Gemm layer the base low_rank_compensation.py
// (and this port) can match between the two models (int4-quantized weight,
// vs. the float model's own full-precision counterpart, matched by output
// name -- see adaround.py's own _find_int4_matmul_candidates, reimplemented
// here directly on raw protobuf since it isn't exposed as a C++ symbol
// outside adaround_entry.cpp's own translation unit), this port computes the
// dequantized weight's error against the float weight, fits that error's
// best rank-r approximation via truncated SVD (Eckart-Young theorem), and
// injects the correction as two new small MatMul nodes plus an Add summed
// into the layer's output:
//
//   Y = X @ Wq_dequant + (X @ B) @ A,  B: [K, r], A: [r, N]
//
// This genuinely needs new graph nodes (unlike a fold-to-initializer port),
// since the correction is additive to an activation, not a weight.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own SVD is a hand-rolled
// one-sided (Hestenes) Jacobi SVD -- see low_rank_compensation_entry.cpp's
// own top-of-file comment for why (no linear-algebra library is linked
// into this codebase) and for the numerical caveat this choice implies
// (the reconstructed rank-r correction matches numpy/LAPACK's own SVD up
// to ordinary floating-point rounding, but individual singular
// vectors/values are not expected to match sign-for-sign or bit-for-bit).

#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>

// Applies Low-Rank Compensation: for every INT4-quantized MatMul/Gemm layer
// matched between `float_model` and `quantized_model` (by output name, the
// same correspondence DAQ/GPTQ/Qronos already make -- see
// adaround.py's own _find_int4_matmul_candidates), computes the
// quantization error matrix, fits its best rank-`rank` approximation via
// truncated SVD, and adds two chained MatMul nodes plus an Add node
// injecting that correction into the layer's output. `rank` is clamped to
// `min(rank, K, N)` per layer (a layer whose clamped rank is <= 0 is left
// untouched). Layers with no INT4/DequantizeLinear counterpart, or whose
// float/quantized shapes don't match, are left untouched. Returns a new
// model; `quantized_model` itself is never modified.
onnx::ModelProto ApplyLowRankCompensation(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, int64_t rank = 8);
