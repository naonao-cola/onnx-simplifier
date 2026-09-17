// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// EasyQuant (Wu, Judd, Isaev, Micikevicius, 2020, "EasyQuant: Post-training
// Quantization via Scale Optimization", https://arxiv.org/abs/2006.16669)
// entry point exposed to Python -- C++ port of onnxsim.easyquant's own
// apply_easyquant (see that module's own docstring for the full technique
// and this port's own scope note below for the exact grid/iteration
// parameters).
//
// Single-model, calibration-driven, plain W8A8 (INT8 weight + INT8
// activation) quantizer using coordinate-descent GRID SEARCH -- no
// gradients, no Hessian, unlike gptq_entry.h's own ApplyGptq or
// awq_entry.h's own ApplyAwq. Closest existing precedent for this
// codebase's own house style of a coarse-to-fine/grid scalar search inside
// a calibration-driven pass is daq_entry.h's own ApplyDaq, though DAQ
// searches one scalar per layer while this module needs a per-output-
// channel grid search for the weight step (see below) -- so this follows
// llm_int8_entry.h's/smoothquant_entry.h's own protobuf-level, single-
// model, node-insertion shape instead (the closest precedent for
// "calibration-driven weight rewrite PLUS new Div/Round/Clip/Mul nodes at
// an activation input").
//
// Matches every MatMul / "vanilla" Gemm (`transA=0`, `alpha=1`, and
// `beta=1` when a bias is present) node with a constant 2-D FLOAT32
// weight -- mirrors onnxsim.llm_int8's own `_match_matmul_like` exactly
// (`Conv` is out of scope, matching apply_easyquant's own scope note).
// For each match, real calibration activations are captured at the
// node's own activation input (`reshape(-1, K)`-flattened across any
// leading batch/sequence axes, exact) and a coordinate-descent grid
// search alternates:
//
//   1. WEIGHT STEP (holding the activation scale fixed): for each output
//      channel independently, grid-search a multiplier around that
//      channel's own current scale minimizing that channel's own
//      quantize-dequantize-round-trip output MSE against the float
//      output -- exact and independent per channel, since column `n` of
//      `Y = X @ W^T` depends only on row `n` of `W`.
//   2. ACTIVATION STEP (holding the just-updated weight scale fixed):
//      grid-search a multiplier on the single per-tensor activation scale
//      maximizing the WHOLE quantized layer output's cosine similarity
//      against the float output (not separable per-channel, so this step
//      evaluates the full output).
//
// repeated for `num_iterations` rounds. `num_candidates` sets the grid
// resolution (`linspace(1 - search_span, 1 + search_span, num_candidates)`,
// filtered to positive multipliers) and `search_span` sets the grid's own
// span around each step's current scale -- mirrors apply_easyquant's own
// parameters of the same names exactly (defaults 3/21/0.5).
//
// The optimized weight is folded into a new float32 initializer (plain
// quantize-dequantize round-trip, `clip(round(w / s), -127, 127) * s`, no
// new graph node -- the same pattern every weight-only `quantize_*`
// function in this repo uses), while the activation side needs
// `Div`/`Round`/`Clip`/`Mul` nodes inserted before the matched node's own
// activation input (a runtime tensor, not a constant) -- mirrors
// apply_easyquant's own graph-rewrite shape exactly, including which
// nodes are minted and their insertion order.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own
// graph inputs throws `std::invalid_argument`. NOT subgraph-aware,
// matching every one of those passes' own scope decision.
//
// A layer whose weight is not a constant 2-D FLOAT32 tensor, whose
// activation was never observed with a feature axis at all (rank < 2), or
// whose activation's feature dimension does not match the weight's own
// reduction size, is left completely untouched -- mirrors
// apply_easyquant's own per-layer skip conditions exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// every other calibration-driven pass in this codebase's own FLOAT32-only
// scope: a FLOAT16/BFLOAT16 activation is never observed (skipped like an
// unobserved one), never converted.
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond ordinary floating-point
// summation-order/accumulation differences from numpy -- this is a
// bounded, deterministic grid search with no RNG or hand-rolled dense
// linear algebra (matmuls here are plain reduction loops, not a Cholesky/
// SVD/eigendecomposition kernel), so this port is expected to track the
// Python reference closely, including exact-tie behavior at every grid
// point boundary (ties broken by first-seen-wins on both sides, since
// neither this port nor apply_easyquant's own `<`/`>` comparisons ever
// replace a tied incumbent). See tests/test_easyquant_cpp.py for the
// measured agreement.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyEasyquant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 3, int64_t num_candidates = 21,
    double search_span = 0.5);
