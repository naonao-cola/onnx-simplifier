// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Calibration-driven FPTQ (Li, Zhang, Li, Yao, Zhang, Chu, Sun, Du and
// Xie, 2023, "FPTQ: Fine-grained Post-Training Quantization for Large
// Language Models", https://arxiv.org/abs/2308.15987) migration entry
// point exposed to Python -- C++ port of fptq.py's own apply_fptq (see
// that module's own docstring for the full technique and closed-form
// derivation of the logarithmic scale below).
//
// Shares smoothquant_entry.h's own ApplySmoothQuant core mechanism almost
// verbatim (this port is smoothquant_entry.cpp's own MatchMatMulLike/
// ComputeChannelAbsmax/node-insertion machinery, transcribed near-total --
// same protobuf-level, single-model, calibration-driven shape) -- for a
// MatMul/Gemm `Y = X @ W`, dividing one activation channel by a
// per-channel scale `s_j` and multiplying the matching weight row by the
// same `s_j` leaves `Y` exactly unchanged while moving quantization
// difficulty from the activation into the weight. FPTQ differs from plain
// SmoothQuant only in HOW `s` is chosen on "intractable" layers (see
// below) -- every other "tractable" layer keeps SmoothQuant's own
// power-law scale (`s_j = max(|X_j|)**alpha / max(|W_j|)**(1-alpha)`)
// unchanged.
//
// A layer is classified "intractable" when its largest per-channel
// activation max sits at least `outlier_ratio_threshold` times its
// per-channel GEOMETRIC MEAN activation max
// (`ref = exp(mean(log(max(|X_j|, epsilon))))`,
// `outlier_ratio = max_j(max(|X_j|, epsilon)) / ref`); such a layer gets
// FPTQ's own "logarithmic equalization" scale instead:
//
//   ref_j  = geometric mean of max(|X_j'|) over every channel j' in the
//            layer (a single per-layer scalar, not per-channel, despite
//            the paper's own `ref_j` notation -- see fptq.py's own
//            docstring)
//   ratio_j = max(|X_j|) / ref_j
//   s_j     = ref_j * log2(1 + ratio_j)
//
// so an ordinary channel (ratio close to 1) is left close to untouched
// (`log2(2) == 1`), while an outlier channel is pulled down
// logarithmically rather than fully linearly.
//
// Like ApplySmoothQuant, this only performs the migration -- returns a
// float model, provably equivalent to the input up to floating-point
// rounding, meant to be fed to onnxsim's own W4 weight-only quantizer and
// an A8 activation quantizer afterward.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see smoothquant_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Migrates activation quantization difficulty into the weight for every
// matched MatMul/vanilla-Gemm node with a constant 2-D FLOAT32 weight,
// using real calibration activations run through `executor`: rescales the
// weight's reduction-dimension columns by `s` in place and inserts a new
// `Mul` node dividing that layer's activation input by the same `s` right
// before it -- SmoothQuant's own power-law `s`, or FPTQ's own logarithmic
// `s` on layers this module's own header comment classifies
// "intractable". Returns a float model.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (smoothquant_entry.h's own ApplySmoothQuant) -- a
// `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument`. NOT subgraph-aware, matching every one
// of those passes' own scope decision.
//
// A candidate whose activation was never observed as a plain 2-D FLOAT32
// tensor across `calibration_data`, or whose activation feature dimension
// does not match the weight's reduction dimension `K`, is left untouched
// -- mirrors apply_fptq's own per-layer skip conditions exactly (including
// the strict 2-D requirement inherited from ApplySmoothQuant: a rank-3+
// activation is skipped, never reduced -- unlike gptq_entry.h's/
// gptaq_entry.h's own rank-agnostic `reshape(-1, K)` convention).
//
// `alpha` is the migration strength used on "tractable" layers, identical
// in meaning to ApplySmoothQuant's own `alpha`; `outlier_ratio_threshold`
// is the "intractable" classification threshold described above;
// `epsilon` floors every per-channel activation/weight max-abs value (and
// the geometric-mean reference `ref`, and `s` itself) before dividing --
// mirrors apply_fptq's own parameters of the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// ApplySmoothQuant's own FLOAT32-only scope exactly.
onnx::ModelProto ApplyFptq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double alpha = 0.5, double outlier_ratio_threshold = 10.0,
    double epsilon = 1e-5);
