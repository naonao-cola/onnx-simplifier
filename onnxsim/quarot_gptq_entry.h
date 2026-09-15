#pragma once

// Calibration-driven QuaRot+GPTQ (Ashkboos et al., 2024) entry point
// exposed to Python -- C++ port of quarot.py's own apply_quarot_gptq (see
// that module's docstring for the full technique: the real QuaRot paper's
// optional, tighter weight quantizer -- identical to apply_quarot in every
// respect except the *weight* is quantized via onnxsim.gptq's Hessian-based
// column algorithm, evaluated in the rotated activation space, instead of
// independent round-to-nearest).
//
// Like gptq_entry.h's own ApplyGptq (the closest existing precedent for
// the calibration/Hessian machinery) and passes/quarot.h's own Quarot pass
// (the closest existing precedent for the rotation/node-emission shape),
// this operates directly on onnx::GraphProto rather than through
// onnxoptimizer's Node/Value IR: unlike ApplyGptq/ApplyAwq, there is no
// separately-quantized "quantized_model" input here -- like passes/quarot.h,
// this pass derives its own rotation and quantizes from the float model
// alone, but (unlike passes/quarot.h, which is a data-free
// PredicateBasedPass) needs a live ModelExecutor plus calibration batches
// to compute each matched layer's rotated-space Hessian, which
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path for -- hence its own protobuf-level entry point,
// following llm_int8_entry.h's single-model calibration-driven shape.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Rotates every matched MatMul/vanilla-Gemm layer's activation by a fresh
// per-layer random orthogonal matrix `U` and quantizes both operands to
// INT4 -- the weight (`W @ U`) via onnxsim.gptq's Hessian-compensated
// column algorithm (the Hessian computed in the *rotated* activation space,
// `H = (X @ U)^T (X @ U)`, from real activations captured from `model`
// through `executor`), the activation via the same data-free, per-token
// round-to-nearest INT4 round trip passes/quarot.h's own ApplyQuarot
// already emits. Matching, weight-eligibility (constant 2-D FLOAT32,
// opset >= 21, reduction dimension K divisible by `block_size`), and the
// emitted node shape (Abs/ReduceMax/Clip/Div/Round/Clip/Mul for the
// activation, DequantizeLinear for the weight, a final MatMul, and the
// original bias if any) are exactly ApplyQuarot's own -- see that pass for
// the full rewrite diagram. Returns `model` with every matched layer that
// has usable calibration data rewritten; a layer without usable
// calibration data (never observed with a feature axis, or whose feature
// dimension doesn't match K) is left completely untouched, unlike
// ApplyQuarot, which never skips a layer for lack of data since it needs
// none. A model with no matching layer, or an opset older than 21, is
// returned unchanged.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. NOT
// subgraph-aware, matching every one of those passes' own scope decision.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// ApplyGptq's own FLOAT32-only calibration scope.
//
// `seed` derives a fresh, deterministic random rotation per matched layer
// -- same ACCEPTED, PERMANENT DIVERGENCE from quarot.py's own RNG
// derivation/orthogonalization algorithm that passes/quarot.h's own
// ApplyQuarot already documents (see that header's top-of-file comment and
// passes/random_orthogonal.h for the full investigation): the same `seed`
// does NOT produce the same rotation, or therefore the same quantized
// weights, as quarot.py's apply_quarot_gptq. `block_size` is the number of
// reduction-dimension (K) elements sharing one weight quantization scale,
// matching ApplyQuarot's own default. `percdamp` is the Hessian damping
// factor and `proc_block_size` is GPTQ's own column-processing block size
// (not the quantization scale's block size) -- mirrors ApplyGptq's own
// parameters of the same names exactly. `epsilon` floors a token's own
// max-abs rotated-activation value before it is used as a quantization
// scale, matching ApplyQuarot's own parameter exactly.
//
// Accepted numerical scope (like ApplyGptq's own): the dense inverse and
// Cholesky factorization at GPTQ's heart are computed with this TU's own
// scalar double-precision kernels rather than LAPACK, so results can differ
// from the reference in the last ulp or two on some inputs -- see
// gptq_entry.h's own identical note.
onnx::ModelProto ApplyQuarotGptq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    uint64_t seed = 0, int64_t block_size = 32, double percdamp = 0.01,
    int64_t proc_block_size = 128, float epsilon = 1e-12f);
