#pragma once

// Calibration-driven GPTVQ (Van Baalen et al., 2024) entry point exposed
// to Python -- C++ port of onnxsim.gptvq's own quantize_weight_only_gptvq
// (see that module's docstring for the full technique: quantizing small
// groups of consecutive input-channel columns against a codebook fit to
// the whole layer's own weight values -- like onnxsim.aqlm's single
// shared codebook, via the same k-means routine -- processing groups
// left to right and propagating each group's resulting per-column
// residual into every not-yet-quantized column via onnxsim.gptq's own
// Cholesky-based Hessian correction).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 and quarot_gptq_entry.h's own
// ApplyQuarotGptq (the closest existing precedents -- the same
// protobuf-level, single-model, calibration-driven shape), this operates
// directly on onnx::GraphProto rather than through onnxoptimizer's
// Node/Value IR: threading a live ModelExecutor plus calibration batches
// through OptimizeFixed's single-node-match PredicateBasedPass model has
// no established path in this codebase. Unlike either of those, this
// pass rewires only the matched node's own weight input (inserting new
// Gather/Reshape[/Transpose] nodes ahead of it) rather than replacing the
// whole node -- the codebook lookup already reconstructs the weight in
// its own units (see gptvq.py's own module docstring on why no scale
// multiply is needed), so the original MatMul/Gemm node (bias included)
// stays untouched.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Quantizes every matched MatMul/vanilla-Gemm layer's constant 2-D
// FLOAT32 weight (reduction dimension K divisible by `vector_dim`) into
// GPTVQ-style Hessian-compensated vector-codebook quantization, using
// real activations captured from `model` through `executor`: fits one
// k-means codebook (`num_centroids` entries of `vector_dim` elements
// each) to the whole layer's own weight values, then quantizes
// consecutive `vector_dim`-wide column groups against it left to right,
// propagating each group's resulting per-column residual into every
// not-yet-quantized column via a GPTQ-style Cholesky-based Hessian
// correction (reusing gptq_entry.cpp's own kernels). Rewires the matched
// node's weight input to `Gather(Codebook, Codes, axis=0)` reshaped back
// to the weight's own shape (transposed back first when the weight was
// not itself stored transposed); the node itself (including any bias) is
// left otherwise unchanged. Returns `model` with every matched, eligible
// layer rewired; a layer with a non-constant, non-2-D weight, a name in
// `skip_names`, a reduction dimension not divisible by `vector_dim`, or
// no usable calibration activation, is left completely untouched.
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
// `seed` derives a fresh, deterministic k-means initialization per
// matched layer -- ACCEPTED, PERMANENT DIVERGENCE from gptvq.py's own RNG
// derivation (a single numpy.random.Generator sequenced across matches in
// graph node order, driving both numpy's own Generator.choice sampling
// and Lloyd's-algorithm iteration): the same `seed` does NOT produce the
// same codebook, or therefore the same quantized weights, as gptvq.py's
// quantize_weight_only_gptvq -- same non-goal ApplyQuarot/ApplyQuarotGptq
// already document for their own random rotations. `vector_dim` is the
// number of consecutive input-channel columns jointly quantized per
// codebook lookup; `num_centroids` is the fitted codebook's own size;
// `num_iterations` is Lloyd's-algorithm's own iteration count;
// `percdamp` is the Hessian damping factor, matching ApplyGptq's own
// parameter and default; `skip_names` lists weight initializer names to
// leave unquantized even if otherwise eligible.
//
// Accepted numerical scope (like ApplyGptq's own): the dense inverse and
// Cholesky factorization at this algorithm's Hessian-correction half use
// this TU's own scalar double-precision kernels rather than LAPACK, so
// results can differ from the reference in the last ulp or two on some
// inputs -- see gptq_entry.h's own identical note. Unlike ApplyGptq, this
// is not expected to be otherwise bit-exact even given the same Hessian,
// since the codebook itself is fit via an independently-seeded k-means
// (see the `seed` note above); the correction math given a *shared*
// codebook is exact-agreement-tested instead (see
// tests/test_gptvq_cpp.py).
onnx::ModelProto ApplyGptvq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    uint64_t seed = 0, int64_t vector_dim = 2, int64_t num_centroids = 256,
    int64_t num_iterations = 10, double percdamp = 0.01,
    const std::unordered_set<std::string>& skip_names = {});
