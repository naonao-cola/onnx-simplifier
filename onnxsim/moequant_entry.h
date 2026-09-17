#pragma once

// Calibration-driven MoEQuant entry point exposed to Python -- C++ port of
// onnxsim.moequant's own apply_moequant (see that module's own docstring
// for the full technique: Hu, Chen et al., 2025's "Expert-Balanced
// Self-Sampling" (EBSS) and "Affinity-Guided Quantization" (AGQ)
// calibration methodology for `com.microsoft::MoE` nodes' per-expert
// weights -- explicitly NOT a new quantization algorithm: the sequential,
// Hessian-compensated column-update procedure itself is exactly GPTQ's own
// (gptq_entry.cpp's own GptqQuantizeColumns, reused here as a direct,
// near-verbatim transcription), reused as-is via a precomputed, per-expert,
// AGQ/EBSS-weighted Hessian).
//
// Single-model, calibration-driven, node-structure-preserving (unlike
// slim_llm_entry.h's own ApplySlimLlm, this never inserts new nodes -- the
// existing `MoE` node, its dtype, and its shapes are all left exactly as
// they are; only the `fc1_experts_weights`/`fc2_experts_weights`
// initializers' own float32 VALUES are overwritten in place with their
// simulated-quantized (dequantized code*scale) reconstruction). Matching
// mirrors structured_pruning_entry.cpp's own MatchMoeProducer/MoEChain
// (the exact same `com.microsoft::MoE` node shape
// onnxsim.pruning._match_moe_producer/_MoEChain define, which this
// header's own Python counterpart, onnxsim.pruning._find_moe_chains,
// reuses directly) -- see this header's own scope note below for the one
// narrowing this port makes relative to that shared matcher.
//
// Candidates (matched MoE nodes) are processed independently, so
// `executor` is invoked exactly once, up front, the same as every other
// calibration-driven pass in this codebase.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// MoEQuant-calibrated INT4 (simulated) quantization of every matched
// `com.microsoft::MoE` node's per-expert `fc1_experts_weights`/
// `fc2_experts_weights`. See onnxsim/moequant.py's own module docstring
// for the full AGQ/EBSS calibration methodology and exact scope.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`.
//
// Matching NARROWS structured_pruning_entry.cpp's own MatchMoeProducer in
// one respect: `fc1_experts_weights`/`fc2_experts_weights` must both be
// FLOAT32 (not FLOAT16/BFLOAT16, which that shared matcher itself accepts)
// for a node to be matched at all here -- apply_moequant's own Python
// reference matches the wider float set via the SAME shared
// `_find_moe_chains` its own pruning.py counterpart uses, then separately
// skips a non-FLOAT32 expert pair inside its own per-chain loop; narrowing
// the match itself rather than reproducing that separate two-step skip
// leaves an identical set of nodes actually rewritten (a FLOAT16/BFLOAT16
// MoE node is left untouched either way) with a simpler one-step matcher.
// A tied/shared `fc1`/`fc2` initializer another already-processed MoE node
// in the SAME model already touched is left untouched for the second node
// (mirrors apply_moequant's own `touched` set).
//
// `quant_block_size`/`percdamp`/`proc_block_size`/`ebss` mirror
// apply_moequant's own parameters of the same names exactly (`seed` seeds
// this port's own independent RNG for EBSS's weighted-without-replacement
// subsampling -- see this header's own accepted numerical scope note
// below for why it does not reproduce numpy's own random stream).
//
// An expert that received no calibration tokens at all (top-k never
// selected it) is left at its original float value; a chain whose
// `input[0]`/`input[1]` (hidden-state/router-probs) activation was never
// observed with the expected shape is left completely untouched -- mirrors
// apply_moequant's own per-chain/per-expert skip conditions.
//
// FLOAT32-only throughout, mirroring ApplyGptq's/ApplyAdaround's own
// FLOAT32-only calibration scope.
//
// ACCEPTED, PERMANENT DIVERGENCE (two, both narrower in scope than
// ApplyGptq's own accepted numerical-scope note, which this port also
// inherits for its own reused Hessian/Cholesky column-update machinery):
//
// 1. EBSS's own weighted-sampling-without-replacement subsampling step
//    uses this TU's own splitmix64-derived RNG and an Efraimidis-Spirakis
//    exponential-key selection, NOT numpy's own `Generator.choice(...,
//    replace=False, p=weights)` algorithm/bit stream -- reproducing
//    numpy's own PCG64-based weighted-without-replacement sampling
//    bit-for-bit is not attempted (the same class of accepted divergence
//    ApplyQuarot's own header documents for its own random rotation).
//    Distributionally equivalent (both are valid weighted-without-
//    replacement samplers), not bit-for-bit reproducible.
// 2. The dense inverse/Cholesky factorization behind GPTQ's own per-column
//    correction (reused unchanged from gptq_entry.cpp's own machinery) is
//    computed with this TU's own scalar double-precision kernels rather
//    than LAPACK -- see gptq_entry.h's own identical note.
//
// See tests/test_moequant_cpp.py for the measured agreement (with
// `ebss=False`, which removes divergence 1, providing the closer of the
// two comparisons).
onnx::ModelProto ApplyMoequant(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t quant_block_size = 32, double percdamp = 0.01,
    int64_t proc_block_size = 128, bool ebss = true, uint64_t seed = 0);
