#pragma once

// OmniQuant (Shao et al., 2023, "OmniQuant: Omnidirectionally Calibrated
// Quantization for Large Language Models", https://arxiv.org/abs/2308.13137)
// entry point exposed to Python -- C++ port of onnxsim.omniquant's own
// apply_omniquant (see onnxsim/omniquant.py's module docstring for the full
// technique: OmniQuant's own Learnable Weight Clipping (LWC) and Learnable
// Equivalent Transformation (LET), here GRID-SEARCHED -- not gradient
// descended through a straight-through relaxation, the way the paper's own
// reference implementation and onnxsim.adaround both do for a different
// parameter -- exactly as that module's own docstring documents and
// justifies).
//
// Two-model (float_model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as
// adaround_entry.h's own ApplyAdaround -- candidates are processed
// independently (no cross-layer dependency), so `executor` is invoked
// exactly once, up front, the same as ApplyAdaround's own shape. Unlike
// ApplyAdaround (an iterative Adam optimization), this is a bounded,
// deterministic coordinate-descent GRID search -- closer in numerical
// character to easyquant_entry.h's own ApplyEasyquant and daq_entry.h's own
// ApplyDaq (both already-ported grid-search-style precedents) -- so no
// "iterative Adam" accepted-numerical-scope caveat applies here; see the
// note below for what accepted scope *does* apply.
//
// Candidate matching mirrors adaround.py's own _find_int4_matmul_candidates
// exactly, transcribed the same way adaround_entry.cpp's own
// FindInt4MatmulCandidates already transcribes it (duplicated here rather
// than shared via a common header, matching this codebase's established
// per-*_entry.cpp self-containment convention -- see gptaq_entry.cpp's own
// top-of-file comment for why).
//
// Graph-rewrite shape: for every layer OmniQuant's own search found *some*
// improvement for, this rewrites that layer's INT4 Wq/Ws initializers in
// place (same as ApplyAdaround/ApplyGptq's own weight-only rewrite) and,
// only for a layer whose best candidate used LET (a per-channel shift AND
// scale, never just one alone -- mirrors apply_omniquant's own
// `channel_scale is None`/`shift is None` pairing exactly), additionally
// inserts a `Sub`/`Mul` pair before that layer's own MatMul/Gemm node
// (transforming its activation input) and a new `Add` right after it
// (folding in the constant bias correction, renaming the node's own old
// output to an internal name first) -- mirrors llm_int8_entry.cpp's own
// node-insertion-before idiom and bias_correction_entry.cpp's own
// rename-and-insert-after idiom, composed.
//
// `num_clip_steps`, `num_alpha_steps`, `min_clip_ratio` mirror
// apply_omniquant's own parameters of the same names exactly (see
// omniquant.py's own docstring for the grid layout: `clip_ratio=1.0` is
// always tried first for LWC, `alpha=0.0` (no LET) is always covered by
// LWC's own stage, so a layer OmniQuant found no improvement for keeps its
// exact plain-RTN codes/scale and gets no inserted nodes at all).
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `float_model`'s own graph inputs throws `std::invalid_argument`. A layer
// whose activation was never observed with a feature axis, or whose feature
// dimension doesn't match the weight's own K, is left completely untouched
// -- mirrors apply_omniquant's own per-layer skip conditions.
//
// FLOAT32-only throughout, mirroring every other calibration-driven pass in
// this codebase's own FLOAT32-only scope.
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond ordinary floating-point
// summation-order/accumulation differences from numpy -- this is a bounded,
// deterministic grid search with no RNG or hand-rolled dense linear algebra
// (matmuls here are plain reduction loops, not a Cholesky/SVD/
// eigendecomposition kernel), so this port is expected to track the Python
// reference closely, including exact-tie behavior at every grid point
// boundary (ties broken by strict-`<`-only-replaces-incumbent on both
// sides, matching apply_omniquant's own `if err < best_err` exactly). See
// tests/test_omniquant_cpp.py for the measured agreement.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyOmniquant(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_clip_steps = 20, int64_t num_alpha_steps = 20,
    double min_clip_ratio = 0.5);
