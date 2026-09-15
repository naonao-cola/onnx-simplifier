#pragma once

// Calibration-driven AdaRound entry point exposed to Python -- C++ port of
// onnxsim.adaround's own apply_adaround (see that module's docstring for
// the full technique: Nagel et al. 2020's rectified-sigmoid relaxation of
// each weight element's floor/ceil rounding decision, optimized by a
// hand-rolled Adam loop to minimize a layer's own reconstruction error
// against real calibration activations, rather than round-to-nearest's
// per-element-independent choice).
//
// Same two-model (float model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as gptq_entry.h's
// own ApplyGptq -- candidates are processed independently (no cross-layer
// dependency), so `executor` is invoked exactly once, up front, the same
// as ApplyGptq's own shape.
//
// Like tesseraq_entry.h's own ApplyTesseraq (the closest existing
// precedent -- the same rectified-sigmoid relaxation and hand-rolled Adam
// loop, minus TesseraQ's own joint scale optimization and progressive
// coarse-to-fine hardening schedule: this is AdaRound's own single,
// uniformly-annealed anneal followed by one hardening step at the end,
// not TesseraQ's multi-round one), this is an *iterative* numerical
// optimization, not a closed-form computation -- see this header's own
// accepted numerical scope note below.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes AdaRound-style adaptive rounding for every
// quantize_weight_only_int4-quantized MatMul/Gemm layer present (by node
// output name) in both `float_model` and `quantized_model`, using real
// activations captured once from `float_model` through `executor`. See
// onnxsim/adaround.py's own module docstring for the full technique.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as ApplyGptq's
// own. A layer whose activation was never observed with a feature axis,
// or whose feature dimension doesn't match the weight's own K, is left
// completely untouched -- mirrors apply_adaround's own per-layer skip
// conditions. `n_min`/`n_max` are fixed at -7/7 (INT4's own full
// symmetric range), matching apply_adaround's own hardcoded values --
// unlike ApplyTesseraq, this has no `num_bits` parameter to narrow them.
//
// `num_iterations`, `learning_rate`, `reg_param`, `warm_start`,
// `beta_start`/`beta_end` (the two ends of apply_adaround's own
// `beta_range` tuple, split into separate parameters here since this
// binding layer has no tuple type) mirror apply_adaround's own parameters
// of the same names exactly.
//
// FLOAT32-only throughout, mirroring ApplyGptq's own FLOAT32-only
// calibration scope.
//
// ACCEPTED NUMERICAL SCOPE (same class as ApplyTesseraq's own): this is a
// `num_iterations`-step Adam optimization, not a single closed-form
// computation -- floating-point summation-order differences between this
// TU's own scalar dense-matmul kernels and numpy's own (possibly
// BLAS-backed) `@` can compound across iterations. Measured empirically
// rather than assumed correct -- see tests/test_adaround_cpp.py for
// exactly how closely (or not) this tracks the pure-Python reference.
onnx::ModelProto ApplyAdaround(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 300, double learning_rate = 0.1,
    double reg_param = 0.01, double warm_start = 0.2, double beta_start = 20.0,
    double beta_end = 2.0);
