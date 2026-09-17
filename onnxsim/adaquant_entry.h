#pragma once

// Calibration-driven AdaQuant entry point exposed to Python -- C++ port of
// onnxsim.adaquant's own apply_adaquant (see that module's docstring for the
// full technique: Hubara, Nahshan, Hanani, Banner, Soudry 2020/2021's
// per-layer, gradient-descent reconstruction-error minimization jointly over
// weight rounding *and* the activation's own quantization clip range -- only
// the paper's own layer-wise reconstruction contribution is ported; its
// separate integer-programming bit-width allocator is out of scope, exactly
// as documented in adaquant.py's own module docstring).
//
// Unlike adaround_entry.h's own ApplyAdaround (which targets
// quantize_weight_only_int4's weight-only INT4 scheme), this targets
// onnxsim.quantize_static's W8A8 QDQ scheme:
//
//   Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: asymmetric uint8
//   Xdq = DequantizeLinear(Xq, Xs, Xzp)
//   Wdq = DequantizeLinear(Wq, Ws, [Wzp], axis=<W's output-channel axis>)
//         -- Wzp, if present, spelled out explicitly, all zeros, same shape
//         as Ws
//   Y   = MatMul(Xdq, Wdq)                  -- or Gemm
//
// so both the found-candidate shape (a plain per-tensor Ws/Xs/Xzp scale, no
// `block_size`, weight coded as INT8 in [-127, 127] rather than blocked
// INT4) and the optimized parameter set (weight-rounding relaxation *plus*
// the activation's own (scale, zero_point), rather than rounding alone)
// differ from ApplyAdaround's own. `calibration_data` is still probed only
// from `float_model` (not `quantized_model` -- unlike e.g. ApplyGptaq),
// matching adaquant.py's own `_optimize_adaquant`'s single-model activation
// capture.
//
// Candidates are processed independently (no cross-layer dependency), so
// `executor` is invoked exactly once, up front, the same as ApplyGptq's/
// ApplyAdaround's own shape.
//
// This is an *iterative* joint Adam optimization over three parameter
// groups at once (the per-element weight-rounding relaxation, the
// activation's log-scale, and its zero-point), not a closed-form
// computation -- see this header's own accepted numerical scope note below,
// the same class as ApplyAdaround's own.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in onnxsim.h,
// which includes this header back).
struct ModelExecutor;

// Optimizes AdaQuant-style joint weight-rounding + activation-clip-range
// calibration for every quantize_static-quantized MatMul/"vanilla" Gemm
// layer present (by node output name) in both `float_model` and
// `quantized_model`, using real activations captured once from
// `float_model` through `executor`. See onnxsim/adaquant.py's own module
// docstring for the full technique and its relationship to
// onnxsim.apply_adaround.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `float_model`'s own graph inputs throws `std::invalid_argument`. A layer
// whose activation was never observed with a feature axis, or whose feature
// dimension doesn't match the weight's own K, is left completely untouched
// -- mirrors apply_adaquant's own per-layer skip conditions.
//
// `num_iterations`, `weight_learning_rate`, `activation_learning_rate`,
// `reg_param`, `warm_start`, `beta_start`/`beta_end` (the two ends of
// apply_adaquant's own `beta_range` tuple, split into separate parameters
// here since this binding layer has no tuple type) mirror apply_adaquant's
// own parameters of the same names exactly. Weight codes are always
// symmetric INT8 in [-127, 127] (`_WEIGHT_N_MIN`/`_WEIGHT_N_MAX`); the
// activation's zero-point is always projected back onto UINT8's own
// [0, 255] range (`_ACT_N_MAX`) at the very end -- both hardcoded, matching
// apply_adaquant's own hardcoded module-level constants (this binding layer
// has no parameter to narrow either range).
//
// FLOAT32-only throughout, mirroring ApplyGptq's/ApplyAdaround's own
// FLOAT32-only calibration scope.
//
// ACCEPTED NUMERICAL SCOPE (same class as ApplyAdaround's own): this is a
// `num_iterations`-step Adam optimization over three parameter groups
// jointly, not a single closed-form computation -- floating-point
// summation-order differences between this TU's own scalar dense-matmul
// kernels and numpy's own (possibly BLAS-backed) `@` can compound across
// iterations. Measured empirically rather than assumed correct -- see
// tests/test_adaquant_cpp.py for exactly how closely (or not) this tracks
// the pure-Python reference.
onnx::ModelProto ApplyAdaquant(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 300, double weight_learning_rate = 0.1,
    double activation_learning_rate = 0.01, double reg_param = 0.01,
    double warm_start = 0.2, double beta_start = 20.0, double beta_end = 2.0);
