#pragma once

// Calibration-driven FlexRound entry point exposed to Python -- C++ port of
// onnxsim.flexround's own apply_flexround (see that module's own module
// docstring for the full technique: Lee et al., 2023's "learnable-division
// rounding" -- the fourth onnxsim-native PTQ technique alongside
// onnxsim.adaround, onnxsim.gptq, and onnxsim.awq, each pulling a
// different lever on the same target scheme
// (quantize_weight_only_int4's block-wise symmetric INT4)).
//
// Same two-model, calibration-driven, protobuf-level shape as
// adaround_entry.h's own ApplyAdaround -- candidates are processed
// independently (no cross-layer dependency), so `executor` is invoked
// exactly once, up front. adaround_entry.h's own ApplyAdaround is this
// file's structural template (candidate matching, tensor <-> flat-buffer
// conversion, the dense scalar matmul kernels, and the overall two-model/
// Adam-loop shape); the actual relaxation math is entirely different, and
// is transcribed from flexround.py's own _optimize_divisor rather than
// adapted from AdaRound's rectified-sigmoid one.
//
// Unlike AdaRound's *additive* perturbation of each weight element's own
// rounding decision, FlexRound reparametrizes the *divisor itself*,
// multiplicatively: the effective per-element divisor is
// `S = scale_nk * S2 * s3`, where `scale_nk` is quantize_weight_only_int4's
// own pre-existing (block-broadcast) scale -- fixed, never optimized, same
// scope restriction ApplyAdaround/ApplyGptq both make -- `S2` is an
// element-wise learnable correction ([N, K], same shape as the weight),
// and `s3` is a per-output-channel learnable correction ([N, 1]). Both
// `S2`/`s3` are parametrized in log-space (`S2 = exp(v2)`, `s3 = exp(v3)`,
// both initialized at 0 so `S2 = s3 = 1` and the very first forward pass
// is exactly round-to-nearest) to keep them positive by construction and
// to algebraically simplify the gradient (`d(S)/d(v2) == S` elementwise,
// `d(S)/d(v3) == S` summed over K). `v2`/`v3` are clamped to
// `[-log_clip, log_clip]` after every Adam step -- a numerical safety
// bound, not part of the paper's own formulation.
//
// Like flexround.py's own implementation, this keeps the *continuous*
// relaxation `clip(w / S, n_min, n_max)` throughout optimization (no
// `round()` every iteration, unlike the paper's own straight-through
// forward pass) and only rounds once at the very end -- the same structure
// ApplyAdaround uses for its own relaxation. There is no
// regularization/annealing schedule here (unlike AdaRound's): the
// reconstruction loss alone shapes `S2`/`s3` from start to finish, so this
// port has no `reg_param`/`warm_start`/`beta_range` parameters at all.
//
// FLOAT32-only throughout, mirroring ApplyAdaround's/ApplyGptq's own
// FLOAT32-only calibration scope. Never rewrites a scale initializer (only
// `wq_name`'s own codes) -- unlike AutoRound, `s1` (the pre-existing block
// scale) is fixed throughout, matching apply_flexround's own module
// docstring ("keeps the block scale quantized_model already computed
// completely unchanged").
//
// ACCEPTED NUMERICAL SCOPE (same class as ApplyAdaround's own, but
// measurably MORE sensitive to it): this is a `num_iterations`-step Adam
// optimization, not a single closed-form computation -- floating-point
// summation-order differences between this TU's own scalar dense-matmul
// kernels and numpy's own (possibly BLAS-backed) `@` can compound across
// iterations. FlexRound's own gradient divides by the effective divisor
// SQUARED (`d(ratio)/ds = -w/s**2`, this port's own docstring's
// "Proposition 3.1" argument), which amplifies rather than damps small
// floating-point differences step over step, so this port measurably
// disagrees with the pure-Python reference on a larger fraction of codes
// than ApplyAdaround's own rectified-sigmoid relaxation does at a matched
// iteration budget -- confirmed to be inherent to the algorithm's own
// reciprocal parametrization (a pure-Python nested-loop transcription that
// sums in the same row-major order this port's own scalar kernels use,
// compared against the vectorized numpy reference, shows a comparable
// mismatch fraction using ONLY numpy on both sides), not a bug in this
// port. UNLIKE ApplyAdaround (whose own mismatches are always the
// immediate grid neighbor, since its annealed regularizer pulls every
// element toward a hard decision by the very end), a FlexRound mismatch
// can occasionally be MORE than one grid point away: an element has no
// such regularizer here, so once its ratio saturates against
// `n_min`/`n_max` its own gradient contribution stops (only the *shared*
// per-output-channel `s3` term can still move it), and two
// implementations that happen to saturate a given element on slightly
// different iterations can settle on genuinely different sides of more
// than one grid line -- observed directly, not hypothesized. Because of
// this, tests/test_flexround_cpp.py does NOT assert tight per-code
// agreement as its primary signal (unlike tests/test_adaround_cpp.py's
// own `max_mismatch_frac`/neighbor-only tolerance) -- it instead checks
// structural validity, reconstruction-error agreement, and a generous,
// informational code-agreement smoke check. See that file's own docstring
// for the full reasoning and measurements.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes FlexRound-style learnable-division rounding for every
// quantize_weight_only_int4-quantized MatMul/Gemm layer present (by node
// output name) in both `float_model` and `quantized_model`, using real
// activations captured once from `float_model` through `executor`. See
// onnxsim/flexround.py's own module docstring for the full technique.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as ApplyAdaround's
// own. A layer whose activation was never observed with a feature axis, or
// whose feature dimension doesn't match the weight's own K, is left
// completely untouched -- mirrors apply_flexround's own per-layer skip
// conditions. `n_min`/`n_max` are fixed at -7/7 (INT4's own full symmetric
// range), matching apply_flexround's own hardcoded values.
//
// `num_iterations`, `learning_rate`, `log_clip` mirror apply_flexround's
// own parameters of the same names exactly (one shared Adam learning rate
// for both `S2`/`s3`, matching the reference's own single
// `learning_rate` -- unlike AutoRound, FlexRound's reference never splits
// this into two separate rates).
onnx::ModelProto ApplyFlexround(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 300, double learning_rate = 0.05,
    double log_clip = 4.0);
