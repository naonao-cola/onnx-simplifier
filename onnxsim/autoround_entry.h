#pragma once

// Calibration-driven AutoRound entry point exposed to Python -- C++ port of
// onnxsim.autoround's own apply_autoround (see that module's own module
// docstring for the full technique: Cheng et al., 2023's signed-gradient-
// descent weight rounding, closing the one gap between
// onnxsim.adaround (AIMET's AdaRound, fixed-scale rounding-only) and
// AutoRound proper that this codebase needs -- a second, jointly optimized
// per-(output-channel, block) clip-ratio parameter that lets the effective
// scale move during the same optimization).
//
// Same two-model (float model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as adaround_entry.h's
// own ApplyAdaround -- candidates are processed independently (no
// cross-layer dependency), so `executor` is invoked exactly once, up front,
// the same as ApplyAdaround's own shape. adaround_entry.h's own
// ApplyAdaround is this file's primary template: candidate matching,
// tensor <-> flat-buffer conversion, the dense scalar matmul kernels, and
// the rounding relaxation's own Adam loop are all transcribed from there
// (this TU cannot include adaround_entry.cpp's own anonymous-namespace
// helpers, so they are duplicated here rather than shared -- the same
// choice every other calibration-driven entry file in this codebase makes;
// see e.g. gptaq_entry.cpp's own near-verbatim duplication of gptq_entry.cpp's
// helpers).
//
// What AutoRound adds on top of AdaRound's own single parameter group (the
// per-element rounding relaxation `v`): a second, per-(output-channel,
// block) parameter `c`, reparametrizing "how much to shrink/grow this
// block's own pre-existing scale" as a bounded sigmoid
// (`clip_ratio(c) = sigmoid(c) * (cmax - cmin) + cmin`), jointly optimized
// with `v` by a second, independent Adam loop sharing only their bias
// corrections (both take their first step on the same iteration). Because
// the scale now moves every step, each element's quantization bin
// `floor(w / scale_eff)` cannot be precomputed once outside the loop the
// way ApplyAdaround's own `floor_base` is -- it is recomputed every
// iteration from the *current* effective scale.
//
// AutoRound's own safety net (`_keep_better_of` in autoround.py): jointly
// optimizing two coupled parameter sets is a harder, non-convex problem
// than AdaRound's own decoupled search over rounding alone, and can,
// at a matched iteration budget, land on a worse local optimum. This port
// therefore always ALSO runs AdaRound's own fixed-scale optimization
// (`clip_ratio == 1` throughout) on the same weight/scale/activation, and
// keeps whichever of the two candidates actually has the lower measured
// reconstruction error on the calibration activations -- so this function
// is guaranteed never to do worse than ApplyAdaround would on the same
// layer and calibration data. A layer whose joint optimization does not
// actually beat the fixed-scale one keeps its ORIGINAL scale untouched
// (byte-identical to `quantized_model`'s own).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see adaround_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes AutoRound-style jointly-optimized rounding AND per-block
// clip-ratio for every quantize_weight_only_int4-quantized MatMul/Gemm
// layer present (by node output name) in both `float_model` and
// `quantized_model`, using real activations captured once from
// `float_model` through `executor`. See onnxsim/autoround.py's own module
// docstring for the full technique.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as ApplyAdaround's
// own. A layer whose activation was never observed with a feature axis, or
// whose feature dimension doesn't match the weight's own K, is left
// completely untouched -- mirrors apply_autoround's own per-layer skip
// conditions. `n_min`/`n_max` are fixed at -7/7 (INT4's own full symmetric
// range), matching apply_autoround's own hardcoded values.
//
// `num_iterations`, `learning_rate`, `reg_param`, `warm_start`,
// `beta_start`/`beta_end` (the two ends of apply_autoround's own
// `beta_range` tuple, split into separate parameters here since this
// binding layer has no tuple type) mirror ApplyAdaround's own parameters of
// the same names exactly, plus `clip_learning_rate` (the Adam learning rate
// for the clip-ratio parameter -- kept separate from, and by default
// smaller than, `learning_rate` since one clip-ratio value is shared by an
// entire block's worth of rounding decisions) and `clip_ratio_min`/
// `clip_ratio_max` (the two ends of apply_autoround's own `clip_ratio_range`
// tuple, split the same way; MUST satisfy
// `clip_ratio_min + clip_ratio_max == 2.0` for the optimization to start at
// the unmodified scale -- not enforced here, matching apply_autoround's own
// undocumented-but-assumed precondition).
//
// FLOAT32-only throughout, mirroring ApplyAdaround's own FLOAT32-only
// calibration scope.
//
// Unlike ApplyAdaround (which never rewrites a layer's scale), a layer
// whose joint optimization DOES beat the AdaRound-only safety net also
// gets its `ws_name` scale initializer rewritten to the optimized
// per-block scale, matching apply_autoround's own two-initializer rewrite
// (`optimized_codes`/`optimized_scale`).
//
// ACCEPTED NUMERICAL SCOPE (same class as ApplyAdaround's own, plus one
// extra source unique to AutoRound): this is a `num_iterations`-step Adam
// optimization over two jointly interacting parameter groups, not a single
// closed-form computation -- floating-point summation-order differences
// between this TU's own scalar dense-matmul kernels and numpy's own
// (possibly BLAS-backed) `@` can compound across iterations, AND because
// the scale moves every step, an element whose ratio sits within an ulp of
// an integer boundary can take a different quantization bin between the
// two implementations -- a *discontinuous* change to that element's
// gradient contribution that then steers the rest of that layer's own run.
// The `_keep_better_of` safety net is exactly what keeps this port's own
// worst case bounded: even a layer whose joint search diverges completely
// from the Python reference can never regress reconstruction error below
// AdaRound's own (verified-agreeing) fixed-scale optimum for that same
// layer. Measured empirically rather than assumed correct -- see
// tests/test_autoround_cpp.py for exactly how closely (or not) this tracks
// the pure-Python reference.
onnx::ModelProto ApplyAutoround(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_iterations = 300, double learning_rate = 0.1,
    double clip_learning_rate = 0.03, double reg_param = 0.01,
    double warm_start = 0.2, double beta_start = 20.0, double beta_end = 2.0,
    double clip_ratio_min = 0.5, double clip_ratio_max = 1.5);
