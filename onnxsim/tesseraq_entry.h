#pragma once

// Calibration-driven TesseraQ entry point exposed to Python -- C++ port of
// onnxsim.tesseraq's own apply_tesseraq (see that module's docstring for
// the full technique: TesseraQ's own "Progressive Adaptive Rounding"
// (PAR) -- onnxsim.adaround's rectified-sigmoid rounding relaxation,
// optimized by a hand-rolled Adam loop jointly with each weight block's
// own dequantization scale (in log-space), with a coarse-to-fine
// element-by-element hardening schedule across a handful of rounds
// instead of adaround's single monolithic anneal).
//
// Same two-model (float model, its quantize_weight_only_int4-quantized
// counterpart), calibration-driven, protobuf-level shape as gptq_entry.h's
// own ApplyGptq -- candidates are processed independently (unlike
// qronos_entry.h's own ApplyQronos, there is no cross-layer dependency:
// every layer's own reconstruction target comes straight from the
// untouched float model), so `executor` is invoked exactly once, up
// front, the same as ApplyGptq's own shape.
//
// Unlike every closed-form port in this codebase (GPTQ, AWQ, Qronos,
// GPTVQ's own correction half), this is an *iterative* numerical
// optimization (a hand-rolled Adam loop run for `num_iterations` steps
// per layer) -- see this header's own accepted numerical scope note
// below for what that means for cross-language agreement.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see gptq_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Optimizes TesseraQ-style Progressive Adaptive Rounding, jointly with
// each weight block's own dequantization scale, for every
// quantize_weight_only_int4-quantized MatMul/Gemm layer present (by node
// output name) in both `float_model` and `quantized_model`, using real
// activations captured once from `float_model` through `executor`. See
// onnxsim/tesseraq.py's own module docstring for the full technique.
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `float_model`'s own graph inputs) shape as ApplyGptq's
// own. A layer whose activation was never observed with a feature axis,
// whose feature dimension doesn't match the weight's own K, or whose K
// isn't divisible by the scale's own block size, is left completely
// untouched -- mirrors apply_tesseraq's own per-layer skip conditions.
//
// `num_bits` (2..4, throws std::invalid_argument otherwise) sets PAR's
// own symmetric rounding range `[-n_max, n_max]`, `n_max = 2^(num_bits-1)
// - 1`; codes are always packed into the INT4 nibble storage
// `quantized_model` already allocated regardless of `num_bits` -- a
// narrower value only constrains PAR's own optimization/rounding range
// (relying on the jointly-optimized scale to make it usable). Every
// other parameter (`num_iterations`, `par_rounds`, `learning_rate`,
// `scale_learning_rate`, `reg_param`, `warm_start`, `beta_start`/
// `beta_end` -- the two ends of apply_tesseraq's own `beta_range` tuple,
// split into separate parameters here since this binding layer has no
// tuple type) mirrors apply_tesseraq's own parameter of the same name
// exactly.
//
// FLOAT32-only throughout, mirroring ApplyGptq's own FLOAT32-only
// calibration scope.
//
// ACCEPTED NUMERICAL SCOPE (unlike every closed-form port in this
// codebase): this is a `num_iterations`-step Adam optimization, not a
// single closed-form computation -- floating-point summation order
// differences between this TU's own scalar dense-matmul kernels and
// numpy's own (possibly BLAS-backed) `@` can compound across iterations
// (Adam's own division/sqrt nonlinearity does not damp small input
// differences the way, say, a single Cholesky solve's rounding error
// stays bounded). Measured empirically rather than assumed correct --
// see tests/test_tesseraq_cpp.py for exactly how closely (or not) this
// tracks the pure-Python reference, and whether that agreement rises to
// the "verified, now aliased" bar every closed-form port in this
// codebase already met.
onnx::ModelProto ApplyTesseraq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t num_bits = 4, int64_t num_iterations = 400, int64_t par_rounds = 4,
    double learning_rate = 0.1, double scale_learning_rate = 0.01,
    double reg_param = 0.01, double warm_start = 0.2, double beta_start = 20.0,
    double beta_end = 2.0);
