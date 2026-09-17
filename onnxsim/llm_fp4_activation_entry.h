#pragma once

// LLM-FP4 (Liu et al., 2023, EMNLP) activation-quantization entry points
// exposed to Python -- C++ ports of onnxsim.llm_fp4's own
// apply_llm_fp4_activation_quantization (data-free, per-token) and
// apply_llm_fp4_activation_quantization_per_tensor (calibration-driven,
// per-tensor). See onnxsim/llm_fp4.py's own module docstring and both
// functions' own docstrings for the full technique, including the
// "Honesty note" explaining that neither reproduces the paper's own
// per-channel-migration design (that migration is a separate, existing
// pass -- onnxsim.apply_smoothquant/apply_outlier_suppression -- the
// caller composes beforehand).
//
// Both functions act only on layers already weight-quantized by
// quantize_weight_only_llm_fp4_cpp (QuantizeWeightOnlyLlmFp4 in
// onnxsim.h, passes/llm_fp4.h): a layer is found by walking its weight
// input backward through that pass's OWN exact dequantization pattern --
// Reshape(Mul(Reshape(Gather(Codebook, Cast(Codes, INT64))),
// Reshape(Scale)), orig_shape) -- and recovering that layer's own
// Codebook initializer NAME (not re-deriving one). A layer whose weight
// isn't fed by exactly that pattern is left completely untouched. Both
// share this matching/recovery machinery and the "nearest codebook
// value" node splice (Unsqueeze/Sub/Abs/ArgMin/Gather), hence one shared
// file -- see llm_fp4.py's own docstring for why they are kept as two
// separate functions rather than one parameterized pass (different scale
// source, different runtime cost, different paper fidelity).
//
// Like billm_entry.h's own ApplyBillm (single-model, protobuf-level),
// this operates directly on onnx::GraphProto rather than through
// onnxoptimizer's Node/Value IR -- for the same reason every other
// calibration-driven pass in this codebase does (no established path to
// thread a live ModelExecutor through OptimizeFixed's PredicateBasedPass
// model). ApplyLlmFp4ActivationQuantization itself needs no calibration
// data at all (a data-free, per-token runtime quantizer -- the codebook
// it reuses is already baked into the graph by the weight-quantization
// pass), so it takes no ModelExecutor/calibration_data parameters; the
// forward declaration below exists only so this header and its
// _per_tensor sibling share one translation unit's includes uniformly
// with every other *_entry.h in this codebase.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see billm_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back). Used only by
// ApplyLlmFp4ActivationQuantizationPerTensor below.
struct ModelExecutor;

// Data-free, per-token FP4 quantize/dequantize round-trip on every
// already-FP4-weight-quantized layer's own activation input -- C++ port
// of llm_fp4.py's own apply_llm_fp4_activation_quantization. For each
// matched layer:
//
//   scale = max(ReduceMax(Abs(X), axis=-1, keepdims=1), epsilon)
//           / max(abs(Codebook))                  -- one scale per token
//   x_normalized = X / scale
//   nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1)
//                                         - Codebook), axis=-1))
//   x_dequant = nearest * scale
//
// spliced in before the matched MatMul/Gemm node, which then reads
// x_dequant instead of X (its weight/bias inputs and own output name are
// left exactly as the weight-quantization pass left them). NOT
// subgraph-aware, matching every pass in this family. Models below
// opset 18 (ReduceMax's axes-as-input form) are returned unchanged.
// `epsilon` floors a token's own max-abs value before it becomes a
// divisor, avoiding a divide-by-zero on an all-zero token.
onnx::ModelProto ApplyLlmFp4ActivationQuantization(
    const onnx::ModelProto& model, double epsilon = 1e-12);

// Calibration-driven, per-tensor FP4 quantize/dequantize round-trip on
// every already-FP4-weight-quantized layer's own activation input -- C++
// port of llm_fp4.py's own
// apply_llm_fp4_activation_quantization_per_tensor: this is the
// quantizer half of the paper's own activation-quantization design (the
// migration half -- pushing per-channel outlier scale into the
// preceding weight/LayerNormalization -- is the caller's job, via
// onnxsim.apply_smoothquant/apply_outlier_suppression run first; this
// function does not check whether that has been done). For each matched
// layer, `X` is captured over `calibration_data` (probe-injection, the
// same shape as every other calibration-driven pass in this codebase),
// concatenated and raveled across every batch (NOT restricted to rank
// >= 2, unlike this file's own per-token sibling -- llm_fp4.py's own
// per-tensor fitting flattens every captured value unconditionally), and
// ONE real-valued scale is grid-searched over `clip_ratios` (17 points
// evenly spaced over [0.5, 1.0] when omitted, matching
// quantize_weight_only_llm_fp4's own weight-side default) minimizing
// direct codebook round-trip MSE -- the same objective
// quantize_weight_only_llm_fp4_cpp's own per-block search already uses,
// here over a single whole-tensor group. That scale is baked into the
// graph as a constant float32 initializer, then:
//
//   x_normalized = X / scale
//   nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1)
//                                         - Codebook), axis=-1))
//   x_dequant = nearest * scale
//
// is spliced in before the matched node (no Abs/ReduceMax/Max
// range-reduction node at all, since the scale is compile-time-constant
// -- 7 inserted nodes versus the per-token sibling's 11). Every scale is
// fit before the graph is touched at all, so a run in which no layer's
// activation turns out usable leaves the model byte-identical to the
// input. A layer whose captured activation is never observed, or whose
// every captured value is zero or non-finite, is left completely
// untouched. Models below opset 13 (Unsqueeze's axes-as-input form) are
// returned unchanged.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `model`'s own graph inputs) shape as every
// other calibration-driven pass in this codebase -- a `calibration_data`
// batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument`.
//
// ACCEPTED, PERMANENT DIVERGENCE: llm_fp4.py's own per-tensor scale fit
// subsamples a calibration sample larger than 2**18 elements via
// `np.random.default_rng(0).choice(..., replace=False)` before
// estimating the MSE objective (the `max_abs` used as the scale's own
// numerator stays exact over the FULL sample either side). This port
// instead takes the first 2**18 raveled elements deterministically when
// over that cap -- a different (not bit-reproducible against the
// Python RNG) but equally valid deterministic subsample of the same
// population; the two sides' fitted scale can differ in this regime.
// Below the cap (every calibration configuration this repo's own test
// suite exercises), no subsampling happens on either side and the two
// implementations agree up to ordinary floating-point summation-order
// differences, the same closed-form-grid-search-with-no-RNG guarantee
// passes/llm_fp4.h's own "ACCEPTED NUMERICAL SCOPE" note already
// documents for the weight side.
onnx::ModelProto ApplyLlmFp4ActivationQuantizationPerTensor(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    std::optional<std::vector<double>> clip_ratios = std::nullopt);
