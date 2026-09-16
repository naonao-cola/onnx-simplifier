#pragma once

// Data-free, two-model DAQ (Delta-Aware Quantization) entry point exposed
// to Python -- C++ port of onnxsim.daq's own apply_daq (see that module's
// docstring for the full technique: rather than choosing a layer's FP8
// scale to minimize ||W_post - W_hat||, DAQ chooses it to best preserve
// the fine-tuning update dW = W_post - W_base against its own
// reconstruction dW_hat = fp8_round_trip(W_post / s) * s - W_base, under
// one of two named delta-fidelity metrics).
//
// Unlike every calibration-driven two-model entry point in this codebase
// (gptq_entry.h's own ApplyGptq, qronos_entry.h's own ApplyQronos), this
// one needs no ModelExecutor and no calibration_data at all -- every
// decision comes from the two weight tensors alone, so this is the
// data-free sibling of that shape: still two full onnx::ModelProto
// arguments matched by node output name (the same correspondence
// assumption ApplyGptq/ApplyQronos already make about their own two
// model arguments), but no activation probing whatsoever.
//
// `base_model`'s and `post_trained_model`'s MatMul/vanilla-Gemm nodes are
// matched by shared node output name; a match needs a same-op-type
// counterpart on both sides, each with a constant 2-D FLOAT32 weight of
// identical shape. For each match, a per-layer scalar FP8 E4M3 scale is
// found by DAQ's own coarse-to-fine search (9 log-spaced multipliers over
// [0.5, 2.0] around the naive absmax scale, then 9 linearly-spaced
// multipliers over [0.9, 1.1] times the coarse winner) that maximizes
// `metric` ("cosine": cosine similarity between dW and dW_hat: the
// default; "sign_preservation": the fraction of elements whose update
// kept its sign) between dW and dW_hat, and
// `fp8_round_trip(W_post / s_best) * s_best` replaces the layer's weight
// as a plain new float32 initializer -- no graph node, no Cast, no
// minimum opset, exactly deepseek_fp8_entry.h's own weight-side shape.
// FP8 E4M3FN round-tripping reuses passes/quantize_fp8.h's own verified
// FloatToFloat8Bits encoder (round-to-nearest, ties-to-even, saturating)
// plus this TU's own straightforward E4M3FN bit-unpacking decoder (no
// rounding ambiguity on the decode side, unlike the encode side).
//
// A layer with no base_model counterpart, a counterpart of a different
// op type, a non-constant/non-2-D/non-FLOAT32 weight on either side, a
// shape mismatch between the two weights, or dW within 1e-12 of the zero
// vector (no fine-tuning signal to preserve) is left completely
// untouched -- mirrors apply_daq's own per-layer skip conditions exactly.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM onnxsim.daq: no accumulation or
// iterative-refinement step beyond DAQ's own bounded, deterministic
// 9+9-candidate grid search, so this port is expected to track the
// Python port's own float64 numpy implementation closely, up to
// floating-point summation-order differences in the cosine-similarity/
// sign-preservation scoring. ApplyDaq and its _cpp Python wrapper and
// apply_daq remain independently-correct, non-interchangeable entry
// points, not aliases.

#include <onnx/onnx_pb.h>

#include <string>
#include <unordered_set>

// DAQ_METRICS in onnxsim.daq: "cosine" (default) or "sign_preservation".
// Throws std::invalid_argument for any other value.
onnx::ModelProto ApplyDaq(
    const onnx::ModelProto& base_model,
    const onnx::ModelProto& post_trained_model,
    const std::string& metric = "cosine",
    const std::unordered_set<std::string>& skip_names = {});
