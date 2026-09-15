#pragma once

// Calibration-driven LLM.int8() (Dettmers et al., 2022) entry point
// exposed to Python -- C++ port of onnxsim.llm_int8's own apply_llm_int8
// (see onnxsim/llm_int8.py's module docstring for the full technique:
// decomposing a MatMul/Gemm into a float32 outlier part plus a
// vector-wise INT8 part computed via MatMulInteger).
//
// Like outlier_suppression_plus_entry.h's own
// ApplyOutlierSuppressionPlus (the closest existing precedent -- the
// same protobuf-level calibration-driven shape, the same
// match-then-probe-then-rewrite flow, the same new-node insertion), this
// operates directly on onnx::GraphProto rather than through
// onnxoptimizer's Node/Value IR: threading a live ModelExecutor plus
// calibration batches through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase
// (every PredicateBasedPass in onnxsim/passes/ is data-free by
// construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see outlier_suppression_plus_entry.h's own
// identical forward declaration for why (the full ModelExecutor interface
// lives in onnxsim.h, which includes this header back).
struct ModelExecutor;

// Decomposes every matched MatMul/vanilla-Gemm node with a constant 2-D
// FLOAT32 weight and a plain 2-D FLOAT32 activation input into an outlier
// float32 part plus a vector-wise INT8 part, using real calibration
// activations run through `executor` to find each layer's outlier
// channels (activation magnitude above `outlier_threshold` anywhere in
// the calibration data). The INT8 part uses one absmax scale per
// activation row (computed at runtime) and one absmax scale per weight
// output channel (computed offline), with the activation stored uint8 at
// zero-point 128 -- see onnxsim/llm_int8.py's own module docstring for
// why uint8 rather than int8. Returns a model whose every rewritten
// layer keeps its original output tensor name.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase (outlier_suppression_plus_entry.h's own
// ApplyOutlierSuppressionPlus, smoothquant_entry.h's ApplySmoothQuant) --
// a `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument`. NOT subgraph-aware, matching every one
// of those passes' own scope decision: calibration_data batches are keyed
// to the top-level graph's own inputs only.
//
// A layer is left untouched when its activation was never observed as a
// plain 2-D FLOAT32 tensor, when the feature dim does not match the
// weight's reduction dimension `K`, when no outlier channel (or every
// channel) is found, or when the non-outlier reduction depth is unsafe
// for MatMulInteger's int32 accumulator -- mirrors
// apply_llm_int8's own per-layer skip conditions exactly. Models below
// opset 18 are returned unchanged (ReduceMax's axes-as-input form needs
// opset >= 18), also mirroring the reference.
//
// `outlier_threshold` marks an input channel an outlier when its
// activation magnitude exceeds it anywhere in the calibration data (the
// paper's own default, 6.0); `epsilon` floors zero row/weight-column
// max-abs values before dividing -- mirrors apply_llm_int8's own
// parameters of the same names exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_llm_int8's own `FLOAT`-only weight requirement and
// ApplySmoothQuant's own FLOAT32-only calibration scope: a FLOAT16/
// BFLOAT16 activation is never observed (skipped like an unobserved one),
// never converted.
onnx::ModelProto ApplyLlmInt8(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double outlier_threshold = 6.0, double epsilon = 1e-8);
