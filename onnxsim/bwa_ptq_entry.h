#pragma once

// Calibration-driven Binary Weight-Activation PTQ (Song et al., 2025, ACL
// Findings, "Achieving binary weight and activation for LLMs using
// Post-Training Quantization") weight-side (W(1+1)) entry point exposed
// to Python -- C++ port of onnxsim.bwa_ptq's own apply_bwa_ptq (see
// onnxsim/bwa_ptq.py's own module docstring for the full technique and
// its documented scope: weight-only, the paper's own A(1x4) bit-serial
// activation decomposition is algebraically equivalent to ordinary
// calibrated INT4 activation quantization and is out of scope here, same
// as onnxsim.billm/onnxsim.pb_llm).
//
// Like billm_entry.h's own ApplyBillm (the closest existing precedent --
// another SINGLE-model, calibration-driven pass at the protobuf level,
// binarizing the ORIGINAL dense float32 weight straight from the float
// model, not a rounding-refinement lever on an already-quantized one),
// this operates directly on onnx::GraphProto rather than through
// onnxoptimizer's Node/Value IR. Unlike BiLLM's own block-wise salient-
// column search plus Cholesky-factored inverse-Hessian error
// compensation, this technique needs only each input channel's own
// Hessian-DIAGONAL (`sum(x**2, axis=0)`, no matrix inverse anywhere) as
// a per-element importance weight for a 2-component weighted-Lloyd/
// k-means EM run per group -- see onnxsim/bwa_ptq.py's own docstring for
// the full "Hessian-aware two-scale binary EM" argument.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see billm_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Binarizes every matched MatMul/vanilla-Gemm node with a constant 2-D
// FLOAT32 weight to exactly 1 sign bit + 1 group-select bit per element,
// using real calibration activations run through `executor` to compute
// each layer's own per-input-channel Hessian diagonal (the EM step's own
// importance weight).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Sign:  initializer, int8, W's own shape -- +1/-1 per element
//   Group: initializer, int8, W's own shape -- 0/1 per element, selects
//          which of a group's two candidate scales that element uses
//   Scale0/Scale1: initializer, float32, per-(output-channel-independent,
//          reduction-axis) group -- the group's two EM-fit scale
//          candidates, Scale0 <= Scale1
//   What_hat = Cast(Sign, float) * (Scale0 + Cast(Group, float) *
//              (Scale1 - Scale0))
//   Y = MatMul(X, What_hat) [+ bias]
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own
// graph inputs throws `std::invalid_argument`. NOT subgraph-aware,
// matching every one of those passes' own scope decision.
//
// A layer whose activation was never observed with a feature axis
// (rank < 2 after the reference's own leading-axis flattening), or whose
// feature dim does not match the weight's reduction dimension `K`, is
// left untouched -- mirrors apply_bwa_ptq's own per-layer skip
// conditions exactly.
//
// `group_size` is the number of contiguous reduction-axis elements
// sharing one pair of candidate scales (matches
// quantize_weight_only_int4's own group convention, apply_bwa_ptq's own
// default 128); `max_em_iters` upper-bounds the EM loop per group (which
// already stops early once the element assignment stops changing) --
// mirrors apply_bwa_ptq's own parameters of the same names/defaults
// exactly.
//
// This port does not expose `skip_names` (a plain initializer-name
// denylist Python callers can already apply themselves by filtering
// candidates before calling this entry point) -- several other *_cpp
// ports in this repo already establish that a C++ port need not mirror
// every optional knob its Python counterpart has.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_bwa_ptq's own FLOAT-only weight requirement and ApplyBillm's/
// ApplyLlmInt8's own FLOAT32-only calibration scope: a FLOAT16/BFLOAT16
// activation is never observed (skipped like an unobserved one), never
// converted.
//
// Accepted numerical scope: the EM loop (a 2-component weighted-Lloyd
// run, no closed-form global optimum) is deterministic given its own
// median-split seed and converges the same way numpy's own vectorized
// implementation does -- both sides run the identical alternating
// weighted-mean/nearest-scale update to the identical stopping
// condition, so this port is expected to track apply_bwa_ptq's own
// float64 numpy implementation closely, up to ordinary floating-point
// summation-order differences (including this port's own
// `std::nth_element`-based median, which can differ from numpy's own
// partition algorithm in which of several exactly-equal median
// candidates it picks -- immaterial to the median VALUE itself). See
// tests/test_bwa_ptq_cpp.py for the measured agreement.
onnx::ModelProto ApplyBwaPtq(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t group_size = 128, int64_t max_em_iters = 10);
