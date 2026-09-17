#pragma once

// Calibration-driven PTQ4ViT twin uniform quantization entry point
// exposed to Python -- C++ port of onnxsim.ptq4vit's own
// apply_ptq4vit_quantization (see onnxsim/ptq4vit.py's module docstring
// for the full technique: splits a matched tensor's value range at a
// calibration-searched threshold `t` into `[lo, t]`/`[t, hi]` and
// quantizes each side independently at the same per-side level count,
// then inserts that twin-uniform quantize/dequantize round trip as new
// graph nodes right after every standalone Softmax output and every GELU
// output).
//
// Single-model, calibration-driven, protobuf-level shape as
// llm_int8_entry.h's own ApplyLlmInt8 (probe real activations through
// `executor`, then insert new nodes) -- but a DIFFERENT calibration shape
// than every weight-quantizing pass ported so far: this file captures
// each matched tensor's own REAL VALUES (a flat double buffer per probe
// name, unfiltered and unreduced -- no absmax/Hessian accumulation),
// mirroring ptq4vit.py's own `collected[name].append(arr.ravel())`
// exactly, since :func:`_search_twin_split`'s own grid search needs the
// full empirical distribution, not a summary statistic.
//
// Candidate matching mirrors ptq4vit.py's own `_find_softmax_targets`/
// `_find_gelu_targets` exactly: every standalone `Softmax` node's output,
// plus every standalone `Gelu` node's output or the final `Mul` of the
// standard `0.5 * x * (1 + Erf(x / sqrt(2)))` export decomposition
// (matched structurally -- an `Erf` feeding an `Add`, feeding a `Mul`,
// feeding a second `Mul`, via each node's OWN first-matching consumer in
// graph-node order, not any check on the intervening constants'
// values) -- excluding any candidate whose own output is already a graph
// output (rewiring one would need renaming a ValueInfoProto, not a node
// input; mirrors the Python reference's own `graph_output_names` filter
// exactly).
//
// The search itself (`_search_twin_split`/`_twin_quantize_dequantize`/
// `_single_uniform_quantize_dequantize`) is transcribed verbatim in
// double precision: a bounded, deterministic grid of `num_candidates`
// (default 97) interior split points, each scored by its own mean
// squared reconstruction error against every finite captured value, kept
// only if it beats an equal-total-bit-budget single quantizer at `2 *
// n_levels` levels (see ptq4vit.py's own docstring for why that
// comparison, not a same-per-side-level-count one, is the fair bar). A
// tensor whose search finds no split clearing that bar is left
// unquantized by this pass -- mirrors the reference's own `None`-return
// skip exactly.
//
// Graph-rewrite shape (`_insert_twin_quantize`): a
// `Less`/`Sub`/`Div`/`Round`/`Clip`/`Mul`/`Add` (per side) /`Where`
// (selecting sides) node sequence spliced in right after the matched
// node's own output, with every existing consumer of that output rewired
// to read the twin round trip's own final `Where` output instead --
// exactly llm_int8_entry.h's/gptq_entry.h's own "quantize a specific
// tensor in place" splice shape, transcribed for this technique's own
// node sequence rather than shared. RELIES on the input model already
// being in the topologically-sorted order onnx.checker requires (every
// consumer of a tensor appears after that tensor's own producer node) to
// safely rewire consumers by post-insertion node INDEX rather than a
// persisted NodeProto identity -- ptq4vit.py's own `_insert_twin_quantize`
// relies on the exact same invariant for the exact same reason (see that
// function's own comment, "breaking the topological order onnx.checker
// requires").
//
// Same `calibration_data` (one `{graph input name: TensorProto}` map per
// batch, keyed to `model`'s own graph inputs) shape as every other
// calibration-driven pass in this codebase -- a batch missing one of
// `model`'s own graph inputs throws `std::invalid_argument`. NOT
// subgraph-aware, matching every one of those passes' own scope decision.
//
// `n_levels` mirrors apply_ptq4vit_quantization's own parameter of the
// same name exactly (uniform levels each sub-quantizer uses). Models
// below opset 11 are returned unchanged (the 3-input Clip form this
// pass's quantize-dequantize round trip needs), also mirroring the
// reference. FLOAT32-only throughout: a tensor observed as anything other
// than FLOAT32 is simply never added to the captured-values map by this
// port's own probe reader (mirrors every other calibration-driven pass's
// own FLOAT32-only scope, though ptq4vit.py's own probe reader has no
// explicit dtype check at all -- onnxruntime/the reference evaluator
// backend only ever returns FLOAT32 for a FLOAT32 graph tensor in
// practice, so this is not an observable scope narrowing).
//
// ACCEPTED, PERMANENT DIVERGENCE: none beyond ordinary floating-point
// summation-order differences from numpy in the reconstruction-error
// scoring -- this is a bounded, deterministic grid search with no RNG,
// so this port is expected to track the Python reference closely,
// including exact-tie behavior at the search's own strict-`<`-only
// improvement rule. See tests/test_ptq4vit_cpp.py for the measured
// agreement.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyPtq4Vit(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t n_levels = 256);
