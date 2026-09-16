#pragma once

// Calibration-driven KV-cache quantization (KIVI, Liu et al. 2024;
// KVQuant, Hooper et al. 2024) entry point exposed to Python -- C++ port
// of onnxsim.kv_cache_quantization's own quantize_kv_cache (see
// onnxsim/kv_cache_quantization.py's module docstring for the full
// technique: every matched `Concat(past, new, axis=seq)` KV-cache stream
// gets quantized to INT8, symmetric -- Key-style (default) with one
// static, calibrated scale per channel (the head-dim axis, shared by
// every cached token for that stream's whole lifetime); Value-style
// (matched by name) with a fresh, data-free scale per token instead,
// computed from that token's own values the instant it is produced).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another SINGLE-model, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather than
// through onnxoptimizer's Node/Value IR: threading a live ModelExecutor
// plus calibration batches through OptimizeFixed's single-node-match
// PredicateBasedPass model has no established path in this codebase
// (every PredicateBasedPass in onnxsim/passes/ is data-free by
// construction). Unlike onnxsim/passes/intactkv.h (already merged, PR
// #1551), which reimplements the SAME structural
// `Concat(past, new, axis=seq)` match directly against onnx-optimizer's
// own Node/Value IR (since IntactKV is a PredicateBasedPass), this port
// re-derives the match fresh against raw onnx::GraphProto/NodeProto --
// the two are NOT code-shared (different IR entirely), though they
// recognize the same underlying graph shape. This port's own match also
// resolves one more field IntactKV's own subset never needed
// (`channel_axis`, the tensor's own last axis) and additionally splits
// matched streams into Key-style/Value-style, neither of which IntactKV's
// own scope requires.
//
// Before (illustrated for Key; Value is handled the other way -- see
// below -- and every candidate matched in a graph is rewritten
// independently, one stream at a time):
//   past_key: graph input, float32 [..., seq_past, head_dim]
//   new_key:  float32 [..., seq_new, head_dim]
//   present_key = Concat(past_key, new_key, axis=seq)  -- graph output,
//                 also consumed by the attention math
//
// After (Key-style -- static, calibrated, per-channel):
//   past_key: graph input, INT8 [..., seq_past, head_dim]  -- dtype
//             changed in place, same shape/name
//   key_scale: initializer, float32 [head_dim]              -- per-channel
//   key_zero_point: initializer, INT8 [head_dim], all zero  -- symmetric
//   new_key_q = QuantizeLinear(new_key, key_scale, key_zero_point,
//                               axis=channel_axis)
//   present_key = Concat(past_key, new_key_q, axis=seq)  -- INT8 graph
//                 output, SAME NAME/producing node as before (only its
//                 own declared dtype and one input changed)
//   present_key_f = DequantizeLinear(present_key, key_scale,
//                                     key_zero_point, axis=channel_axis)
//   <every other NODE consumer of the old float present_key (the
//    attention math) is rewired to present_key_f; the graph's own output
//    BINDING itself is untouched -- it keeps resolving to the Concat
//    node's own (now INT8) output by name, unchanged>
//
// After (Value-style -- data-free, per-token, matched by name):
//   past_value: graph input, INT8 [..., seq_past, head_dim]
//   past_value_scale: graph input, float32 [..., seq_past, 1]  -- NEW
//                      input, one scale per already-cached token
//   new_scale = Max(ReduceMax(Abs(new_value), axes=[channel_axis],
//                              keepdims=1), eps) / 127  -- per NEW token,
//               no calibration data involved
//   new_value_q = Cast(Clip(Round(new_value / new_scale), -128, 127),
//                       INT8)
//   present_value = Concat(past_value, new_value_q, axis=seq)  -- INT8
//   present_value_scale = Concat(past_value_scale, new_scale, axis=seq)
//               -- NEW output, float32, grows in lockstep with
//               present_value itself
//   present_value_f = Cast(present_value, float32) * present_value_scale
//   <every other NODE consumer of the old float present_value is rewired
//    to present_value_f, same output-binding convention as Key-style>
//
// SCOPE NARROWING (matching quantize_kv_cache's own documented scope
// exactly, not an additional one this port adds): KIVI's own
// residual-window bookkeeping (the most recent R tokens kept in full
// precision until they age out) is out of scope here too, exactly as it
// is in the Python reference (see that module's own docstring) -- deciding
// which tokens have "aged out" is cross-step, host-side bookkeeping, not
// something one exported ONNX graph can express on its own. This port
// also omits exposing `num_samples`/`seed`/`providers` (Python-side-only
// concerns for generating `calibration_data` when it is omitted, mirrored
// by the Python wrapper this entry point is called through, not by this
// function's own signature -- the same convention llm_int8_entry.h's own
// ApplyLlmInt8 already establishes for this codebase).
//
// A stream matched as Value-style is left completely untouched (not
// downgraded to Key-style) when the model's opset is below 18
// (ReduceMax's axes-as-input form) -- mirrors quantize_kv_cache's own
// per-stream skip exactly. A model whose opset is below 13
// (QuantizeLinear/DequantizeLinear's own per-channel `axis`) is returned
// completely unchanged.
//
// NUMERICAL SCOPE: this is a closed-form calibration statistic (a
// per-channel max-abs reduction) with no fitting algorithm, matrix
// inversion, or RNG anywhere in it, so this port is expected to track
// quantize_kv_cache's own float64 numpy implementation closely, up to
// ordinary floating-point summation/reduction-order differences. No
// ACCEPTED, PERMANENT DIVERGENCE note applies here.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Quantizes every `Concat(past, new, axis=seq)` KV-cache stream matched in
// `model` to INT8 -- see this header's own top-of-file comment for the
// full Key-style/Value-style rewrite. `calibration_data` (one
// `{graph input name: TensorProto}` map per batch, keyed to `model`'s own
// graph inputs) calibrates Key-style streams' per-channel scale only --
// Value-style streams need none; a batch missing one of `model`'s own
// graph inputs throws `std::invalid_argument` (but ONLY when at least one
// Key-style candidate needs calibrating at all -- exactly like every
// other calibration-driven pass in this codebase, matching
// quantize_kv_cache's own `if channel_candidates: ...` gate, this port
// never even touches `executor`/`calibration_data` when every matched
// stream is Value-style). `value_output_names`, when non-empty, is the
// exact set of matched streams' own `present` output names that get
// Value-style treatment; when empty (the default), any matched stream
// whose `present` output name contains ".value" is treated as Value-style
// automatically (mirrors quantize_kv_cache's own name-based default
// exactly), every other stream gets Key-style.
//
// SCOPE NARROWING: quantize_kv_cache's own `value_output_names` is
// `Optional[Sequence[str]]`, so Python can distinguish "omitted entirely"
// (falls back to the ".value"-in-name heuristic) from "explicitly passed
// an empty list" (forces every stream to Key-style, overriding that
// heuristic even for a stream whose name happens to contain ".value").
// This entry point's own `value_output_names` has no such None sentinel
// -- an empty vector here always means "omitted," so that second,
// narrower case (explicitly overriding the heuristic to nothing) is not
// expressible through this signature. No known caller needs it; several
// other *_cpp ports in this repo already establish that a C++ port need
// not mirror every optional knob its Python counterpart has.
onnx::ModelProto ApplyKvCacheQuantization(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    const std::vector<std::string>& value_output_names = {});
