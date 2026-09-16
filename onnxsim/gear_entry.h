#pragma once

// Calibration-driven GEAR (Kang et al., 2024) entry point exposed to
// Python -- C++ port of onnxsim.gear's own apply_gear (see
// onnxsim/gear.py's module docstring for the full technique: low-rank-
// plus-sparse residual compensation layered on top of
// onnxsim.kv_cache_quantization's own static per-channel INT8 base
// quantization, applied only to a freshly-produced ("new") KV-cache
// token, never to an already-cached ("past") one -- see that docstring's
// own "Why this has to be calibration-fit and static" section for why a
// past token's true residual is gone forever once it is written).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another SINGLE-model, calibration-driven pass at the
// protobuf level), this operates directly on onnx::GraphProto rather
// than through onnxoptimizer's Node/Value IR: threading a live
// ModelExecutor plus calibration batches through OptimizeFixed's
// single-node-match PredicateBasedPass model has no established path in
// this codebase (every PredicateBasedPass in onnxsim/passes/ is
// data-free by construction).
//
// Candidate matching is a fresh, protobuf-level transcription of
// onnxsim.kv_cache_quantization's own _find_kv_cache_candidates (gear.py
// itself imports and reuses that exact function -- see this file's own
// gear_entry.cpp for why this port re-derives it independently rather
// than depending on any C++ port of kv_cache_quantization.py, which may
// or may not exist yet): a `Concat(past, new, axis=seq)` node whose
// output is a graph output, where `past` is a FLOAT32 graph input
// consumed by nothing else; `seq_axis`/`channel_axis` (the tensor's own
// last axis) are resolved the same way, and a candidate whose resolved
// axes coincide (no distinct channel axis left to quantize per-channel
// on) is skipped, exactly like the Python reference.
//
// Before (illustrated for Key; Value is handled identically -- unlike
// onnxsim.kv_cache_quantization's own Key/Value asymmetry, this module
// deliberately applies the SAME treatment to every matched stream, per
// its own documented second simplification):
//   past: graph input, float32 [..., seq_past, head_dim]
//   new:  float32 [..., seq_new, head_dim]        -- this step's own K/V
//   present = Concat(past, new, axis=seq)          -- graph output, and
//             consumed by the attention math
// After (opset 13+ only -- QuantizeLinear/DequantizeLinear's per-channel
// `axis` needs it):
//   past: graph input, INT8 [..., seq_past, head_dim]     -- dtype changed
//   scale: initializer, float32 [head_dim]                   -- per-channel
//   zero_point: initializer, INT8 [head_dim], all zero       -- symmetric
//   p: initializer, float32 [head_dim, head_dim]      -- fitted rank-r
//      projector (present only when rank > 0)
//   sparse_mask: initializer, float32 [head_dim], 0/1  -- fitted outlier
//      mask (present only when outlier_fraction selects at least one
//      channel)
//   new_q = QuantizeLinear(new, scale, zero_point, axis=channel_axis)
//   present = Concat(past, new_q, axis=seq_axis)     -- INT8 graph output,
//             unchanged from onnxsim.kv_cache_quantization's own rewrite
//             (the SAME Concat node, its own input rewired -- not
//             replaced)
//   past_dequant = DequantizeLinear(past, scale, zero_point, axis)
//   new_dequant = DequantizeLinear(new_q, scale, zero_point, axis)
//   new_residual = new - new_dequant                -- this step's *true*
//                  residual, exact (still has the live float `new` tensor)
//   low_rank_term = new_residual @ p                 -- coherent part
//   remainder = new_residual - low_rank_term
//   sparse_term = remainder * sparse_mask             -- outlier-channel part
//   new_corrected = new_dequant + low_rank_term + sparse_term
//   present_corrected = Concat(past_dequant, new_corrected, axis=seq_axis)
//   <every OTHER consumer of the old float `present` -- everything except
//    the Concat node itself, which still produces the raw INT8 `present`
//    above -- now reads present_corrected instead>
//
// SCOPE NARROWING (matching gear.py's own documented scope exactly, not
// an additional one this port adds): a past token's own already-written
// reconstruction is never revisited/improved after the fact -- only a
// freshly-produced ("new") token ever receives the low-rank/sparse
// compensation, exactly as gear.py's own docstring documents and for the
// same static-graph-rewrite reason. `num_samples`/`seed`/`providers` are
// Python-side-only concerns for generating `calibration_data` when it is
// omitted, mirrored by the Python wrapper this C++ entry point is called
// through, not by this function's own signature.
//
// ACCEPTED, PERMANENT DIVERGENCE: this port's own low-rank projector fit
// reuses the SAME hand-rolled one-sided (Hestenes) Jacobi SVD this
// session's own low_rank_compensation_entry.cpp/kbvq_moe.h already
// establish as this codebase's precedent (no linear-algebra library is
// linked into this codebase) -- kept as a deliberate, file-local
// TRANSCRIBED COPY here too, not a shared dependency (see
// gear_entry.cpp's own top-of-file comment for why). As those files'
// own notes already explain: individual singular vectors are not
// expected to match numpy's own LAPACK-backed SVD sign-for-sign, but the
// reconstructed projector `P = V_r @ V_r.T` is basis- and sign-invariant
// (Eckart-Young uniqueness whenever the r-th and (r+1)-th singular
// values of the calibration residual are well separated), so this port
// is expected to track the Python reference's own fitted projector
// closely. The outlier-channel SELECTION also uses a full deterministic
// sort where the reference uses `np.argsort` (not guaranteed stable) --
// the same vanishingly-unlikely-tie caveat SpQR's/BiLLM's own headers
// already document, changing nothing about the SET of channels chosen
// except at an exact sensitivity tie.
#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Applies GEAR-style low-rank-plus-sparse residual compensation, layered
// on top of static per-channel INT8 quantization, to every
// `Concat(past, new, axis=seq)` KV-cache stream this port's own
// `FindKvCacheCandidates` can find -- see this header's own top-of-file
// comment for the exact match/rewrite/scope. `rank` is the low-rank
// correction's own rank (clamped to `min(rank, head_dim,
// num_calibration_rows)` per stream; `0` disables the low-rank term
// entirely); `outlier_fraction` is the fraction of each stream's channels
// (by count, rounded half-to-even to the nearest whole channel) kept as
// an explicit sparse correction after the low-rank term is subtracted
// (`0.0` disables the sparse term entirely) -- mirrors apply_gear's own
// parameters of the same names/defaults exactly. A stream whose
// calibration activation never appeared in any batch is left untouched,
// as is the whole model when no stream matches at all or `model`'s
// opset is older than 13.
onnx::ModelProto ApplyGear(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t rank = 4, double outlier_fraction = 0.05);
