#pragma once

// Calibration-driven BiLLM (Huang et al., 2024, ICML) entry point exposed
// to Python -- C++ port of onnxsim.billm's own quantize_weight_only_billm
// (see onnxsim/billm.py's module docstring for the full technique: a
// genuine ~1-bit-average weight binarizer, not another rounding-
// refinement lever on top of an already quantize_weight_only_int4-
// quantized layer the way GPTQ/AWQ are. Every matched MatMul/vanilla-Gemm
// layer's ORIGINAL dense float32 weight is processed block-wise: within
// each block, a small, data-dependent set of "salient" columns (found via
// a bounded search over Hessian-based column sensitivity) get a two-level
// binary residual approximation (~2 bits/element), every other column
// gets a single flat binary level (~1 bit/element), and the leftover
// per-block error is charged forward into not-yet-processed columns via
// the same OBC/GPTQ-style Cholesky-factored-inverse-Hessian mechanism
// onnxsim.gptq already uses).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- another SINGLE-model, calibration-driven pass at the
// protobuf level: BiLLM, like LLM.int8(), takes one model and calibration
// data, not a float/already-quantized model PAIR the way GPTQ/AWQ do),
// this operates directly on onnx::GraphProto rather than through
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

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Reuses onnxsim.gptq's own _inverse_hessian_cholesky machinery (billm.py
// itself imports it directly: `from onnxsim.gptq import
// _inverse_hessian_cholesky`) -- this port's own billm_entry.cpp keeps a
// deliberate, header-file-local TRANSCRIBED COPY of gptq_entry.cpp's own
// scalar double-precision CholeskyLower/InverseSPD/InverseHessianCholesky
// kernels, not a shared dependency between the two translation units,
// mirroring the exact same "don't refactor an already-tested file as a
// side effect of an unrelated port" convention this session's own
// kbvq_moe.h already establishes for a header-only copy of
// low_rank_compensation_entry.cpp's own Jacobi SVD. If a third port needs
// this same dense-Hessian machinery, that is the point to actually
// extract a shared file.
//
// Candidate matching mirrors onnxsim.quip_sharp's own _match_matmul_like
// (which billm.py itself imports and reuses): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1, whose weight (input
// 1) is a constant 2-D FLOAT32 tensor. The bias, when present, is never
// touched by this pass (mirrors billm.py's own candidate tuple, which
// discards the bias name entirely).
//
// Before (illustrated for MatMul; a "vanilla" Gemm is handled the same
// way, its bias left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Code1: initializer, int8, [K, N], values in {-1, +1} -- sign(W), or
//          sign(B1) for a salient column
//   Code2: initializer, int8, [K, N], values in {-1, 0, +1} -- sign of the
//          salient residual for a salient column, exactly 0 (no
//          correction) for a non-salient column
//   Scale1: initializer, float32, per-column (broadcastable against
//           [K, N]) -- this block's alpha for a salient column's own
//           level-1 binarization, or a non-salient column's own flat
//           binarization
//   Scale2: initializer, float32, per-column, exactly 0 for a non-salient
//           column
//   What_hat = Cast(Code1, float) * Scale1 + Cast(Code2, float) * Scale2
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
// left untouched -- mirrors quantize_weight_only_billm's own per-layer
// skip conditions exactly.
//
// `block_size` is BiLLM's own salient-column-search/OBC-compensation
// block width (paper default 128); `percdamp` is the Hessian damping
// factor, matching apply_gptq's own parameter of the same name/default;
// `max_salient_search` bounds how many leading (most Hessian-salient)
// columns of a block the search tries as "the salient group" (the
// paper's own stated 3-30 range; this port searches 1..min(
// max_salient_search, block_size - 1) and lets the search settle on
// however few or many minimize reconstruction error) -- mirrors
// quantize_weight_only_billm's own parameters of the same names exactly.
//
// This port does not expose `skip_names` (a plain initializer-name
// denylist Python callers can already apply themselves by filtering
// candidates before calling this entry point) -- several other *_cpp
// ports in this repo already establish that a C++ port need not mirror
// every optional knob its Python counterpart has.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// quantize_weight_only_billm's own FLOAT-only weight requirement and
// ApplyLlmInt8's/ApplyGptq's own FLOAT32-only calibration scope: a
// FLOAT16/BFLOAT16 activation is never observed (skipped like an
// unobserved one), never converted.
//
// Accepted numerical scope (same caveat as gptq_entry.h's own, since this
// port reuses the identical Hessian/Cholesky machinery): the dense
// inverse and Cholesky factorization at this algorithm's heart are
// computed with this TU's own scalar double-precision kernels rather than
// LAPACK, so the inverse-Hessian factor can differ from numpy's in the
// last ulp or two on some inputs. Additionally, this port's own salient-
// column ranking uses `std::stable_sort` where quantize_weight_only_billm
// uses `np.argsort` (whose default `quicksort` kind is not guaranteed
// stable) -- an exact tie in column sensitivity (vanishingly unlikely
// with real floating-point activations/weights) could pick a different
// salient-column SET on the two sides for the same candidate count,
// diverging from that point on for that one block. Reconstruction error
// tracks the reference's to well within quantization noise in every case
// observed. See tests/test_billm_cpp.py for the measured agreement.
onnx::ModelProto ApplyBillm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size = 128, double percdamp = 0.01,
    int64_t max_salient_search = 30);
