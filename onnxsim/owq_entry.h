#pragma once

// Calibration-driven OWQ (Lee, Park, Kim, Kim and Sung, 2023, AAAI 2024)
// entry point exposed to Python -- C++ port of onnxsim.owq's own apply_owq
// (see onnxsim/owq.py's module docstring for the full technique: a fifth
// lever targeting quantize_weight_only_int4's own output, alongside
// adaround/gptq/awq/quantease -- but unlike all four of those (which only
// ever change *which integer* an already-fixed-scale quantizer rounds a
// column to), OWQ instead rescues a small number of columns from
// quantization entirely, restoring them to exact float32 precision via a
// runtime correction term, using the classic Optimal Brain Surgeon (OBS)
// saliency metric `sensitivity_j = mean_n[(W[n,j] - RTN(W[n,j]))^2] /
// [H^-1]_jj`).
//
// Like gptq_entry.h's own ApplyGptq (the closest existing precedent --
// another two-model, calibration-driven pass at the protobuf level: OWQ,
// like GPTQ/AWQ, takes a float model AND an already-quantized model, not
// the single-model shape SpQR/PB-LLM/SqueezeLLM/BiLLM use), this operates
// directly on onnx::GraphProto rather than through onnxoptimizer's
// Node/Value IR: threading a live ModelExecutor plus calibration batches
// through OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).
//
// Before (illustrated for MatMul; a "vanilla" Gemm is handled the same
// way, its bias left untouched):
//   Y = MatMul(X, Wq_dequant) [+ bias]   -- the already-quantized node,
//                                            untouched apart from a rename
// After:
//   WeakIdx: initializer, int64, [num_weak] -- ascending column indices
//   DeltaW:  initializer, float32, [num_weak, N] -- exact residual
//            (W_float - W_rtn) for the rescued columns only, transposed
//            ready for a right-multiply
//   XWeak = Gather(X, WeakIdx, axis=-1)
//   Correction = MatMul(XWeak, DeltaW)
//   Y' = Add(Y, Correction)     -- Y' takes over the ORIGINAL output name;
//                                   the quantized node itself is renamed to
//                                   feed this Add instead
//
// `quantized_model`'s own INT4 codes are NEVER modified anywhere by this
// pass -- OWQ's whole point is leaving the existing quantization alone and
// compensating for its worst columns with an exact additive correction,
// not changing how those columns round.
//
// Candidate matching mirrors adaround.py's own _find_int4_matmul_candidates
// exactly (the same matcher gptq_entry.cpp's own FindInt4MatmulCandidates
// already reimplements for this codebase, transcribed again here as a
// local copy per this session's own "don't share a header across ports"
// convention -- see billm_entry.cpp's identical rationale for its own copy
// of gptq_entry.cpp's Hessian/Cholesky kernels): a MatMul/Gemm node (same
// op type on both sides) sharing its output tensor name across the two
// models, whose float weight is a constant 2-D FLOAT tensor and whose
// quantized weight arrives through a DequantizeLinear (with a block_size
// attribute) from a same-shaped INT4 initializer.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch, keyed to `float_model`'s own graph inputs) shape as every
// other calibration-driven pass in this codebase -- a `calibration_data`
// batch missing one of `float_model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware, matching every one of those
// passes' own scope decision.
//
// A layer whose activation was never observed with a feature axis
// (rank < 2 after the reference's own leading-axis flattening), whose
// feature dim does not match the weight's reduction dimension `K`, whose
// `K` is not divisible by the quantized scale's own block_size, whose
// clamped `outlier_fraction * K` rounds to fewer than 1 weak column, or
// whose selected columns already round-trip exactly (RTN already exact on
// every rescued column, so the correction would be an all-zero no-op), is
// left untouched entirely -- mirrors apply_owq's own per-layer skip
// conditions exactly.
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_owq's own FLOAT-only float-weight requirement and ApplyGptq's own
// FLOAT32-only calibration scope: a FLOAT16/BFLOAT16 activation is never
// observed (skipped like an unobserved one), never converted.
//
// This port does not expose a `skip_names`-style knob (apply_owq itself
// has none either) and hardcodes no additional parameters beyond what
// apply_owq's own signature already has; `num_samples`/`seed`/`providers`
// are Python-side-only concerns for generating `calibration_data` when it
// is omitted, the same convention every other calibration-driven `*_cpp`
// entry point in this codebase already establishes.
//
// Accepted numerical scope (same caveat as gptq_entry.h's own, since this
// port reuses the identical Hessian/Cholesky machinery): the dense inverse
// and Cholesky factorization at this algorithm's heart are computed with
// this TU's own scalar double-precision kernels rather than LAPACK, so the
// inverse-Hessian factor -- and therefore `h_inv_diag`, the denominator of
// OWQ's own sensitivity score -- can differ from numpy's in the last ulp or
// two on some inputs. Additionally, this port's own weak-column ranking
// uses `std::stable_sort` where apply_owq uses `np.argsort` (whose default
// `quicksort` kind is not guaranteed stable) -- an exact tie in column
// sensitivity (vanishingly unlikely with real floating-point
// activations/weights) could pick a different weak-column SET on the two
// sides for the same candidate count, diverging from that point on for
// that one layer. Since the correction is otherwise an EXACT closed-form
// residual (no rounding/fitting step at all once the column set is
// decided), reconstruction is bit-exact-in-practice for every column
// actually selected on both sides. See tests/test_owq_cpp.py for the
// measured agreement.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical forward
// declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

onnx::ModelProto ApplyOwq(
    const onnx::ModelProto& float_model,
    const onnx::ModelProto& quantized_model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double outlier_fraction = 0.01, double percdamp = 0.01);
