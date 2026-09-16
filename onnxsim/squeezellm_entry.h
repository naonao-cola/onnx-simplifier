#pragma once

// Calibration-driven SqueezeLLM (Kim et al., 2023) entry point exposed to
// Python -- C++ port of onnxsim.squeezellm's own
// quantize_weight_only_squeezellm (see onnxsim/squeezellm.py's module
// docstring for the full technique: sensitivity-weighted k-means, over a
// per-group non-uniform codebook, plus a dense-and-sparse outlier
// correction).
//
// Like llm_int8_entry.h's own ApplyLlmInt8 (the closest existing
// precedent -- the same SINGLE-model, calibration-driven, protobuf-level
// shape: `model` in, `model` out, no separate already-quantized second
// model the way gptq_entry.h/awq_entry.h take), this operates directly on
// onnx::GraphProto rather than through onnxoptimizer's Node/Value IR:
// threading a live ModelExecutor plus calibration batches through
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).
//
// Unlike every fold-to-initializer *_cpp port elsewhere in this repo,
// this pass keeps squeezellm.py's own real graph rewrite visible (a
// per-group codebook lookup via GatherND, not a folded float32
// initializer) -- the codebook/codes split IS the point of the technique
// (a real deployment would ship the small codebook plus the packed codes,
// not a materialized dense reconstruction), matching squeezellm.py's own
// choice exactly rather than this repo's more common weight-only
// fold-to-a-single-initializer convention (kmeans_quantization.h's own
// port narrows scope the other way, folding a *plain, unweighted* k-means
// fit -- SqueezeLLM's own sensitivity-weighted fit and outlier
// decomposition genuinely need the real graph, since GatherND/Reshape/Add/
// Transpose is exactly what a real INT4-storage-format deployment would
// keep, not merely an intermediate fold this port could discard).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched and never
// consumed by this pass):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After (opset 12+ only -- GatherND's own batch_dims support):
//   Codebook = per-group [num_levels] centroids     -- new initializer,
//                                                        [num_groups, L]
//   Codes    = per-element nearest-centroid index    -- new initializer,
//                                                        [num_groups,
//                                                        block_size, 1]
//   SparseDiff = w - dequant at outlier positions, 0 elsewhere -- new
//                                                        initializer,
//                                                        dense [N, K]
//   Gathered = GatherND(Codebook, Codes, batch_dims=1)  -- [num_groups,
//                                                           block_size]
//   Unblocked = Reshape(Gathered, [N, K])
//   Corrected = Add(Unblocked, SparseDiff)
//   W'       = Corrected, or Transpose(Corrected, perm=[1, 0]) when the
//              original weight was NOT already stored [N, K] (i.e. a
//              plain MatMul's own [K, N] layout) -- restoring the
//              node's own original weight orientation
//   Y = MatMul(X, W') [+ bias]     the ORIGINAL node, kept in place: only
//                                   its own weight input is rewired, no
//                                   node is replaced or destroyed
//
// SCOPE NARROWING: `num_samples`/`seed`/`providers` are Python-side-only
// concerns for generating `calibration_data` when it is omitted --
// mirrored by the Python wrapper this C++ entry point is called through,
// not by this function's own signature, the same convention
// llm_int8_entry.h's own ApplyLlmInt8 already establishes for this
// codebase's calibration-driven ports.
//
// NUMERICAL SCOPE: unlike this repo's k-means-family *_cpp ports
// (kmeans_quantization.h, aqlm.h, quip_sharp.h), squeezellm.py's own
// weighted k-means fit initializes its centroids *deterministically*
// (evenly-spaced order statistics of each group's own sorted values --
// `np.linspace(0, group_size - 1, num_levels).round()` -- no random
// sampling anywhere in the reference), so this port's own fit is expected
// to track the Python reference numerically closely -- not merely
// structurally -- up to ordinary floating-point summation-order
// differences and any ties in either the k-means assignment step's own
// argmin or the outlier threshold's own quantile computation (both
// broken the same way the reference breaks them -- see
// squeezellm_entry.cpp's own comments). No ACCEPTED, PERMANENT
// DIVERGENCE note applies here, unlike this repo's other k-means-family
// ports.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- see llm_int8_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D FLOAT32
// weight (reduction dimension `K` divisible by `block_size`) and a plain
// 2-D FLOAT32 activation input into SqueezeLLM-style dense-and-sparse
// non-uniform quantization -- see this header's own top-of-file comment.
// `calibration_data` is the same `{graph input name: TensorProto}`
// map-per-batch shape as every other calibration-driven pass in this
// codebase (llm_int8_entry.h's own ApplyLlmInt8, gptq_entry.h's own
// ApplyGptq) -- a batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware, matching every one of
// those passes' own scope decision.
//
// `block_size` is the number of contiguous reduction-dimension elements
// sharing one codebook; `bits` sets the codebook size to `2 ** bits`
// centroids per group; `outlier_fraction` is the fraction of weight
// elements (by magnitude, across the whole tensor) excluded from the
// k-means fit and corrected back to their exact original value instead;
// `num_kmeans_iterations` is the weighted Lloyd's-algorithm iteration
// budget -- mirrors quantize_weight_only_squeezellm's own parameters of
// the same names exactly.
//
// A layer is left untouched when its activation was never observed as a
// plain 2-D FLOAT32 tensor, when the feature dim does not match the
// weight's reduction dimension `K`, or when `K` is not divisible by
// `block_size` -- mirrors the reference's own per-layer skip conditions
// exactly. A model whose opset is older than 12 (GatherND's own
// `batch_dims` support) is returned unchanged.
onnx::ModelProto ApplySqueezeLlm(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t block_size = 32, int64_t bits = 4, double outlier_fraction = 0.0045,
    int64_t num_kmeans_iterations = 20);
