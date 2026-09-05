#pragma once

// Structured (channel) and attention-head pruning entry points exposed to
// Python, mirroring pruning_entry.h's own "not a Quantize* scheme"
// rationale for its separate file. Unlike every pass in onnxsim/passes/
// (which run through onnxoptimizer's Node/Value IR via OptimizeFixed),
// both operate directly on onnx::GraphProto: the algorithm needs
// whole-graph, multi-hop forward/backward tensor-name-based analysis
// (producer/consumer maps, chain walking) that maps far more directly onto
// the same protobuf-level approach onnxsim/pruning.py's own reference
// implementation takes than onto the optimizer's PredicateBasedPass
// single-node-match model -- see structured_pruning_entry.cpp's own
// top-of-file comment for the details and scope of both ports (attention-
// head pruning lives in the same translation unit specifically to reuse
// its producer/consumer/slicing helpers directly, rather than duplicating
// them). See onnxsim.h for the documentation this mirrors.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration only -- the full ``ModelExecutor`` interface (and the
// DLPack machinery its ``Run`` signature needs) lives in onnxsim.h, which
// itself includes *this* header (to reuse EmbeddingVocabPruningResult
// below), so including onnxsim.h back from here would be circular. A
// reference to an incomplete type is all ApplyStructuredWandaPruning's own
// declaration needs; structured_pruning_entry.cpp includes onnxsim.h itself
// for the definition, where ``executor.Run(...)`` is actually called.
struct ModelExecutor;

onnx::ModelProto ApplyStructuredPruning(const onnx::ModelProto& model,
                                        double sparsity);

onnx::ModelProto ApplyAttentionHeadPruning(const onnx::ModelProto& model,
                                           double sparsity);

// Removes intermediate (`inter_size`) channels from every expert of a
// matched `com.microsoft::MoE` node at once -- real structural pruning
// (smaller fc1/fc2 weight tensors, smaller per-expert matmuls on any
// runtime), the C++ port of pruning.py's own
// `apply_moe_expert_channel_pruning`. `num_experts` (whole-expert pruning,
// which needs runtime calibration data this build has no ONNX Runtime to
// provide -- see CLAUDE.md) is out of scope; see
// structured_pruning_entry.cpp's own "MoE (com.microsoft::MoE) expert-
// intermediate-channel pruning" section comment for the full scope and
// safety argument.
onnx::ModelProto ApplyMoeExpertChannelPruning(const onnx::ModelProto& model,
                                              double sparsity);

onnx::ModelProto ApplyQMoEExpertChannelPruning(const onnx::ModelProto& model,
                                               double sparsity);

// Removes whole experts (shrinks the `num_experts` leading axis) from a
// matched `com.microsoft::MoE` node and its upstream router projection at
// once -- the complementary technique to ApplyMoeExpertChannelPruning's own
// `inter_size` pruning. The C++ port of pruning.py's own
// `apply_moe_whole_expert_pruning`: experts are ranked by their mean router
// *gate weight* (`softmax(router_probs)` averaged over every calibration
// token, via the shared MoeRouterGateCalibrationStats helper in
// structured_pruning_entry.cpp -- see that function's own comment, and
// pruning.py's own `_moe_router_gate_calibration_stats`, for the full
// safety argument for why Softmax-then-mean, not raw logit magnitude or
// exact top-k frequency), falling back to each expert's own combined
// `fc1`/`fc2`(+`fc1_experts_bias`) L2 weight norm when a chain's
// `router_probs` was never observed during calibration (including
// `calibration_data=[]`, or a chain matched only inside a nested subgraph --
// see structured_pruning_entry.cpp's own "MoE whole-expert pruning" section
// comment). `num_experts_to_keep` is silently floored at the matched node's
// own `k` attribute (`k` itself is NEVER modified) -- pruning below `k`
// experts remaining is a hard onnxruntime execution failure, not merely
// suboptimal.
//
// Subgraph-aware (IterSubgraphs) for matching/slicing, exactly like
// ApplyMoeExpertChannelPruning -- but the calibration-based *ranking* only
// ever runs over chains matched in the TOP-LEVEL graph (mirrors
// pruning.py's own scope decision: `_add_probe_outputs` only ever appends
// to the top-level graph's own `output` list), so a chain matched only
// inside a nested If/Loop/Scan/BeamSearch-family subgraph always falls back
// to the weight-norm-only ranking above -- still correctly pruned, never
// silently skipped.
//
// Same calibration_data contract as ApplyStructuredWandaPruning: a batch
// missing one of `model`'s own graph inputs throws `std::invalid_argument`.
onnx::ModelProto ApplyMoeWholeExpertPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity);

// The quantized-weight (`com.microsoft::QMoE`) counterpart of
// ApplyMoeWholeExpertPruning -- the C++ port of pruning.py's own
// `apply_qmoe_whole_expert_pruning`. Reuses the exact same
// MoeRouterGateCalibrationStats helper unchanged (`router_probs` is QMoE's
// own second input too, upstream of and oblivious to its quantized
// `fc1`/`fc2`), and the exact same `k`-floor safety property. Unlike
// ApplyQMoEExpertChannelPruning's own `inter_size` slicing, whole-expert
// pruning needs no packed-axis unpack/re-pack anywhere: `num_experts` is
// every per-expert QMoE tensor's own LEADING axis regardless of
// `quant_type`/`block_size` (packing always lives on a later axis), so
// every `fc1`/`fc2` weight/scale/bias/zero_point (and, for
// `quant_type='nvfp4'`, `fc1`/`fc2`'s own `global_scale`) is a plain
// raw-element axis-0 index-select -- see structured_pruning_entry.cpp's own
// "QMoE whole-expert pruning" section comment.
onnx::ModelProto ApplyQMoEWholeExpertPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity);

// Return type of ApplyEmbeddingVocabPruning/ApplyEmbeddingVocabMagnitude
// Pruning -- the C++ mirror of pruning.py's own `EmbeddingPruningResult`
// dataclass (see structured_pruning_entry.cpp's own "Embedding vocabulary
// pruning" section comment for the full rationale). Unlike every other
// entry point in this file/onnxsim.h, these two passes change what counts
// as a valid model *input* (a vocabulary-pruned model only accepts
// remapped token ids), so -- exactly like the Python original -- neither
// one returns a bare `onnx::ModelProto`.
//
// Deliberately narrower than pruning.py's own dataclass: `id_map` (the
// old-token-id -> new-token-id mapping) is NOT carried across this
// boundary -- it is exactly `{kept_token_ids[i]: i for i in
// range(len(kept_token_ids))}`, trivial for the Python wrapper
// (onnx_simplifier.py) to reconstruct from `kept_token_ids` alone, so
// there is no reason to also serialize an int64->int64 map through
// nanobind. The Python wrapper builds the real, public
// `onnxsim.pruning.EmbeddingPruningResult` (the exact same dataclass the
// pure-Python entry points already return) from this struct's fields,
// rather than inventing a second, C++-only result type Python callers
// would need to know about -- one canonical return shape for both the
// pure-Python and C++-backed entry points.
struct EmbeddingVocabPruningResult {
  onnx::ModelProto model;
  bool matched = false;
  // Ascending, original-vocabulary token ids that survive. Empty when
  // `matched` is false (mirrors `kept_token_ids: Optional[List[int]] =
  // None` in Python -- an empty vector here is likewise only ever
  // meaningful when `matched` is true, since a non-empty keep-set is
  // required by both entry points below).
  std::vector<int64_t> kept_token_ids;
  bool lm_head_pruned = false;
};

// Shrinks a matched token-embedding table's vocabulary axis (a plain
// `Gather`'s `data` input feeding a graph input's token-id tensor, plus,
// where a tied or confidently-auto-identified untied `lm_head` exists, its
// own vocab-logits projection too) down to a caller-supplied, explicit
// keep-set. The C++ port of pruning.py's own `apply_embedding_vocab_
// pruning` -- give exactly one of `keep_token_ids`/`drop_token_ids`
// (`std::nullopt` means "not given"; an empty-but-present vector is a
// caller value, not "omitted"). See structured_pruning_entry.cpp's own
// "Embedding vocabulary pruning" section comment for the full matched
// topology/scope and structured_pruning_entry.cpp's own
// `MatchEmbeddingChain` for exactly what is/isn't recognized.
//
// `input_name`, when given, names which graph input (by name) the target
// `Gather`'s indices operand must resolve to -- required whenever more
// than one structurally-eligible `Gather` producer exists; a name that
// matches none throws `std::invalid_argument`. When omitted, the whole
// call declines (`matched=false`) rather than guessing if more than one
// eligible producer exists anywhere in the model (including nested
// If/Loop/Scan/BeamSearch-family subgraphs, at any depth -- see
// `IterSubgraphs`).
//
// Matches all three of pruning.py's own producer shapes -- a plain
// `Gather`, `com.microsoft::EmbedLayerNormalization`, or `com.microsoft::
// GatherBlockQuantized` (the block-quantized embedding shape -- verified
// TRUE full parity across both of its own sub-8-bit packing conventions;
// see structured_pruning_entry.cpp's own section comment for the full
// empirical detail). `lm_head` auto-detection recognizes `MatMul`/vanilla
// `Gemm`/`com.microsoft::FusedGemm`/`GemmFastGelu`, and the embedding
// table/`lm_head` weight/bias may be FLOAT, FLOAT16, OR BFLOAT16 -- all
// matching pruning.py's own scope exactly. See structured_pruning_entry.cpp's
// own section comment for the full matched topology.
EmbeddingVocabPruningResult ApplyEmbeddingVocabPruning(
    const onnx::ModelProto& model,
    const std::optional<std::vector<int64_t>>& keep_token_ids,
    const std::optional<std::vector<int64_t>>& drop_token_ids,
    const std::optional<std::string>& input_name);

// The importance-ranked variant of ApplyEmbeddingVocabPruning: drops the
// lowest-L2-norm `sparsity` fraction of vocabulary rows (combined,
// root-sum-square, with a matched untied `lm_head`'s own per-row weight
// norm when one is identified), never dropping any id in
// `protect_token_ids`. The C++ port of pruning.py's own
// `apply_embedding_vocab_magnitude_pruning` -- same weaker-safety-bar
// caveat as the Python original applies here too (see that function's own
// docstring): a small row norm means small weights, not that a token is
// safe to drop from a real deployment.
EmbeddingVocabPruningResult ApplyEmbeddingVocabMagnitudePruning(
    const onnx::ModelProto& model, double sparsity,
    const std::optional<std::vector<int64_t>>& protect_token_ids,
    const std::optional<std::string>& input_name);

// The calibration-driven (Wanda-style) upgrade of ApplyStructuredPruning --
// the FIRST calibration-driven pass in this file, and the reusable pattern
// every later one (SparseGPT, MoE whole-expert, transformer-block pruning)
// is meant to build on. Mirrors pruning.py's own
// apply_structured_wanda_pruning: same chain finding as
// ApplyStructuredPruning's own plain (FindChains/FindGatedChains/
// FindConvChains/FindConvResidualChains/FindMatmulResidualChains/
// FindMatmulConcatChains/FindConvConcatChains/FindSplitGatedChains) chain
// families -- deliberately NOT the additional quantized-weight families
// (MatMulNBits, QDQ, Bnb4, Fp8/Fp4 block-quantized, QOperator,
// DynamicQuantizeMatMul) ApplyStructuredPruning also folds in, since
// pruning.py's own apply_structured_wanda_pruning has no quantized-weight
// counterpart either -- but each chain's output channels are ranked by
// ``||W_row||_2 * ||X||_2`` (weight-row/filter L2 norm times the L2 norm of
// the *calibrated activation* actually flowing through that channel) rather
// than weight magnitude alone. See structured_pruning_entry.cpp's own
// "Wanda calibration" section comment for:
//   - the calibration-crossing design (how `calibration_data` -- a plain
//     {graph input name -> TensorProto} map per batch, the same shape as
//     pruning.py's own `Sequence[Tensors]`, crossing the pybind boundary via
//     the ordinary onnx::TensorProto nanobind caster rather than a bespoke
//     bytes encoding -- gets reordered into ModelExecutor::Run's own
//     positional contract), and
//   - how the per-channel activation-norm multiplier threads into
//     ApplyChains/ApplyConcatChains' existing importance computation
//     (a new optional trailing parameter on each, defaulted to nullptr so
//     ApplyStructuredPruning's own call sites are unchanged) rather than
//     duplicating either function.
//
// Unlike ApplyStructuredPruning, this is NOT subgraph-aware (mirrors
// pruning.py's own apply_structured_wanda_pruning, which likewise never
// calls `_iter_subgraphs` and only ever prunes `model.graph` directly) --
// calibration_data batches are keyed to the top-level graph's own inputs,
// and a nested If/Loop/Scan/BeamSearch-family subgraph body has no
// standalone way to be Run at all (it depends on its enclosing loop/branch's
// own runtime state), so there is no meaningful calibrated variant of the
// subgraph recursion ApplyStructuredPruning/ApplyAttentionHeadPruning/etc.
// otherwise share.
//
// A `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument` (ModelExecutor::Run needs every positional
// input filled; there is no meaningful partial-batch fallback). An
// otherwise-empty `calibration_data` (or one where every batch's probed
// activation never resolves -- see the .cpp's own dtype/rank scope notes)
// makes every matched chain fall back to ApplyStructuredPruning's own plain
// ``||W_row||_2`` ranking, exactly mirroring pruning.py's own per-chain "no
// matching activation observed" fallback, just triggered for every chain at
// once instead of one at a time.
onnx::ModelProto ApplyStructuredWandaPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, double epsilon = 1e-8);

// The calibration-driven (Wanda-style) upgrade of ApplyAttentionHeadPruning
// -- same relationship, and same reused pattern, as
// ApplyStructuredWandaPruning is to ApplyStructuredPruning above. Mirrors
// pruning.py's own apply_attention_head_wanda_pruning: same chain finding
// as ApplyAttentionHeadPruning's own three matched families
// (FindAttentionChains/FindGqaChains/FindOnnxAttentionChains) --
// deliberately NOT the fused `com.microsoft::MatMulNBitsQkv` variant
// ApplyAttentionHeadPruning additionally handles (via
// FindMatMulNBitsQkvChains/ApplyMatMulNBitsQkvChains), since pruning.py's
// own apply_attention_head_wanda_pruning has no quantized-weight
// counterpart either -- but each matched block's head (or, for
// GroupQueryAttention/plain ai.onnx::Attention, whole-KV-group) importance
// is ``||W||_F * ||X||_2`` -- the existing plain Frobenius-norm weight
// score times the combined (root-sum-square) L2 norm of that unit's own
// slice of the *calibrated* output-projection input activation -- rather
// than weight magnitude alone. See structured_pruning_entry.cpp's own
// "Wanda calibration" section comment (WandaCalibrationStats, reused
// as-is here: the attention output-projection's own input, at probe axis
// -1, is exactly the same "probe name -> per-channel activation L2 norm"
// shape that helper already produces for ApplyStructuredWandaPruning) for:
//   - the calibration-crossing design (unchanged from
//     ApplyStructuredWandaPruning -- same `calibration_data` shape, same
//     name -> ModelExecutor::Run-positional reordering), and
//   - how the resulting per-channel activation-norm map threads into
//     ApplyOnePlainAttentionChain/ApplyOneGqaChain's existing per-head/
//     per-group importance computation (new optional trailing parameters
//     on each, and on the ApplyAttentionChains dispatcher between them,
//     defaulted to nullptr so every ApplyAttentionHeadPruning call site is
//     unchanged) rather than duplicating the ranking/top-k/slicing logic.
//
// Unlike ApplyAttentionHeadPruning, this is NOT subgraph-aware (mirrors
// pruning.py's own apply_attention_head_wanda_pruning, which likewise
// never calls `_iter_subgraphs` and only ever prunes `model.graph`
// directly) -- same reasoning as ApplyStructuredWandaPruning's own
// declaration comment above: calibration_data batches are keyed to the
// top-level graph's own inputs, and a nested subgraph body has no
// standalone way to be Run at all.
//
// A `calibration_data` batch missing one of `model`'s own graph inputs
// throws `std::invalid_argument` (see ApplyStructuredWandaPruning's own
// declaration comment). An otherwise-empty `calibration_data` (or one
// where every batch's probed activation never resolves) makes every
// matched block fall back to ApplyAttentionHeadPruning's own plain
// ``||W||_F`` ranking, exactly mirroring pruning.py's own per-block "no
// matching activation observed" fallback.
onnx::ModelProto ApplyAttentionHeadWandaPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, double epsilon = 1e-8);

// Depth/block-level pruning: drops whole redundant pre-norm transformer
// residual sub-blocks (``x = x + SelfAttn(LN(x))`` or ``x = x + MLP(LN(x))``)
// wholesale, rather than shrinking every block a little the way every
// other entry point in this header does. The C++ port of pruning.py's own
// `apply_transformer_block_pruning` -- a GENUINELY DIFFERENT KIND of pass
// from every other entry point above: it deletes whole nodes and rewires
// their consumers (graph surgery, changing the graph's own topology), not
// just tensors resized in place. See structured_pruning_entry.cpp's own
// "Transformer block (depth) pruning" section comment (IsEntryLnNode
// through CommitTransformerBlockDrops, and ApplyTransformerBlockPruning's
// own comment right above its definition) for:
//   - the exact matched pattern (a bare-Add merge; a plain, unfused entry
//     norm -- LayerNormalization/RMSNormalization/
//     SimplifiedLayerNormalization -- OR a fused SkipLayerNormalization/
//     SkipSimplifiedLayerNormalization node's own optional fourth output,
//     standing in for x_in, so a model already run through onnxruntime's
//     own transformer optimizer is matched too);
//   - why attention and MLP/FFN blocks are matched and dropped fully
//     independently, never only as a paired "whole layer";
//   - why a KV-cache-bearing attention block needs no dedicated handling
//     to always decline safely on its own;
//   - the calibration-driven ranking (mean cosine similarity between each
//     candidate's own x_in/x_out over `calibration_data` -- the
//     literature-standard "Block Influence"/ShortGPT-style depth-pruning
//     signal -- via TransformerBlockSimilarity, reusing
//     WandaCalibrationStats' own probe-injection/batch-iteration pattern
//     with its own dedicated accumulator), which drops the HIGHEST-
//     similarity (most redundant) candidates first, up to the target
//     count, SKIPPING (never failing the whole call on) any candidate
//     whose own interior overlaps an already-committed one's;
//   - the subgraph-aware pooled-selection-but-per-graph-commit scope
//     (mirrors pruning.py's own docstring exactly: `sparsity`/
//     `num_blocks_to_drop` are pooled ACROSS THE WHOLE MODEL, but the
//     calibration-based ranking only ever runs over TOP-LEVEL-graph
//     candidates, exactly like ApplyMoeWholeExpertPruning's own scope
//     note above).
//
// `num_blocks_to_drop`, when given, is an explicit block count (silently
// capped at however many candidates were actually matched); otherwise
// `sparsity` is interpreted as a FRACTION OF MATCHED CANDIDATES (rounded
// to the nearest whole block), not of the model's total layer count --
// the same "fraction of what was actually found eligible" meaning
// ApplyMoeWholeExpertPruning's own `sparsity` already has for experts.
//
// Same `calibration_data` contract, and the same batch-missing-a-required-
// graph-input `std::invalid_argument`, as ApplyStructuredWandaPruning.
onnx::ModelProto ApplyTransformerBlockPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, std::optional<int64_t> num_blocks_to_drop);

// SparseGPT (Frantar & Alistarh, 2023, "SparseGPT: Massive Language Models
// Can Be Accurately Pruned in One-Shot", https://arxiv.org/abs/2301.00774):
// zeros the least-important entries of every matched layer's constant 2-D
// FLOAT32/FLOAT16/BFLOAT16 weight to an unstructured or N:M sparsity
// pattern, using a sequential, Hessian-error-compensating algorithm ported
// from GPTQ (same authors) rather than magnitude/Wanda's one-shot static
// importance score -- see structured_pruning_entry.cpp's own "SparseGPT
// (unstructured / N:M) pruning" section comment for the full technique, and
// pruning.py's own module-level docstring / `apply_sparsegpt_pruning`
// docstring for the Python reference this ports. Unlike every other pass in
// this file, this NEVER changes any tensor's shape: it only rewrites
// individual weight entries in place (zeroed when pruned, Hessian-
// compensated -- so possibly changed even when KEPT -- otherwise), and
// every matched layer is processed completely independently, with no
// producer/consumer chain-walking at all.
//
// Matches exactly: every plain `MatMul`/vanilla-`Gemm` node (MatchMatMulLikeRaw
// -- NOT pruning.py's own widened `_match_matmul_like`, which also
// recognizes `com.microsoft::FusedGemm`/`GemmFastGelu`; this already covers
// `com.microsoft::GroupQueryAttention`'s own separate Q/K/V projections,
// which are ordinary MatMul/Gemm nodes feeding it, not a weight the op
// itself owns) with a constant 2-D FLOAT/FLOAT16/BFLOAT16 weight
// (IsSupportedFloatDtype -- matches pruning.py's own
// `_is_supported_float_dtype`; read/written via ReadTensorAsF64/
// WriteF64TensorAs, preserving each weight's own original dtype, exactly
// mirroring pruning.py's own `_to_f64`/`_from_f64` convention), plus every
// `com.microsoft::Attention` node's constant 2-D FLOAT/FLOAT16/BFLOAT16
// merged QKV weight (MatchAttentionProducerAnyFloat -- a narrow, SparseGPT-
// local dtype-widened duplicate of the shared, still-FLOAT32-only
// MatchAttentionProducer, which also backs this file's own structural
// Attention-head pruning and was deliberately left untouched -- see that
// function's own comment in structured_pruning_entry.cpp). Deliberately
// narrower than pruning.py's own `apply_sparsegpt_pruning` in one remaining
// way (previously two -- FLOAT16/BFLOAT16 support was closed in the same
// round that added this comment, following this file's own established
// MoE/QMoE FP16/BFLOAT16-widening precedent):
//   - 2-D `Conv` (ordinary/depthwise/general-grouped) is NOT matched at
//     all -- pruning.py's own Conv support needs an entirely new, from-
//     first-principles im2col cross-covariance Hessian (`H = patches.T @
//     patches`, plus a genuinely per-group Hessian/column-processing split
//     for grouped/depthwise Conv) that pruning.py's own module docstring
//     documents at length as having no correct upstream reference to port
//     from at all (the original SparseGPT repository's own `add_batch`
//     never actually exercises a Conv layer). Reaching that bar is
//     materially bigger than the rest of this pass and was deliberately
//     left out of scope for this port -- a Conv node here is left
//     completely untouched (never guessed at, never crashes), exactly as
//     if it were any other unmatched node type. THIS IS THE ONLY REMAINING
//     SCOPE GAP versus pruning.py's own `apply_sparsegpt_pruning` -- until
//     it closes, `apply_sparsegpt_pruning` itself must stay pure Python
//     (not aliased to this port), since a Conv-bearing model would
//     otherwise silently get a different (Conv-less) result from the two
//     implementations.
//
// Same real numerical linear algebra pruning.py's own `apply_sparsegpt_
// pruning`/:mod:`onnxsim.gptq` needs, hand-written here rather than reused
// from any external BLAS/LAPACK dependency (structured_pruning_entry.cpp's
// own CholeskyLower/InvertLowerTriangular/InverseHessianCholesky): a dense
// Cholesky decomposition and triangular solve are used to turn each
// matched layer's own calibration Hessian into the "one Cholesky-factored
// inverse Hessian, reused via slicing for every column and block" GPTQ's
// own reformulation both this and :mod:`onnxsim.gptq` key their whole
// efficiency argument on.
//
// `n`/`m` (both std::nullopt, or both given: N:M semi-structured pruning,
// keep the `n` highest-importance entries per group of `m`, `0 < n <= m`)
// and `sparsity` (target unstructured-sparsity fraction, ignored when
// `n`/`m` are given, `[0, 1)`) mirror pruning.py's own identical
// parameters and validation (`_validate_pattern`) exactly, including the
// ONE deliberate departure from every other unstructured-pruning function
// in pruning.py that function's own docstring calls out: for unstructured
// sparsity, the pruning threshold is shared across every output row
// within each `proc_block_size`-wide column block (the reference
// implementation's own behavior), not chosen per row -- faithfully
// reproduced here (see structured_pruning_entry.cpp's own
// SparseGptPruneColumns) rather than "corrected" to match every other
// pass, since the whole point of this port is to reproduce SparseGPT
// specifically. `percdamp` (Hessian damping fraction) and
// `proc_block_size` (column-processing block size -- both the lazy-update
// granularity and, for unstructured sparsity only, the width each shared
// per-block threshold is computed over) mirror pruning.py's own identical
// parameters and defaults.
//
// Same executor-as-first-argument, `calibration_data` (one
// `{graph input name: TensorProto}` map per batch) shape as every other
// calibration-driven pass in this file (ApplyStructuredWandaPruning and
// friends) -- see that function's own declaration comment and
// structured_pruning_entry.cpp's own "Wanda calibration" section comment
// for the full calibration-crossing design this reuses. A `calibration_data`
// batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware (mirrors pruning.py's own
// `apply_sparsegpt_pruning`, which likewise never calls `_iter_subgraphs`)
// -- same reasoning as ApplyStructuredWandaPruning's own declaration
// comment: calibration_data batches are keyed to the top-level graph's own
// inputs, and a nested If/Loop/Scan/BeamSearch-family subgraph body has no
// standalone way to be Run at all.
//
// A matched layer with no observed 2-D-or-higher-rank-with-a-trailing-
// feature-axis calibration activation for its own input (dead input, an
// otherwise-empty `calibration_data`, or every batch's own activation isn't
// FLOAT/FLOAT16/BFLOAT16) is left completely untouched -- unlike Wanda,
// there is no data-free fallback for a technique whose entire mechanism is
// the Hessian.
onnx::ModelProto ApplySparseGptPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, const std::optional<int64_t>& n,
    const std::optional<int64_t>& m, double percdamp = 0.01,
    int64_t proc_block_size = 128);

// Wanda pruning (Sun et al., 2023, "A Simple and Effective Pruning Approach
// for Large Language Models", https://arxiv.org/abs/2306.11695): the
// calibration-driven upgrade of magnitude pruning's data-free baseline
// (onnxsim.h's ApplyMagnitudePruning/PruneMagnitude) to Wanda's own
// importance metric -- ``|W_ij| * ||X_j||_2`` (weight magnitude times its
// reduction-dimension entry's CALIBRATED activation L2-norm) -- applied
// ELEMENT-WISE (unstructured, or N:M semi-structured) rather than as a
// whole-row/channel keep/drop decision. The C++ port of pruning.py's own
// `apply_wanda_pruning`. Named ApplyWandaPruning, deliberately distinct
// from the already-existing ApplyStructuredWandaPruning above (Wanda-ranked
// whole-CHANNEL structural pruning, a completely different output shape
// change): like ApplySparseGptPruning immediately above, this NEVER changes
// any tensor's shape -- only individual weight entries are zeroed in place.
//
// Matches EXACTLY the same candidate set as ApplySparseGptPruning, via the
// same MatchMatMulLikeRaw matcher for the MatMul/Gemm case, PLUS a local,
// dtype-widened copy of MatchAttentionProducer for the Attention case (see
// structured_pruning_entry.cpp's own "Wanda unstructured (element-wise)
// pruning" section comment, MatchAttentionProducerWideDtype, for why this
// needs its own copy rather than reusing MatchAttentionProducer directly):
// every plain `MatMul`/vanilla-`Gemm` node with a constant 2-D FLOAT32/
// FLOAT16/BFLOAT16 weight (already covering
// `com.microsoft::GroupQueryAttention`'s own separate Q/K/V projections --
// ordinary MatMul/Gemm nodes feeding it, not a weight the op itself owns),
// plus every `com.microsoft::Attention` node's constant 2-D FLOAT32/FLOAT16/
// BFLOAT16 merged QKV weight -- mirrors pruning.py's own
// `_is_supported_float_dtype` widening of `_match_attention_producer`
// exactly. Like ApplySparseGptPruning immediately above (also widened this
// same round), FLOAT16/BFLOAT16 support is now at TRUE parity with
// pruning.py's own `apply_wanda_pruning` (read out upcast to float64 via
// ReadTensorAsF64, written back down via WriteF64TensorAs -- this file's own
// MoE/QMoE-established conversion trio, see IsSupportedFloatDtype's own
// declaration comment -- exactly mirroring pruning.py's own `_to_f64`/
// `_from_f64` round trip; masking never changes a surviving entry's own
// value, only zeros dropped ones, so this reproduces every kept entry's
// exact original bit pattern). Only remaining, deliberate gap vs.
// pruning.py's own `apply_wanda_pruning`:
//   - 2-D `Conv` (ordinary/depthwise/general-grouped) is NOT matched at
//     all -- pruning.py's own Conv support for THIS function needs the same
//     from-first-principles im2col per-receptive-field-offset activation
//     norm (`_conv_patch_sq_sum`, `_conv_group_relative_norm`) that
//     ApplySparseGptPruning's own declaration comment explains is out of
//     scope here for the analogous Hessian case -- materially bigger than
//     the rest of this pass (a brand-new im2col activation-collection engine
//     this file has nowhere else, plus the grouped/depthwise group-relative
//     norm expansion, with real numeric-correctness risk if gotten subtly
//     wrong), and deliberately left to the pure-Python implementation rather
//     than risked this round. A Conv node here simply never matches either
//     candidate case and is left completely untouched, never guessed at.
//     Because of this ONE remaining gap, `apply_wanda_pruning` is NOT
//     aliased to this port in pruning.py -- see that function's own
//     docstring there.
//
// Wanda's own calibration statistic -- one shared per-reduction-channel
// activation L2-norm per matched layer's own input, reduced over every
// leading axis (so a rank-3 `com.microsoft::Attention` input is handled
// exactly like a rank-2 MatMul/Gemm one, mirroring pruning.py's own
// `x.reshape(-1, x.shape[-1])`) -- is exactly WandaCalibrationStats' own
// output shape, reused verbatim: probe axis -1 for every candidate, keyed
// by each candidate's own activation tensor name (`x_name`) rather than by
// weight name the way pruning.py's own `attn_act_norm` is keyed for
// Attention -- mirroring ApplyAttentionHeadWandaPruning's own already-
// established simplification of that same distinction (two distinct
// Attention nodes sharing one activation tensor name is a purely
// theoretical edge case neither C++ port bothers distinguishing). See
// structured_pruning_entry.cpp's own "Wanda unstructured (element-wise)
// pruning" section comment for the masking mechanics.
//
// `n`/`m` (both std::nullopt, or both given together: N:M semi-structured
// pruning, `0 < n <= m`) and `sparsity` (target unstructured-sparsity
// fraction, ignored when `n`/`m` are given, `[0, 1)`) mirror pruning.py's
// own identical parameters and validation (`_validate_pattern`) exactly.
// `epsilon` floors the per-channel activation norm before multiplying it
// into the importance score, avoiding every entry of an all-zero-activation
// channel tying at exactly-zero importance. `global_sparsity` pools every
// matched layer's own already-computed importance into one ranking across
// the WHOLE model and picks a single keep-count from `sparsity`'s fraction
// of that pooled total -- mirrors pruning.py's own `apply_wanda_pruning`/
// `apply_magnitude_pruning` `global_sparsity` mode (same honestly-noted
// cross-layer-scale caveat; see pruning.py's own docstring). Incompatible
// with `n`/`m` (same reasoning as the Python original): throws
// `std::invalid_argument` when both are given.
//
// A matched layer with NO observed calibration activation for its own
// input (dead input, an otherwise-empty `calibration_data`, every batch's
// own activation isn't FLOAT32/FLOAT16/BFLOAT16, or an observed width
// mismatched with the weight's own reduction dimension) falls back to PLAIN
// MAGNITUDE
// importance (``|W_ij|`` alone) for that one layer -- mirrors pruning.py's
// own per-layer `_wanda_importance` fallback exactly. This is the one
// meaningful behavioral difference from ApplySparseGptPruning's own
// declaration comment: SparseGPT has no data-free fallback (its entire
// mechanism IS the Hessian) and leaves an unobserved layer completely
// untouched, whereas Wanda always has a well-defined, data-free importance
// score to fall back to, so an unobserved layer is still pruned (to plain
// magnitude, i.e. `apply_magnitude_pruning`'s own metric) rather than
// skipped.
//
// Same executor-as-first-argument, `calibration_data` (one
// `{graph input name: TensorProto}` map per batch) shape as every other
// calibration-driven pass in this file -- see ApplyStructuredWandaPruning's
// own declaration comment for the full calibration-crossing design. A
// `calibration_data` batch missing one of `model`'s own graph inputs throws
// `std::invalid_argument`. NOT subgraph-aware (mirrors pruning.py's own
// `apply_wanda_pruning`, which likewise never calls `_iter_subgraphs`) --
// same reasoning as every other calibration-driven pass's own declaration
// comment: calibration_data batches are keyed to the top-level graph's own
// inputs, and a nested If/Loop/Scan/BeamSearch-family subgraph body has no
// standalone way to be Run at all.
onnx::ModelProto ApplyWandaPruning(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    double sparsity, const std::optional<int64_t>& n,
    const std::optional<int64_t>& m, double epsilon = 1e-8,
    bool global_sparsity = false);
