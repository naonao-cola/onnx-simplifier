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
#include <vector>

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
// Unlike pruning.py's own version, this port only ever matches a plain
// `Gather` producer -- not `com.microsoft::EmbedLayerNormalization` or
// `com.microsoft::GatherBlockQuantized` -- and only ever matches a bare
// `MatMul`/vanilla-`Gemm` `lm_head` -- not `com.microsoft::FusedGemm`/
// `GemmFastGelu` -- and only ever admits a plain FLOAT (float32) embedding
// table/lm_head weight/bias, not also FLOAT16/BFLOAT16, matching this
// file's own established narrower-than-pruning.py C++-port scope decision
// elsewhere (e.g. `ApplyMoeExpertChannelPruning`'s own FLOAT32-only
// restriction). See structured_pruning_entry.cpp's own section comment for
// the full list of deliberately out-of-scope shapes and why.
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
