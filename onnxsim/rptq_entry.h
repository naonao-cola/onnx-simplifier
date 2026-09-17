#pragma once

// Calibration-driven RPTQ (Yuan et al., 2023) reorder entry point exposed
// to Python -- C++ port of onnxsim.rptq's own apply_rptq_reorder (see
// onnxsim/rptq.py's module docstring for the full technique: cluster each
// matched MatMul/vanilla-Gemm layer's input channels by their own
// calibration abs-max, via a plain Lloyd's-algorithm k-means, then permute
// the layer's activation input (a new `Gather`) and the weight's matching
// `K`-axis rows by the permutation that sorts channels by cluster --
// EXACT, not a quantization: `Gather(X, perm, axis=-1) @ Gather(W, perm,
// axis=0) == X @ W` for any permutation).
//
// Like smoothquant_entry.h's own ApplySmoothQuant (the closest existing
// precedent -- another calibration-driven, purely-reordering/rescaling
// MatMul/Gemm rewrite at the protobuf level, sharing this module's own
// matcher, onnxsim.smoothquant._match_matmul_like), this operates directly
// on onnx::GraphProto rather than through onnxoptimizer's Node/Value IR:
// threading a live ModelExecutor plus calibration batches through
// OptimizeFixed's single-node-match PredicateBasedPass model has no
// established path in this codebase (every PredicateBasedPass in
// onnxsim/passes/ is data-free by construction).

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration only -- see smoothquant_entry.h's own identical
// forward declaration for why (the full ModelExecutor interface lives in
// onnxsim.h, which includes this header back).
struct ModelExecutor;

// Per-cluster metadata for one RPTQ-reordered layer -- the C++-side
// equivalent of onnxsim.rptq.RptqLayerInfo, marshalled into that real
// dataclass by the Python wrapper (onnx_simplifier.py's own
// apply_rptq_reorder_cpp) rather than crossing the nanobind boundary as a
// bound class itself (the same "reconstruct the real public dataclass in
// the Python wrapper" pattern apply_embedding_vocab_pruning_cpp/
// EmbeddingPruningResult already establishes -- see
// structured_pruning_entry.h). `cluster_bounds` is the permuted axis's own
// half-open `[start, end)` per-cluster slice list, same meaning as
// RptqLayerInfo's own field.
struct RptqLayerInfo {
  std::string x_name;
  std::string w_name;
  std::string gather_output;
  std::vector<int64_t> permutation;
  std::vector<std::pair<int64_t, int64_t>> cluster_bounds;
};

struct RptqReorderResult {
  onnx::ModelProto model;
  // One entry per matched-and-reordered layer, in graph node order (mirrors
  // apply_rptq_reorder's own `layer_info` dict's insertion order -- Python
  // dicts preserve insertion order, so iterating this vector and building a
  // dict keyed by `x_name` reproduces it exactly, including the "last
  // write wins" behavior on a hypothetical repeated `x_name`).
  std::vector<RptqLayerInfo> layers;
};

// Clusters and reorders every matched MatMul/vanilla-Gemm layer's input
// channels using real calibration activations run through `executor`.
// Returns a float model -- an exact reordering, not a quantization -- plus
// per-layer cluster metadata for a downstream per-cluster-aware quantizer
// (see onnxsim/rptq.py's own module docstring for how far that composition
// is, and isn't, wired up today).
//
// Candidate matching mirrors onnxsim.smoothquant's own _match_matmul_like
// exactly (reused, unmodified, by rptq.py itself): a MatMul, or a Gemm with
// transA=0, alpha=1 and (when it has a bias) beta=1 -- the bias itself is
// never read or rewritten, only the activation (input 0) and weight
// (input 1) operands. Only a constant 2-D FLOAT32 weight is matched.
//
// Same `executor`/`calibration_data` (one `{graph input name: TensorProto}`
// map per batch) shape as every other calibration-driven pass in this
// codebase -- a `calibration_data` batch missing one of `model`'s own graph
// inputs throws `std::invalid_argument`. Activation capture is 2-D-only
// (mirrors `if x.ndim != 2: continue` exactly -- NOT the rank-agnostic
// `_activation_rows` flattening spqr_entry.h/gptq_entry.h use), matching
// apply_rptq_reorder's own per-channel abs-max scope. NOT subgraph-aware,
// matching every other calibration-driven pass's own scope decision.
//
// A candidate whose activation was never observed as a plain 2-D FLOAT32
// tensor, or whose activation feature dimension does not match the
// weight's reduction dimension `K`, is left untouched and gets no entry in
// `layers` -- mirrors apply_rptq_reorder's own per-layer skip conditions
// exactly.
//
// `seed` seeds the k-means centroid tie-breaking/duplicate-jitter; `num_
// -clusters` is the number of channel clusters to reorder each matched
// layer's input into -- mirrors apply_rptq_reorder's own parameters of the
// same names exactly (`num_samples` is a Python-side-only default-
// -calibration-data knob, not part of this entry point's own signature,
// the same convention every other calibration-driven *_entry.h in this
// codebase uses).
//
// FLOAT32-only throughout (weights and activations alike), mirroring
// apply_rptq_reorder's own `FLOAT`-only weight requirement.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM rptq.py: the k-means fit here is a
// genuinely different (though algorithmically identical -- percentile-
// -seeded Lloyd's algorithm) fit from apply_rptq_reorder's own, because
// this port's per-candidate RNG (used only to jitter apart duplicate
// percentile-derived centroids and to break exact-tie assignments) is a
// single sequentially-advancing std::mt19937_64 rather than reproducing
// numpy's own PCG64 bit generator and Generator.normal sampler bit-for-bit
// -- the same "comparable, not bit-identical" judgment
// kmeans_quantization.h/random_orthogonal.h already establish for this
// codebase's other k-means/RNG-seeded ports. This does not change WHAT the
// permutation is optimizing for (channels grouped by calibration-range
// similarity) -- for calibration data with clearly separated magnitude
// clusters, the deterministic percentile seeding alone (not the RNG
// jitter, which is scaled to ~1e-9 of the value range specifically so it
// never decides a non-tied assignment) already drives both ports to the
// same clustering; only a genuine near-tie between two candidate
// assignments could ever make the two ports' permutations differ. The
// reordered model's own correctness (the Gather/weight-permutation
// algebraic identity) holds regardless of clustering quality -- see this
// header's own top-of-file comment. See tests/test_rptq_cpp.py for how
// clustering quality is verified (by permutation validity and cluster-
// -bound partition properties on well-separated synthetic data, not exact
// index equality with the Python reference).
RptqReorderResult ApplyRptqReorder(
    const onnx::ModelProto& model, const ModelExecutor& executor,
    const std::vector<std::unordered_map<std::string, onnx::TensorProto>>&
        calibration_data,
    int64_t seed = 0, int64_t num_clusters = 4);
