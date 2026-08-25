#pragma once

// C++ port of onnxsim/precision_estimator.py's static INT8-quantization
// precision analysis for MatMul/Gemm/Conv/Attention -- see that module's
// docstring for the full rationale (accumulator-overflow bound, outlier-ratio
// resolution loss, float32-cast exactness, and analytically-known activation
// ranges). Kept in sync with the Python implementation by design: every
// threshold, formula and recommendation string here should read as a
// line-for-line translation, not an independent reimplementation, so the two
// never quietly drift apart. This exists (rather than reusing the Python
// module) so the WASM build -- which has no Python runtime -- can offer the
// same quantization-risk pre-check in the browser UI, entirely client-side
// and before the user even runs a quantize pass.
//
// Like precision_estimator.py, this is read-only: it never modifies the
// model.

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <string>
#include <vector>

namespace onnxsim {

// Mirrors precision_estimator.py's MatMulGemmPrecisionEstimate and
// ConvPrecisionEstimate -- identical field sets in Python (only the dataclass
// name differs), so this C++ port unifies them into one struct with an
// ``op_type`` discriminator ("MatMul" | "Gemm" | "Conv").
struct WeightPrecisionEstimate {
  std::string node_name;
  std::string op_type;
  int64_t reduction_depth = 0;
  int64_t num_channels = 0;
  bool int32_accumulator_safe = false;
  bool float32_cast_exact = false;  // false just means routine float rounding
  double max_outlier_ratio = 0.0;   // NaN if not computable (see .cpp)
  bool outlier_risk = false;
  std::string activation_producer_op;  // "" if not recognized
  bool has_activation_range = false;
  double activation_range_lo = 0.0;  // valid iff has_activation_range
  double activation_range_hi = 0.0;
  std::string recommendation;
};

// Mirrors precision_estimator.py's AttentionPrecisionEstimate. Every
// "unknown" field (couldn't be determined statically) is flagged by its own
// has_* bool rather than a sentinel, except max/default/actual scale, which
// use NaN (mirroring Python's Optional[float] -> None there, since NaN
// already is this module's "unknown float" convention elsewhere).
struct AttentionPrecisionEstimate {
  std::string node_name;
  bool has_num_query_heads = false;
  int64_t num_query_heads = 0;
  bool has_num_kv_heads = false;
  int64_t num_kv_heads = 0;
  bool has_head_dim = false;
  int64_t head_dim = 0;
  double default_scale = 0.0;  // NaN if head_dim unknown
  double actual_scale = 0.0;   // NaN if the node has no `scale` attribute
  // -1 = unknown (default_scale or actual_scale unavailable), 0 = false,
  // 1 = true -- mirrors Python's Optional[bool].
  int scale_matches_default = -1;
  std::string recommendation;
};

// Mirrors precision_estimator.py's ModelQuantizationEstimate: the whole-model
// rollup of every analyzed node's estimate, plus the per-node detail so a
// caller (e.g. the WASM UI) can render both a headline risk figure and a
// drill-down table.
struct ModelQuantizationEstimate {
  std::vector<WeightPrecisionEstimate> weight_estimates;
  std::vector<AttentionPrecisionEstimate> attention_estimates;
  int64_t total_nodes_analyzed = 0;
  std::vector<std::string> unsafe_nodes;
  std::vector<std::string> outlier_risk_nodes;
  double worst_outlier_ratio = 0.0;      // NaN if no node had one
  double estimated_relative_error = 0.0;  // NaN if any unsafe_nodes
  std::string risk_level;                 // "unsafe" | "degraded" | "safe"
};

// Per-node precision estimates for `model` -- see precision_estimator.py's
// ``estimate_quantization_precision`` for the full algorithm. Only the
// top-level graph is walked (subgraphs -- e.g. inside If/Loop/Scan -- are not
// descended into), matching the Python version.
void EstimateQuantizationPrecision(
    const onnx::ModelProto& model,
    std::vector<WeightPrecisionEstimate>* weight_estimates,
    std::vector<AttentionPrecisionEstimate>* attention_estimates);

// Aggregates EstimateQuantizationPrecision's per-node estimates into a single
// whole-model INT8-quantization risk summary -- see
// precision_estimator.py's ``estimate_model_quantization_drop`` for the full
// risk_level / estimated_relative_error semantics, which this mirrors
// exactly (same thresholds, same root-sum-square combination).
ModelQuantizationEstimate EstimateModelQuantizationDrop(
    const onnx::ModelProto& model);

}  // namespace onnxsim
