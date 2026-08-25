/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises precision_estimator.{h,cpp} -- the C++ port of
 * onnxsim/precision_estimator.py's static INT8-quantization risk analysis.
 * Mirrors tests/test_precision_estimator.py's cases (same models, same
 * expected values) so the two implementations are checked against the same
 * ground truth. Needs onnx configured, so this links the fully-configured
 * `onnxsim` CMake target rather than building standalone.
 */
#include "precision_estimator.h"

#include <onnx/onnx_pb.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using onnxsim::AttentionPrecisionEstimate;
using onnxsim::EstimateModelQuantizationDrop;
using onnxsim::EstimateQuantizationPrecision;
using onnxsim::WeightPrecisionEstimate;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckClose(double a, double b, const std::string& what,
                double rtol = 1e-6) {
  Check(std::fabs(a - b) <= rtol * std::max(std::fabs(a), std::fabs(b)), what);
}

// Same formulas precision_estimator.cpp itself uses (kept independent here,
// as a from-scratch cross-check rather than importing the .cpp's private
// constants).
int64_t MaxSafeInt32ReductionDepth() {
  return (int64_t{2147483647}) / (127 * 255);
}
int64_t MaxExactFloat32ReductionDepth() {
  return (int64_t{1} << 24) / (127 * 255);
}
constexpr double kOutlierRatioThreshold = 127.0;

onnx::TensorProto MakeFloatInitializer(const std::string& name,
                                       const std::vector<int64_t>& dims,
                                       const std::vector<float>& data,
                                       bool raw) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) t.add_dims(d);
  if (raw) {
    t.set_raw_data(std::string(reinterpret_cast<const char*>(data.data()),
                               data.size() * sizeof(float)));
  } else {
    for (float v : data) t.add_float_data(v);
  }
  return t;
}

onnx::ValueInfoProto MakeValueInfo(const std::string& name,
                                   const std::vector<int64_t>& dims) {
  onnx::ValueInfoProto vi;
  vi.set_name(name);
  auto* tt = vi.mutable_type()->mutable_tensor_type();
  tt->set_elem_type(onnx::TensorProto::FLOAT);
  auto* shape = tt->mutable_shape();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
  return vi;
}

onnx::NodeProto MakeNode(const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs) {
  onnx::NodeProto n;
  n.set_op_type(op_type);
  for (const auto& i : inputs) n.add_input(i);
  for (const auto& o : outputs) n.add_output(o);
  return n;
}

void AddIntAttr(onnx::NodeProto& n, const std::string& name, int64_t v) {
  auto* a = n.add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INT);
  a->set_i(v);
}

void AddFloatAttr(onnx::NodeProto& n, const std::string& name, float v) {
  auto* a = n.add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::FLOAT);
  a->set_f(v);
}

onnx::ModelProto MakeModel(const std::vector<onnx::NodeProto>& nodes,
                           const std::vector<onnx::ValueInfoProto>& inputs,
                           const std::vector<onnx::ValueInfoProto>& outputs,
                           const std::vector<onnx::TensorProto>& initializers,
                           int opset = 17) {
  onnx::ModelProto m;
  m.set_ir_version(10);
  auto* opset_import = m.add_opset_import();
  opset_import->set_domain("");
  opset_import->set_version(opset);
  auto* graph = m.mutable_graph();
  graph->set_name("g");
  for (const auto& n : nodes) *graph->add_node() = n;
  for (const auto& i : inputs) *graph->add_input() = i;
  for (const auto& o : outputs) *graph->add_output() = o;
  for (const auto& t : initializers) *graph->add_initializer() = t;
  return m;
}

std::vector<float> RandomVec(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n);
  for (auto& v : out) v = dist(rng) * scale;
  return out;
}

void TestMatMulSmallReductionDepthSafeAndExact() {
  const int64_t K = 16, N = 4;
  auto w = MakeFloatInitializer("W", {K, N}, RandomVec(K * N, 0.1f, 0), true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "matmul small: one estimate");
  if (weights.size() == 1) {
    const auto& est = weights[0];
    Check(est.op_type == "MatMul", "matmul small: op_type");
    Check(est.reduction_depth == K, "matmul small: reduction_depth");
    Check(est.num_channels == N, "matmul small: num_channels");
    Check(est.int32_accumulator_safe, "matmul small: int32 safe");
    Check(est.float32_cast_exact, "matmul small: float32 exact");
    Check(!est.outlier_risk, "matmul small: no outlier risk");
  }
}

void TestMatMulPastInt32BoundIsUnsafe() {
  const int64_t k = MaxSafeInt32ReductionDepth() + 1;
  auto w = MakeFloatInitializer("W", {k, 1}, RandomVec(k, 0.01f, 1), true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, k})},
                         {MakeValueInfo("Y", {1, 1})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "matmul unsafe: one estimate");
  if (!weights.empty()) {
    Check(!weights[0].int32_accumulator_safe, "matmul unsafe: flagged unsafe");
    Check(
        weights[0].recommendation.find("int32-safe bound") != std::string::npos,
        "matmul unsafe: recommendation mentions bound");
  }
}

void TestMatMulPastFloat32ExactButInt32Safe() {
  const int64_t k = MaxExactFloat32ReductionDepth() + 1;
  Check(k <= MaxSafeInt32ReductionDepth(), "sanity: still int32-safe");
  auto w = MakeFloatInitializer("W", {k, 1}, RandomVec(k, 0.01f, 2), true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, k})},
                         {MakeValueInfo("Y", {1, 1})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "float32 exact: one estimate");
  if (!weights.empty()) {
    Check(weights[0].int32_accumulator_safe, "float32 exact: int32 safe");
    Check(!weights[0].float32_cast_exact, "float32 exact: not exact");
    Check(weights[0].recommendation.find("not bit-exact") != std::string::npos,
          "float32 exact: recommendation mentions rounding");
  }
}

void TestMatMulOutlierChannelIsFlagged() {
  const int64_t K = 32, N = 2;
  auto data = RandomVec(K * N, 0.05f, 3);
  for (int64_t i = 0; i < K; ++i) data[static_cast<size_t>(i * N + 1)] = 0.01f;
  data[1] = 10.0f;  // channel 1, row 0: extreme outlier
  auto w = MakeFloatInitializer("W", {K, N}, data, true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "outlier: one estimate");
  if (!weights.empty()) {
    Check(weights[0].outlier_risk, "outlier: flagged");
    Check(weights[0].max_outlier_ratio > kOutlierRatioThreshold,
          "outlier: ratio above threshold");
  }
}

void TestGemmTransBUsesTransposedLayout() {
  const int64_t N = 4, K = 20;
  auto w = MakeFloatInitializer("W", {N, K}, RandomVec(N * K, 0.1f, 4), true);
  auto node = MakeNode("Gemm", {"X", "W"}, {"Y"});
  AddIntAttr(node, "transB", 1);
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "gemm transB: one estimate");
  if (!weights.empty()) {
    Check(weights[0].op_type == "Gemm", "gemm transB: op_type");
    Check(weights[0].reduction_depth == K, "gemm transB: reduction_depth");
    Check(weights[0].num_channels == N, "gemm transB: num_channels");
  }
}

void TestConvReductionDepthIsCinTimesKernelVolume() {
  const int64_t cout = 6, cin = 3, kh = 5, kw = 5;
  auto w = MakeFloatInitializer("W", {cout, cin, kh, kw},
                                RandomVec(cout * cin * kh * kw, 0.1f, 5), true);
  auto node = MakeNode("Conv", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, cin, 16, 16})},
                         {MakeValueInfo("Y", {1, cout, 12, 12})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "conv: one estimate");
  if (!weights.empty()) {
    Check(weights[0].op_type == "Conv", "conv: op_type");
    Check(weights[0].reduction_depth == cin * kh * kw, "conv: reduction_depth");
    Check(weights[0].num_channels == cout, "conv: num_channels");
  }
}

void TestAttentionReportsHeadDimAndFlagsScaleMismatch() {
  const int64_t q_heads = 4, kv_heads = 4, head_dim = 8, sq = 6, skv = 6;
  auto node = MakeNode("Attention", {"Q", "K", "V"}, {"O"});
  AddIntAttr(node, "q_num_heads", q_heads);
  AddIntAttr(node, "kv_num_heads", kv_heads);
  AddFloatAttr(node, "scale", 1.0f);  // deliberately not 1/sqrt(head_dim)
  auto model =
      MakeModel({node},
                {MakeValueInfo("Q", {1, sq, q_heads * head_dim}),
                 MakeValueInfo("K", {1, skv, kv_heads * head_dim}),
                 MakeValueInfo("V", {1, skv, kv_heads * head_dim})},
                {MakeValueInfo("O", {1, sq, q_heads * head_dim})}, {}, 23);

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(attn.size() == 1, "attention mismatch: one estimate");
  if (!attn.empty()) {
    Check(attn[0].head_dim == head_dim, "attention mismatch: head_dim");
    Check(attn[0].num_query_heads == q_heads,
          "attention mismatch: num_query_heads");
    Check(attn[0].num_kv_heads == kv_heads, "attention mismatch: num_kv_heads");
    CheckClose(attn[0].default_scale, 1.0 / std::sqrt(double(head_dim)),
               "attention mismatch: default_scale");
    Check(attn[0].scale_matches_default == 0,
          "attention mismatch: flagged mismatched");
    Check(attn[0].recommendation.find("does not match") != std::string::npos,
          "attention mismatch: recommendation");
  }
}

void TestAttentionScaleMatchingDefaultIsNotFlagged() {
  const int64_t q_heads = 2, kv_heads = 2, head_dim = 16, sq = 4, skv = 4;
  auto node = MakeNode("Attention", {"Q", "K", "V"}, {"O"});
  AddIntAttr(node, "q_num_heads", q_heads);
  AddIntAttr(node, "kv_num_heads", kv_heads);
  auto model =
      MakeModel({node},
                {MakeValueInfo("Q", {1, sq, q_heads * head_dim}),
                 MakeValueInfo("K", {1, skv, kv_heads * head_dim}),
                 MakeValueInfo("V", {1, skv, kv_heads * head_dim})},
                {MakeValueInfo("O", {1, sq, q_heads * head_dim})}, {}, 23);

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(attn.size() == 1, "attention no-mismatch: one estimate");
  if (!attn.empty()) {
    Check(std::isnan(attn[0].actual_scale),
          "attention no-mismatch: no actual_scale");
    Check(attn[0].scale_matches_default == -1,
          "attention no-mismatch: unknown (no actual scale to compare)");
    Check(attn[0].recommendation.find("no int8 accumulator applies") !=
              std::string::npos,
          "attention no-mismatch: recommendation");
  }
}

void TestMatMulFedBySoftmaxReportsKnownActivationRange() {
  const int64_t K = 32, N = 4;
  auto w = MakeFloatInitializer("W", {K, N}, RandomVec(K * N, 0.1f, 7), true);
  auto softmax = MakeNode("Softmax", {"X"}, {"S"});
  auto matmul = MakeNode("MatMul", {"S", "W"}, {"Y"});
  auto model = MakeModel({softmax, matmul}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "softmax range: one estimate");
  if (!weights.empty()) {
    Check(weights[0].activation_producer_op == "Softmax",
          "softmax range: producer op");
    Check(weights[0].has_activation_range, "softmax range: has range");
    Check(weights[0].activation_range_lo == 0.0 &&
              weights[0].activation_range_hi == 1.0,
          "softmax range: (0, 1)");
    Check(weights[0].int32_accumulator_safe, "softmax range: still int32 safe");
    Check(weights[0].recommendation.find(
              "fixed static scale would quantize it exactly") !=
              std::string::npos,
          "softmax range: recommendation");
  }
}

void TestConvFedByClipWithConstantBoundsReportsRange() {
  const int64_t cout = 3, cin = 2, kh = 3, kw = 3;
  auto w = MakeFloatInitializer("W", {cout, cin, kh, kw},
                                RandomVec(cout * cin * kh * kw, 0.1f, 8), true);
  auto lo = MakeFloatInitializer("lo", {}, {0.0f}, true);
  auto hi = MakeFloatInitializer("hi", {}, {6.0f}, true);
  auto clip = MakeNode("Clip", {"X", "lo", "hi"}, {"C"});
  auto conv = MakeNode("Conv", {"C", "W"}, {"Y"});
  auto model = MakeModel({clip, conv}, {MakeValueInfo("X", {1, cin, 8, 8})},
                         {MakeValueInfo("Y", {1, cout, 6, 6})}, {w, lo, hi});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "clip range: one estimate");
  if (!weights.empty()) {
    Check(weights[0].activation_producer_op == "Clip", "clip range: producer");
    Check(weights[0].activation_range_lo == 0.0 &&
              weights[0].activation_range_hi == 6.0,
          "clip range: (0, 6)");
  }
}

void TestClipWithNonConstantBoundIsNotReportedAsKnownRange() {
  auto w = MakeFloatInitializer("W", {16, 4}, RandomVec(16 * 4, 0.1f, 9), true);
  auto clip = MakeNode("Clip", {"X", "lo", "hi"}, {"C"});
  auto matmul = MakeNode("MatMul", {"C", "W"}, {"Y"});
  auto lo = MakeFloatInitializer("lo", {}, {0.0f}, true);
  auto model = MakeModel({clip, matmul},
                         {MakeValueInfo("X", {1, 16}), MakeValueInfo("hi", {})},
                         {MakeValueInfo("Y", {1, 4})}, {w, lo});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "clip non-const: one estimate");
  if (!weights.empty()) {
    Check(weights[0].activation_producer_op.empty(),
          "clip non-const: no producer reported");
    Check(!weights[0].has_activation_range, "clip non-const: no range");
  }
}

void TestNonConstantWeightIsSkipped() {
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel(
      {node}, {MakeValueInfo("X", {1, 8}), MakeValueInfo("W", {8, 4})},
      {MakeValueInfo("Y", {1, 4})}, {});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.empty() && attn.empty(), "non-constant weight: skipped");
}

void TestModelDropSafeWithNoOutliers() {
  const int64_t K = 16, N = 4;
  std::vector<float> data(static_cast<size_t>(K * N));
  for (int64_t i = 0; i < K; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      data[static_cast<size_t>(i * N + j)] = (i % 2 == 0) ? 0.1f : -0.1f;
    }
  }
  auto w = MakeFloatInitializer("W", {K, N}, data, true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  auto est = EstimateModelQuantizationDrop(model);
  Check(est.total_nodes_analyzed == 1, "model drop safe: total nodes");
  Check(est.unsafe_nodes.empty(), "model drop safe: no unsafe nodes");
  Check(est.outlier_risk_nodes.empty(), "model drop safe: no outlier nodes");
  Check(est.risk_level == "safe", "model drop safe: risk_level");
  CheckClose(est.worst_outlier_ratio, 1.0, "model drop safe: worst ratio",
             1e-9);
  CheckClose(est.estimated_relative_error, 1.0 / (127.0 * std::sqrt(12.0)),
             "model drop safe: estimated error", 1e-9);
}

void TestModelDropDegradedWhenOutlierChannelPresent() {
  const int64_t K = 32, N = 2;
  auto data = RandomVec(K * N, 0.05f, 21);
  for (int64_t i = 0; i < K; ++i) data[static_cast<size_t>(i * N + 1)] = 0.01f;
  data[1] = 10.0f;
  auto w = MakeFloatInitializer("W", {K, N}, data, true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  auto est = EstimateModelQuantizationDrop(model);
  Check(est.risk_level == "degraded", "model drop degraded: risk_level");
  Check(est.unsafe_nodes.empty(), "model drop degraded: no unsafe nodes");
  Check(est.outlier_risk_nodes.size() == 1,
        "model drop degraded: one outlier node");
  Check(est.worst_outlier_ratio > kOutlierRatioThreshold,
        "model drop degraded: worst ratio above threshold");
  Check(est.estimated_relative_error > 1.0 / (127.0 * std::sqrt(12.0)),
        "model drop degraded: error above baseline");
}

void TestModelDropUnsafeReportsNanErrorAndListsTheNode() {
  const int64_t k = MaxSafeInt32ReductionDepth() + 1;
  auto w = MakeFloatInitializer("W", {k, 1}, RandomVec(k, 0.01f, 22), true);
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, k})},
                         {MakeValueInfo("Y", {1, 1})}, {w});

  auto est = EstimateModelQuantizationDrop(model);
  Check(est.risk_level == "unsafe", "model drop unsafe: risk_level");
  Check(est.unsafe_nodes.size() == 1, "model drop unsafe: one unsafe node");
  Check(std::isnan(est.estimated_relative_error),
        "model drop unsafe: NaN error");
}

void TestModelDropMoreUnsafeNodesWidenTheAggregateError() {
  const int64_t K = 16, N = 4;
  auto w1 =
      MakeFloatInitializer("W1", {K, N}, RandomVec(K * N, 0.1f, 23), true);
  auto w2 =
      MakeFloatInitializer("W2", {K, N}, RandomVec(K * N, 0.1f, 24), true);
  auto n1 = MakeNode("MatMul", {"X", "W1"}, {"Y1"});
  auto n2 = MakeNode("MatMul", {"X", "W2"}, {"Y2"});
  auto model_two = MakeModel(
      {n1, n2}, {MakeValueInfo("X", {1, K})},
      {MakeValueInfo("Y1", {1, N}), MakeValueInfo("Y2", {1, N})}, {w1, w2});
  auto model_one = MakeModel({n1}, {MakeValueInfo("X", {1, K})},
                             {MakeValueInfo("Y1", {1, N})}, {w1});

  auto est_two = EstimateModelQuantizationDrop(model_two);
  auto est_one = EstimateModelQuantizationDrop(model_one);
  Check(est_two.total_nodes_analyzed == 2, "model drop widen: two nodes");
  Check(est_two.estimated_relative_error > est_one.estimated_relative_error,
        "model drop widen: two-node error exceeds one-node error");
}

void TestModelDropNoAnalyzableNodesIsSafeWithZeroError() {
  auto node = MakeNode("Relu", {"X"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, 4})},
                         {MakeValueInfo("Y", {1, 4})}, {});

  auto est = EstimateModelQuantizationDrop(model);
  Check(est.total_nodes_analyzed == 0, "model drop empty: zero nodes");
  Check(est.risk_level == "safe", "model drop empty: safe");
  Check(est.estimated_relative_error == 0.0, "model drop empty: zero error");
}

// One case built via the typed float_data field instead of raw_data, to
// exercise ReadFloatTensorFlat's other branch (see precision_estimator.cpp).
void TestFloatDataFieldStorageIsReadCorrectly() {
  const int64_t K = 16, N = 4;
  auto w = MakeFloatInitializer("W", {K, N}, RandomVec(K * N, 0.1f, 30), false);
  Check(!w.has_raw_data(), "float_data path: no raw_data set");
  auto node = MakeNode("MatMul", {"X", "W"}, {"Y"});
  auto model = MakeModel({node}, {MakeValueInfo("X", {1, K})},
                         {MakeValueInfo("Y", {1, N})}, {w});

  std::vector<WeightPrecisionEstimate> weights;
  std::vector<AttentionPrecisionEstimate> attn;
  EstimateQuantizationPrecision(model, &weights, &attn);
  Check(weights.size() == 1, "float_data path: one estimate");
  if (!weights.empty()) {
    Check(weights[0].reduction_depth == K, "float_data path: reduction_depth");
  }
}

}  // namespace

int main() {
  TestMatMulSmallReductionDepthSafeAndExact();
  TestMatMulPastInt32BoundIsUnsafe();
  TestMatMulPastFloat32ExactButInt32Safe();
  TestMatMulOutlierChannelIsFlagged();
  TestGemmTransBUsesTransposedLayout();
  TestConvReductionDepthIsCinTimesKernelVolume();
  TestAttentionReportsHeadDimAndFlagsScaleMismatch();
  TestAttentionScaleMatchingDefaultIsNotFlagged();
  TestMatMulFedBySoftmaxReportsKnownActivationRange();
  TestConvFedByClipWithConstantBoundsReportsRange();
  TestClipWithNonConstantBoundIsNotReportedAsKnownRange();
  TestNonConstantWeightIsSkipped();
  TestModelDropSafeWithNoOutliers();
  TestModelDropDegradedWhenOutlierChannelPresent();
  TestModelDropUnsafeReportsNanErrorAndListsTheNode();
  TestModelDropMoreUnsafeNodesWidenTheAggregateError();
  TestModelDropNoAnalyzableNodesIsSafeWithZeroError();
  TestFloatDataFieldStorageIsReadCorrectly();

  if (g_failures == 0) {
    std::printf("precision_estimator_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "precision_estimator_test: %d failure(s)\n", g_failures);
  return 1;
}
