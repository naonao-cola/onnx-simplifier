/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises contrib_schemas.cpp's MoE/QMoE registration and
 * BuildMoEFunctionBody directly through the real ONNX C++ schema API
 * (OpSchemaRegistry::Schema, OpSchema::BuildContextDependentFunction) --
 * the same path ONNX Runtime or a Python `onnx.reference.ReferenceEvaluator`
 * would use to execute an otherwise-opaque MoE node when no native kernel is
 * available. Needs onnx configured (schema registry, function building), so
 * this links the fully-configured `onnxsim` CMake target rather than
 * building standalone, mirroring precision_estimator_test.cpp.
 *
 * This only checks structural properties (does a body get built when it
 * should, does it decline when it shouldn't, is the result a well-formed
 * function referencing the opsets it uses). The actual per-op arithmetic --
 * in particular, that `router_probs` needs an internal Softmax despite its
 * name, which is not obvious from ONNX Runtime's own docs -- was verified
 * out-of-band by running the generated FunctionProto (loaded as a model-
 * level local function) through a real onnxruntime InferenceSession and
 * diffing against a bare `com.microsoft.MoE` node executed by ONNX
 * Runtime's own native kernel, for relu/identity/silu/gelu activations and
 * with/without fc1/fc2 bias, all matching to float32 precision. That
 * cross-check needs onnxruntime (a Python-only, opt-in dependency for
 * onnxsim -- see CLAUDE.md), which is why it isn't reproduced as a C++ test
 * here.
 */
#include <onnx/defs/function.h>
#include <onnx/defs/schema.h>

#include <cstdio>
#include <string>
#include <vector>

#include "contrib_schemas.h"

using onnx::AttributeProto;
using onnx::FunctionBodyBuildContextImpl;
using onnx::FunctionProto;
using onnx::NodeProto;
using onnx::OpSchema;
using onnx::OpSchemaRegistry;
using onnx::TensorProto;
using onnx::TypeProto;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// TEMPORARY diagnostic: the s390x (big-endian) CI job reports this test as
// failing in 0.00 sec with zero captured output (not even the final
// printf on a normal, non-crashing failure) -- consistent with a crash
// early enough that nothing was ever flushed. This breadcrumb, printed
// and flushed before/after each step, is here purely to localize which
// step doesn't return, and should be removed once that's identified.
void Breadcrumb(const char* what) {
  std::fprintf(stderr, "[breadcrumb] %s\n", what);
  std::fflush(stderr);
}

TypeProto MakeFloatType(const std::vector<int64_t>& dims,
                        const std::vector<std::string>& dim_params = {}) {
  TypeProto t;
  auto* tensor = t.mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  auto* shape = tensor->mutable_shape();
  for (size_t i = 0; i < dims.size(); ++i) {
    auto* dim = shape->add_dim();
    if (!dim_params.empty() && !dim_params[i].empty()) {
      dim->set_dim_param(dim_params[i]);
    } else {
      dim->set_dim_value(dims[i]);
    }
  }
  return t;
}

NodeProto MakeMoENode(int64_t k, const std::string& activation_type,
                      int64_t normalize, int64_t use_sparse_mixer = 0,
                      int64_t swiglu_fusion = 0, bool with_fc3 = false) {
  NodeProto node;
  node.set_op_type("MoE");
  node.set_domain("com.microsoft");
  node.add_input("input");
  node.add_input("router_probs");
  node.add_input("fc1_experts_weights");
  node.add_input("");  // fc1_experts_bias: absent
  node.add_input("fc2_experts_weights");
  node.add_input("");  // fc2_experts_bias: absent
  if (with_fc3) {
    node.add_input("fc3_experts_weights");
  }
  node.add_output("output");
  auto add_int_attr = [&](const char* name, int64_t value) {
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::INT);
    attr->set_i(value);
  };
  auto* act = node.add_attribute();
  act->set_name("activation_type");
  act->set_type(AttributeProto::STRING);
  act->set_s(activation_type);
  add_int_attr("k", k);
  add_int_attr("normalize_routing_weights", normalize);
  add_int_attr("use_sparse_mixer", use_sparse_mixer);
  add_int_attr("swiglu_fusion", swiglu_fusion);
  return node;
}

std::vector<TypeProto> MakeInputTypes(int64_t num_experts, int64_t hidden_size,
                                      int64_t inter_size, bool dynamic_experts,
                                      bool with_fc3) {
  std::vector<TypeProto> types;
  types.push_back(MakeFloatType({-1, hidden_size}, {"N", ""}));
  types.push_back(MakeFloatType({-1, num_experts}, {"N", ""}));
  if (dynamic_experts) {
    types.push_back(
        MakeFloatType({-1, inter_size, hidden_size}, {"E", "", ""}));
  } else {
    types.push_back(MakeFloatType({num_experts, inter_size, hidden_size}));
  }
  types.push_back(TypeProto());  // fc1_experts_bias: absent
  types.push_back(MakeFloatType({num_experts, hidden_size, inter_size}));
  types.push_back(TypeProto());  // fc2_experts_bias: absent
  if (with_fc3) {
    types.push_back(MakeFloatType({num_experts, inter_size, hidden_size}));
  }
  return types;
}

void TestMoESchemaIsRegistered() {
  Breadcrumb("TestMoESchemaIsRegistered: before RegisterContribOpSchemas");
  onnxsim::RegisterContribOpSchemas();
  Breadcrumb("TestMoESchemaIsRegistered: after RegisterContribOpSchemas");
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  Check(schema != nullptr, "MoE schema should be registered");
  Check(schema != nullptr && schema->HasContextDependentFunction(),
        "MoE schema should have a context-dependent function body");

  const OpSchema* qmoe = OpSchemaRegistry::Schema("QMoE", 1, "com.microsoft");
  Check(qmoe != nullptr, "QMoE schema should be registered");
  Check(qmoe != nullptr && !qmoe->HasContextDependentFunction(),
        "QMoE should stay schema-only (no reference decomposition)");
}

void TestBuildsForPlainReluCase() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "relu", /*normalize=*/1);
  auto input_types =
      MakeInputTypes(/*num_experts=*/4, /*hidden_size=*/6, /*inter_size=*/8,
                     /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  Breadcrumb(
      "TestBuildsForPlainReluCase: before BuildContextDependentFunction");
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Breadcrumb("TestBuildsForPlainReluCase: after BuildContextDependentFunction");
  Check(built,
        "should build a body for the plain relu/no-bias/static-shape case");
  if (!built) return;

  Check(function_proto.node_size() > 0, "built function should have nodes");
  bool has_default_domain_opset = false, has_ms_domain_opset = false;
  for (const auto& opset : function_proto.opset_import()) {
    if (opset.domain().empty()) has_default_domain_opset = true;
    if (opset.domain() == "com.microsoft") has_ms_domain_opset = true;
  }
  Check(has_default_domain_opset,
        "function must declare an opset_import for the \"\" domain ops it "
        "uses (Softmax, TopK, Gemm, ...)");
  Check(has_ms_domain_opset,
        "function must declare its own com.microsoft opset_import");

  // Every input the underlying MakeMoENode call actually wired up (input,
  // router_probs, fc1/fc2 weights) must be a real, non-empty node input
  // somewhere in the body -- i.e. the formal parameters resolve to
  // something, not dangling references.
  bool references_input = false, references_router_probs = false;
  for (const auto& n : function_proto.node()) {
    for (const auto& in : n.input()) {
      if (in == "input") references_input = true;
      if (in == "router_probs") references_router_probs = true;
    }
  }
  Check(references_input, "body should reference the formal 'input' parameter");
  Check(references_router_probs,
        "body should reference the formal 'router_probs' parameter");
}

void TestDeclinesForSwiglu() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built, "swiglu is not decomposed by this reference body");
}

void TestDeclinesForSparseMixer() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "relu", /*normalize=*/0, /*use_sparse_mixer=*/1);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built,
        "use_sparse_mixer=1 uses a different combination rule, not "
        "decomposed here");
}

void TestDeclinesForFc3() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "relu", /*normalize=*/0, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/0, /*with_fc3=*/true);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/true);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built, "fc3_experts_weights present is only meaningful for swiglu");
}

void TestDeclinesForDynamicExpertCount() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "relu", /*normalize=*/0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/true, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built,
        "an unrolled per-expert body needs num_experts known statically");
}

void TestNodeCountScalesWithExpertCount() {
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  int prev_nodes = -1;
  for (int64_t num_experts : {2, 4, 8}) {
    NodeProto node = MakeMoENode(/*k=*/1, "relu", /*normalize=*/0);
    auto input_types = MakeInputTypes(num_experts, 6, 8,
                                      /*dynamic_experts=*/false,
                                      /*with_fc3=*/false);
    FunctionBodyBuildContextImpl ctx(node, input_types);
    FunctionProto function_proto;
    bool built = schema->BuildContextDependentFunction(ctx, function_proto);
    Check(built, "should build for every tested static expert count");
    if (!built) continue;
    Check(prev_nodes < 0 || function_proto.node_size() > prev_nodes,
          "node count should strictly grow with num_experts (confirms the "
          "body is actually unrolled per expert, not a fixed-size "
          "approximation)");
    prev_nodes = function_proto.node_size();
  }
}

}  // namespace

int main() {
  Breadcrumb("main: start");
  Breadcrumb("main: before TestMoESchemaIsRegistered");
  TestMoESchemaIsRegistered();
  Breadcrumb("main: after TestMoESchemaIsRegistered");
  TestBuildsForPlainReluCase();
  Breadcrumb("main: after TestBuildsForPlainReluCase");
  TestDeclinesForSwiglu();
  Breadcrumb("main: after TestDeclinesForSwiglu");
  TestDeclinesForSparseMixer();
  Breadcrumb("main: after TestDeclinesForSparseMixer");
  TestDeclinesForFc3();
  Breadcrumb("main: after TestDeclinesForFc3");
  TestDeclinesForDynamicExpertCount();
  Breadcrumb("main: after TestDeclinesForDynamicExpertCount");
  TestNodeCountScalesWithExpertCount();
  Breadcrumb("main: after TestNodeCountScalesWithExpertCount");

  if (g_failures == 0) {
    std::printf("contrib_schemas_moe_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "contrib_schemas_moe_test: %d failure(s)\n", g_failures);
  return 1;
}
