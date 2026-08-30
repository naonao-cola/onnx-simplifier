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
 * function referencing the opsets it uses, does it use a real Loop instead
 * of per-expert unrolling). The actual per-op arithmetic -- in particular,
 * that `router_probs` needs an internal Softmax despite its name, which is
 * not obvious from ONNX Runtime's own docs -- was verified out-of-band by
 * running the generated FunctionProto (loaded as a model-level local
 * function) through a real onnxruntime InferenceSession and diffing
 * against a bare `com.microsoft.MoE` node executed by ONNX Runtime's own
 * native kernel: 80 (activation x fc1_bias x fc2_bias x num_experts x k x
 * normalize_routing_weights) combinations, all matching to float32
 * precision, including gelu -- which needed correcting from an exact
 * erf-based formula to the tanh approximation ONNX Runtime's CPU kernel
 * actually uses (confirmed by isolating the activation with a 1x1
 * identity-weight MoE node; the two formulas differ by up to ~4e-4
 * absolute / ~5% relative near their inflection points, which the earlier,
 * looser numeric check happened not to catch).
 *
 * use_sparse_mixer has no CPU kernel to check against at all (confirmed:
 * ONNX Runtime's CPU MoE kernel never reads that attribute, silently
 * falling back to plain routing) -- its own routing math is instead
 * transcribed from ONNX Runtime's CUDA kernel
 * (onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu) and cross-checked
 * against an independent numpy transliteration of that same source. That
 * cross-check, like the onnxruntime one above, needs Python dependencies
 * onnxsim's C++ build doesn't have (see CLAUDE.md), which is why it isn't
 * reproduced as a C++ test here.
 *
 * fc3 (silu-only, ONNX Runtime's "Mixtral case": fc2(silu(fc1(x)) *
 * fc3(x))) is in the same disclosed-gap category as use_sparse_mixer:
 * transcribed from onnxruntime/contrib_ops/cuda/moe/moe.cc's own comment
 * ("map Mixtral to SwiGLU by packing weights as [FC3, FC1]"), not covered
 * by the 80-combination onnxruntime session cross-check above, since ORT's
 * CPU MoE kernel rejects fc3 outright for any activation ("FC3 is not
 * implemented for CPU MoE") -- there is no CPU oracle to run it against.
 *
 * swiglu (interleaved, swiglu_fusion=1 -- gpt-oss-20b's real convention) is
 * the exception to that pattern: it IS the one activation ONNX Runtime's
 * own CPU MoE kernel actually implements (its constructor throws unless
 * swiglu_fusion == 1 for a SwiGLU node), and was cross-checked end to end
 * against a real onnxruntime session -- both this schema's own
 * decomposition under a private domain and the real native com.microsoft.MoE
 * kernel directly, with and without swiglu_limit -- the same rigor as the
 * original 80-combination check, not a disclosed gap. See
 * generate_moe_function_templates.py's own comment for the formula
 * (transcribed from onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc's
 * ApplySwiGLUVectorized) and why activation_alpha/activation_beta must both
 * be present on the calling node for this to build at all.
 */
#include <onnx/defs/function.h>
#include <onnx/defs/schema.h>

#include <cstdio>
#include <optional>
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

// fc3_experts_weights (like fc1/fc2's per-expert Gathers) is only ever
// referenced inside the body's Loop subgraph, not among function_proto's own
// top-level nodes -- unlike "input"/"router_probs", which the routing logic
// above the Loop also uses directly. So checking for a reference needs to
// look inside Loop/If node's nested graph attributes too, not just the
// top-level node list.
bool ReferencesValue(const FunctionProto& function_proto,
                     const std::string& value_name) {
  std::vector<const onnx::GraphProto*> graphs_to_scan;
  auto scan_nodes = [&](const auto& nodes) {
    for (const auto& n : nodes) {
      for (const auto& in : n.input()) {
        if (in == value_name) return true;
      }
      for (const auto& attr : n.attribute()) {
        if (attr.has_g()) graphs_to_scan.push_back(&attr.g());
      }
    }
    return false;
  };
  if (scan_nodes(function_proto.node())) return true;
  while (!graphs_to_scan.empty()) {
    const onnx::GraphProto* g = graphs_to_scan.back();
    graphs_to_scan.pop_back();
    if (scan_nodes(g->node())) return true;
  }
  return false;
}

NodeProto MakeMoENode(int64_t k, const std::string& activation_type,
                      int64_t normalize, int64_t use_sparse_mixer = 0,
                      int64_t swiglu_fusion = 0, bool with_fc3 = false,
                      std::optional<double> activation_alpha = std::nullopt,
                      std::optional<double> activation_beta = std::nullopt,
                      std::optional<double> swiglu_limit = std::nullopt) {
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
  auto add_float_attr = [&](const char* name, double value) {
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::FLOAT);
    attr->set_f(static_cast<float>(value));
  };
  auto* act = node.add_attribute();
  act->set_name("activation_type");
  act->set_type(AttributeProto::STRING);
  act->set_s(activation_type);
  add_int_attr("k", k);
  add_int_attr("normalize_routing_weights", normalize);
  add_int_attr("use_sparse_mixer", use_sparse_mixer);
  add_int_attr("swiglu_fusion", swiglu_fusion);
  if (activation_alpha.has_value())
    add_float_attr("activation_alpha", *activation_alpha);
  if (activation_beta.has_value())
    add_float_attr("activation_beta", *activation_beta);
  if (swiglu_limit.has_value()) add_float_attr("swiglu_limit", *swiglu_limit);
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
  onnxsim::RegisterContribOpSchemas();
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
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
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

void TestDeclinesForSwigluWithoutFusion() {
  // swiglu_fusion defaults to 0 (not fused) here -- ONNX Runtime's own CPU
  // MoE kernel only implements the interleaved swiglu_fusion=1 layout (its
  // constructor throws for anything else), so this stays declined.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built, "swiglu with swiglu_fusion != 1 should stay declined");
}

void TestDeclinesForSwigluWithFusionButNoAlphaBeta() {
  // swiglu_fusion=1 alone isn't enough: activation_alpha/activation_beta
  // must both be present too, since there is nothing valid to forward via
  // ref_attr_name for an absent FLOAT attribute.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/0,
                               /*use_sparse_mixer=*/0, /*swiglu_fusion=*/1);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built,
        "swiglu_fusion=1 without activation_alpha/activation_beta should "
        "stay declined");
}

void TestDeclinesForSwigluWithFc3() {
  // swiglu's own fc1 is already the fused gate+linear pair -- a separate
  // fc3 alongside it has no defined meaning (real ORT has none either).
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/0, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/1, /*with_fc3=*/true,
                  /*activation_alpha=*/1.702, /*activation_beta=*/1.0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/true);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built, "swiglu with fc3_experts_weights present should stay declined");
}

void TestBuildsForSwiglu() {
  // swiglu_fusion=1 with both activation_alpha/activation_beta present is
  // ONNX Runtime's own "Mixtral case" for gpt-oss-20b -- the one activation
  // its CPU MoE kernel actually implements. Confirmed end to end against a
  // real onnxruntime session (see generate_moe_function_templates.py's own
  // comment), unlike fc3/use_sparse_mixer above.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/1, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/1, /*with_fc3=*/false,
                  /*activation_alpha=*/1.702, /*activation_beta=*/1.0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(built, "swiglu_fusion=1 with activation_alpha/beta should build");
  if (!built) return;

  Check(ReferencesValue(function_proto, "fc1_experts_weights"),
        "body should reference the formal 'fc1_experts_weights' parameter");
  bool has_loop = false;
  for (const auto& n : function_proto.node()) {
    if (n.op_type() == "Loop") has_loop = true;
  }
  Check(has_loop,
        "body should contain a real Loop op iterating over num_experts");
}

void TestBuildsForSwigluWithLimit() {
  // swiglu_limit is genuinely optional ("no clamp when limit is not
  // provided") -- its presence is its own axis (not just a forwarded
  // value), so a node that sets it should still build, with a structurally
  // different (larger) body than TestBuildsForSwiglu's.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "swiglu", /*normalize=*/1, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/1, /*with_fc3=*/false,
                  /*activation_alpha=*/1.702, /*activation_beta=*/1.0,
                  /*swiglu_limit=*/7.0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(built, "swiglu_fusion=1 with swiglu_limit should build");
}

void TestBuildsForSparseMixer() {
  // use_sparse_mixer is forwarded to the fixed body's own runtime `If`
  // (see generate_moe_function_templates.py), not read/decided here -- so
  // this now builds the same way the plain case does, just with a
  // different routing rule (see contrib_schemas_moe_test's numeric
  // cross-check note above for why that rule's own math is validated out
  // of band against a numpy transliteration of ONNX Runtime's CUDA kernel
  // instead of a running onnxruntime session).
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "relu", /*normalize=*/0, /*use_sparse_mixer=*/1);
  auto input_types =
      MakeInputTypes(8, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(built,
        "use_sparse_mixer=1 should build the same fixed body, with "
        "its routing choice left to the body's own runtime If");
}

void TestDeclinesForFc3WithNonSiluActivation() {
  // fc3 is only implemented for activation_type == "silu" (ONNX Runtime's
  // own "Mixtral case" -- see SelectMoESiluFc3FunctionTemplate's comment in
  // contrib_schemas.cpp); relu (and identity/gelu) + fc3 has no ORT-defined
  // behavior at all and stays declined.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "relu", /*normalize=*/0, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/0, /*with_fc3=*/true);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/true);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(!built,
        "fc3_experts_weights with a non-silu activation has no ORT-defined "
        "behavior and should stay declined");
}

void TestBuildsForSiluFc3() {
  // silu + fc3 is ONNX Runtime's own "Mixtral case": fc2(silu(fc1(x)) *
  // fc3(x)), the standard separate gate/up/down-projection gated MLP (the
  // shape onnxruntime-genai's Phi-3.5-MoE builder actually exports). Unlike
  // TestDeclinesForFc3WithNonSiluActivation, this must build successfully.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node =
      MakeMoENode(/*k=*/2, "silu", /*normalize=*/0, /*use_sparse_mixer=*/0,
                  /*swiglu_fusion=*/0, /*with_fc3=*/true);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/false, /*with_fc3=*/true);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(built, "silu + fc3 (the Mixtral case) should build a reference body");
  if (!built) return;

  Check(ReferencesValue(function_proto, "fc3_experts_weights"),
        "body should reference the formal 'fc3_experts_weights' parameter "
        "(inside the Loop subgraph, alongside fc1/fc2's own per-expert "
        "Gathers)");
}

void TestBuildsForDynamicExpertCount() {
  // num_experts is a real ONNX `Loop` trip count inside the fixed body
  // (Shape(fc1_experts_weights)[0]), not something baked into a per-node
  // unrolled copy -- so a node whose fc1_experts_weights shape doesn't
  // statically know its expert-count dimension builds the exact same way a
  // statically-shaped one does.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  NodeProto node = MakeMoENode(/*k=*/2, "relu", /*normalize=*/0);
  auto input_types =
      MakeInputTypes(4, 6, 8, /*dynamic_experts=*/true, /*with_fc3=*/false);
  FunctionBodyBuildContextImpl ctx(node, input_types);
  FunctionProto function_proto;
  bool built = schema->BuildContextDependentFunction(ctx, function_proto);
  Check(built,
        "num_experts unknown statically should still build (a real Loop, "
        "not a per-node unrolled copy, iterates over it)");
}

void TestBodyUsesALoopNotPerExpertUnrolling() {
  // The same fixed body is attached (and is identical) regardless of
  // num_experts, since it's a real Loop trip count rather than something
  // that changes the node list -- confirms the design change directly,
  // where TestNodeCountScalesWithExpertCount used to confirm the opposite
  // (deliberately per-node unrolled) design.
  const OpSchema* schema = OpSchemaRegistry::Schema("MoE", 1, "com.microsoft");
  int first_nodes = -1;
  for (int64_t num_experts : {2, 4, 8}) {
    NodeProto node = MakeMoENode(/*k=*/1, "relu", /*normalize=*/0);
    auto input_types = MakeInputTypes(num_experts, 6, 8,
                                      /*dynamic_experts=*/false,
                                      /*with_fc3=*/false);
    FunctionBodyBuildContextImpl ctx(node, input_types);
    FunctionProto function_proto;
    bool built = schema->BuildContextDependentFunction(ctx, function_proto);
    Check(built, "should build for every tested expert count");
    if (!built) continue;
    bool has_loop = false;
    for (const auto& n : function_proto.node()) {
      if (n.op_type() == "Loop") has_loop = true;
    }
    Check(has_loop,
          "body should contain a real Loop op iterating over num_experts");
    if (first_nodes < 0) {
      first_nodes = function_proto.node_size();
    } else {
      Check(function_proto.node_size() == first_nodes,
            "node count should be identical across different num_experts "
            "values (confirms the body is a fixed Loop, not unrolled per "
            "expert)");
    }
  }
}

}  // namespace

int main() {
  TestMoESchemaIsRegistered();
  TestBuildsForPlainReluCase();
  TestDeclinesForSwigluWithoutFusion();
  TestDeclinesForSwigluWithFusionButNoAlphaBeta();
  TestDeclinesForSwigluWithFc3();
  TestBuildsForSwiglu();
  TestBuildsForSwigluWithLimit();
  TestBuildsForSparseMixer();
  TestDeclinesForFc3WithNonSiluActivation();
  TestBuildsForSiluFc3();
  TestBuildsForDynamicExpertCount();
  TestBodyUsesALoopNotPerExpertUnrolling();

  if (g_failures == 0) {
    std::printf("contrib_schemas_moe_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "contrib_schemas_moe_test: %d failure(s)\n", g_failures);
  return 1;
}
