/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises xnnpack_codegen.{h,cpp} -- the C++ core behind
 * onnxsim.generate_xnnpack_c. Mirrors tests/test_xnnpack_codegen.py's cases
 * at the C++ layer directly. Needs onnx (ModelProto/shape inference), like
 * precision_estimator_test.cpp, so this links the fully-configured `onnxsim`
 * CMake target rather than building standalone.
 */
#include "xnnpack_codegen.h"

#include <onnx/onnx_pb.h>

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using onnxsim::xnnpack_backend::GenerateXnnpackC;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckContains(const std::string& haystack, const std::string& needle,
                   const std::string& what) {
  Check(haystack.find(needle) != std::string::npos,
        what + " (expected to find '" + needle + "')");
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

void AddIntsAttr(onnx::NodeProto& n, const std::string& name,
                 const std::vector<int64_t>& v) {
  auto* a = n.add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INTS);
  for (int64_t x : v) a->add_ints(x);
}

onnx::TensorProto MakeFloatTensor(const std::string& name,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<float>& data) {
  onnx::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : shape) tp.add_dims(d);
  for (float v : data) tp.add_float_data(v);
  return tp;
}

std::vector<float> RandomVec(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> out(n);
  for (auto& v : out) v = dist(rng) * scale;
  return out;
}

onnx::ModelProto MakeModel(const std::vector<onnx::NodeProto>& nodes,
                           const std::vector<onnx::ValueInfoProto>& inputs,
                           const std::vector<onnx::ValueInfoProto>& outputs,
                           const std::vector<onnx::TensorProto>& initializers) {
  onnx::ModelProto m;
  m.set_ir_version(8);
  auto* opset_import = m.add_opset_import();
  opset_import->set_domain("");
  opset_import->set_version(17);
  auto* graph = m.mutable_graph();
  graph->set_name("g");
  for (const auto& n : nodes) *graph->add_node() = n;
  for (const auto& i : inputs) *graph->add_input() = i;
  for (const auto& o : outputs) *graph->add_output() = o;
  for (const auto& t : initializers) *graph->add_initializer() = t;
  return m;
}

// Conv (groups=1): NHWC input/output, OHWI filter, correct padding args.
void TestConvRegular() {
  onnx::NodeProto conv = MakeNode("Conv", {"x", "w", "b"}, {"y"});
  AddIntsAttr(conv, "pads", {1, 1, 1, 1});
  const onnx::ModelProto model = MakeModel(
      {conv}, {MakeValueInfo("x", {1, 3, 8, 8})},
      {MakeValueInfo("y", {1, 4, 8, 8})},
      {MakeFloatTensor("w", {4, 3, 3, 3}, RandomVec(4 * 3 * 3 * 3, 0.1f, 1)),
       MakeFloatTensor("b", {4}, RandomVec(4, 0.1f, 2))});

  const std::string src = GenerateXnnpackC(model, "m");
  CheckContains(
      src,
      "xnn_define_convolution_2d(sg, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 4,",
      "TestConvRegular: conv call args");
  CheckContains(src, "size_t dims[] = {1, 8, 8, 3};",
                "TestConvRegular: NHWC input shape");
  CheckContains(src, "size_t dims[] = {4, 3, 3, 3};",
                "TestConvRegular: OHWI filter shape");
}

// Conv with groups == Cin, one input channel per group -> depthwise path.
void TestConvDepthwise() {
  onnx::NodeProto conv = MakeNode("Conv", {"x", "w"}, {"y"});
  AddIntsAttr(conv, "pads", {1, 1, 1, 1});
  AddIntAttr(conv, "group", 4);
  const onnx::ModelProto model = MakeModel(
      {conv}, {MakeValueInfo("x", {1, 4, 8, 8})},
      {MakeValueInfo("y", {1, 8, 8, 8})},
      {MakeFloatTensor("w", {8, 1, 3, 3}, RandomVec(8 * 1 * 3 * 3, 0.1f, 3))});

  const std::string src = GenerateXnnpackC(model, "m");
  CheckContains(src, "xnn_define_depthwise_convolution_2d(",
                "TestConvDepthwise: uses depthwise op");
  Check(src.find("xnn_define_convolution_2d(") == std::string::npos,
        "TestConvDepthwise: must not also use the regular conv op");
  CheckContains(src, "size_t dims[] = {1, 3, 3, 8};",
                "TestConvDepthwise: [1,KH,KW,Cin*mult] filter shape");
}

void TestUnsupportedOpThrows() {
  const onnx::ModelProto model = MakeModel(
      {MakeNode("Identity", {"x"}, {"y"})}, {MakeValueInfo("x", {1, 3, 4, 4})},
      {MakeValueInfo("y", {1, 3, 4, 4})}, {});
  bool threw = false;
  try {
    GenerateXnnpackC(model, "m");
  } catch (const std::runtime_error& e) {
    threw = true;
    CheckContains(e.what(), "Identity",
                  "TestUnsupportedOpThrows: message names the op");
  }
  Check(threw, "TestUnsupportedOpThrows: must throw");
}

void TestDynamicShapeThrows() {
  onnx::ValueInfoProto x = MakeValueInfo("x", {});
  x.mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_param("batch");
  x.mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_value(3);
  x.mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_value(4);
  x.mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_value(4);
  onnx::ValueInfoProto y = x;
  y.set_name("y");
  const onnx::ModelProto model =
      MakeModel({MakeNode("Relu", {"x"}, {"y"})}, {x}, {y}, {});
  bool threw = false;
  try {
    GenerateXnnpackC(model, "m");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "TestDynamicShapeThrows: must throw for a symbolic dimension");
}

void TestInvalidFunctionPrefixThrows() {
  const onnx::ModelProto model = MakeModel(
      {MakeNode("Relu", {"x"}, {"y"})}, {MakeValueInfo("x", {1, 3, 4, 4})},
      {MakeValueInfo("y", {1, 3, 4, 4})}, {});
  bool threw = false;
  try {
    GenerateXnnpackC(model, "1bad");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw,
        "TestInvalidFunctionPrefixThrows: must throw std::invalid_argument");
}

// Multi-pixel spatial Reshape after a Conv is a layout-unsafe pattern (see
// xnnpack_codegen.h's module comment) and must be rejected, not silently
// mis-ordered.
void TestLayoutUnsafeReshapeThrows() {
  onnx::NodeProto conv = MakeNode("Conv", {"x", "w"}, {"c1"});
  AddIntsAttr(conv, "pads", {1, 1, 1, 1});
  onnx::NodeProto reshape = MakeNode("Reshape", {"c1", "shape"}, {"y"});
  const onnx::ModelProto model = MakeModel(
      {conv, reshape}, {MakeValueInfo("x", {1, 3, 8, 8})},
      {MakeValueInfo("y", {1, 128})},
      {MakeFloatTensor("w", {2, 3, 3, 3}, RandomVec(2 * 3 * 3 * 3, 0.1f, 4)),
       [] {
         onnx::TensorProto tp;
         tp.set_name("shape");
         tp.set_data_type(onnx::TensorProto::INT64);
         tp.add_dims(2);
         tp.add_int64_data(1);
         tp.add_int64_data(128);
         return tp;
       }()});
  bool threw = false;
  try {
    GenerateXnnpackC(model, "m");
  } catch (const std::runtime_error& e) {
    threw = true;
    CheckContains(e.what(), "not supported in v1",
                  "TestLayoutUnsafeReshapeThrows: explanatory message");
  }
  Check(threw, "TestLayoutUnsafeReshapeThrows: must throw");
}

}  // namespace

int main() {
  TestConvRegular();
  TestConvDepthwise();
  TestUnsupportedOpThrows();
  TestDynamicShapeThrows();
  TestInvalidFunctionPrefixThrows();
  TestLayoutUnsafeReshapeThrows();
  if (g_failures == 0) {
    std::printf("all xnnpack_codegen tests passed\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", g_failures);
  return 1;
}
