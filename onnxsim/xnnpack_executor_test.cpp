/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises GetXnnpackModelExecutor() (onnxsim/xnnpack_executor.cpp) end to
 * end: builds a small ONNX ModelProto by hand (mirroring
 * precision_estimator_test.cpp's MakeNode/MakeModel helpers), feeds it
 * through the real ModelExecutor::Run interface with DLManagedTensor inputs,
 * and checks the resulting DLManagedTensor outputs against values computed
 * independently in plain C++ -- a from-scratch numeric cross-check of the
 * ONNX -> XNNPACK Subgraph API lowering (onnx_to_xnnpack_subgraph.cpp), not
 * just "did it not crash". Needs onnx + XNNPACK configured, so this links
 * the fully-configured `onnxsim` CMake target like precision_estimator_test.
 */
#include <onnx/onnx_pb.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"
#include "onnxsim.h"

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckClose(double a, double b, const std::string& what,
               double atol = 1e-4) {
  Check(std::fabs(a - b) <= atol, what + " (got " + std::to_string(a) +
                                      ", want " + std::to_string(b) + ")");
}

onnx::TensorProto MakeFloatInitializer(const std::string& name,
                                       const std::vector<int64_t>& dims,
                                       const std::vector<float>& data) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(std::string(reinterpret_cast<const char*>(data.data()),
                             data.size() * sizeof(float)));
  return t;
}

onnx::TensorProto MakeInt64Initializer(const std::string& name,
                                       const std::vector<int64_t>& dims,
                                       const std::vector<int64_t>& data) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(std::string(reinterpret_cast<const char*>(data.data()),
                             data.size() * sizeof(int64_t)));
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

onnx::ModelProto MakeModel(const std::vector<onnx::NodeProto>& nodes,
                           const std::vector<onnx::ValueInfoProto>& inputs,
                           const std::vector<onnx::ValueInfoProto>& outputs,
                           const std::vector<onnx::TensorProto>& initializers) {
  onnx::ModelProto m;
  m.set_ir_version(10);
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

// Wraps a caller-owned buffer as a non-owning DLManagedTensor (deleter left
// null): the test always keeps the backing std::vector alive for at least as
// long as the Run() call, matching ModelExecutor::Run's "inputs are borrowed"
// contract, so nothing needs to free it via DLPack.
DLManagedTensor MakeInputTensor(std::vector<int64_t>& shape,
                                std::vector<float>& data) {
  DLManagedTensor t{};
  t.dl_tensor.data = data.data();
  t.dl_tensor.device = DLDevice{kDLCPU, 0};
  t.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  t.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  t.dl_tensor.shape = shape.data();
  t.dl_tensor.strides = nullptr;
  t.dl_tensor.byte_offset = 0;
  return t;
}

DLManagedTensor MakeInputTensorI64(std::vector<int64_t>& shape,
                                   std::vector<int64_t>& data) {
  DLManagedTensor t{};
  t.dl_tensor.data = data.data();
  t.dl_tensor.device = DLDevice{kDLCPU, 0};
  t.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  t.dl_tensor.dtype = DLDataType{kDLInt, 64, 1};
  t.dl_tensor.shape = shape.data();
  t.dl_tensor.strides = nullptr;
  t.dl_tensor.byte_offset = 0;
  return t;
}

const float* OutData(const DLManagedTensorPtr& t) {
  return static_cast<const float*>(t->dl_tensor.data);
}

void TestAddThenReluBroadcast() {
  // X: [2,3], Y: [3] (broadcasts over the leading dim) -> Add -> Relu.
  auto model = MakeModel(
      {MakeNode("Add", {"X", "Y"}, {"sum"}), MakeNode("Relu", {"sum"}, {"Z"})},
      {MakeValueInfo("X", {2, 3}), MakeValueInfo("Y", {3})},
      {MakeValueInfo("Z", {2, 3})}, {});

  std::vector<int64_t> x_shape{2, 3};
  std::vector<float> x_data{1, -5, 3, -1, 2, -3};
  std::vector<int64_t> y_shape{3};
  std::vector<float> y_data{0, 1, -1};
  DLManagedTensor x = MakeInputTensor(x_shape, x_data);
  DLManagedTensor y = MakeInputTensor(y_shape, y_data);

  auto outputs =
      GetXnnpackModelExecutor()->Run(model, {&x, &y});
  Check(outputs.size() == 1, "Add+Relu: one output");
  if (outputs.empty()) return;
  Check(outputs[0]->dl_tensor.ndim == 2, "Add+Relu: output rank");
  const float* out = OutData(outputs[0]);
  const float expected[6] = {1, 0, 2, 0, 3, 0};
  for (int i = 0; i < 6; ++i) {
    CheckClose(out[i], expected[i],
              "Add+Relu: element " + std::to_string(i));
  }
}

void TestSigmoid() {
  auto model = MakeModel({MakeNode("Sigmoid", {"X"}, {"Y"})},
                         {MakeValueInfo("X", {3})}, {MakeValueInfo("Y", {3})},
                         {});
  std::vector<int64_t> x_shape{3};
  std::vector<float> x_data{0.0f, 2.0f, -2.0f};
  DLManagedTensor x = MakeInputTensor(x_shape, x_data);

  auto outputs = GetXnnpackModelExecutor()->Run(model, {&x});
  Check(outputs.size() == 1, "Sigmoid: one output");
  if (outputs.empty()) return;
  const float* out = OutData(outputs[0]);
  for (int i = 0; i < 3; ++i) {
    const double expected = 1.0 / (1.0 + std::exp(-x_data[i]));
    CheckClose(out[i], expected, "Sigmoid: element " + std::to_string(i));
  }
}

void TestGemmWithBiasAndTransB() {
  // A: [2,3] input. B: [4,3] initializer (transB=1, so B is already
  // [output_channels, input_channels]). C (bias): [4] initializer.
  // Y = A @ B^T + C, shape [2,4].
  onnx::NodeProto gemm = MakeNode("Gemm", {"A", "B", "C"}, {"Y"});
  AddIntAttr(gemm, "transB", 1);
  const std::vector<float> b_data{
      1, 0, 0,   //
      0, 1, 0,   //
      0, 0, 1,   //
      1, 1, 1};  // [4,3]
  const std::vector<float> c_data{10, 20, 30, 40};
  auto model =
      MakeModel({gemm}, {MakeValueInfo("A", {2, 3})}, {MakeValueInfo("Y", {2, 4})},
               {MakeFloatInitializer("B", {4, 3}, b_data),
                MakeFloatInitializer("C", {4}, c_data)});

  std::vector<int64_t> a_shape{2, 3};
  std::vector<float> a_data{1, 2, 3, 4, 5, 6};
  DLManagedTensor a = MakeInputTensor(a_shape, a_data);

  auto outputs = GetXnnpackModelExecutor()->Run(model, {&a});
  Check(outputs.size() == 1, "Gemm: one output");
  if (outputs.empty()) return;
  Check(outputs[0]->dl_tensor.shape[0] == 2 &&
           outputs[0]->dl_tensor.shape[1] == 4,
       "Gemm: output shape");
  const float* out = OutData(outputs[0]);
  // Row 0 of A is [1,2,3]: B^T columns are the standard basis e0,e1,e2 plus
  // an all-ones column, so A@B^T row 0 is [1, 2, 3, 6], plus bias.
  const float expected[8] = {1 + 10, 2 + 20, 3 + 30, 6 + 40,
                             4 + 10, 5 + 20, 6 + 30, 15 + 40};
  for (int i = 0; i < 8; ++i) {
    CheckClose(out[i], expected[i], "Gemm: element " + std::to_string(i));
  }
}

void TestMatMul2D() {
  // A: [2,3] input, B: [3,2] initializer -> Y: [2,2].
  const std::vector<float> b_data{1, 4, 2, 5, 3, 6};  // [3,2]
  auto model = MakeModel({MakeNode("MatMul", {"A", "B"}, {"Y"})},
                         {MakeValueInfo("A", {2, 3})},
                         {MakeValueInfo("Y", {2, 2})},
                         {MakeFloatInitializer("B", {3, 2}, b_data)});

  std::vector<int64_t> a_shape{2, 3};
  std::vector<float> a_data{1, 0, 0, 0, 1, 0};
  DLManagedTensor a = MakeInputTensor(a_shape, a_data);

  auto outputs = GetXnnpackModelExecutor()->Run(model, {&a});
  Check(outputs.size() == 1, "MatMul: one output");
  if (outputs.empty()) return;
  const float* out = OutData(outputs[0]);
  // Row 0 = [1,0,0] selects B's first row [1,4]; row 1 = [0,1,0] selects
  // B's second row [2,5].
  const float expected[4] = {1, 4, 2, 5};
  for (int i = 0; i < 4; ++i) {
    CheckClose(out[i], expected[i], "MatMul: element " + std::to_string(i));
  }
}

void TestReshapeWithInferredDim() {
  // X: [2,3,4] (24 elements) reshaped via a [-1, 4] initializer to [6,4].
  // Reshape never reorders memory, so the flattened data must be unchanged.
  auto model = MakeModel(
      {MakeNode("Reshape", {"X", "shape"}, {"Y"})},
      {MakeValueInfo("X", {2, 3, 4})}, {MakeValueInfo("Y", {6, 4})},
      {MakeInt64Initializer("shape", {2}, {-1, 4})});

  std::vector<int64_t> x_shape{2, 3, 4};
  std::vector<float> x_data(24);
  for (int i = 0; i < 24; ++i) x_data[i] = static_cast<float>(i);
  DLManagedTensor x = MakeInputTensor(x_shape, x_data);

  auto outputs = GetXnnpackModelExecutor()->Run(model, {&x});
  Check(outputs.size() == 1, "Reshape: one output");
  if (outputs.empty()) return;
  Check(outputs[0]->dl_tensor.ndim == 2 &&
           outputs[0]->dl_tensor.shape[0] == 6 &&
           outputs[0]->dl_tensor.shape[1] == 4,
       "Reshape: resolved output shape");
  const float* out = OutData(outputs[0]);
  for (int i = 0; i < 24; ++i) {
    CheckClose(out[i], x_data[i], "Reshape: element " + std::to_string(i));
  }
}

void TestUnsupportedOpThrows() {
  auto model = MakeModel({MakeNode("Conv", {"X", "W"}, {"Y"})},
                         {MakeValueInfo("X", {1, 1, 4, 4})},
                         {MakeValueInfo("Y", {1, 1, 4, 4})},
                         {MakeFloatInitializer("W", {1, 1, 3, 3},
                                               std::vector<float>(9, 1.0f))});
  std::vector<int64_t> x_shape{1, 1, 4, 4};
  std::vector<float> x_data(16, 1.0f);
  DLManagedTensor x = MakeInputTensor(x_shape, x_data);

  bool threw = false;
  try {
    GetXnnpackModelExecutor()->Run(model, {&x});
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw, "unsupported op (Conv): Run() throws rather than "
              "silently mis-executing");
}

}  // namespace

int main() {
  TestAddThenReluBroadcast();
  TestSigmoid();
  TestGemmWithBiasAndTransB();
  TestMatMul2D();
  TestReshapeWithInferredDim();
  TestUnsupportedOpThrows();

  if (g_failures == 0) {
    std::printf("xnnpack_executor_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "xnnpack_executor_test: %d failure(s)\n", g_failures);
  return 1;
}
