// Standalone test for the ONNX-free metric core:
//   g++ -std=c++20 model_metrics.cpp sym_expr.cpp model_metrics_test.cpp -o t
//   && ./t
#include "model_metrics.h"

#include <cassert>
#include <iostream>

#include "sym_expr.h"

namespace {

using onnxsim::ComputeMetrics;
using onnxsim::ElemSize;
using onnxsim::GraphView;
using onnxsim::HumanReadableDensity;
using onnxsim::HumanReadableNum;
using onnxsim::HumanReadableSize;
using onnxsim::Metrics;
using onnxsim::NodeView;
using onnxsim::PeakMemoryFootprint;
using onnxsim::Shape;
using onnxsim::SymExpr;
using onnxsim::SymRatio;

SymExpr Sym(const std::string& n) { return SymExpr::Symbol(n); }

void TestConvMacs() {
  // output [1,4,8,8], weight [4,3,3,3]: MACs = prod(out)*in_per_group*prod(k)
  //   = (1*4*8*8) * 3 * (3*3) = 1*4*8*8*3*9, matching test_model_info.py.
  GraphView g;
  g.shapes["w"] = {SymExpr(4), SymExpr(3), SymExpr(3), SymExpr(3)};
  g.shapes["y"] = {SymExpr(1), SymExpr(4), SymExpr(8), SymExpr(8)};
  NodeView conv;
  conv.op_type = "Conv";
  conv.inputs = {"x", "w"};
  conv.outputs = {"y"};
  g.nodes = {conv};
  const Metrics m = ComputeMetrics(g);
  assert(!m.macs.is_symbolic());
  assert(m.macs.to_int() == 1LL * 4 * 8 * 8 * 3 * 9);
}

void TestGemmMacs() {
  // a [M,K], b [K,N] (no transpose) -> M*N*K.
  GraphView g;
  g.shapes["a"] = {SymExpr(16), SymExpr(32)};
  g.shapes["b"] = {SymExpr(32), SymExpr(8)};
  NodeView gemm;
  gemm.op_type = "Gemm";
  gemm.inputs = {"a", "b"};
  gemm.outputs = {"y"};
  g.nodes = {gemm};
  assert(ComputeMetrics(g).macs.to_int() == 16LL * 8 * 32);

  // transB=1: b is [N,K].
  g.shapes["b"] = {SymExpr(8), SymExpr(32)};
  g.nodes[0].attr_ints["transB"] = 1;
  assert(ComputeMetrics(g).macs.to_int() == 16LL * 8 * 32);
}

void TestMatMulSymbolic() {
  // a [batch, M, K], output [batch, M, N] -> prod(out)*K = batch*M*N*K.
  GraphView g;
  g.shapes["a"] = {Sym("batch"), SymExpr(4), SymExpr(32)};
  g.shapes["y"] = {Sym("batch"), SymExpr(4), SymExpr(8)};
  NodeView mm;
  mm.op_type = "MatMul";
  mm.inputs = {"a", "b"};
  mm.outputs = {"y"};
  g.nodes = {mm};
  const SymExpr macs = ComputeMetrics(g).macs;
  assert(macs.is_symbolic());
  // batch * (4*8*32) = 1024*batch.
  assert(macs.str() == "1024*batch");
}

void TestAttention4D() {
  // q,k,v all [B,H,S,D] -> B*H*S*S*D (QK^T) + B*H*S*S*D (PV) = 2*B*H*S*S*D.
  GraphView g;
  g.shapes["q"] = {SymExpr(2), SymExpr(4), SymExpr(8), SymExpr(16)};
  g.shapes["k"] = {SymExpr(2), SymExpr(4), SymExpr(8), SymExpr(16)};
  g.shapes["v"] = {SymExpr(2), SymExpr(4), SymExpr(8), SymExpr(16)};
  NodeView attn;
  attn.op_type = "Attention";
  attn.inputs = {"q", "k", "v"};
  attn.outputs = {"y"};
  g.nodes = {attn};
  assert(ComputeMetrics(g).macs.to_int() == 2LL * 2 * 4 * 8 * 8 * 16);
}

void TestMemAccessAndPeak() {
  // A single Conv: x [1,3,8,8] f32, w [4,3,3,3] f32 (weight), y [1,4,8,8] f32.
  GraphView g;
  g.shapes["x"] = {SymExpr(1), SymExpr(3), SymExpr(8), SymExpr(8)};
  g.shapes["w"] = {SymExpr(4), SymExpr(3), SymExpr(3), SymExpr(3)};
  g.shapes["y"] = {SymExpr(1), SymExpr(4), SymExpr(8), SymExpr(8)};
  for (const char* n : {"x", "w", "y"}) g.dtypes[n] = 4;  // float32
  g.inputs = {"x"};
  g.outputs = {"y"};
  g.initializers = {"w"};
  NodeView conv;
  conv.op_type = "Conv";
  conv.inputs = {"x", "w"};
  conv.outputs = {"y"};
  g.nodes = {conv};

  const int64_t xb = 1LL * 3 * 8 * 8 * 4;  // 768
  const int64_t wb = 4LL * 3 * 3 * 3 * 4;  // 432
  const int64_t yb = 1LL * 4 * 8 * 8 * 4;  // 1024
  const Metrics m = ComputeMetrics(g);
  assert(m.mem_access.to_int() == xb + wb + yb);  // read x + w, write y

  // Peak: weight resident (wb) + x live + y live once produced = wb + xb + yb.
  assert(PeakMemoryFootprint(g).to_int() == wb + xb + yb);
}

void TestFormatting() {
  assert(HumanReadableNum(SymExpr(1500)) == "1.5K");
  assert(HumanReadableSize(SymExpr(1536)) == "1.5KiB");
  // symbolic -> factored formula
  const SymExpr s = SymExpr(512) * Sym("batch") * Sym("seq") * Sym("seq") +
                    SymExpr(5419008) * Sym("batch");
  assert(HumanReadableNum(s) == "512*batch*(seq**2 + 10584)");
  // density: numeric and symbolic
  assert(HumanReadableDensity(SymRatio(SymExpr(300), SymExpr(100))) ==
         "3.00 FLOP/Byte");
  assert(HumanReadableDensity(SymRatio(SymExpr(2) * Sym("seq"), SymExpr(1))) ==
         "2*seq FLOP/Byte");

  assert(ElemSize(1) == 4);  // FLOAT
  assert(ElemSize(7) == 8);  // INT64
  assert(!ElemSize(8));      // STRING -> no fixed width
}

}  // namespace

int main() {
  TestConvMacs();
  TestGemmMacs();
  TestMatMulSymbolic();
  TestAttention4D();
  TestMemAccessAndPeak();
  TestFormatting();
  std::cout << "all model_metrics tests passed\n";
  return 0;
}
