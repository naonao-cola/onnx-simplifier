#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sym_expr.h"

// ONNX-free core of the C++ model_info metrics (MACs / FLOPs and the memory
// figures). Everything here operates on ``GraphView`` -- a graph reduced to
// just the tensor shapes, dtypes and node connectivity the metrics need -- so
// it carries no dependency on the ONNX C++ headers and is unit-testable on its
// own. The ONNX glue that builds a ``GraphView`` from a ``GraphProto`` (running
// shape inference first) lives in model_info.cpp.
//
// This mirrors the compute-dominant counters of the Python onnxsim.model_info,
// using SymExpr in place of sympy so a dynamic dimension (dim_param, e.g.
// "batch") stays symbolic and the totals become formulas. The one metric that
// is a ratio, compute_density, uses SymRatio.
namespace onnxsim {

// A tensor's shape: one SymExpr per dimension, concrete (SymExpr(int)) or
// symbolic (SymExpr::Symbol(dim_param)). A tensor whose shape is not fully
// known -- missing, rank-0, or with any dimension that is neither a value nor a
// dim_param -- is simply absent from the ShapeMap, so a name's presence there
// means exactly the Python ``_known`` predicate (all dims known, rank >= 1).
using Shape = std::vector<SymExpr>;
using ShapeMap = std::map<std::string, Shape>;
// Bytes per element of each named tensor; absent when the dtype has no fixed
// width (STRING) or is unmapped, which disables the memory metrics for it.
using DTypeMap = std::map<std::string, int>;

struct GraphView;  // defined below; a NodeView owns its control-flow subgraphs.

// One node reduced to what the metrics need. Free of ONNX types.
struct NodeView {
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::map<std::string, int64_t> attr_ints;  // only the INT attributes
  std::vector<GraphView> subgraphs;          // bodies of If / Loop / Scan ...

  int64_t AttrInt(const std::string& name, int64_t dflt) const;
  // Input/output name at an index, or "" when out of range (optional/omitted
  // operands read as "", which never resolves to a shape).
  const std::string& Input(std::size_t i) const;
  const std::string& Output(std::size_t i) const;
};

// A graph reduced for metric computation: nodes in topological order (as ONNX
// requires), the resolved shapes/dtypes of every fully-known tensor, and the
// input/output/initializer name lists the liveness pass needs.
struct GraphView {
  std::vector<NodeView> nodes;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> initializers;
  ShapeMap shapes;
  DTypeMap dtypes;
};

// MACs and memory-access traffic aggregated over a graph and all its subgraphs.
// FLOPs are 2 * macs; peak footprint is computed separately (see below).
struct Metrics {
  SymExpr macs;
  SymExpr mem_access;
};

// Total MACs and forward-pass memory traffic of a graph (recursing subgraphs).
Metrics ComputeMetrics(const GraphView& graph);

// MACs of a single node (0 for ops without a counter or with unknown shapes),
// and the bytes it reads+writes in a forward pass. Exposed so a caller can
// annotate per-node metrics; ComputeMetrics is these summed over the graph.
SymExpr NodeMacs(const NodeView& node, const ShapeMap& shapes);
SymExpr NodeMemAccess(const NodeView& node, const ShapeMap& shapes,
                      const DTypeMap& dtypes);

// Peak bytes resident during a forward pass: initializers stay resident, each
// activation lives from where it is produced to its last use, and a node's
// control-flow subgraphs add their own peak on top of the live set at that node
// (a conservative bound). Unknown-size tensors contribute 0.
SymExpr PeakMemoryFootprint(const GraphView& graph);

// Size in bytes of a named tensor, or nullopt when its shape or dtype is
// unknown. Symbolic when a dimension is a dim_param.
std::optional<SymExpr> TensorBytes(const std::string& name,
                                   const ShapeMap& shapes,
                                   const DTypeMap& dtypes);

// Byte width of an ONNX TensorProto element type (passed as its raw int enum,
// so this header needs no ONNX include), or nullopt when it has no fixed width
// (STRING / UNDEFINED) or is not mapped (e.g. the sub-byte types).
std::optional<int> ElemSize(int onnx_elem_type);

// Human-readable renderings mirroring model_info.py: a symbolic value prints as
// its factored formula, a concrete one as a scaled number.
std::string HumanReadableNum(const SymExpr& value);   // 1000-based, no suffix
std::string HumanReadableSize(const SymExpr& value);  // 1024-based, "B" suffix
std::string HumanReadableDensity(const SymRatio& value);  // " FLOP/Byte"

}  // namespace onnxsim
