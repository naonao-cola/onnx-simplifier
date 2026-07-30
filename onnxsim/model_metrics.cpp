#include "model_metrics.h"

#include <cmath>
#include <cstdio>
#include <set>

namespace onnxsim {

namespace {

// --- shape helpers -----------------------------------------------------------

// Shape of a named tensor, or nullptr when it is unknown (absent from the map).
// An empty name (optional/omitted operand) never resolves.
const Shape* Lookup(const ShapeMap& shapes, const std::string& name) {
  if (name.empty()) return nullptr;
  auto it = shapes.find(name);
  return it == shapes.end() ? nullptr : &it->second;
}

SymExpr ProdRange(const Shape& shape, std::size_t begin, std::size_t end) {
  SymExpr result(1);
  for (std::size_t i = begin; i < end && i < shape.size(); ++i) {
    result = result * shape[i];
  }
  return result;
}

SymExpr Prod(const Shape& shape) { return ProdRange(shape, 0, shape.size()); }

// --- per-op MAC counters -----------------------------------------------------
// Each returns the MACs of a single node, or 0 when a required shape is
// unknown. These mirror onnxsim.model_info's counters one-for-one.

// weight: [out_channels, in_channels / group, *kernel]; output has the batch
// and out_channels. QLinearConv puts the weight at input[3]; everything else
// at input[1].
SymExpr ConvImpl(const NodeView& node, const ShapeMap& shapes,
                 std::size_t weight_idx) {
  const Shape* weight = Lookup(shapes, node.Input(weight_idx));
  const Shape* output = Lookup(shapes, node.Output(0));
  if (weight == nullptr || output == nullptr || weight->size() < 2) {
    return SymExpr(0);
  }
  const SymExpr in_channels_per_group = (*weight)[1];
  return Prod(*output) * in_channels_per_group *
         ProdRange(*weight, 2, weight->size());
}

SymExpr ConvMacs(const NodeView& node, const ShapeMap& shapes) {
  return ConvImpl(node, shapes, /*weight_idx=*/1);
}

SymExpr QLinearConvMacs(const NodeView& node, const ShapeMap& shapes) {
  return ConvImpl(node, shapes, /*weight_idx=*/3);
}

SymExpr ConvTransposeMacs(const NodeView& node, const ShapeMap& shapes) {
  // weight: [in_channels, out_channels / group, *kernel]; input has the batch.
  const Shape* x = Lookup(shapes, node.Input(0));
  const Shape* weight = Lookup(shapes, node.Input(1));
  if (x == nullptr || weight == nullptr || weight->size() < 2) {
    return SymExpr(0);
  }
  const SymExpr out_channels_per_group = (*weight)[1];
  return Prod(*x) * out_channels_per_group *
         ProdRange(*weight, 2, weight->size());
}

SymExpr GemmMacs(const NodeView& node, const ShapeMap& shapes) {
  const Shape* a = Lookup(shapes, node.Input(0));
  const Shape* b = Lookup(shapes, node.Input(1));
  if (a == nullptr || b == nullptr || a->size() != 2 || b->size() != 2) {
    return SymExpr(0);
  }
  const bool trans_a = node.AttrInt("transA", 0) != 0;
  const bool trans_b = node.AttrInt("transB", 0) != 0;
  const SymExpr m = trans_a ? (*a)[1] : (*a)[0];
  const SymExpr k = trans_a ? (*a)[0] : (*a)[1];
  const SymExpr n = trans_b ? (*b)[0] : (*b)[1];
  return m * n * k;
}

SymExpr MatMulMacs(const NodeView& node, const ShapeMap& shapes) {
  // output: [*batch, M, N]; the contraction dim K is the last dim of input A.
  const Shape* a = Lookup(shapes, node.Input(0));
  const Shape* output = Lookup(shapes, node.Output(0));
  if (a == nullptr || output == nullptr || a->empty()) {
    return SymExpr(0);
  }
  const SymExpr k = a->back();
  return Prod(*output) * k;
}

// Scaled dot-product attention (ai.onnx opset 23+): the two batched matmuls
// QK^T and (softmax)V dominate. See the Python _attention_macs for the shape
// conventions; in the 3D packed form the per-head sizes require concrete head
// dims, so a symbolic packed dimension falls back to 0 (best-effort).
SymExpr AttentionMacs(const NodeView& node, const ShapeMap& shapes) {
  const Shape* q = Lookup(shapes, node.Input(0));
  const Shape* k = Lookup(shapes, node.Input(1));
  const Shape* v = Lookup(shapes, node.Input(2));
  if (q == nullptr || k == nullptr || v == nullptr) {
    return SymExpr(0);
  }

  SymExpr batch, q_heads, sq, skv, d, dv;
  if (q->size() == 4 && k->size() == 4 && v->size() == 4) {
    batch = (*q)[0];
    q_heads = (*q)[1];
    sq = (*q)[2];
    d = (*q)[3];
    skv = (*k)[2];
    dv = (*v)[3];
  } else if (q->size() == 3 && k->size() == 3 && v->size() == 3) {
    const int64_t qh = node.AttrInt("q_num_heads", 0);
    const int64_t kvh = node.AttrInt("kv_num_heads", 0);
    if (qh <= 0 || kvh <= 0) return SymExpr(0);  // head split unknowable
    const SymExpr& k_packed = (*k)[2];
    const SymExpr& v_packed = (*v)[2];
    if (k_packed.is_symbolic() || v_packed.is_symbolic()) {
      return SymExpr(0);  // cannot split a symbolic packed (heads*size) dim
    }
    batch = (*q)[0];
    sq = (*q)[1];
    skv = (*k)[1];
    q_heads = SymExpr(qh);
    d = SymExpr(k_packed.to_int() / kvh);
    dv = SymExpr(v_packed.to_int() / kvh);
  } else {
    return SymExpr(0);
  }

  const SymExpr qk = batch * q_heads * sq * skv * d;
  const SymExpr pv = batch * q_heads * sq * skv * dv;
  return qk + pv;
}

using Counter = SymExpr (*)(const NodeView&, const ShapeMap&);

const std::map<std::string, Counter>& Counters() {
  // Built once; const after init, so lookups are thread-safe.
  static const std::map<std::string, Counter> kCounters = {
      {"Conv", ConvMacs},
      {"ConvInteger", ConvMacs},
      {"QLinearConv", QLinearConvMacs},
      {"ConvTranspose", ConvTransposeMacs},
      {"Gemm", GemmMacs},
      {"MatMul", MatMulMacs},
      {"MatMulInteger", MatMulMacs},
      {"QLinearMatMul", MatMulMacs},
      {"Attention", AttentionMacs},
  };
  return kCounters;
}

// Larger of two possibly-symbolic values by representative magnitude (all free
// dims -> 1); the winner is returned intact so a symbolic peak stays a formula.
SymExpr Max(const SymExpr& a, const SymExpr& b) {
  return a.representative() >= b.representative() ? a : b;
}

}  // namespace

SymExpr NodeMacs(const NodeView& node, const ShapeMap& shapes) {
  const auto& counters = Counters();
  auto it = counters.find(node.op_type);
  return it == counters.end() ? SymExpr(0) : it->second(node, shapes);
}

// Bytes a node moves: every input read plus every output written; unknown-size
// tensors contribute nothing.
SymExpr NodeMemAccess(const NodeView& node, const ShapeMap& shapes,
                      const DTypeMap& dtypes) {
  SymExpr total(0);
  for (const std::string& name : node.inputs) {
    if (auto bytes = TensorBytes(name, shapes, dtypes)) total = total + *bytes;
  }
  for (const std::string& name : node.outputs) {
    if (auto bytes = TensorBytes(name, shapes, dtypes)) total = total + *bytes;
  }
  return total;
}

int64_t NodeView::AttrInt(const std::string& name, int64_t dflt) const {
  auto it = attr_ints.find(name);
  return it == attr_ints.end() ? dflt : it->second;
}

const std::string& NodeView::Input(std::size_t i) const {
  static const std::string kEmpty;
  return i < inputs.size() ? inputs[i] : kEmpty;
}

const std::string& NodeView::Output(std::size_t i) const {
  static const std::string kEmpty;
  return i < outputs.size() ? outputs[i] : kEmpty;
}

std::optional<SymExpr> TensorBytes(const std::string& name,
                                   const ShapeMap& shapes,
                                   const DTypeMap& dtypes) {
  const Shape* shape = Lookup(shapes, name);
  auto dtype = dtypes.find(name);
  if (shape == nullptr || dtype == dtypes.end()) return std::nullopt;
  return Prod(*shape) * SymExpr(dtype->second);
}

std::optional<int> ElemSize(int t) {
  switch (t) {
    case 1:   // FLOAT
    case 6:   // INT32
    case 12:  // UINT32
      return 4;
    case 2:   // UINT8
    case 3:   // INT8
    case 9:   // BOOL
    case 17:  // FLOAT8E4M3FN
    case 18:  // FLOAT8E4M3FNUZ
    case 19:  // FLOAT8E5M2
    case 20:  // FLOAT8E5M2FNUZ
      return 1;
    case 4:   // UINT16
    case 5:   // INT16
    case 10:  // FLOAT16
    case 16:  // BFLOAT16
      return 2;
    case 7:   // INT64
    case 11:  // DOUBLE
    case 13:  // UINT64
    case 14:  // COMPLEX64
      return 8;
    case 15:  // COMPLEX128
      return 16;
    default:  // UNDEFINED, STRING, and the sub-byte / unmapped types
      return std::nullopt;
  }
}

Metrics ComputeMetrics(const GraphView& graph) {
  Metrics metrics;  // macs / mem_access default to 0
  for (const NodeView& node : graph.nodes) {
    metrics.macs = metrics.macs + NodeMacs(node, graph.shapes);
    metrics.mem_access =
        metrics.mem_access + NodeMemAccess(node, graph.shapes, graph.dtypes);
    for (const GraphView& sub : node.subgraphs) {
      const Metrics sub_metrics = ComputeMetrics(sub);
      metrics.macs = metrics.macs + sub_metrics.macs;
      metrics.mem_access = metrics.mem_access + sub_metrics.mem_access;
    }
  }
  return metrics;
}

SymExpr PeakMemoryFootprint(const GraphView& graph) {
  auto nbytes = [&](const std::string& name) -> SymExpr {
    if (auto bytes = TensorBytes(name, graph.shapes, graph.dtypes))
      return *bytes;
    return SymExpr(0);
  };

  const std::set<std::string> weights(graph.initializers.begin(),
                                      graph.initializers.end());
  SymExpr resident(0);
  for (const std::string& w : weights) resident = resident + nbytes(w);

  // Last node index that consumes each tensor; graph outputs are "consumed" at
  // the end so they stay live for the whole pass.
  std::map<std::string, int> last_use;
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    for (const std::string& name : graph.nodes[i].inputs) {
      if (!name.empty()) last_use[name] = i;
    }
  }
  const int end = static_cast<int>(graph.nodes.size());
  for (const std::string& out : graph.outputs) last_use[out] = end;

  auto live_bytes = [&](const std::set<std::string>& live) -> SymExpr {
    SymExpr total = resident;
    for (const std::string& name : live) total = total + nbytes(name);
    return total;
  };

  std::set<std::string> live;
  for (const std::string& in : graph.inputs) {
    if (weights.find(in) == weights.end()) live.insert(in);
  }
  SymExpr peak = live_bytes(live);

  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    const NodeView& node = graph.nodes[i];
    for (const std::string& name : node.outputs) {
      if (!name.empty() && weights.find(name) == weights.end())
        live.insert(name);
    }
    SymExpr current = live_bytes(live);
    for (const GraphView& sub : node.subgraphs) {
      current = current + PeakMemoryFootprint(sub);
    }
    peak = Max(peak, current);
    std::vector<std::string> expired;
    for (const std::string& name : live) {
      auto it = last_use.find(name);
      if (it != last_use.end() && it->second == i) expired.push_back(name);
    }
    for (const std::string& name : expired) live.erase(name);
  }
  return peak;
}

std::string HumanReadableNum(const SymExpr& value) {
  if (value.is_symbolic()) return value.str_factored();
  double num = static_cast<double>(value.to_int());
  static const char* const kUnits[] = {"", "K", "M", "G", "T", "P", "E"};
  char buf[64];
  for (const char* unit : kUnits) {
    if (std::fabs(num) < 1000.0) {
      std::snprintf(buf, sizeof(buf), "%.1f%s", num, unit);
      return buf;
    }
    num /= 1000.0;
  }
  std::snprintf(buf, sizeof(buf), "%.1fZ", num);
  return buf;
}

std::string HumanReadableSize(const SymExpr& value) {
  if (value.is_symbolic()) return value.str_factored();
  double num = static_cast<double>(value.to_int());
  static const char* const kUnits[] = {"",   "Ki", "Mi", "Gi",
                                       "Ti", "Pi", "Ei", "Zi"};
  char buf[64];
  for (const char* unit : kUnits) {
    if (std::fabs(num) < 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.1f%sB", num, unit);
      return buf;
    }
    num /= 1024.0;
  }
  std::snprintf(buf, sizeof(buf), "%.1fYiB", num);
  return buf;
}

std::string HumanReadableDensity(const SymRatio& value) {
  if (value.is_symbolic()) return value.str() + " FLOP/Byte";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f FLOP/Byte", value.representative());
  return buf;
}

}  // namespace onnxsim
