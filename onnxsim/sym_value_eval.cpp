#include "sym_value_eval.h"

#include <algorithm>
#include <functional>

namespace onnxsim {
namespace {

// A SymExpr collapses to a concrete int only when it carries no symbol.
std::optional<int64_t> ConcreteInt(const SymExpr& e) {
  if (e.is_symbolic()) return std::nullopt;
  return e.to_int();
}

// The i-th element of an operand, applying numpy broadcasting: a scalar or a
// length-1 vector repeats, so index 0 is reused for every output position.
const SymExpr& Broadcast(const SymTensor& t, int64_t i) {
  return t.data.size() == 1 ? t.data[0] : t.data[static_cast<std::size_t>(i)];
}

// The broadcast output length for a set of operands, and whether the result is
// a scalar. Returns nullopt when two genuine vectors disagree in length (an
// input the shape scaffolding never produces, so it is left unevaluated rather
// than guessed at). A result is scalar only when *every* operand is a scalar.
struct BroadcastShape {
  int64_t length;
  bool scalar;
};
std::optional<BroadcastShape> BroadcastOf(
    std::initializer_list<const SymTensor*> operands) {
  int64_t length = 1;
  bool all_scalar = true;
  for (const SymTensor* t : operands) {
    if (!t->scalar) all_scalar = false;
    const int64_t n = t->size();
    if (n == 1) continue;  // scalar / length-1 vector broadcasts
    if (length != 1 && length != n) return std::nullopt;
    length = n;
  }
  return BroadcastShape{length, all_scalar};
}

// Apply an elementwise binary op with numpy broadcasting. `f` returns nullopt
// for an undecidable element (e.g. a symbolic Div that does not divide evenly),
// which aborts the whole output so the tensor is left unresolved.
std::optional<SymTensor> BinaryElementwise(
    const SymTensor& a, const SymTensor& b,
    const std::function<std::optional<SymExpr>(const SymExpr&, const SymExpr&)>&
        f) {
  const auto bshape = BroadcastOf({&a, &b});
  if (!bshape) return std::nullopt;
  std::vector<SymExpr> out;
  out.reserve(static_cast<std::size_t>(bshape->length));
  for (int64_t i = 0; i < bshape->length; ++i) {
    auto v = f(Broadcast(a, i), Broadcast(b, i));
    if (!v) return std::nullopt;
    out.push_back(std::move(*v));
  }
  if (bshape->scalar) return SymTensor::Scalar(std::move(out[0]));
  return SymTensor::Vector(std::move(out));
}

class Evaluator {
 public:
  explicit Evaluator(const SymGraph& graph) : graph_(graph) {
    for (const auto& [name, value] : graph.initializer) values_[name] = value;
  }

  std::map<std::string, SymTensor> Run() {
    for (const SymNode& node : graph_.node) {
      auto value = Eval(node);
      if (value && node.output.size() == 1 && !node.output[0].empty()) {
        values_[node.output[0]] = std::move(*value);
      }
    }
    return values_;
  }

 private:
  // A resolved value for an operand: an initializer, a Constant, or an already
  // evaluated node output. Absent name / empty ("" optional-input sentinel) /
  // unresolved tensor all read as "not available".
  const SymTensor* Value(const std::string& name) const {
    if (name.empty()) return nullptr;
    auto it = values_.find(name);
    return it == values_.end() ? nullptr : &it->second;
  }

  // The int list carried by attribute `attr_name`, else by input `input_index`
  // (opset >= 13 moved several of these axis/starts/ends operands to inputs).
  // Requires every element to be concrete.
  std::optional<std::vector<int64_t>> IntListOperand(
      const SymNode& node, const std::string& attr_name,
      std::size_t input_index) const {
    if (const SymAttr* a = node.attr(attr_name)) return a->ints;
    if (input_index < node.input.size()) {
      if (const SymTensor* t = Value(node.input[input_index])) {
        std::vector<int64_t> out;
        out.reserve(t->data.size());
        for (const SymExpr& e : t->data) {
          auto c = ConcreteInt(e);
          if (!c) return std::nullopt;
          out.push_back(*c);
        }
        return out;
      }
    }
    return std::nullopt;
  }

  std::optional<int64_t> IntAttr(const SymNode& node, const std::string& name,
                                 int64_t fallback) const {
    if (const SymAttr* a = node.attr(name)) {
      if (a->i) return *a->i;
    }
    return fallback;
  }

  std::optional<SymTensor> Eval(const SymNode& node) {
    const std::string& op = node.op_type;
    if (op == "Constant") return EvalConstant(node);
    if (op == "Shape") return EvalShape(node);
    if (op == "Size") return EvalSize(node);
    if (op == "Gather") return EvalGather(node);
    if (op == "Concat") return EvalConcat(node);
    if (op == "Unsqueeze") return EvalUnsqueeze(node);
    if (op == "Squeeze") return EvalSqueeze(node);
    if (op == "Slice") return EvalSlice(node);
    if (op == "Cast") return EvalCast(node);
    if (op == "Reshape") return EvalReshape(node);
    if (op == "ConstantOfShape") return EvalConstantOfShape(node);
    if (op == "Range") return EvalRange(node);
    if (op == "ReduceProd") return EvalReduceProd(node);
    if (op == "Tile") return EvalTile(node);
    if (op == "Transpose")
      return EvalUnary(node, [](const SymExpr& e) {
        return e;  // rank <= 1 transpose is the identity
      });
    if (op == "Expand") return EvalExpand(node);
    if (op == "Neg")
      return EvalUnary(node, [](const SymExpr& e) { return SymExpr(0) - e; });
    if (op == "Floor")
      return EvalUnary(node,
                       [](const SymExpr& e) { return e; });  // integer data
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div")
      return EvalArithmetic(node);
    if (op == "Equal") return EvalEqual(node);
    if (op == "Where") return EvalWhere(node);
    if (op == "Min" || op == "Max") return EvalMinMax(node);
    return std::nullopt;
  }

  // --- seeds --------------------------------------------------------------

  std::optional<SymTensor> EvalConstant(const SymNode& node) {
    if (const SymAttr* a = node.attr("value")) {
      if (a->t) return *a->t;
    }
    if (const SymAttr* a = node.attr("value_ints")) {
      std::vector<SymExpr> v;
      for (int64_t x : a->ints) v.emplace_back(x);
      return SymTensor::Vector(std::move(v));
    }
    if (const SymAttr* a = node.attr("value_int")) {
      if (a->i) return SymTensor::Scalar(SymExpr(*a->i));
    }
    return std::nullopt;
  }

  std::optional<SymTensor> EvalShape(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto it = graph_.shape.find(node.input[0]);
    if (it == graph_.shape.end()) return std::nullopt;
    const std::vector<SymExpr>& dims = it->second;
    const int64_t rank = static_cast<int64_t>(dims.size());
    // Optional start/end slice of the shape (opset 15+). Defaults span the
    // rank.
    int64_t start = IntAttr(node, "start", 0).value();
    int64_t end = IntAttr(node, "end", rank).value();
    if (start < 0) start += rank;
    if (end < 0) end += rank;
    start = std::clamp<int64_t>(start, 0, rank);
    end = std::clamp<int64_t>(end, 0, rank);
    std::vector<SymExpr> out;
    for (int64_t i = start; i < end; ++i)
      out.push_back(dims[static_cast<std::size_t>(i)]);
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalSize(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto it = graph_.shape.find(node.input[0]);
    if (it == graph_.shape.end()) return std::nullopt;
    SymExpr prod(1);
    for (const SymExpr& d : it->second) prod = prod * d;
    return SymTensor::Scalar(std::move(prod));
  }

  // --- shape family -------------------------------------------------------

  std::optional<SymTensor> EvalGather(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    const SymTensor* indices = Value(node.input[1]);
    if (!data || !indices || !data->is_vector()) return std::nullopt;
    const int64_t axis = IntAttr(node, "axis", 0).value();
    if (!(axis == 0 || axis == -1)) return std::nullopt;  // rank-1 data only
    const int64_t len = data->size();
    std::vector<SymExpr> out;
    out.reserve(indices->data.size());
    for (const SymExpr& idx_e : indices->data) {
      auto idx = ConcreteInt(idx_e);
      if (!idx) return std::nullopt;
      int64_t i = *idx < 0 ? *idx + len : *idx;
      if (i < 0 || i >= len) return std::nullopt;
      out.push_back(data->data[static_cast<std::size_t>(i)]);
    }
    // A scalar index selects a scalar; a 1-D index tensor yields a 1-D result.
    if (indices->scalar) return SymTensor::Scalar(std::move(out[0]));
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalConcat(const SymNode& node) {
    const int64_t axis = IntAttr(node, "axis", 0).value();
    if (!(axis == 0 || axis == -1)) return std::nullopt;  // rank-1 inputs
    std::vector<SymExpr> out;
    for (const std::string& name : node.input) {
      const SymTensor* t = Value(name);
      if (!t || !t->is_vector()) return std::nullopt;
      out.insert(out.end(), t->data.begin(), t->data.end());
    }
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalUnsqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data) return std::nullopt;
    auto axes = IntListOperand(node, "axes", 1);
    // Only the scalar -> length-1 vector case is representable in the rank<=1
    // model this pass tracks; anything raising rank above 1 is left alone.
    if (data->scalar && axes && axes->size() == 1 &&
        ((*axes)[0] == 0 || (*axes)[0] == -1)) {
      return SymTensor::Vector({data->data[0]});
    }
    return std::nullopt;
  }

  std::optional<SymTensor> EvalSqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data) return std::nullopt;
    if (data->scalar) return *data;  // already rank 0
    if (data->size() == 1)
      return SymTensor::Scalar(data->data[0]);  // [1] -> ()
    return std::nullopt;
  }

  std::optional<SymTensor> EvalSlice(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data || !data->is_vector()) return std::nullopt;
    auto starts = IntListOperand(node, "starts", 1);
    auto ends = IntListOperand(node, "ends", 2);
    auto axes = IntListOperand(node, "axes", 3);
    auto steps = IntListOperand(node, "steps", 4);
    if (!starts || !ends || starts->size() != 1 || ends->size() != 1)
      return std::nullopt;
    if (axes && !(axes->size() == 1 && ((*axes)[0] == 0 || (*axes)[0] == -1)))
      return std::nullopt;  // rank-1 data, axis 0 only
    const int64_t step = (steps && steps->size() == 1) ? (*steps)[0] : 1;
    if (step == 0) return std::nullopt;
    const int64_t len = data->size();
    int64_t start = (*starts)[0];
    int64_t end = (*ends)[0];
    // ONNX Slice normalization / clamping, honouring the direction of `step`.
    if (start < 0) start += len;
    if (end < 0) end += len;
    std::vector<SymExpr> out;
    if (step > 0) {
      start = std::clamp<int64_t>(start, 0, len);
      end = std::clamp<int64_t>(end, 0, len);
      for (int64_t i = start; i < end; i += step)
        out.push_back(data->data[static_cast<std::size_t>(i)]);
    } else {
      start = std::clamp<int64_t>(start, 0, len - 1);
      end = std::clamp<int64_t>(end, -1, len - 1);
      for (int64_t i = start; i > end; i += step)
        out.push_back(data->data[static_cast<std::size_t>(i)]);
    }
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalCast(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data) return std::nullopt;
    // Casting between integer / bool representations preserves the exact
    // integer value the shape scaffolding carries, so the symbolic value is
    // unchanged. A float target is not modelled (later float arithmetic would
    // be lost), so leave it unresolved.
    const int64_t to = IntAttr(node, "to", 0).value();
    // onnx::TensorProto data types: FLOAT=1, INT8=3, INT16=5, INT32=6, INT64=7,
    // BOOL=9, UINT8=2, UINT16=4, UINT32=12, UINT64=13, INT4=22, UINT4=21.
    static const std::vector<int64_t> kIntegerTypes = {2, 3,  4,  5,  6, 7,
                                                       9, 12, 13, 21, 22};
    if (std::find(kIntegerTypes.begin(), kIntegerTypes.end(), to) ==
        kIntegerTypes.end())
      return std::nullopt;
    return *data;
  }

  std::optional<SymTensor> EvalReshape(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    auto shape = IntListOperand(node, "shape", 1);
    if (!data || !shape) return std::nullopt;
    const int64_t count = data->size();
    // Only rank <= 1 targets are representable here: a 1-D value flattened /
    // kept 1-D (the `Shape -> Reshape([-1]) -> Gather` pattern), or collapsed
    // to a scalar. Higher-rank targets are left alone.
    if (shape->empty()) {  // reshape to scalar
      if (count != 1) return std::nullopt;
      return SymTensor::Scalar(data->data[0]);
    }
    if (shape->size() != 1) return std::nullopt;
    const int64_t dim = (*shape)[0];
    if (dim == -1 || dim == 0 || dim == count) {
      return SymTensor::Vector(data->data);  // element order is preserved
    }
    return std::nullopt;
  }

  std::optional<SymTensor> EvalConstantOfShape(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* shape = Value(node.input[0]);
    // The shape input lists the output dims. A single-entry shape gives a
    // rank-1 output; multi-entry shapes exceed the rank<=1 model.
    if (!shape || !shape->is_vector() || shape->size() != 1)
      return std::nullopt;
    auto len = ConcreteInt(shape->data[0]);
    if (!len || *len < 0) return std::nullopt;
    SymExpr fill(0);  // ONNX default is a single 0
    if (const SymAttr* a = node.attr("value")) {
      if (a->t && a->t->data.size() == 1)
        fill = a->t->data[0];
      else
        return std::nullopt;
    }
    return SymTensor::Vector(
        std::vector<SymExpr>(static_cast<std::size_t>(*len), fill));
  }

  std::optional<SymTensor> EvalRange(const SymNode& node) {
    if (node.input.size() < 3) return std::nullopt;
    const SymTensor* start = Value(node.input[0]);
    const SymTensor* limit = Value(node.input[1]);
    const SymTensor* delta = Value(node.input[2]);
    if (!start || !limit || !delta) return std::nullopt;
    auto s = ConcreteInt(start->data[0]);
    auto l = ConcreteInt(limit->data[0]);
    auto d = ConcreteInt(delta->data[0]);
    if (!s || !l || !d || *d == 0) return std::nullopt;
    std::vector<SymExpr> out;
    if (*d > 0)
      for (int64_t v = *s; v < *l; v += *d) out.emplace_back(v);
    else
      for (int64_t v = *s; v > *l; v += *d) out.emplace_back(v);
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalReduceProd(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data || !data->is_vector()) return std::nullopt;
    SymExpr prod(1);
    for (const SymExpr& e : data->data) prod = prod * e;
    const int64_t keepdims = IntAttr(node, "keepdims", 1).value();
    if (keepdims) return SymTensor::Vector({std::move(prod)});
    return SymTensor::Scalar(std::move(prod));
  }

  std::optional<SymTensor> EvalTile(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    const SymTensor* repeats = Value(node.input[1]);
    if (!data || !data->is_vector() || !repeats || repeats->size() != 1)
      return std::nullopt;
    auto r = ConcreteInt(repeats->data[0]);
    if (!r || *r < 0) return std::nullopt;
    std::vector<SymExpr> out;
    out.reserve(data->data.size() * static_cast<std::size_t>(*r));
    for (int64_t k = 0; k < *r; ++k)
      out.insert(out.end(), data->data.begin(), data->data.end());
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalExpand(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    const SymTensor* shape = Value(node.input[1]);
    if (!data || !shape || !shape->is_vector()) return std::nullopt;
    // Only rank <= 1 targets are representable. An empty target shape keeps the
    // (necessarily length-1) input; a single-entry target gives a rank-1
    // result.
    if (shape->size() == 0) {
      if (data->size() != 1) return std::nullopt;
      return data->scalar ? *data : SymTensor::Scalar(data->data[0]);
    }
    if (shape->size() != 1) return std::nullopt;
    auto n = ConcreteInt(shape->data[0]);
    if (!n || *n < 0) return std::nullopt;
    const int64_t in = data->size();
    // numpy broadcast of the 1-D input against the length-`n` target: the two
    // must be equal or one of them 1; the output takes the larger length.
    if (in != 1 && *n != 1 && in != *n) return std::nullopt;
    const int64_t length = std::max<int64_t>(in, *n);
    std::vector<SymExpr> out;
    out.reserve(static_cast<std::size_t>(length));
    for (int64_t i = 0; i < length; ++i) out.push_back(Broadcast(*data, i));
    return SymTensor::Vector(std::move(out));
  }

  // --- arithmetic + the four data-prop gap ops ----------------------------

  std::optional<SymTensor> EvalUnary(
      const SymNode& node, const std::function<SymExpr(const SymExpr&)>& f) {
    if (node.input.empty()) return std::nullopt;
    const SymTensor* data = Value(node.input[0]);
    if (!data) return std::nullopt;
    SymTensor out = *data;
    for (SymExpr& e : out.data) e = f(e);
    return out;
  }

  std::optional<SymTensor> EvalArithmetic(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* a = Value(node.input[0]);
    const SymTensor* b = Value(node.input[1]);
    if (!a || !b) return std::nullopt;
    const std::string& op = node.op_type;
    return BinaryElementwise(
        *a, *b,
        [&op](const SymExpr& x, const SymExpr& y) -> std::optional<SymExpr> {
          if (op == "Add") return x + y;
          if (op == "Sub") return x - y;
          if (op == "Mul") return x * y;
          // Div: exact integer division only; a symbolic remainder is
          // undecidable and leaves the value unresolved (never rounds/guesses).
          return TryExactDivide(x, y);
        });
  }

  std::optional<SymTensor> EvalEqual(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    const SymTensor* a = Value(node.input[0]);
    const SymTensor* b = Value(node.input[1]);
    if (!a || !b) return std::nullopt;
    return BinaryElementwise(
        *a, *b,
        [](const SymExpr& x, const SymExpr& y) -> std::optional<SymExpr> {
          auto eq = TryEqual(x, y);
          if (!eq) return std::nullopt;  // undecidable comparison
          return SymExpr(*eq ? 1 : 0);
        });
  }

  std::optional<SymTensor> EvalWhere(const SymNode& node) {
    if (node.input.size() < 3) return std::nullopt;
    const SymTensor* cond = Value(node.input[0]);
    const SymTensor* x = Value(node.input[1]);
    const SymTensor* y = Value(node.input[2]);
    if (!cond || !x || !y) return std::nullopt;
    const auto bshape = BroadcastOf({cond, x, y});
    if (!bshape) return std::nullopt;
    std::vector<SymExpr> out;
    out.reserve(static_cast<std::size_t>(bshape->length));
    for (int64_t i = 0; i < bshape->length; ++i) {
      // The branch is chosen only when the condition is a concrete bool; a
      // symbolic condition is undecidable, so the whole Where stays unresolved
      // rather than picking a branch that might be wrong.
      auto c = ConcreteInt(Broadcast(*cond, i));
      if (!c) return std::nullopt;
      out.push_back(*c != 0 ? Broadcast(*x, i) : Broadcast(*y, i));
    }
    if (bshape->scalar) return SymTensor::Scalar(std::move(out[0]));
    return SymTensor::Vector(std::move(out));
  }

  std::optional<SymTensor> EvalMinMax(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    const bool is_min = node.op_type == "Min";
    const SymTensor* acc_src = Value(node.input[0]);
    if (!acc_src) return std::nullopt;
    SymTensor acc = *acc_src;
    for (std::size_t k = 1; k < node.input.size(); ++k) {
      const SymTensor* b = Value(node.input[k]);
      if (!b) return std::nullopt;
      auto merged = BinaryElementwise(
          acc, *b,
          [is_min](const SymExpr& x,
                   const SymExpr& y) -> std::optional<SymExpr> {
            // Pick a side only when the difference is a concrete number; a
            // symbolic difference has unknown sign and is left undecidable.
            const SymExpr diff = x - y;
            if (diff.is_symbolic()) return std::nullopt;
            const bool x_le_y = diff.to_int() <= 0;
            return (is_min == x_le_y) ? x : y;
          });
      if (!merged) return std::nullopt;
      acc = std::move(*merged);
    }
    return acc;
  }

  const SymGraph& graph_;
  std::map<std::string, SymTensor> values_;
};

}  // namespace

std::map<std::string, SymTensor> EvaluateSymbolicValues(const SymGraph& graph) {
  return Evaluator(graph).Run();
}

}  // namespace onnxsim
