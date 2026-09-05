/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * See xnnpack_codegen.h for the public API and v1 scope summary.
 *
 * ---- The NHWC layout convention, in detail ----
 *
 * XNNPACK's convolution Nodes (xnn_define_convolution_2d,
 * xnn_define_depthwise_convolution_2d) hard-require NHWC activations and an
 * OHWI-ish filter layout (confirmed against the pinned XNNPACK commit --
 * cmake/build_xnnpack.cmake's ONNXSIM_XNNPACK_GIT_TAG -- both from its public
 * header's own doc comments and, for the exact channel ordering a regular vs.
 * depthwise filter's data must already be in, its
 * test/subgraph/{convolution-2d,depthwise-convolution-2d}.cc reference
 * implementations). ONNX is NCHW/OIHW by convention. Rather than insert a
 * real, data-moving Transpose Node at every NCHW/NHWC boundary (XNNPACK's
 * Subgraph API has no generic N-D transpose Node to do this with, only
 * xnn_define_static_reshape's flat reinterpretation, which is not the same
 * operation), this generator instead:
 *
 *   1. Emits every rank-4 tensor's XNNPACK Value already in NHWC -- so a
 *      Conv's own input/output Values, and any purely-elementwise activation
 *      immediately upstream/downstream of one (Add/Sub/Mul/Div/Relu/Sigmoid
 *      do not care what order their axes are in; they only require producer
 *      and consumer to agree, which permuting *every* rank-4 tensor the same
 *      way guarantees), never need a real data-moving transpose at all.
 *   2. Permutes Conv/depthwise-Conv filter (and, being rank-4, image-shaped)
 *      constant data exactly once, here, at generation time -- a free,
 *      one-time transpose of already-known constant bytes, emitted directly
 *      as the (permuted) C array literal. This is *not* a runtime cost.
 *   3. Requires the model's own graph inputs/outputs to be supplied/read
 *      already in NHWC order by the emitted code's caller -- which is also
 *      not a real cost in the intended use case: a cv::Mat holding an
 *      interleaved image is *already* row-major HWC (see
 *      onnxsim/xnnpack_cv_mat.hpp), so the common "image in, tensor out"
 *      pipeline needs no conversion at that boundary either.
 *
 * This is airtight as long as no op in between actually depends on axis
 * *order* rather than just per-element values -- true for every op this
 * generator supports except Reshape, which reinterprets a flat/row-major
 * byte sequence and therefore only produces the same result under NHWC
 * physical order as ONNX's own (NCHW-assuming) semantics intended when
 * either side of the reshape has no real spatial extent to reorder: rank
 * != 4, or rank == 4 with H == W == 1 (immediately after a
 * GlobalAveragePool, e.g. the standard "backbone -> classifier head"
 * pattern). EmitReshape below enforces exactly that condition and throws,
 * rather than silently mis-ordering data, otherwise.
 */
#include "xnnpack_codegen.h"

#include <onnx/shape_inference/implementation.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnxsim {
namespace xnnpack_backend {

namespace {

// ---------------------------------------------------------------------- //
// Generic attribute / shape / text helpers
// ---------------------------------------------------------------------- //

const onnx::AttributeProto* FindAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) return &attr;
  }
  return nullptr;
}

float AttrFloat(const onnx::NodeProto& node, const std::string& name,
                float dflt) {
  const auto* a = FindAttr(node, name);
  return a != nullptr ? a->f() : dflt;
}

int64_t AttrInt(const onnx::NodeProto& node, const std::string& name,
                int64_t dflt) {
  const auto* a = FindAttr(node, name);
  return a != nullptr ? a->i() : dflt;
}

std::vector<int64_t> AttrInts(const onnx::NodeProto& node,
                              const std::string& name,
                              std::vector<int64_t> dflt) {
  const auto* a = FindAttr(node, name);
  if (a == nullptr) return dflt;
  return std::vector<int64_t>(a->ints().begin(), a->ints().end());
}

std::string AttrString(const onnx::NodeProto& node, const std::string& name,
                       const std::string& dflt) {
  const auto* a = FindAttr(node, name);
  return a != nullptr ? a->s() : dflt;
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

// ONNX/numpy multidirectional broadcasting of two shapes (same rule
// onnx_to_xnnpack_subgraph.cpp's BroadcastShape implements, duplicated here
// rather than shared -- see that file's own header comment on why this
// generator is a separate, independent implementation).
std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b,
                                    const std::string& context) {
  const size_t n = std::max(a.size(), b.size());
  std::vector<int64_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    const int64_t ad = i < n - a.size() ? 1 : a[i - (n - a.size())];
    const int64_t bd = i < n - b.size() ? 1 : b[i - (n - b.size())];
    if (ad != bd && ad != 1 && bd != 1) {
      throw std::runtime_error("xnnpack codegen: " + context +
                               ": shapes are not broadcast-compatible");
    }
    out[i] = std::max(ad, bd);
  }
  return out;
}

// Permutes `shape` (or, in the data overload, row-major `data` of that
// shape) so that output axis i holds original axis perm[i] -- i.e. out[i] =
// shape[perm[i]]. Used both for NCHW->NHWC activation shapes (perm =
// {0,2,3,1}) and, at the same time, for OIHW->OHWI-ish filter layouts (see
// the module comment above): {0,2,3,1} for a regular/grouped Conv filter,
// {1,2,3,0} for a depthwise one.
std::vector<int64_t> PermuteShape(const std::vector<int64_t>& shape,
                                  const std::vector<int>& perm) {
  std::vector<int64_t> out(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) out[i] = shape[perm[i]];
  return out;
}

std::vector<float> PermuteData(const float* data,
                               const std::vector<int64_t>& shape,
                               const std::vector<int>& perm) {
  const int rank = static_cast<int>(shape.size());
  std::vector<int64_t> out_shape = PermuteShape(shape, perm);
  std::vector<int64_t> in_strides(rank);
  int64_t acc = 1;
  for (int i = rank - 1; i >= 0; --i) {
    in_strides[i] = acc;
    acc *= shape[i];
  }
  const int64_t total = acc;
  std::vector<float> out(static_cast<size_t>(total));
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat_out = 0; flat_out < total; ++flat_out) {
    int64_t in_flat = 0;
    for (int i = 0; i < rank; ++i) in_flat += idx[i] * in_strides[perm[i]];
    out[static_cast<size_t>(flat_out)] = data[in_flat];
    for (int i = rank - 1; i >= 0; --i) {
      if (++idx[i] < out_shape[i]) break;
      idx[i] = 0;
    }
  }
  return out;
}

// A valid, non-empty C identifier: [A-Za-z_][A-Za-z0-9_]*.
bool IsValidCIdent(const std::string& s) {
  if (s.empty()) return false;
  if (!std::isalpha(static_cast<unsigned char>(s[0])) && s[0] != '_') {
    return false;
  }
  return std::all_of(s.begin(), s.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  });
}

// Turns an arbitrary ONNX tensor/node name into a valid C identifier
// fragment: non-alnum bytes become '_', and a leading digit gets a '_'
// prefix. Uniqueness (across the whole generated file) is the caller's job
// -- see Emitter::MakeUniqueIdent.
std::string SanitizeIdentFragment(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  for (unsigned char c : raw) {
    out.push_back((std::isalnum(c) || c == '_') ? static_cast<char>(c) : '_');
  }
  if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0]))) {
    out.insert(out.begin(), '_');
  }
  return out;
}

std::string FormatSizeArrayLiteral(const std::vector<int64_t>& dims) {
  std::ostringstream oss;
  oss << "{";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) oss << ", ";
    oss << dims[i];
  }
  oss << "}";
  return oss.str();
}

// A float formatted so the C compiler round-trips it exactly (17 significant
// decimal digits is always enough to uniquely identify an IEEE754 binary32
// value), suffixed `f` so it is an `float` literal, not a `double` one.
std::string FormatFloatLiteral(float v) {
  if (std::isinf(v)) return v > 0 ? "INFINITY" : "-INFINITY";
  if (std::isnan(v)) return "NAN";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9gf", v);
  return buf;
}

// Emits a `static const float <c_name>[] = { ... };` definition, wrapped
// at a reasonable line width so a large weight array does not become one
// gigantic unreadable line.
void EmitFloatArrayDef(std::ostringstream& out, const std::string& c_name,
                       const std::vector<float>& data) {
  out << "static const float " << c_name << "[] = {\n ";
  for (size_t i = 0; i < data.size(); ++i) {
    out << " " << FormatFloatLiteral(data[i]) << ",";
    if ((i + 1) % 12 == 0) out << "\n ";
  }
  out << "\n};\n";
}

// ---------------------------------------------------------------------- //
// Shape collection (post shape-inference), concrete shapes only
// ---------------------------------------------------------------------- //

std::optional<std::vector<int64_t>> ConcreteShapeOf(
    const onnx::TypeProto& type) {
  if (!type.has_tensor_type() || !type.tensor_type().has_shape()) {
    return std::nullopt;
  }
  std::vector<int64_t> dims;
  for (const auto& d : type.tensor_type().shape().dim()) {
    if (!d.has_dim_value()) return std::nullopt;
    dims.push_back(d.dim_value());
  }
  return dims;
}

std::unordered_map<std::string, std::vector<int64_t>> CollectShapes(
    const onnx::GraphProto& graph) {
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  auto add = [&](const onnx::ValueInfoProto& vi) {
    if (auto s = ConcreteShapeOf(vi.type())) shapes[vi.name()] = std::move(*s);
  };
  for (const auto& vi : graph.input()) add(vi);
  for (const auto& vi : graph.output()) add(vi);
  for (const auto& vi : graph.value_info()) add(vi);
  for (const auto& tp : graph.initializer()) {
    shapes[tp.name()] =
        std::vector<int64_t>(tp.dims().begin(), tp.dims().end());
  }
  return shapes;
}

// ---------------------------------------------------------------------- //
// Emitter
// ---------------------------------------------------------------------- //

struct TensorInfo {
  std::string id_var;  // C variable name holding this Value's uint32_t id
  std::vector<int64_t>
      xnn_shape;  // shape as declared to XNNPACK (permuted if 4-D)
  bool is_4d_permuted = false;  // true if xnn_shape is NCHW->NHWC-permuted
};

const std::vector<int> kNCHWToNHWC = {0, 2, 3, 1};
const std::vector<int> kDepthwiseFilterPerm = {1, 2, 3, 0};

class Emitter {
 public:
  Emitter(const onnx::ModelProto& model, std::string prefix)
      : model_(model), prefix_(std::move(prefix)) {
    for (const auto& tp : model_.graph().initializer()) {
      initializers_[tp.name()] = &tp;
    }
    shapes_ = CollectShapes(model_.graph());
  }

  std::string Run() {
    const auto& graph = model_.graph();
    const int num_inputs = graph.input_size();
    const int num_outputs = graph.output_size();

    for (int i = 0; i < num_outputs; ++i) {
      pending_output_ids_[graph.output(i).name()] =
          static_cast<uint32_t>(num_inputs + i);
    }

    build_ << "static enum xnn_status " << prefix_
           << "_build_subgraph(xnn_subgraph_t* subgraph_out) {\n";
    build_ << "  xnn_subgraph_t sg;\n";
    build_ << "  ONNXSIM_XNN_CHECK(xnn_create_subgraph(/*external_value_ids=*/"
           << (num_inputs + num_outputs) << ", /*flags=*/0, &sg));\n\n";

    for (int i = 0; i < num_inputs; ++i) {
      const auto& vi = graph.input(i);
      DefineExternalValue(vi.name(), RequireShape(vi.name(), "graph input"),
                          static_cast<uint32_t>(i),
                          "XNN_VALUE_FLAG_EXTERNAL_INPUT");
    }

    for (const auto& node : graph.node()) EmitNode(node);

    if (!pending_output_ids_.empty()) {
      throw std::runtime_error("xnnpack codegen: graph output '" +
                               pending_output_ids_.begin()->first +
                               "' was never produced by any supported node");
    }

    build_ << "\n  *subgraph_out = sg;\n  return xnn_status_success;\n}\n";

    return AssembleFile(num_inputs, num_outputs);
  }

 private:
  const onnx::ModelProto& model_;
  std::string prefix_;
  std::unordered_map<std::string, const onnx::TensorProto*> initializers_;
  std::unordered_map<std::string, std::vector<int64_t>> shapes_;
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::unordered_map<std::string, uint32_t> pending_output_ids_;
  std::unordered_set<std::string> used_idents_;
  std::ostringstream data_;   // static const float ... definitions
  std::ostringstream build_;  // body of <prefix>_build_subgraph

  std::string MakeUniqueIdent(const std::string& hint) {
    std::string base = SanitizeIdentFragment(hint);
    std::string candidate = base;
    int suffix = 1;
    while (!used_idents_.insert(candidate).second) {
      candidate = base + "_" + std::to_string(++suffix);
    }
    return candidate;
  }

  const std::vector<int64_t>& RequireShape(const std::string& name,
                                           const std::string& context) {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) {
      throw std::runtime_error("xnnpack codegen: " + context + " '" + name +
                               "' has no concrete (fully static) shape");
    }
    return it->second;
  }

  const onnx::TensorProto& RequireInitializer(const std::string& name,
                                              const std::string& context) {
    auto it = initializers_.find(name);
    if (it == initializers_.end()) {
      throw std::runtime_error("xnnpack codegen: " + context + " '" + name +
                               "' must be a constant (initializer) -- "
                               "generated code cannot depend on a value only "
                               "known at runtime for this parameter");
    }
    return *it->second;
  }

  // Raw fp32 data of a constant initializer; throws if it is not fp32 or its
  // element count does not match `shape`.
  const float* RequireFp32Data(const onnx::TensorProto& tp,
                               const std::vector<int64_t>& shape,
                               const std::string& context) {
    if (tp.data_type() != onnx::TensorProto::FLOAT) {
      throw std::runtime_error("xnnpack codegen: " + context + " '" +
                               tp.name() +
                               "' is not fp32 (only fp32 is supported)");
    }
    if (tp.has_raw_data()) {
      if (static_cast<int64_t>(tp.raw_data().size()) !=
          NumElements(shape) * static_cast<int64_t>(sizeof(float))) {
        throw std::runtime_error("xnnpack codegen: " + context + " '" +
                                 tp.name() +
                                 "': raw_data size does not match its shape");
      }
      return reinterpret_cast<const float*>(tp.raw_data().data());
    }
    if (tp.float_data_size() != NumElements(shape)) {
      throw std::runtime_error("xnnpack codegen: " + context + " '" +
                               tp.name() +
                               "': float_data size does not match its shape");
    }
    return tp.float_data().data();
  }

  // Defines a graph input/output Value at a fixed, externally-reserved id.
  TensorInfo& DefineExternalValue(const std::string& name,
                                  const std::vector<int64_t>& onnx_shape,
                                  uint32_t external_id, const char* flag) {
    TensorInfo info;
    info.id_var = MakeUniqueIdent("id_" + name);
    const bool is_4d = onnx_shape.size() == 4;
    info.xnn_shape = is_4d ? PermuteShape(onnx_shape, kNCHWToNHWC) : onnx_shape;
    info.is_4d_permuted = is_4d;

    build_ << "  uint32_t " << info.id_var << ";\n";
    build_ << "  {\n    size_t dims[] = "
           << FormatSizeArrayLiteral(info.xnn_shape) << ";\n";
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_tensor_value(sg, "
              "xnn_datatype_fp32, "
           << info.xnn_shape.size() << ", dims, NULL, /*external_id=*/"
           << external_id << ", " << flag << ", &" << info.id_var
           << "));\n  }\n";

    tensors_[name] = info;
    return tensors_[name];
  }

  // Defines (memoized) a plain constant Value: embeds `name`'s initializer
  // data verbatim (no layout permutation) -- used for anything that is not
  // a Conv/depthwise-Conv filter (Gemm/MatMul operands and biases, an
  // elementwise operand, a Conv's own bias).
  const TensorInfo& GetOrDefineConstant(const std::string& name,
                                        const std::string& context) {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return it->second;

    const auto& shape = RequireShape(name, context);
    const auto& tp = RequireInitializer(name, context);
    const float* data = RequireFp32Data(tp, shape, context);
    // A rank-4 constant is just as much an NCHW/NHWC tensor as any
    // activation (e.g. a per-channel bias reshaped to [1, C, 1, 1], meant to
    // broadcast against an NHWC-permuted activation in an Add/Mul) -- it
    // must be permuted exactly like DefineExternalValue's graph inputs are,
    // or its declared shape (and, for a genuinely non-1x1-spatial constant,
    // its element order) would silently disagree with everything else this
    // generator treats as NHWC. This is a plain axis permutation, distinct
    // from DefineFilterConstant's OIHW->OHWI-ish filter relayout below.
    const bool is_4d = shape.size() == 4;
    const std::vector<int64_t> xnn_shape =
        is_4d ? PermuteShape(shape, kNCHWToNHWC) : shape;
    const std::vector<float> data_vec =
        is_4d ? PermuteData(data, shape, kNCHWToNHWC)
              : std::vector<float>(data, data + NumElements(shape));

    TensorInfo info;
    info.id_var = MakeUniqueIdent("id_" + name);
    info.xnn_shape = xnn_shape;
    info.is_4d_permuted = is_4d;
    const std::string c_name = MakeUniqueIdent("g_" + prefix_ + "_" + name);
    EmitFloatArrayDef(data_, c_name, data_vec);

    build_ << "  uint32_t " << info.id_var << ";\n";
    build_ << "  {\n    size_t dims[] = "
           << FormatSizeArrayLiteral(info.xnn_shape) << ";\n";
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_tensor_value(sg, "
              "xnn_datatype_fp32, "
           << info.xnn_shape.size() << ", dims, " << c_name
           << ", XNN_INVALID_VALUE_ID, 0, &" << info.id_var << "));\n  }\n";

    tensors_[name] = info;
    return tensors_[name];
  }

  // Defines a Conv/depthwise-Conv filter Value: permutes both the shape and
  // the underlying data by `perm` (see the module comment) before embedding.
  // Always defines a fresh Value (not memoized under `name`) -- a filter is
  // essentially never reused by another node, and if it somehow were, two
  // independent Values for the same source tensor is harmless.
  TensorInfo DefineFilterConstant(const std::string& name,
                                  const std::vector<int>& perm,
                                  const std::string& context) {
    const auto& shape = RequireShape(name, context);
    if (shape.size() != 4) {
      throw std::runtime_error("xnnpack codegen: " + context + " '" + name +
                               "' must be a 4-D filter");
    }
    const auto& tp = RequireInitializer(name, context);
    const float* data = RequireFp32Data(tp, shape, context);
    const std::vector<float> permuted = PermuteData(data, shape, perm);
    const std::vector<int64_t> xnn_shape = PermuteShape(shape, perm);

    TensorInfo info;
    info.id_var = MakeUniqueIdent("id_" + name);
    info.xnn_shape = xnn_shape;
    const std::string c_name = MakeUniqueIdent("g_" + prefix_ + "_" + name);
    EmitFloatArrayDef(data_, c_name, permuted);

    build_ << "  uint32_t " << info.id_var << ";\n";
    build_ << "  {\n    size_t dims[] = " << FormatSizeArrayLiteral(xnn_shape)
           << ";\n";
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_tensor_value(sg, "
              "xnn_datatype_fp32, "
           << xnn_shape.size() << ", dims, " << c_name
           << ", XNN_INVALID_VALUE_ID, 0, &" << info.id_var << "));\n  }\n";
    return info;
  }

  // Looks up an already-defined Value (a graph input or a prior node's
  // output) or falls back to defining it as a plain constant. Throws if
  // `name` is neither.
  const TensorInfo& ValueOf(const std::string& name,
                            const std::string& context) {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return it->second;
    if (initializers_.count(name)) return GetOrDefineConstant(name, context);
    throw std::runtime_error("xnnpack codegen: " + context + ": '" + name +
                             "' is neither a graph input, a constant, nor a "
                             "prior node's output (out-of-order or unsupported "
                             "producer?)");
  }

  // Defines a node's output Value: at the graph's reserved external id (and
  // flag) if `name` is a declared graph output, else an ordinary internal
  // Value. `onnx_rank4` says whether the *ONNX* tensor this output
  // represents is rank 4 (in which case its XNNPACK shape/data convention is
  // NHWC, per the module comment) -- distinct from `xnn_shape`'s own rank,
  // since e.g. GlobalAveragePool's ONNX-side output is rank 4 ([N,C,1,1])
  // but this generator represents it directly as the already-rank-2
  // `xnn_shape` [N,C] (see EmitGlobalAveragePool).
  TensorInfo& DefineOutputValue(const std::string& name,
                                std::vector<int64_t> xnn_shape,
                                bool is_4d_permuted) {
    TensorInfo info;
    info.id_var = MakeUniqueIdent("id_" + name);
    info.xnn_shape = std::move(xnn_shape);
    info.is_4d_permuted = is_4d_permuted;

    auto out_it = pending_output_ids_.find(name);
    const bool is_graph_output = out_it != pending_output_ids_.end();
    const std::string flag =
        is_graph_output ? "XNN_VALUE_FLAG_EXTERNAL_OUTPUT" : "0";
    const std::string external_id = is_graph_output
                                        ? std::to_string(out_it->second)
                                        : "XNN_INVALID_VALUE_ID";

    build_ << "  uint32_t " << info.id_var << ";\n";
    build_ << "  {\n    size_t dims[] = "
           << FormatSizeArrayLiteral(info.xnn_shape) << ";\n";
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_tensor_value(sg, "
              "xnn_datatype_fp32, "
           << info.xnn_shape.size() << ", dims, NULL, " << external_id << ", "
           << flag << ", &" << info.id_var << "));\n  }\n";

    if (is_graph_output) pending_output_ids_.erase(out_it);
    tensors_[name] = info;
    return tensors_[name];
  }

  void EmitNode(const onnx::NodeProto& node) {
    const std::string& op = node.op_type();
    if (op == "Add") return EmitBinary(node, "xnn_binary_add");
    if (op == "Sub") return EmitBinary(node, "xnn_binary_subtract");
    if (op == "Mul") return EmitBinary(node, "xnn_binary_multiply");
    if (op == "Div") return EmitBinary(node, "xnn_binary_divide");
    if (op == "Relu") {
      return EmitUnary(node, "xnn_unary_clamp",
                       "params.clamp.min = 0.0f; params.clamp.max = INFINITY;");
    }
    if (op == "Sigmoid") return EmitUnary(node, "xnn_unary_sigmoid", "");
    if (op == "Gemm") return EmitGemm(node);
    if (op == "MatMul") return EmitMatMul(node);
    if (op == "Reshape") return EmitReshape(node);
    if (op == "Conv") return EmitConv(node);
    if (op == "GlobalAveragePool") return EmitGlobalAveragePool(node);
    throw std::runtime_error("xnnpack codegen: unsupported op '" + op +
                             "' (node '" + node.name() + "')");
  }

  void EmitBinary(const onnx::NodeProto& node, const std::string& xnn_op) {
    if (node.input_size() != 2 || node.output_size() != 1) {
      throw std::runtime_error("xnnpack codegen: " + node.op_type() +
                               " must have 2 inputs and 1 output");
    }
    const auto& a = ValueOf(node.input(0), node.op_type());
    const auto& b = ValueOf(node.input(1), node.op_type());
    // Both operands were already permuted consistently if 4-D (see the
    // module comment), so broadcasting can be checked directly against
    // their (possibly-permuted) xnn shapes.
    const auto out_shape =
        BroadcastShape(a.xnn_shape, b.xnn_shape, node.op_type());
    auto& out = DefineOutputValue(node.output(0), out_shape,
                                  a.is_4d_permuted || b.is_4d_permuted);
    build_ << "  {\n    struct xnn_binary_params params = {-INFINITY, "
              "INFINITY};\n";
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_binary(sg, " << xnn_op
           << ", &params, " << a.id_var << ", " << b.id_var << ", "
           << out.id_var << ", 0));\n  }\n";
  }

  // `params_stmts` is zero or more complete, semicolon-terminated C
  // statements initializing `params` (e.g. Relu's clamp min/max); empty
  // means the op takes no parameters (Sigmoid), zeroed via memset instead.
  void EmitUnary(const onnx::NodeProto& node, const std::string& xnn_op,
                 const std::string& params_stmts) {
    if (node.input_size() != 1 || node.output_size() != 1) {
      throw std::runtime_error("xnnpack codegen: " + node.op_type() +
                               " must have 1 input and 1 output");
    }
    const auto& in = ValueOf(node.input(0), node.op_type());
    auto& out =
        DefineOutputValue(node.output(0), in.xnn_shape, in.is_4d_permuted);
    build_ << "  {\n    union xnn_unary_params params;\n";
    if (params_stmts.empty()) {
      build_ << "    memset(&params, 0, sizeof(params));\n";
    } else {
      build_ << "    " << params_stmts << "\n";
    }
    build_ << "    ONNXSIM_XNN_CHECK(xnn_define_unary(sg, " << xnn_op
           << ", &params, " << in.id_var << ", " << out.id_var
           << ", 0));\n  }\n";
  }

  // Shared by Gemm (transA=0, transB in {0,1}) and MatMul (both 2-D, no
  // transpose/bias) -- both lower to xnn_define_fully_connected, mirroring
  // onnx_to_xnnpack_subgraph.cpp's LowerMatMulLike exactly (see that
  // function's own comment for the flag/shape reasoning).
  void EmitMatMulLike(const onnx::NodeProto& node, const std::string& a_name,
                      const std::string& b_name, const std::string& bias_name,
                      bool b_is_transposed) {
    const auto& a = ValueOf(a_name, node.op_type());
    const auto& b = ValueOf(b_name, node.op_type());
    if (a.xnn_shape.size() != 2 || b.xnn_shape.size() != 2) {
      throw std::runtime_error("xnnpack codegen: " + node.op_type() +
                               " only supports 2-D operands");
    }
    const int64_t m = a.xnn_shape[0], k = a.xnn_shape[1];
    const int64_t b_k = b_is_transposed ? b.xnn_shape[1] : b.xnn_shape[0];
    const int64_t n = b_is_transposed ? b.xnn_shape[0] : b.xnn_shape[1];
    if (b_k != k) {
      throw std::runtime_error("xnnpack codegen: " + node.op_type() +
                               ": incompatible operand shapes");
    }
    std::string bias_id_var = "XNN_INVALID_VALUE_ID";
    if (!bias_name.empty()) {
      const auto& bias = ValueOf(bias_name, node.op_type());
      if (!(bias.xnn_shape.size() == 1 && bias.xnn_shape[0] == n)) {
        throw std::runtime_error(
            "xnnpack codegen: " + node.op_type() +
            ": bias must be 1-D with shape [N] (a broadcasting scalar or "
            "[M, N] bias is not supported)");
      }
      bias_id_var = bias.id_var;
    }
    const std::string flags =
        b_is_transposed ? "0" : "XNN_FLAG_TRANSPOSE_WEIGHTS";
    auto& out =
        DefineOutputValue(node.output(0), {m, n}, /*is_4d_permuted=*/false);
    build_ << "  ONNXSIM_XNN_CHECK(xnn_define_fully_connected(sg, -INFINITY, "
              "INFINITY, "
           << a.id_var << ", " << b.id_var << ", " << bias_id_var << ", "
           << out.id_var << ", " << flags << "));\n";
  }

  void EmitGemm(const onnx::NodeProto& node) {
    if (node.input_size() < 2 || node.input_size() > 3 ||
        node.output_size() != 1) {
      throw std::runtime_error(
          "xnnpack codegen: Gemm must have 2 or 3 inputs and 1 output");
    }
    const float alpha = AttrFloat(node, "alpha", 1.0f);
    const float beta = AttrFloat(node, "beta", 1.0f);
    const int64_t transA = AttrInt(node, "transA", 0);
    const int64_t transB = AttrInt(node, "transB", 0);
    if (alpha != 1.0f) {
      throw std::runtime_error(
          "xnnpack codegen: Gemm alpha != 1 is not supported");
    }
    if (transA != 0) {
      throw std::runtime_error(
          "xnnpack codegen: Gemm transA != 0 is not supported");
    }
    std::string bias_name;
    if (node.input_size() == 3) {
      if (beta == 1.0f) {
        bias_name = node.input(2);
      } else if (beta != 0.0f) {
        throw std::runtime_error("xnnpack codegen: Gemm beta must be 0 or 1");
      }
    }
    EmitMatMulLike(node, node.input(0), node.input(1), bias_name, transB != 0);
  }

  void EmitMatMul(const onnx::NodeProto& node) {
    if (node.input_size() != 2 || node.output_size() != 1) {
      throw std::runtime_error(
          "xnnpack codegen: MatMul must have 2 inputs and 1 output");
    }
    EmitMatMulLike(node, node.input(0), node.input(1), "",
                   /*b_is_transposed=*/false);
  }

  // See the module comment: only safe when neither side has a real (>1)
  // spatial extent to reorder.
  void EmitReshape(const onnx::NodeProto& node) {
    if (node.input_size() < 1 || node.output_size() != 1) {
      throw std::runtime_error(
          "xnnpack codegen: Reshape must have >=1 input and 1 output");
    }
    const auto& in = ValueOf(node.input(0), "Reshape");
    const auto& out_onnx_shape = RequireShape(node.output(0), "Reshape output");

    const bool in_is_flatten_safe =
        !in.is_4d_permuted || (in.xnn_shape[1] == 1 && in.xnn_shape[2] == 1);
    const bool out_is_flatten_safe =
        out_onnx_shape.size() != 4 ||
        (out_onnx_shape[2] == 1 && out_onnx_shape[3] == 1);
    if (!in_is_flatten_safe || !out_is_flatten_safe) {
      throw std::runtime_error(
          "xnnpack codegen: Reshape '" + node.name() +
          "' would reorder a real (non-1x1) spatial map between ONNX's NCHW "
          "and this generator's NHWC layout -- not supported in v1. This is "
          "safe immediately after GlobalAveragePool (H=W=1, where flatten "
          "order does not depend on layout); route a genuine multi-pixel "
          "flatten through that instead.");
    }
    if (NumElements(in.xnn_shape) != NumElements(out_onnx_shape)) {
      throw std::runtime_error("xnnpack codegen: Reshape '" + node.name() +
                               "': element count mismatch");
    }
    auto& out = DefineOutputValue(node.output(0), out_onnx_shape,
                                  /*is_4d_permuted=*/false);
    build_ << "  ONNXSIM_XNN_CHECK(xnn_define_static_reshape(sg, "
           << out.xnn_shape.size() << ", (size_t[]) "
           << FormatSizeArrayLiteral(out.xnn_shape) << ", " << in.id_var << ", "
           << out.id_var << ", 0));\n";
  }

  void EmitGlobalAveragePool(const onnx::NodeProto& node) {
    if (node.input_size() != 1 || node.output_size() != 1) {
      throw std::runtime_error(
          "xnnpack codegen: GlobalAveragePool must have 1 input and 1 output");
    }
    const auto& in = ValueOf(node.input(0), "GlobalAveragePool");
    if (in.xnn_shape.size() != 4) {
      throw std::runtime_error("xnnpack codegen: GlobalAveragePool '" +
                               node.name() + "' input must be 4-D");
    }
    // Reduce over the (already-NHWC) H, W axes (1, 2), producing [N, C]
    // directly -- see DefineOutputValue's comment on why this is rank 2
    // rather than the [N, C, 1, 1] ONNX itself declares.
    const std::vector<int64_t> out_shape = {in.xnn_shape[0], in.xnn_shape[3]};
    auto& out =
        DefineOutputValue(node.output(0), out_shape, /*is_4d_permuted=*/false);
    build_
        << "  ONNXSIM_XNN_CHECK(xnn_define_static_reduce(sg, xnn_reduce_mean, "
           "2, (size_t[]){1, 2}, "
        << in.id_var << ", " << out.id_var << ", 0));\n";
  }

  void EmitConv(const onnx::NodeProto& node) {
    if (node.input_size() < 2 || node.input_size() > 3 ||
        node.output_size() != 1) {
      throw std::runtime_error(
          "xnnpack codegen: Conv must have 2 or 3 inputs and 1 output");
    }
    const auto& in = ValueOf(node.input(0), "Conv");
    if (!in.is_4d_permuted) {
      throw std::runtime_error("xnnpack codegen: Conv '" + node.name() +
                               "' input must be 4-D");
    }
    const auto& filter_shape = RequireShape(node.input(1), "Conv filter");
    if (filter_shape.size() != 4) {
      throw std::runtime_error("xnnpack codegen: Conv '" + node.name() +
                               "' filter must be 4-D");
    }
    const int64_t cin = in.xnn_shape[3];
    const int64_t cout = filter_shape[0];
    const int64_t icg = filter_shape[1];  // per ONNX: in_channels / groups
    const int64_t kh = filter_shape[2];
    const int64_t kw = filter_shape[3];
    const int64_t groups = AttrInt(node, "group", 1);
    if (cin != icg * groups) {
      throw std::runtime_error(
          "xnnpack codegen: Conv '" + node.name() +
          "': filter's input-channel dimension is "
          "inconsistent with the input tensor and group count");
    }

    const auto strides = AttrInts(node, "strides", {1, 1});
    const auto dilations = AttrInts(node, "dilations", {1, 1});
    const auto& out_onnx_shape = RequireShape(node.output(0), "Conv output");
    if (out_onnx_shape.size() != 4) {
      throw std::runtime_error("xnnpack codegen: Conv '" + node.name() +
                               "' output must be 4-D");
    }
    const int64_t out_h = out_onnx_shape[2], out_w = out_onnx_shape[3];

    int64_t pad_top, pad_left, pad_bottom, pad_right;
    ResolveConvPadding(node, in.xnn_shape[1], in.xnn_shape[2], kh, kw, strides,
                       dilations, out_h, out_w, &pad_top, &pad_left,
                       &pad_bottom, &pad_right);

    std::string bias_id_var = "XNN_INVALID_VALUE_ID";
    if (node.input_size() == 3 && !node.input(2).empty()) {
      const auto& bias = ValueOf(node.input(2), "Conv bias");
      if (bias.xnn_shape.size() != 1 || bias.xnn_shape[0] != cout) {
        throw std::runtime_error(
            "xnnpack codegen: Conv '" + node.name() +
            "': bias must be 1-D with shape [out_channels]");
      }
      bias_id_var = bias.id_var;
    }

    const bool is_depthwise = groups == cin && icg == 1;
    const std::vector<int64_t> out_xnn_shape = {out_onnx_shape[0], out_h, out_w,
                                                cout};
    if (is_depthwise) {
      const int64_t depth_multiplier = cout / cin;
      TensorInfo filter = DefineFilterConstant(
          node.input(1), kDepthwiseFilterPerm, "Conv filter");
      auto& out = DefineOutputValue(node.output(0), out_xnn_shape,
                                    /*is_4d_permuted=*/true);
      build_ << "  ONNXSIM_XNN_CHECK(xnn_define_depthwise_convolution_2d(sg, "
             << pad_top << ", " << pad_right << ", " << pad_bottom << ", "
             << pad_left << ", " << kh << ", " << kw << ", " << strides[0]
             << ", " << strides[1] << ", " << dilations[0] << ", "
             << dilations[1] << ", " << depth_multiplier << ", " << cin
             << ", -INFINITY, INFINITY, " << in.id_var << ", " << filter.id_var
             << ", " << bias_id_var << ", " << out.id_var << ", 0));\n";
    } else {
      TensorInfo filter =
          DefineFilterConstant(node.input(1), kNCHWToNHWC, "Conv filter");
      auto& out = DefineOutputValue(node.output(0), out_xnn_shape,
                                    /*is_4d_permuted=*/true);
      build_ << "  ONNXSIM_XNN_CHECK(xnn_define_convolution_2d(sg, " << pad_top
             << ", " << pad_right << ", " << pad_bottom << ", " << pad_left
             << ", " << kh << ", " << kw << ", " << strides[0] << ", "
             << strides[1] << ", " << dilations[0] << ", " << dilations[1]
             << ", " << groups << ", " << icg << ", " << (cout / groups)
             << ", -INFINITY, INFINITY, " << in.id_var << ", " << filter.id_var
             << ", " << bias_id_var << ", " << out.id_var << ", 0));\n";
    }
  }

  // ONNX Conv padding (NOTSET/VALID/SAME_UPPER/SAME_LOWER) resolved to the
  // four explicit XNNPACK padding parameters. `out_h`/`out_w` come from this
  // model's own shape inference (already-known, so SAME_*'s "pad_needed"
  // formula -- which the ONNX spec defines in terms of the *output* size --
  // needs no separate re-derivation here).
  void ResolveConvPadding(const onnx::NodeProto& node, int64_t in_h,
                          int64_t in_w, int64_t kh, int64_t kw,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations, int64_t out_h,
                          int64_t out_w, int64_t* pad_top, int64_t* pad_left,
                          int64_t* pad_bottom, int64_t* pad_right) {
    const std::string auto_pad = AttrString(node, "auto_pad", "NOTSET");
    if (auto_pad == "NOTSET") {
      const auto pads = AttrInts(node, "pads", {0, 0, 0, 0});
      if (pads.size() != 4) {
        throw std::runtime_error("xnnpack codegen: Conv '" + node.name() +
                                 "': pads must have 4 values for a 2-D Conv");
      }
      *pad_top = pads[0];
      *pad_left = pads[1];
      *pad_bottom = pads[2];
      *pad_right = pads[3];
      return;
    }
    if (auto_pad == "VALID") {
      *pad_top = *pad_left = *pad_bottom = *pad_right = 0;
      return;
    }
    if (auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") {
      throw std::runtime_error("xnnpack codegen: Conv '" + node.name() +
                               "': unsupported auto_pad '" + auto_pad + "'");
    }
    auto pad_needed = [&](int64_t in_dim, int64_t out_dim, int64_t stride,
                          int64_t kernel, int64_t dilation) {
      const int64_t effective_kernel = (kernel - 1) * dilation + 1;
      const int64_t needed = (out_dim - 1) * stride + effective_kernel - in_dim;
      return std::max<int64_t>(0, needed);
    };
    const int64_t need_h =
        pad_needed(in_h, out_h, strides[0], kh, dilations[0]);
    const int64_t need_w =
        pad_needed(in_w, out_w, strides[1], kw, dilations[1]);
    if (auto_pad == "SAME_UPPER") {
      *pad_top = need_h / 2;
      *pad_bottom = need_h - *pad_top;
      *pad_left = need_w / 2;
      *pad_right = need_w - *pad_left;
    } else {
      *pad_bottom = need_h / 2;
      *pad_top = need_h - *pad_bottom;
      *pad_right = need_w / 2;
      *pad_left = need_w - *pad_right;
    }
  }

  std::string AssembleFile(int num_inputs, int num_outputs) {
    const auto& graph = model_.graph();
    std::ostringstream out;
    out << "/* Generated by onnxsim's XNNPACK C emitter -- DO NOT EDIT BY "
           "HAND.\n"
        << " *\n"
        << " * Layout convention: every 4-D input/output tensor listed below "
           "is\n"
        << " * NHWC (batch, height, width, channels) -- NOT ONNX's own NCHW "
           "--\n"
        << " * matching XNNPACK's native convolution layout and the natural "
           "memory\n"
        << " * layout of an interleaved image buffer (e.g. a cv::Mat -- see\n"
        << " * onnxsim/xnnpack_cv_mat.hpp). Tensors of any other rank are "
           "unchanged\n"
        << " * from the source model's own shape.\n"
        << " *\n"
        << " * Inputs (positional, same order as " << prefix_
        << "_run's `inputs`):\n";
    for (int i = 0; i < num_inputs; ++i) {
      const auto& vi = graph.input(i);
      out << " *   [" << i << "] \"" << vi.name() << "\"\n";
    }
    out << " * Outputs (positional, same order as " << prefix_
        << "_run's `outputs`):\n";
    for (int i = 0; i < num_outputs; ++i) {
      const auto& vi = graph.output(i);
      out << " *   [" << i << "] \"" << vi.name() << "\"\n";
    }
    out << " */\n"
        << "#include <xnnpack.h>\n"
        << "#include <math.h>\n"
        << "#include <stddef.h>\n"
        << "#include <stdint.h>\n"
        << "#include <string.h>\n\n"
        << "#define ONNXSIM_XNN_CHECK(expr) do { enum xnn_status _s = (expr); "
           "if (_s != xnn_status_success) return _s; } while (0)\n\n";

    out << data_.str() << "\n";
    out << build_.str() << "\n";

    out << "typedef struct {\n  xnn_subgraph_t subgraph;\n  xnn_runtime_t "
           "runtime;\n} "
        << prefix_ << "_model_t;\n\n";

    out << "int " << prefix_ << "_create(" << prefix_ << "_model_t* model) {\n"
        << "  ONNXSIM_XNN_CHECK(xnn_initialize(NULL));\n"
        << "  model->subgraph = NULL;\n  model->runtime = NULL;\n"
        << "  ONNXSIM_XNN_CHECK(" << prefix_
        << "_build_subgraph(&model->subgraph));\n"
        << "  ONNXSIM_XNN_CHECK(xnn_create_runtime_v4(model->subgraph, NULL, "
           "NULL, "
           "NULL, 0, &model->runtime));\n"
        << "  /* Every shape in this model is static (fixed at generation "
           "time), so\n"
           "   * one reshape here is enough for the runtime's whole lifetime "
           "-- unlike\n"
           "   * a dynamic-shape XNNPACK user, "
        << prefix_ << "_run never needs to call xnn_reshape_runtime again. */\n"
        << "  ONNXSIM_XNN_CHECK(xnn_reshape_runtime(model->runtime));\n"
        << "  return xnn_status_success;\n}\n\n";

    out << "int " << prefix_ << "_run(" << prefix_
        << "_model_t* model, const float* const* inputs, float* const* "
           "outputs) {\n"
        << "  struct xnn_external_value values[" << (num_inputs + num_outputs)
        << "];\n";
    for (int i = 0; i < num_inputs; ++i) {
      out << "  values[" << i << "].id = " << i << "; values[" << i
          << "].data = (void*)inputs[" << i << "];\n";
    }
    for (int i = 0; i < num_outputs; ++i) {
      out << "  values[" << (num_inputs + i) << "].id = " << (num_inputs + i)
          << "; values[" << (num_inputs + i) << "].data = (void*)outputs[" << i
          << "];\n";
    }
    out << "  ONNXSIM_XNN_CHECK(xnn_setup_runtime_v2(model->runtime, "
        << (num_inputs + num_outputs) << ", values));\n"
        << "  ONNXSIM_XNN_CHECK(xnn_invoke_runtime(model->runtime));\n"
        << "  return xnn_status_success;\n}\n\n";

    out << "void " << prefix_ << "_destroy(" << prefix_
        << "_model_t* model) {\n"
        << "  if (model->runtime) xnn_delete_runtime(model->runtime);\n"
        << "  if (model->subgraph) xnn_delete_subgraph(model->subgraph);\n"
        << "  model->runtime = NULL;\n  model->subgraph = NULL;\n}\n";
    return out.str();
  }
};

}  // namespace

std::string GenerateXnnpackC(const onnx::ModelProto& model,
                             const std::string& function_prefix) {
  if (!IsValidCIdent(function_prefix)) {
    throw std::invalid_argument(
        "xnnpack codegen: function_prefix must be a valid C identifier, got '" +
        function_prefix + "'");
  }
  onnx::ModelProto inferred = model;
  onnx::shape_inference::InferShapes(inferred);
  Emitter emitter(inferred, function_prefix);
  return emitter.Run();
}

}  // namespace xnnpack_backend
}  // namespace onnxsim
