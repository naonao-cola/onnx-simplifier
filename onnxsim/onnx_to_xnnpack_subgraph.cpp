/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * See onnx_to_xnnpack_subgraph.h for the design/scope of this lowering.
 */
#include "onnx_to_xnnpack_subgraph.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "dlpack_bridge.h"
#include "dlpack_dtype.h"

namespace onnxsim {
namespace xnnpack_backend {

namespace {

const char* StatusToString(xnn_status s) {
  switch (s) {
    case xnn_status_success:
      return "success";
    case xnn_status_uninitialized:
      return "uninitialized";
    case xnn_status_invalid_parameter:
      return "invalid_parameter";
    case xnn_status_invalid_state:
      return "invalid_state";
    case xnn_status_unsupported_parameter:
      return "unsupported_parameter";
    case xnn_status_unsupported_hardware:
      return "unsupported_hardware";
    case xnn_status_out_of_memory:
      return "out_of_memory";
    case xnn_status_reallocation_required:
      return "reallocation_required";
    default:
      return "unknown xnn_status";
  }
}

void CheckStatus(xnn_status s, const std::string& what) {
  if (s != xnn_status_success) {
    throw std::runtime_error("xnnpack backend: " + what +
                             " failed: " + StatusToString(s));
  }
}

std::vector<size_t> ToSizeVec(const std::vector<int64_t>& dims) {
  std::vector<size_t> out(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      throw std::runtime_error(
          "xnnpack backend: negative dimension in a resolved shape");
    }
    out[i] = static_cast<size_t>(dims[i]);
  }
  return out;
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

// ONNX/numpy multidirectional broadcasting of two shapes.
std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a,
                                    const std::vector<int64_t>& b) {
  const size_t n = std::max(a.size(), b.size());
  std::vector<int64_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    const int64_t ad = i < n - a.size() ? 1 : a[i - (n - a.size())];
    const int64_t bd = i < n - b.size() ? 1 : b[i - (n - b.size())];
    if (ad != bd && ad != 1 && bd != 1) {
      throw std::runtime_error(
          "xnnpack backend: shapes are not broadcast-compatible");
    }
    out[i] = std::max(ad, bd);
  }
  return out;
}

// Resolves Reshape's -1 (infer) / 0 (copy from input) target-dim conventions
// (allowzero=0, the ONNX default) against the reshaped tensor's input shape.
std::vector<int64_t> ResolveReshapeTarget(const std::vector<int64_t>& in_shape,
                                          const std::vector<int64_t>& target) {
  std::vector<int64_t> out(target.size());
  int64_t infer_axis = -1;
  int64_t known_product = 1;
  for (size_t i = 0; i < target.size(); ++i) {
    if (target[i] == -1) {
      if (infer_axis != -1) {
        throw std::runtime_error(
            "xnnpack backend: Reshape target has more than one -1");
      }
      infer_axis = static_cast<int64_t>(i);
      out[i] = -1;
    } else if (target[i] == 0) {
      if (i >= in_shape.size()) {
        throw std::runtime_error(
            "xnnpack backend: Reshape target uses 0 (copy from input) past "
            "the input tensor's rank");
      }
      out[i] = in_shape[i];
      known_product *= out[i];
    } else {
      out[i] = target[i];
      known_product *= out[i];
    }
  }
  if (infer_axis != -1) {
    const int64_t total = NumElements(in_shape);
    if (known_product == 0 || total % known_product != 0) {
      throw std::runtime_error(
          "xnnpack backend: Reshape target's -1 dimension is not evenly "
          "determined by the input's element count");
    }
    out[infer_axis] = total / known_product;
  }
  return out;
}

const onnx::AttributeProto* FindAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == name) return &attr;
  }
  return nullptr;
}

float AttrFloat(const onnx::NodeProto& node, const std::string& name,
                float default_value) {
  const auto* attr = FindAttr(node, name);
  return attr != nullptr ? attr->f() : default_value;
}

int64_t AttrInt(const onnx::NodeProto& node, const std::string& name,
                int64_t default_value) {
  const auto* attr = FindAttr(node, name);
  return attr != nullptr ? attr->i() : default_value;
}

// Maps an ONNX quantized element dtype to the matching XNNPACK datatype.
// Only int8/uint8 -- the two ONNX quantization uses (QuantizeLinear/
// DequantizeLinear/QLinearMatMul's int8 or uint8 tensors), never int32
// (QLinearMatMul-style bias accumulation is not part of this lowering's
// quantized op set) or the sub-8-bit/float8 types XNNPACK also has
// datatypes for.
xnn_datatype MapQuantDtype(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case onnx::TensorProto::INT8:
      return xnn_datatype_qint8;
    case onnx::TensorProto::UINT8:
      return xnn_datatype_quint8;
    default:
      throw std::runtime_error(
          "xnnpack backend: a quantized tensor must be int8 or uint8");
  }
}

// Threaded through every per-node lowering function below. Owns nothing
// itself -- `subgraph` and `owned_tensors` are borrowed from the
// LoweredSubgraph the caller (Lower(), at the bottom of this file) is
// building, and outlive this context.
struct LoweringContext {
  const onnx::ModelProto& model;
  const std::vector<const DLManagedTensor*>& inputs;
  xnn_subgraph_t subgraph = nullptr;

  std::unordered_map<std::string, uint32_t> value_ids;
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  std::unordered_map<std::string, const onnx::TensorProto*> initializers;
  std::unordered_map<std::string, int> graph_input_index;
  // Graph-output name -> its reserved XNNPACK external id. Erased as each
  // output gets defined, so anything left once every node has been lowered
  // means the graph declared an output no node in it actually produces.
  std::unordered_map<std::string, uint32_t> pending_outputs;
  std::vector<DLManagedTensorPtr>* owned_tensors = nullptr;
  // See LoweredSubgraph::owned_scale_arrays -- backs a per-channel
  // quantized Value's scale array, which XNNPACK stores as a bare pointer.
  std::vector<std::vector<float>>* owned_scale_arrays = nullptr;
  // Set once in Lower() before any node is lowered. Needed to turn an
  // external output id back into an index into `*output_dtypes` (output ids
  // are reserved as [num_inputs, num_inputs + num_outputs), one per
  // model.graph().output() position -- see Lower()).
  uint32_t num_inputs = 0;
  std::vector<DLDataType>* output_dtypes = nullptr;

  const std::vector<int64_t>& ShapeOf(const std::string& name) const {
    auto it = shapes.find(name);
    if (it == shapes.end()) {
      throw std::runtime_error(
          "xnnpack backend: internal error, no known shape for '" + name + "'");
    }
    return it->second;
  }

  // Returns the XNNPACK Value ID for `name`, defining it as an internal
  // constant Value (borrowing the initializer's data, via dlpack_bridge so a
  // big-endian host gets a byte-swapped copy) the first time it is asked for.
  // Throws if `name` is neither already-defined nor a graph initializer --
  // i.e. it would have to be the output of a node not yet lowered, which
  // cannot happen for a topologically-ordered graph feeding only forward
  // references, or is simply unknown.
  uint32_t GetOrDefineValueId(const std::string& name) {
    auto it = value_ids.find(name);
    if (it != value_ids.end()) return it->second;

    auto init_it = initializers.find(name);
    if (init_it == initializers.end()) {
      throw std::runtime_error(
          "xnnpack backend: value '" + name +
          "' is neither a graph input, an initializer, nor a prior node "
          "output");
    }
    const onnx::TensorProto& tp = *init_it->second;
    if (tp.data_type() != onnx::TensorProto::FLOAT) {
      throw std::runtime_error("xnnpack backend: initializer '" + name +
                               "' is not fp32 (only fp32 is supported)");
    }
    DLManagedTensor* dlt = dlpack::FromTensorProtoBorrowing(tp);
    owned_tensors->emplace_back(dlt);
    std::vector<int64_t> shape(dlt->dl_tensor.shape,
                               dlt->dl_tensor.shape + dlt->dl_tensor.ndim);
    auto dims = ToSizeVec(shape);
    uint32_t id;
    CheckStatus(xnn_define_tensor_value(
                    subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
                    dlt->dl_tensor.data, XNN_INVALID_VALUE_ID, 0, &id),
                "define initializer '" + name + "'");
    value_ids[name] = id;
    shapes[name] = std::move(shape);
    return id;
  }

  // Like GetOrDefineValueId, but for a Value that is either already defined
  // (in which case `scale_name`/`zero_point_name`/`axis` are ignored -- the
  // Value keeps whatever quantization it was first defined with) or a
  // constant int8/uint8 initializer, quantized either per-tensor
  // (`scale_name` resolves to exactly one value) or per-channel
  // (`scale_name` resolves to one value per channel along `axis`, and
  // `zero_point_name` must then be all zero --
  // xnn_define_channelwise_quantized_tensor_value has no per-channel
  // zero-point parameter, i.e. only symmetric per-channel quantization is
  // supported).
  uint32_t GetOrDefineQuantizedValueId(const std::string& name,
                                       const std::string& scale_name,
                                       const std::string& zero_point_name,
                                       int64_t axis) {
    auto it = value_ids.find(name);
    if (it != value_ids.end()) return it->second;

    auto init_it = initializers.find(name);
    if (init_it == initializers.end()) {
      throw std::runtime_error(
          "xnnpack backend: value '" + name +
          "' is neither a graph input, an initializer, nor a prior node "
          "output");
    }
    const onnx::TensorProto& tp = *init_it->second;
    const xnn_datatype dtype = MapQuantDtype(tp.data_type());
    DLManagedTensor* dlt = dlpack::FromTensorProtoBorrowing(tp);
    owned_tensors->emplace_back(dlt);
    std::vector<int64_t> shape(dlt->dl_tensor.shape,
                               dlt->dl_tensor.shape + dlt->dl_tensor.ndim);
    auto dims = ToSizeVec(shape);

    auto scales = ReadFloatValues(scale_name);
    uint32_t id;
    if (scales.size() == 1) {
      int32_t zp_dtype;
      const auto zps = ReadZeroPointValues(zero_point_name, &zp_dtype);
      if (zps.size() != 1) {
        throw std::runtime_error("xnnpack backend: '" + zero_point_name +
                                 "' must have exactly one element for "
                                 "per-tensor quantization of '" +
                                 name + "'");
      }
      if (zp_dtype != tp.data_type()) {
        throw std::runtime_error("xnnpack backend: '" + name +
                                 "' and its zero point '" + zero_point_name +
                                 "' must have the same int8/uint8 dtype");
      }
      CheckStatus(
          xnn_define_quantized_tensor_value(
              subgraph, dtype, zps[0], scales[0], dims.size(), dims.data(),
              dlt->dl_tensor.data, XNN_INVALID_VALUE_ID, 0, &id),
          "define quantized initializer '" + name + "'");
    } else {
      if (axis < 0 || static_cast<size_t>(axis) >= shape.size() ||
          shape[static_cast<size_t>(axis)] !=
              static_cast<int64_t>(scales.size())) {
        throw std::runtime_error("xnnpack backend: '" + scale_name +
                                 "' does not match '" + name +
                                 "'s shape at axis " + std::to_string(axis));
      }
      // XNNPACK's per-channel datatypes are a distinct enum from their
      // per-tensor counterparts (xnn_datatype_qcint8, not
      // xnn_datatype_qint8) and only exist for signed data -- there is no
      // per-channel unsigned-int8 datatype.
      if (tp.data_type() != onnx::TensorProto::INT8) {
        throw std::runtime_error(
            "xnnpack backend: per-channel quantization of '" + name +
            "' is only supported for int8 (not uint8)");
      }
      int32_t zp_dtype;
      const auto zps = ReadZeroPointValues(zero_point_name, &zp_dtype);
      if (!std::all_of(zps.begin(), zps.end(),
                       [](int32_t z) { return z == 0; })) {
        throw std::runtime_error(
            "xnnpack backend: per-channel zero point for '" + name +
            "' must be all zero (only symmetric per-channel quantization "
            "is supported)");
      }
      // xnn_define_channelwise_quantized_tensor_value stores `scale` as a
      // bare pointer (xnn_value.quantization.channelwise_scale), read again
      // whenever a runtime is later created from this subgraph -- so the
      // array must outlive this function call. Move it into
      // owned_scale_arrays (parallel to owned_tensors' handling of tensor
      // data) rather than passing the local `scales`' pointer directly,
      // which would dangle the moment this function returns.
      owned_scale_arrays->push_back(std::move(scales));
      const float* scale_ptr = owned_scale_arrays->back().data();
      CheckStatus(xnn_define_channelwise_quantized_tensor_value(
                      subgraph, xnn_datatype_qcint8, scale_ptr, dims.size(),
                      static_cast<size_t>(axis), dims.data(),
                      dlt->dl_tensor.data, XNN_INVALID_VALUE_ID, 0, &id),
                  "define channelwise-quantized initializer '" + name + "'");
    }
    value_ids[name] = id;
    shapes[name] = std::move(shape);
    return id;
  }

  // Resolves `name` to its concrete DLTensor -- either a graph input's
  // tensor directly, or (for an initializer) a freshly materialized
  // DLManagedTensor owned via `owned_tensors`. `name` must be a graph input
  // or an initializer, not the output of another node in this fold group:
  // every Read*Values helper below reads a compile-time-static argument
  // (a shape, a permutation, a quantization scale/zero-point), never a
  // runtime tensor. Shared by every Read*Values helper; each does its own
  // dtype check since the expected dtype differs per caller.
  const DLTensor& ResolveStaticTensor(const std::string& name) {
    auto gi = graph_input_index.find(name);
    if (gi != graph_input_index.end()) {
      return inputs[gi->second]->dl_tensor;
    }
    auto init_it = initializers.find(name);
    if (init_it != initializers.end()) {
      DLManagedTensor* dlt = dlpack::FromTensorProtoBorrowing(*init_it->second);
      owned_tensors->emplace_back(dlt);
      return owned_tensors->back()->dl_tensor;
    }
    throw std::runtime_error(
        "xnnpack backend: '" + name +
        "' must be a constant or a graph input to be used as a static value "
        "(this lowering does not support one produced by another node)");
  }

  // Reads out the actual int64 values of `name` -- used for Reshape's target
  // shape.
  std::vector<int64_t> ReadInt64Values(const std::string& name) {
    const DLTensor& t = ResolveStaticTensor(name);
    if (!(t.dtype.code == kDLInt && t.dtype.bits == 64)) {
      throw std::runtime_error("xnnpack backend: '" + name +
                               "' must be int64 to be used as a static shape");
    }
    const auto* data = reinterpret_cast<const int64_t*>(
        static_cast<const uint8_t*>(t.data) + t.byte_offset);
    return std::vector<int64_t>(data,
                                data + dlpack::NumElements(t.shape, t.ndim));
  }

  // Reads out the actual fp32 values of `name` -- used for QuantizeLinear /
  // DequantizeLinear / QLinearMatMul's scale tensors (per-tensor: one
  // element; per-channel: one per output channel).
  std::vector<float> ReadFloatValues(const std::string& name) {
    const DLTensor& t = ResolveStaticTensor(name);
    if (!(t.dtype.code == kDLFloat && t.dtype.bits == 32)) {
      throw std::runtime_error("xnnpack backend: '" + name +
                               "' must be fp32 to be used as a static scale");
    }
    const auto* data = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(t.data) + t.byte_offset);
    return std::vector<float>(data,
                              data + dlpack::NumElements(t.shape, t.ndim));
  }

  // Reads out the values of `name` (int8 or uint8) widened to int32 -- used
  // for QuantizeLinear/DequantizeLinear/QLinearMatMul's zero-point tensors.
  // `*onnx_dtype_out` is set to the tensor's own ONNX dtype (TensorProto::
  // INT8 or ::UINT8), which callers use (via MapQuantDtype) to determine a
  // quantized Value's own datatype.
  std::vector<int32_t> ReadZeroPointValues(const std::string& name,
                                           int32_t* onnx_dtype_out) {
    const DLTensor& t = ResolveStaticTensor(name);
    const int64_t n = dlpack::NumElements(t.shape, t.ndim);
    const auto* data = static_cast<const uint8_t*>(t.data) + t.byte_offset;
    std::vector<int32_t> out(n);
    if (t.dtype.code == kDLInt && t.dtype.bits == 8) {
      *onnx_dtype_out = onnx::TensorProto::INT8;
      for (int64_t i = 0; i < n; ++i) {
        out[i] = reinterpret_cast<const int8_t*>(data)[i];
      }
    } else if (t.dtype.code == kDLUInt && t.dtype.bits == 8) {
      *onnx_dtype_out = onnx::TensorProto::UINT8;
      for (int64_t i = 0; i < n; ++i) out[i] = data[i];
    } else {
      throw std::runtime_error("xnnpack backend: '" + name +
                               "' must be int8 or uint8 to be used as a "
                               "zero point");
    }
    return out;
  }

  // Resolves the external id (and its flag) `name` should use when it is
  // first defined as an output Value: the reserved external-output id if
  // `name` is one of the graph's declared outputs, or XNN_INVALID_VALUE_ID
  // (auto-assigned internal id) otherwise. Shared by DefineOutputValue and
  // DefineQuantizedOutputValue below.
  std::pair<uint32_t, uint32_t> ResolveOutputExternalId(
      const std::string& name) {
    auto it = pending_outputs.find(name);
    if (it == pending_outputs.end()) {
      return {XNN_INVALID_VALUE_ID, 0u};
    }
    const uint32_t external_id = it->second;
    pending_outputs.erase(it);
    return {external_id, static_cast<uint32_t>(XNN_VALUE_FLAG_EXTERNAL_OUTPUT)};
  }

  // Defines the output Value for a just-lowered node: as the reserved
  // external-output Value if `name` is one of the graph's declared outputs,
  // or as an internal Value otherwise. Records its id/shape either way so
  // later nodes can reference it as an input.
  uint32_t DefineOutputValue(const std::string& name,
                             const std::vector<int64_t>& shape) {
    auto dims = ToSizeVec(shape);
    const auto [external_id, flags] = ResolveOutputExternalId(name);
    if (flags != 0) {
      (*output_dtypes)[external_id - num_inputs] = DLDataType{kDLFloat, 32, 1};
    }
    uint32_t id;
    CheckStatus(
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                dims.data(), nullptr, external_id, flags, &id),
        "define value '" + name + "'");
    value_ids[name] = id;
    shapes[name] = shape;
    return id;
  }

  // Like DefineOutputValue, but for a per-tensor quantized (qint8/quint8)
  // Value -- QuantizeLinear's output, or QLinearMatMul's (this lowering only
  // supports a per-tensor y_scale/y_zero_point for both; per-axis output
  // quantization is unusual in practice and unsupported here).
  uint32_t DefineQuantizedOutputValue(const std::string& name,
                                      const std::vector<int64_t>& shape,
                                      xnn_datatype dtype, float scale,
                                      int32_t zero_point) {
    auto dims = ToSizeVec(shape);
    const auto [external_id, flags] = ResolveOutputExternalId(name);
    if (flags != 0) {
      DLDataType dl_dtype;
      if (dtype == xnn_datatype_qint8) {
        dl_dtype = DLDataType{kDLInt, 8, 1};
      } else if (dtype == xnn_datatype_quint8) {
        dl_dtype = DLDataType{kDLUInt, 8, 1};
      } else {
        throw std::runtime_error(
            "xnnpack backend: internal error, unexpected quantized output "
            "dtype");
      }
      (*output_dtypes)[external_id - num_inputs] = dl_dtype;
    }
    uint32_t id;
    CheckStatus(xnn_define_quantized_tensor_value(
                    subgraph, dtype, zero_point, scale, dims.size(),
                    dims.data(), nullptr, external_id, flags, &id),
                "define value '" + name + "'");
    value_ids[name] = id;
    shapes[name] = shape;
    return id;
  }
};

void LowerBinary(LoweringContext& ctx, const onnx::NodeProto& node,
                 xnn_binary_operator type) {
  if (node.input_size() != 2 || node.output_size() != 1) {
    throw std::runtime_error("xnnpack backend: " + node.op_type() +
                             " must have exactly 2 inputs and 1 output");
  }
  const uint32_t in1 = ctx.GetOrDefineValueId(node.input(0));
  const uint32_t in2 = ctx.GetOrDefineValueId(node.input(1));
  const auto out_shape =
      BroadcastShape(ctx.ShapeOf(node.input(0)), ctx.ShapeOf(node.input(1)));
  const uint32_t out = ctx.DefineOutputValue(node.output(0), out_shape);
  const xnn_binary_params params{-std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()};
  CheckStatus(xnn_define_binary(ctx.subgraph, type, &params, in1, in2, out, 0),
              node.op_type() + " '" + node.name() + "'");
}

void LowerUnary(LoweringContext& ctx, const onnx::NodeProto& node,
                xnn_unary_operator type, const xnn_unary_params& params) {
  if (node.input_size() != 1 || node.output_size() != 1) {
    throw std::runtime_error("xnnpack backend: " + node.op_type() +
                             " must have exactly 1 input and 1 output");
  }
  const uint32_t in = ctx.GetOrDefineValueId(node.input(0));
  const uint32_t out =
      ctx.DefineOutputValue(node.output(0), ctx.ShapeOf(node.input(0)));
  CheckStatus(xnn_define_unary(ctx.subgraph, type, &params, in, out, 0),
              node.op_type() + " '" + node.name() + "'");
}

// Shared by Gemm (transA=0, transB in {0,1}) and MatMul (both 2D, no
// transpose/bias) -- both lower to xnn_define_fully_connected, which is
// XNNPACK's only matrix-multiply Node. `b_is_transposed` mirrors ONNX Gemm's
// transB: false means B is [K, N] (needs XNN_FLAG_TRANSPOSE_WEIGHTS, since
// xnn_define_fully_connected's default filter layout is [N, K]); true means B
// is already [N, K].
void LowerMatMulLike(LoweringContext& ctx, const onnx::NodeProto& node,
                     const std::string& a_name, const std::string& b_name,
                     const std::string& bias_name /* may be empty */,
                     bool b_is_transposed) {
  const uint32_t a_id = ctx.GetOrDefineValueId(a_name);
  const uint32_t b_id = ctx.GetOrDefineValueId(b_name);
  const auto& a_shape = ctx.ShapeOf(a_name);
  const auto& b_shape = ctx.ShapeOf(b_name);
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    throw std::runtime_error("xnnpack backend: " + node.op_type() +
                             " only supports 2D operands");
  }
  const int64_t m = a_shape[0], k = a_shape[1];
  const int64_t b_k = b_is_transposed ? b_shape[1] : b_shape[0];
  const int64_t n = b_is_transposed ? b_shape[0] : b_shape[1];
  if (b_k != k) {
    throw std::runtime_error("xnnpack backend: " + node.op_type() +
                             ": incompatible operand shapes");
  }
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (!bias_name.empty()) {
    bias_id = ctx.GetOrDefineValueId(bias_name);
    const auto& c_shape = ctx.ShapeOf(bias_name);
    if (!(c_shape.size() == 1 && c_shape[0] == n)) {
      throw std::runtime_error(
          "xnnpack backend: " + node.op_type() +
          ": bias must be 1-D with shape [N] (broadcasting a scalar or "
          "[M, N] bias is not supported)");
    }
  }
  const uint32_t flags = b_is_transposed ? 0 : XNN_FLAG_TRANSPOSE_WEIGHTS;
  const uint32_t out = ctx.DefineOutputValue(node.output(0), {m, n});
  CheckStatus(xnn_define_fully_connected(
                  ctx.subgraph, -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(), a_id, b_id, bias_id,
                  out, flags),
              node.op_type() + " '" + node.name() + "'");
}

void LowerGemm(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() < 2 || node.input_size() > 3 ||
      node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: Gemm must have 2 or 3 inputs and 1 output");
  }
  const float alpha = AttrFloat(node, "alpha", 1.0f);
  const float beta = AttrFloat(node, "beta", 1.0f);
  const int64_t transA = AttrInt(node, "transA", 0);
  const int64_t transB = AttrInt(node, "transB", 0);
  if (alpha != 1.0f) {
    throw std::runtime_error(
        "xnnpack backend: Gemm alpha != 1 is not supported");
  }
  if (transA != 0) {
    throw std::runtime_error(
        "xnnpack backend: Gemm transA != 0 is not supported");
  }
  std::string bias_name;
  if (node.input_size() == 3) {
    if (beta == 1.0f) {
      bias_name = node.input(2);
    } else if (beta != 0.0f) {
      throw std::runtime_error("xnnpack backend: Gemm beta must be 0 or 1");
    }
    // beta == 0: C is present but contributes nothing; drop it, matching the
    // ONNX spec ("if beta is 0, C is not required to be defined ... its
    // values are ignored").
  }
  LowerMatMulLike(ctx, node, node.input(0), node.input(1), bias_name,
                  /*b_is_transposed=*/transB != 0);
}

void LowerMatMul(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() != 2 || node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: MatMul must have 2 inputs and 1 output");
  }
  LowerMatMulLike(ctx, node, node.input(0), node.input(1), /*bias_name=*/"",
                  /*b_is_transposed=*/false);
}

void LowerReshape(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() != 2 || node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: Reshape must have 2 inputs and 1 output");
  }
  if (AttrInt(node, "allowzero", 0) != 0) {
    throw std::runtime_error(
        "xnnpack backend: Reshape allowzero=1 is not supported");
  }
  const uint32_t in_id = ctx.GetOrDefineValueId(node.input(0));
  const auto& in_shape = ctx.ShapeOf(node.input(0));
  const auto target = ctx.ReadInt64Values(node.input(1));
  const auto out_shape = ResolveReshapeTarget(in_shape, target);
  if (NumElements(out_shape) != NumElements(in_shape)) {
    throw std::runtime_error(
        "xnnpack backend: Reshape target does not preserve element count");
  }
  const uint32_t out = ctx.DefineOutputValue(node.output(0), out_shape);
  const auto dims = ToSizeVec(out_shape);
  CheckStatus(xnn_define_static_reshape(ctx.subgraph, dims.size(), dims.data(),
                                        in_id, out, 0),
              "Reshape '" + node.name() + "'");
}

// QuantizeLinear(x, y_scale, y_zero_point) -> y (int8/uint8). Lowers to an
// XNNPACK "convert" Node between an fp32 Value and a quantized Value --
// XNNPACK's unified subgraph ops (unlike DLPack) determine int8-vs-fp32
// behavior from each Value's own declared datatype, not a separate op
// variant. Per-tensor only (y_scale/y_zero_point are always required here,
// so the output dtype is unambiguous; per-axis activation quantization is
// unusual in practice and unsupported).
void LowerQuantizeLinear(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() != 3 || node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QuantizeLinear must have 3 inputs (x, y_scale, "
        "y_zero_point) and 1 output -- y_zero_point is required here so "
        "the output dtype (int8 vs uint8) is unambiguous");
  }
  const uint32_t in_id = ctx.GetOrDefineValueId(node.input(0));
  const auto& in_shape = ctx.ShapeOf(node.input(0));
  const auto scales = ctx.ReadFloatValues(node.input(1));
  if (scales.size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QuantizeLinear: per-axis quantization is not "
        "supported (y_scale must have exactly one element)");
  }
  int32_t onnx_dtype;
  const auto zps = ctx.ReadZeroPointValues(node.input(2), &onnx_dtype);
  if (zps.size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QuantizeLinear: y_zero_point must have exactly "
        "one element");
  }
  const uint32_t out = ctx.DefineQuantizedOutputValue(
      node.output(0), in_shape, MapQuantDtype(onnx_dtype), scales[0], zps[0]);
  xnn_unary_params params{};
  CheckStatus(
      xnn_define_unary(ctx.subgraph, xnn_unary_convert, &params, in_id, out, 0),
      "QuantizeLinear '" + node.name() + "'");
}

// DequantizeLinear(x, x_scale, x_zero_point) -> y (fp32). The inverse
// convert Node of LowerQuantizeLinear above. `x` is usually the output of a
// preceding QuantizeLinear in this fold group, but may also be a constant
// int8/uint8 initializer dequantized directly (handled the same way
// QLinearMatMul's operands are, via GetOrDefineQuantizedValueId) -- per-
// tensor only, like QuantizeLinear.
void LowerDequantizeLinear(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() != 3 || node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: DequantizeLinear must have 3 inputs (x, x_scale, "
        "x_zero_point) and 1 output");
  }
  const auto scales = ctx.ReadFloatValues(node.input(1));
  if (scales.size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: DequantizeLinear: per-axis quantization is not "
        "supported (x_scale must have exactly one element)");
  }
  const uint32_t in_id = ctx.GetOrDefineQuantizedValueId(
      node.input(0), node.input(1), node.input(2), /*axis=*/1);
  const auto& in_shape = ctx.ShapeOf(node.input(0));
  const uint32_t out = ctx.DefineOutputValue(node.output(0), in_shape);
  xnn_unary_params params{};
  CheckStatus(
      xnn_define_unary(ctx.subgraph, xnn_unary_convert, &params, in_id, out, 0),
      "DequantizeLinear '" + node.name() + "'");
}

// QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale,
// y_zero_point) -> y. Unlike QuantizeLinear/DequantizeLinear (a bridging
// convert Node), this is a genuine int8 compute op -- ONNX's fused
// quantized-matmul, operating directly on already-quantized a/b/y with no
// float in between. Maps to the same xnn_define_fully_connected used for
// Gemm/MatMul above (XNNPACK dispatches on the Values' quantized datatype
// internally); QLinearMatMul has no bias input.
void LowerQLinearMatMul(LoweringContext& ctx, const onnx::NodeProto& node) {
  if (node.input_size() != 8 || node.output_size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QLinearMatMul must have 8 inputs (a, a_scale, "
        "a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point) "
        "and 1 output");
  }
  const std::string& a = node.input(0);
  const std::string& a_scale = node.input(1);
  const std::string& a_zp = node.input(2);
  const std::string& b = node.input(3);
  const std::string& b_scale = node.input(4);
  const std::string& b_zp = node.input(5);
  const std::string& y_scale = node.input(6);
  const std::string& y_zp = node.input(7);

  // a is always per-tensor quantized per the ONNX spec (a_scale is a single
  // value). b may be per-tensor or per-column (b_scale one value per output
  // channel). b's layout here is [K, N] -- QLinearMatMul has no transpose
  // attribute, unlike Gemm -- so its output-channel axis is 1, matching the
  // XNN_FLAG_TRANSPOSE_WEIGHTS passed to xnn_define_fully_connected below
  // (the same flag/layout LowerMatMul above uses for plain fp32 MatMul).
  const uint32_t a_id =
      ctx.GetOrDefineQuantizedValueId(a, a_scale, a_zp, /*axis=*/1);
  const uint32_t b_id =
      ctx.GetOrDefineQuantizedValueId(b, b_scale, b_zp, /*axis=*/1);
  const auto& a_shape = ctx.ShapeOf(a);
  const auto& b_shape = ctx.ShapeOf(b);
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    throw std::runtime_error(
        "xnnpack backend: QLinearMatMul only supports 2D operands");
  }
  const int64_t m = a_shape[0], k = a_shape[1];
  if (b_shape[0] != k) {
    throw std::runtime_error(
        "xnnpack backend: QLinearMatMul: incompatible operand shapes");
  }
  const int64_t n = b_shape[1];

  const auto y_scales = ctx.ReadFloatValues(y_scale);
  if (y_scales.size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QLinearMatMul: per-axis output quantization is "
        "not supported (y_scale must have exactly one element)");
  }
  int32_t y_onnx_dtype;
  const auto y_zps = ctx.ReadZeroPointValues(y_zp, &y_onnx_dtype);
  if (y_zps.size() != 1) {
    throw std::runtime_error(
        "xnnpack backend: QLinearMatMul: y_zero_point must have exactly "
        "one element");
  }
  const uint32_t out = ctx.DefineQuantizedOutputValue(
      node.output(0), {m, n}, MapQuantDtype(y_onnx_dtype), y_scales[0],
      y_zps[0]);
  CheckStatus(xnn_define_fully_connected(
                  ctx.subgraph, -std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::infinity(), a_id, b_id,
                  XNN_INVALID_VALUE_ID, out, XNN_FLAG_TRANSPOSE_WEIGHTS),
              "QLinearMatMul '" + node.name() + "'");
}

void LowerNode(LoweringContext& ctx, const onnx::NodeProto& node) {
  const std::string& op = node.op_type();
  if (op == "Add") return LowerBinary(ctx, node, xnn_binary_add);
  if (op == "Sub") return LowerBinary(ctx, node, xnn_binary_subtract);
  if (op == "Mul") return LowerBinary(ctx, node, xnn_binary_multiply);
  if (op == "Div") return LowerBinary(ctx, node, xnn_binary_divide);
  if (op == "Relu") {
    xnn_unary_params p{};
    p.clamp.min = 0.0f;
    p.clamp.max = std::numeric_limits<float>::infinity();
    return LowerUnary(ctx, node, xnn_unary_clamp, p);
  }
  if (op == "Sigmoid") {
    xnn_unary_params p{};
    return LowerUnary(ctx, node, xnn_unary_sigmoid, p);
  }
  if (op == "Gemm") return LowerGemm(ctx, node);
  if (op == "MatMul") return LowerMatMul(ctx, node);
  if (op == "Reshape") return LowerReshape(ctx, node);
  if (op == "QuantizeLinear") return LowerQuantizeLinear(ctx, node);
  if (op == "DequantizeLinear") return LowerDequantizeLinear(ctx, node);
  if (op == "QLinearMatMul") return LowerQLinearMatMul(ctx, node);
  throw std::runtime_error("xnnpack backend: unsupported op '" + op +
                           "' (node '" + node.name() + "')");
}

}  // namespace

LoweredSubgraph::~LoweredSubgraph() {
  // Delete the subgraph (and any runtime built from it must already have
  // been deleted by the caller -- see onnxsim/xnnpack_executor.cpp) before
  // `owned_tensors` releases the buffers it may still be pointing at.
  if (subgraph != nullptr) {
    xnn_delete_subgraph(subgraph);
  }
}

LoweredSubgraph Lower(const onnx::ModelProto& model,
                      const std::vector<const DLManagedTensor*>& inputs) {
  const auto& graph = model.graph();
  if (static_cast<size_t>(graph.input_size()) != inputs.size()) {
    throw std::invalid_argument("xnnpack backend: Lower() got " +
                                std::to_string(inputs.size()) +
                                " inputs for a graph declaring " +
                                std::to_string(graph.input_size()) + " inputs");
  }

  LoweredSubgraph result;
  LoweringContext ctx{model, inputs};
  ctx.owned_tensors = &result.owned_tensors;
  ctx.owned_scale_arrays = &result.owned_scale_arrays;
  for (const auto& tp : graph.initializer()) {
    ctx.initializers[tp.name()] = &tp;
  }

  const uint32_t num_inputs = static_cast<uint32_t>(graph.input_size());
  const uint32_t num_outputs = static_cast<uint32_t>(graph.output_size());
  ctx.num_inputs = num_inputs;
  result.output_dtypes.assign(num_outputs, DLDataType{});
  ctx.output_dtypes = &result.output_dtypes;
  CheckStatus(xnn_create_subgraph(num_inputs + num_outputs, 0, &ctx.subgraph),
              "create subgraph");
  // Own the subgraph from this point on, so a throw below still cleans it up
  // via ~LoweredSubgraph.
  result.subgraph = ctx.subgraph;

  for (uint32_t i = 0; i < num_inputs; ++i) {
    const auto& vi = graph.input(i);
    const DLTensor& t = inputs[i]->dl_tensor;
    if (!(t.dtype.code == kDLFloat && t.dtype.bits == 32)) {
      throw std::runtime_error("xnnpack backend: graph input '" + vi.name() +
                               "' is not fp32 (only fp32 is supported)");
    }
    if (t.device.device_type != kDLCPU) {
      throw std::runtime_error("xnnpack backend: graph input '" + vi.name() +
                               "' is not a CPU tensor");
    }
    std::vector<int64_t> shape(t.shape, t.shape + t.ndim);
    auto dims = ToSizeVec(shape);
    uint32_t id;
    CheckStatus(xnn_define_tensor_value(ctx.subgraph, xnn_datatype_fp32,
                                        dims.size(), dims.data(), nullptr, i,
                                        XNN_VALUE_FLAG_EXTERNAL_INPUT, &id),
                "define graph input '" + vi.name() + "'");
    ctx.value_ids[vi.name()] = id;
    ctx.shapes[vi.name()] = std::move(shape);
    ctx.graph_input_index[vi.name()] = static_cast<int>(i);
  }

  for (uint32_t j = 0; j < num_outputs; ++j) {
    const std::string& name = graph.output(j).name();
    if (ctx.value_ids.count(name) != 0) {
      throw std::runtime_error(
          "xnnpack backend: graph output '" + name +
          "' aliases a graph input directly, which this lowering does not "
          "support");
    }
    ctx.pending_outputs[name] = num_inputs + j;
  }

  for (const auto& node : graph.node()) {
    LowerNode(ctx, node);
  }

  if (!ctx.pending_outputs.empty()) {
    throw std::runtime_error("xnnpack backend: graph output '" +
                             ctx.pending_outputs.begin()->first +
                             "' is not produced by any node in the graph");
  }

  result.input_value_ids.resize(num_inputs);
  std::iota(result.input_value_ids.begin(), result.input_value_ids.end(), 0);
  result.output_value_ids.resize(num_outputs);
  std::iota(result.output_value_ids.begin(), result.output_value_ids.end(),
            num_inputs);
  return result;
}

}  // namespace xnnpack_backend
}  // namespace onnxsim
