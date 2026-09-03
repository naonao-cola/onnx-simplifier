/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Lowers an onnx::ModelProto (one of constant folding's throwaway fold-group
 * sub-models, see onnxsim/constant_folding.cpp) into an XNNPACK
 * `xnn_subgraph_t` -- the "ONNX to XNNPACK's Subgraph API" translation.
 *
 * This is deliberately NOT a general ONNX importer: it supports a small,
 * explicit set of ops (see kSupportedOps below) and throws
 * std::runtime_error with a specific reason for anything else -- an
 * unsupported op, an unsupported tensor dtype, a shape it cannot resolve
 * statically, an attribute combination it does not implement.
 * onnxsim/xnnpack_executor.h is the ModelExecutor adapter that calls this,
 * creates a runtime from the result, and is the piece that actually invokes
 * XNNPACK.
 *
 * Dtypes: graph inputs/outputs and most ops are fp32 only. QuantizeLinear,
 * DequantizeLinear, and QLinearMatMul additionally support standard ONNX
 * int8/uint8 quantization (XNNPACK's qint8/quint8/qcint8 datatypes) --
 * QuantizeLinear/DequantizeLinear per-tensor only, QLinearMatMul's "b"
 * operand per-tensor or per-column (matching how int8 weights are commonly
 * quantized). There is no support for ONNX Runtime's `com.microsoft`
 * contrib quantized ops (QGemm, QLinearConv, QLinearAdd, ...) -- only
 * standard ONNX ops, matching this lowering's fp32 op set.
 *
 * Shapes: unlike ONNX Runtime (which re-derives shapes from the model itself),
 * XNNPACK's `xnn_define_tensor_value` requires every Value's shape up front,
 * including intermediate node outputs. Rather than depend on `model` already
 * carrying populated ValueInfoProto shape annotations for every intermediate
 * (constant_folding.cpp's fold-group models may or may not), this lowering
 * computes each supported op's output shape itself from its already-known
 * input shapes, in the same pass that walks the graph -- effectively a tiny,
 * self-contained ONNX shape inference limited to the ops this backend
 * supports.
 */
#ifndef ONNXSIM_ONNX_TO_XNNPACK_SUBGRAPH_H_
#define ONNXSIM_ONNX_TO_XNNPACK_SUBGRAPH_H_

#include <onnx/onnx_pb.h>
#include <xnnpack.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "onnxsim.h"

namespace onnxsim {
namespace xnnpack_backend {

// Op types this lowering understands. Anything else in the fold-group graph
// makes Lower() throw std::runtime_error naming the unsupported op.
inline constexpr const char* kSupportedOps[] = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "Sigmoid",
    "Gemm",
    "MatMul",
    "Reshape",
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearMatMul",
};

// Owns an xnn_subgraph_t plus everything it borrows: `subgraph` holds
// pointers into `model`'s initializers (safe -- Lower's caller, and this
// struct's caller in turn, keep `model` alive for at least as long as this
// struct) and into `input_bytes_holders` (owned here, for any tensor this
// lowering itself materializes, e.g. a big-endian-swapped initializer).
struct LoweredSubgraph {
  xnn_subgraph_t subgraph = nullptr;

  // Value IDs for model.graph().input(i) / model.graph().output(j), in the
  // same positional order ModelExecutor::Run uses for `inputs` and its
  // returned tensors. Reserved as XNNPACK external IDs [0, num_inputs) and
  // [num_inputs, num_inputs + num_outputs) respectively -- see Lower()'s
  // implementation.
  std::vector<uint32_t> input_value_ids;
  std::vector<uint32_t> output_value_ids;
  // DLPack dtype of each output_value_ids[i] -- always fp32 before this
  // lowering supported QuantizeLinear/QLinearMatMul, but a quantized op can
  // now make a graph output int8/uint8, so the ModelExecutor adapter
  // (xnnpack_executor.cpp) needs this to allocate the right kind of output
  // buffer and set the right dtype on the DLManagedTensor it returns; it
  // cannot recover this from xnn_get_external_value_shape (shape only).
  std::vector<DLDataType> output_dtypes;

  // Backing storage for any tensor materialized during lowering (currently:
  // only a big-endian host's byte-swapped copy of an initializer -- see
  // dlpack_bridge.h's kRawDataIsHostOrder). Must outlive `subgraph` and any
  // xnn_runtime_t created from it.
  std::vector<DLManagedTensorPtr> owned_tensors;

  // Backing storage for per-channel quantization scale arrays.
  // xnn_define_channelwise_quantized_tensor_value stores the `scale`
  // pointer it is given as-is (xnn_value.quantization.channelwise_scale) --
  // it does not copy the array -- and that pointer is read again whenever a
  // runtime is created from this subgraph, so it needs the same "outlive
  // `subgraph` and any runtime built from it" lifetime `owned_tensors`
  // above documents, just for float scale arrays instead of tensor data.
  std::vector<std::vector<float>> owned_scale_arrays;

  LoweredSubgraph() = default;
  // Not = default: a defaulted move would shallow-copy `subgraph` without
  // nulling the source, so both the moved-from and moved-to object would
  // call xnn_delete_subgraph on the same handle at destruction time.
  LoweredSubgraph(LoweredSubgraph&& other) noexcept
      : subgraph(other.subgraph),
        input_value_ids(std::move(other.input_value_ids)),
        output_value_ids(std::move(other.output_value_ids)),
        output_dtypes(std::move(other.output_dtypes)),
        owned_tensors(std::move(other.owned_tensors)),
        owned_scale_arrays(std::move(other.owned_scale_arrays)) {
    other.subgraph = nullptr;
  }
  LoweredSubgraph& operator=(LoweredSubgraph&& other) noexcept {
    if (this != &other) {
      if (subgraph != nullptr) xnn_delete_subgraph(subgraph);
      subgraph = other.subgraph;
      input_value_ids = std::move(other.input_value_ids);
      output_value_ids = std::move(other.output_value_ids);
      output_dtypes = std::move(other.output_dtypes);
      owned_tensors = std::move(other.owned_tensors);
      owned_scale_arrays = std::move(other.owned_scale_arrays);
      other.subgraph = nullptr;
    }
    return *this;
  }
  LoweredSubgraph(const LoweredSubgraph&) = delete;
  LoweredSubgraph& operator=(const LoweredSubgraph&) = delete;
  ~LoweredSubgraph();
};

// Translate `model`'s graph into an XNNPACK subgraph. `inputs[i]` must
// positionally match model.graph().input(i) (same contract as
// ModelExecutor::Run) and are read (not retained) during lowering, to learn
// graph-input shapes and, for a Reshape target-shape input that is itself a
// graph input rather than an initializer, its actual values.
//
// Throws std::invalid_argument for a malformed call (size mismatch with
// `inputs`) and std::runtime_error for anything this lowering does not
// support -- an op outside kSupportedOps, a non-fp32 tensor, an attribute
// combination not implemented (e.g. Gemm transA=1), or a shape this lowering
// cannot resolve statically (e.g. Reshape's shape tensor produced by another
// node in the same fold group rather than a constant/feed).
LoweredSubgraph Lower(const onnx::ModelProto& model,
                      const std::vector<const DLManagedTensor*>& inputs);

}  // namespace xnnpack_backend
}  // namespace onnxsim

#endif  // ONNXSIM_ONNX_TO_XNNPACK_SUBGRAPH_H_
