/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelExecutor backed by Google's XNNPACK. onnx_to_xnnpack_subgraph.h does
 * the ONNX -> XNNPACK Subgraph API translation; this file is the adapter that
 * turns the resulting xnn_subgraph_t into an xnn_runtime_t, feeds it this
 * call's `inputs`, invokes it, and hands the outputs back as DLManagedTensors
 * -- the same role onnxsim/constant_folding.cpp's CppModelExecutor plays for
 * ONNX Runtime (see docs/dlpack-executor.md's adapters table).
 *
 * Like CppModelExecutor's per-call Ort::Session, this builds a fresh
 * xnn_subgraph_t + xnn_runtime_t on every Run() call: each call is one
 * throwaway constant-folding fold group, not a model meant to be executed
 * repeatedly, so there is nothing to gain from caching either across calls.
 */
#include <xnnpack.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"
#include "onnx_to_xnnpack_subgraph.h"
#include "onnxsim.h"

namespace {

void CheckXnnStatus(xnn_status s, const char* what) {
  if (s != xnn_status_success) {
    throw std::runtime_error(std::string("xnnpack backend: ") + what +
                             " failed (xnn_status=" +
                             std::to_string(static_cast<int>(s)) + ")");
  }
}

void EnsureXnnpackInitialized() {
  static std::once_flag once;
  static xnn_status status = xnn_status_uninitialized;
  std::call_once(once, [] { status = xnn_initialize(/*allocator=*/nullptr); });
  CheckXnnStatus(status, "xnn_initialize");
}

// Backing store for one output, allocated and owned by this executor: unlike
// ONNX Runtime (see dlpack_bridge.h's FromOrtValue), XNNPACK writes external
// outputs into caller-supplied memory rather than handing back its own
// buffer, so Run() below allocates `data` itself and this struct is what the
// returned DLManagedTensor's deleter frees.
struct XnnpackOutputBuffer {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

DLManagedTensor* WrapOutputBuffer(
    std::unique_ptr<XnnpackOutputBuffer> ctx) {
  auto* managed = new DLManagedTensor;
  managed->dl_tensor.data = ctx->data.data();
  managed->dl_tensor.device = DLDevice{kDLCPU, 0};
  managed->dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
  managed->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  managed->dl_tensor.shape = ctx->shape.data();
  managed->dl_tensor.strides = nullptr;
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = ctx.release();
  managed->deleter = [](DLManagedTensor* self) {
    delete static_cast<XnnpackOutputBuffer*>(self->manager_ctx);
    delete self;
  };
  return managed;
}

// xnn_delete_runtime must run before the LoweredSubgraph that built it (which
// owns the subgraph and the static data backing every constant Value,
// including packed-weight source buffers) is destroyed -- see
// LoweredSubgraph's own destructor comment in onnx_to_xnnpack_subgraph.h.
// Declaring a RuntimeGuard after the LoweredSubgraph in the same scope gets
// this for free from C++'s reverse-order local destruction.
struct RuntimeGuard {
  xnn_runtime_t runtime = nullptr;
  ~RuntimeGuard() {
    if (runtime != nullptr) xnn_delete_runtime(runtime);
  }
};

struct XnnpackModelExecutor : public ModelExecutor {
  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    EnsureXnnpackInitialized();

    onnxsim::xnnpack_backend::LoweredSubgraph lowered =
        onnxsim::xnnpack_backend::Lower(model, inputs);

    RuntimeGuard rt;
    CheckXnnStatus(
        xnn_create_runtime_v4(lowered.subgraph, /*weights_cache=*/nullptr,
                              /*workspace=*/nullptr, /*threadpool=*/nullptr,
                              /*flags=*/0, &rt.runtime),
        "xnn_create_runtime_v4");
    CheckXnnStatus(xnn_reshape_runtime(rt.runtime), "xnn_reshape_runtime");

    std::vector<xnn_external_value> external_values;
    external_values.reserve(lowered.input_value_ids.size() +
                            lowered.output_value_ids.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const DLTensor& t = inputs[i]->dl_tensor;
      external_values.push_back(xnn_external_value{
          lowered.input_value_ids[i],
          const_cast<void*>(static_cast<const void*>(
              static_cast<const uint8_t*>(t.data) + t.byte_offset))});
    }

    // Reshape has already run, so every external value's shape (including
    // outputs) is settled; query it rather than re-deriving it, per
    // xnn_get_external_value_shape's documented contract ("Output tensor
    // shapes are returned by xnn_get_external_value_shape").
    std::vector<std::unique_ptr<XnnpackOutputBuffer>> out_bufs;
    out_bufs.reserve(lowered.output_value_ids.size());
    for (uint32_t out_id : lowered.output_value_ids) {
      size_t ndim = 0;
      size_t dims[XNN_MAX_TENSOR_DIMS];
      CheckXnnStatus(
          xnn_get_external_value_shape(rt.runtime, out_id, &ndim, dims),
          "xnn_get_external_value_shape");
      auto buf = std::make_unique<XnnpackOutputBuffer>();
      buf->shape.assign(dims, dims + ndim);
      size_t nelem = 1;
      for (size_t d : buf->shape) nelem *= d;
      buf->data.assign(nelem, 0.0f);
      external_values.push_back(xnn_external_value{out_id, buf->data.data()});
      out_bufs.push_back(std::move(buf));
    }

    CheckXnnStatus(xnn_setup_runtime_v2(rt.runtime, external_values.size(),
                                        external_values.data()),
                   "xnn_setup_runtime_v2");
    CheckXnnStatus(xnn_invoke_runtime(rt.runtime), "xnn_invoke_runtime");

    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(out_bufs.size());
    for (auto& buf : out_bufs) {
      outputs.emplace_back(WrapOutputBuffer(std::move(buf)));
    }
    return outputs;
  }
};

}  // namespace

std::shared_ptr<const ModelExecutor> GetXnnpackModelExecutor() {
  static std::shared_ptr<const ModelExecutor> executor =
      std::make_shared<XnnpackModelExecutor>();
  return executor;
}
