/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bridges between onnxsim's tensor representations and DLPack DLManagedTensor,
 * used at the ModelExecutor boundary so constant folding can hand tensors to an
 * executor (and take results back) without serializing them to TensorProto.
 *
 * Two layers:
 *   1. onnx::TensorProto <-> DLManagedTensor (always available). The input
 *      direction *borrows* the initializer's raw_data (zero copy); the output
 *      direction writes results back with a single set_raw_data (one memcpy,
 *      versus the old element-by-element add_*_data loop).
 *   2. Ort::Value <-> DLManagedTensor (only with the built-in ONNX Runtime).
 *      The input direction wraps the DLPack buffer as an ORT tensor with the
 *      borrowing CreateTensor overload (zero copy); the output direction hands
 *      out ORT's own output buffer by moving the Ort::Value into the managed
 *      tensor's manager_ctx (zero copy) and freeing it in the deleter.
 *
 * DLManagedTensors returned here are always CPU (kDLCPU), contiguous, and in
 * host byte order. TensorProto::raw_data is little-endian by spec on every
 * host, so on a big-endian host both directions swap the payload (which also
 * costs the input direction its zero copy); on a little-endian host the two
 * layouts coincide and nothing is swapped. Callers own every returned
 * DLManagedTensor* and must release it exactly once via `self->deleter(self)`
 * (see DLPack's contract).
 *
 * See docs/dlpack-executor.md for the boundary design and the separate-heap
 * analysis that applies to a JS/onnxruntime-web executor.
 */
#ifndef ONNXSIM_DLPACK_BRIDGE_H_
#define ONNXSIM_DLPACK_BRIDGE_H_

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "dlpack_dtype.h"

#ifndef NO_BUILTIN_ORT
#ifdef ONNXSIM_ORT_FLAT_HEADERS
#include "onnxruntime_cxx_api.h"
#else
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#endif
#endif

namespace onnxsim {
namespace dlpack {

// Number of elements described by a DLPack shape.
inline int64_t NumElements(const int64_t* shape, int32_t ndim) {
  int64_t n = 1;
  for (int32_t i = 0; i < ndim; ++i) n *= shape[i];
  return n;
}

// True when `t` is laid out row-major contiguous (or has no strides, which
// DLPack defines as contiguous). onnxsim only exchanges contiguous tensors.
inline bool IsContiguous(const DLTensor& t) {
  if (t.strides == nullptr) return true;
  int64_t expected = 1;
  for (int32_t i = t.ndim - 1; i >= 0; --i) {
    if (t.shape[i] == 1) continue;  // stride is irrelevant for unit dims
    if (t.strides[i] != expected) return false;
    expected *= t.shape[i];
  }
  return true;
}

// Backing store for a DLManagedTensor built from an onnx::TensorProto.
//   * `shape` outlives the DLTensor (whose shape pointer aliases it).
//   * `owned_proto` is populated only for the *owning* conversion, where the
//     managed tensor must keep the source proto alive itself (its raw_data is
//     what the DLTensor points at).
//   * `owned_bytes` holds materialized bytes when the source proto stored its
//     data in a typed repeated field rather than raw_data.
struct TensorProtoManagerCtx {
  onnx::TensorProto owned_proto;
  std::vector<int64_t> shape;
  std::vector<uint8_t> owned_bytes;
};

// Pack an onnx::TensorProto that keeps its data in a typed repeated field (no
// raw_data) into host-order bytes. Mirrors the dtype set the previous
// TensorProto->ORT converter handled; FLOAT16/BFLOAT16 without raw_data are
// rejected (they were unsupported before too and essentially never occur --
// half-precision initializers are always raw_data).
inline std::vector<uint8_t> PackTypedFields(const onnx::TensorProto& tp) {
  std::vector<uint8_t> out;
  auto blit = [&out](const void* src, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    out.insert(out.end(), p, p + n);
  };
  switch (tp.data_type()) {
#define ONNXSIM_PACK(onnx_dtype, storage, cpp_type)             \
  case onnx::TensorProto::onnx_dtype: {                         \
    std::vector<cpp_type> vec;                                  \
    vec.reserve(tp.storage##_data_size());                      \
    for (const auto& x : tp.storage##_data()) vec.push_back(x); \
    blit(vec.data(), vec.size() * sizeof(cpp_type));            \
    break;                                                      \
  }
    ONNXSIM_PACK(FLOAT, float, float)
    ONNXSIM_PACK(DOUBLE, double, double)
    ONNXSIM_PACK(INT64, int64, int64_t)
    ONNXSIM_PACK(UINT64, uint64, uint64_t)
    ONNXSIM_PACK(INT32, int32, int32_t)
    ONNXSIM_PACK(UINT8, int32, uint8_t)
    ONNXSIM_PACK(INT8, int32, int8_t)
    ONNXSIM_PACK(UINT16, int32, uint16_t)
    ONNXSIM_PACK(INT16, int32, int16_t)
    ONNXSIM_PACK(BOOL, int32, int8_t)
#undef ONNXSIM_PACK
    default:
      throw std::invalid_argument(
          "dlpack bridge: TensorProto with dtype " +
          std::to_string(tp.data_type()) +
          " has no raw_data and cannot be materialized");
  }
  return out;
}

// Fill a DLManagedTensor from `src`, using `ctx` (already allocated, possibly
// already owning a moved-in proto) as backing store. `src` must reference a
// buffer that stays valid for the managed tensor's lifetime -- either an
// external proto (borrowing) or ctx->owned_proto (owning).
inline DLManagedTensor* BuildFromProto(const onnx::TensorProto& src,
                                       TensorProtoManagerCtx* ctx) {
  if (src.data_location() == onnx::TensorProto::EXTERNAL) {
    delete ctx;
    throw std::invalid_argument(
        "dlpack bridge: external-data TensorProto is not supported");
  }
  DLDataType dt;
  if (!TryOnnxToDL(src.data_type(), &dt)) {
    delete ctx;
    throw std::invalid_argument("dlpack bridge: unsupported dtype " +
                                std::to_string(src.data_type()));
  }

  ctx->shape.assign(src.dims().begin(), src.dims().end());
  const void* data_ptr;
  if (src.has_raw_data()) {
    if constexpr (kRawDataIsHostOrder) {
      data_ptr = src.raw_data().data();  // borrow src's buffer
    } else {
      // Big endian: raw_data is little-endian by spec, so it cannot be handed
      // to the executor as-is. Copy into ctx-owned storage and swap there,
      // leaving `src` untouched (callers of the borrowing entry point below
      // hand us initializers they still own). The branch above is the
      // little-endian fast path and stays zero copy.
      const std::string& raw = src.raw_data();
      ctx->owned_bytes.assign(raw.begin(), raw.end());
      SwapElementBytes(ctx->owned_bytes.data(), ctx->owned_bytes.size(),
                       SizeOf(dt));
      data_ptr = ctx->owned_bytes.data();
    }
  } else {
    // Typed repeated fields arrive already decoded into host-order scalars, so
    // PackTypedFields' blit needs no swap on either byte order.
    ctx->owned_bytes = PackTypedFields(src);
    data_ptr = ctx->owned_bytes.data();
  }

  auto* managed = new DLManagedTensor;
  managed->dl_tensor.data = const_cast<void*>(data_ptr);
  managed->dl_tensor.device = DLDevice{kDLCPU, 0};
  managed->dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
  managed->dl_tensor.dtype = dt;
  managed->dl_tensor.shape = ctx->shape.data();
  managed->dl_tensor.strides = nullptr;
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = ctx;
  managed->deleter = [](DLManagedTensor* self) {
    delete static_cast<TensorProtoManagerCtx*>(self->manager_ctx);
    delete self;
  };
  return managed;
}

// Build a DLManagedTensor that BORROWS `tp`'s buffer with no copy (when `tp`
// stores raw_data and the host is little-endian; typed fields, and any tensor
// on a big-endian host, are materialized once). `tp` must outlive the returned
// tensor. Use for feeds whose TensorProto the caller keeps alive (e.g.
// initializer tensors held by RunOps). Caller frees via deleter.
inline DLManagedTensor* FromTensorProtoBorrowing(const onnx::TensorProto& tp) {
  return BuildFromProto(tp, new TensorProtoManagerCtx);
}

// Build a DLManagedTensor that OWNS its source proto (moved into the managed
// tensor), so it is safe standalone. Use when the source proto is a temporary
// (e.g. a TensorProto just parsed from bytes in the Python trampoline).
inline DLManagedTensor* FromTensorProtoOwning(onnx::TensorProto tp) {
  auto* ctx = new TensorProtoManagerCtx;
  ctx->owned_proto = std::move(tp);
  return BuildFromProto(ctx->owned_proto, ctx);
}

// Copy a DLPack tensor into a fresh onnx::TensorProto, writing the payload as
// raw_data in a single copy (plus an in-place byte swap on a big-endian host).
// The tensor must be CPU, contiguous, and of a supported dtype. `name` is
// optional (RunOps sets output names afterwards).
inline onnx::TensorProto ToTensorProto(const DLTensor& t,
                                       const std::string& name = "") {
  if (t.device.device_type != kDLCPU) {
    throw std::invalid_argument(
        "dlpack bridge: only kDLCPU tensors can be converted to TensorProto");
  }
  if (!IsContiguous(t)) {
    throw std::invalid_argument(
        "dlpack bridge: non-contiguous DLTensor cannot be converted");
  }
  int32_t onnx_dtype;
  if (!TryDLToOnnx(t.dtype, &onnx_dtype)) {
    throw std::invalid_argument("dlpack bridge: unsupported DLDataType code=" +
                                std::to_string(t.dtype.code) +
                                " bits=" + std::to_string(t.dtype.bits));
  }
  onnx::TensorProto tp;
  if (!name.empty()) tp.set_name(name);
  tp.set_data_type(onnx_dtype);
  for (int32_t i = 0; i < t.ndim; ++i) tp.add_dims(t.shape[i]);

  const size_t nbytes = NumElements(t.shape, t.ndim) * SizeOf(t.dtype);
  const auto* base = static_cast<const uint8_t*>(t.data) + t.byte_offset;
  if constexpr (kRawDataIsHostOrder) {
    tp.set_raw_data(base, nbytes);  // single copy into the persisted model
  } else {
    // raw_data is little-endian whatever the host is, so a big-endian result
    // has to be swapped on the way out. Still one copy: build the string, then
    // swap it in place before it is moved into the proto.
    std::string raw(reinterpret_cast<const char*>(base), nbytes);
    SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()), raw.size(),
                     SizeOf(t.dtype));
    tp.set_raw_data(std::move(raw));
  }
  return tp;
}

#ifndef NO_BUILTIN_ORT
// A process-wide CPU OrtMemoryInfo for wrapping/borrowing host buffers.
inline Ort::MemoryInfo& CpuMemoryInfo() {
  static Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  return info;
}

// Wrap a DLPack tensor as an Ort::Value that BORROWS its buffer -- no copy. The
// DLManagedTensor's storage must outlive the returned Ort::Value. Used to feed
// initializer tensors into a constant-folding session.
inline Ort::Value BorrowAsOrtValue(const DLTensor& t) {
  if (t.device.device_type != kDLCPU) {
    throw std::invalid_argument("dlpack bridge: ORT input must be kDLCPU");
  }
  if (!IsContiguous(t)) {
    throw std::invalid_argument("dlpack bridge: ORT input must be contiguous");
  }
  int32_t onnx_dtype;
  if (!TryDLToOnnx(t.dtype, &onnx_dtype)) {
    throw std::invalid_argument(
        "dlpack bridge: unsupported DLDataType for ORT");
  }
  const size_t nbytes = NumElements(t.shape, t.ndim) * SizeOf(t.dtype);
  auto* base = static_cast<uint8_t*>(t.data) + t.byte_offset;
  return Ort::Value::CreateTensor(
      CpuMemoryInfo(), base, nbytes, t.shape, static_cast<size_t>(t.ndim),
      static_cast<ONNXTensorElementDataType>(onnx_dtype));
}

// Manager context that keeps an Ort::Value (and thus its output buffer) alive
// for as long as the DLManagedTensor that hands the buffer out.
struct OrtValueManagerCtx {
  Ort::Value value{nullptr};
  std::vector<int64_t> shape;
};

// Hand out an ORT output tensor as a DLManagedTensor without copying: ownership
// of the Ort::Value moves into the managed tensor, whose deleter releases it.
// Throws for non-tensor or unsupported-dtype outputs.
inline DLManagedTensor* FromOrtValue(Ort::Value&& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  DLDataType dt;
  if (!TryOnnxToDL(static_cast<int32_t>(info.GetElementType()), &dt)) {
    throw std::invalid_argument(
        "dlpack bridge: unsupported ORT output element type");
  }
  auto* ctx = new OrtValueManagerCtx;
  ctx->shape = info.GetShape();
  ctx->value = std::move(value);

  auto* managed = new DLManagedTensor;
  managed->dl_tensor.data = ctx->value.GetTensorMutableData<void>();
  managed->dl_tensor.device = DLDevice{kDLCPU, 0};
  managed->dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
  managed->dl_tensor.dtype = dt;
  managed->dl_tensor.shape = ctx->shape.data();
  managed->dl_tensor.strides = nullptr;
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = ctx;
  managed->deleter = [](DLManagedTensor* self) {
    delete static_cast<OrtValueManagerCtx*>(self->manager_ctx);
    delete self;
  };
  return managed;
}
#endif  // NO_BUILTIN_ORT

}  // namespace dlpack
}  // namespace onnxsim

#endif  // ONNXSIM_DLPACK_BRIDGE_H_
