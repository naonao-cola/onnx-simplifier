/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure, dependency-free mapping between ONNX tensor element types and DLPack
 * DLDataType. This header intentionally depends on nothing but dlpack.h: it
 * operates on the *integer* ONNX dtype codes (the stable wire numbers of
 * onnx::TensorProto::DataType) rather than on the onnx headers, so it compiles
 * and is unit-tested on its own -- including under Emscripten and in the wheel
 * build -- without configuring ONNX or ONNX Runtime.
 *
 * The onnx::TensorProto glue that actually reads/writes tensor buffers lives in
 * dlpack_bridge.h and calls into these functions with `tp.data_type()`.
 *
 * Supported set (bijective ONNX code <-> DLDataType, lanes == 1):
 *   FLOAT16/FLOAT/DOUBLE/BFLOAT16, all signed/unsigned 8/16/32/64 ints, BOOL.
 * Everything else (STRING, COMPLEX*, FLOAT8*, INT4/UINT4/FLOAT4, UNDEFINED) is
 * rejected here -- these have no raw contiguous little-endian layout onnxsim's
 * constant folder exchanges, mirroring the dtype set the old TensorProto<->ORT
 * converters handled.
 */
#ifndef ONNXSIM_DLPACK_DTYPE_H_
#define ONNXSIM_DLPACK_DTYPE_H_

#include <bit>
#include <cstdint>
#include <utility>

#include "dlpack/dlpack.h"

namespace onnxsim {
namespace dlpack {

// Stable ONNX TensorProto::DataType wire numbers, mirrored so this header does
// not need the onnx protobuf headers. Kept in sync with onnx/onnx.proto3.
enum OnnxDtype : int32_t {
  ONNX_UNDEFINED = 0,
  ONNX_FLOAT = 1,
  ONNX_UINT8 = 2,
  ONNX_INT8 = 3,
  ONNX_UINT16 = 4,
  ONNX_INT16 = 5,
  ONNX_INT32 = 6,
  ONNX_INT64 = 7,
  ONNX_STRING = 8,
  ONNX_BOOL = 9,
  ONNX_FLOAT16 = 10,
  ONNX_DOUBLE = 11,
  ONNX_UINT32 = 12,
  ONNX_UINT64 = 13,
  ONNX_COMPLEX64 = 14,
  ONNX_COMPLEX128 = 15,
  ONNX_BFLOAT16 = 16,
};

// Number of bytes one element of `dt` occupies. Uses the common array-library
// convention that storage size is ceil(bits*lanes / 8) (so bool -> 1 byte).
inline size_t SizeOf(DLDataType dt) {
  return (static_cast<size_t>(dt.bits) * dt.lanes + 7) / 8;
}

inline bool operator==(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// ONNX fixes the byte order of TensorProto::raw_data: "elements MUST be stored
// in as fixed-width, little-endian order", on every host. DLPack carries no
// byte-order field, so DLTensors are host order by convention. The two agree on
// a little-endian machine and disagree on a big-endian one, where the payload
// has to be swapped at each TensorProto <-> DLPack crossing.
inline constexpr bool kRawDataIsHostOrder =
    std::endian::native == std::endian::little;

// Reverse the bytes of each `elem_size`-wide element of `data` in place,
// converting between raw_data's little-endian layout and host order (the
// conversion is its own inverse, so one function serves both directions).
// A no-op for 1-byte elements. A trailing partial element is left alone.
inline void SwapElementBytes(uint8_t* data, size_t nbytes, size_t elem_size) {
  if (elem_size < 2) return;
  for (size_t off = 0; off + elem_size <= nbytes; off += elem_size) {
    for (size_t i = 0, j = elem_size - 1; i < j; ++i, --j) {
      std::swap(data[off + i], data[off + j]);
    }
  }
}

// Map an ONNX dtype code to a DLDataType. Returns true and fills `*out` for a
// supported type; returns false (leaving `*out` untouched) otherwise.
inline bool TryOnnxToDL(int32_t onnx_dtype, DLDataType* out) {
  auto set = [out](uint8_t code, uint8_t bits) {
    out->code = code;
    out->bits = bits;
    out->lanes = 1;
  };
  switch (onnx_dtype) {
    case ONNX_FLOAT:
      set(kDLFloat, 32);
      return true;
    case ONNX_DOUBLE:
      set(kDLFloat, 64);
      return true;
    case ONNX_FLOAT16:
      set(kDLFloat, 16);
      return true;
    case ONNX_BFLOAT16:
      set(kDLBfloat, 16);
      return true;
    case ONNX_INT8:
      set(kDLInt, 8);
      return true;
    case ONNX_INT16:
      set(kDLInt, 16);
      return true;
    case ONNX_INT32:
      set(kDLInt, 32);
      return true;
    case ONNX_INT64:
      set(kDLInt, 64);
      return true;
    case ONNX_UINT8:
      set(kDLUInt, 8);
      return true;
    case ONNX_UINT16:
      set(kDLUInt, 16);
      return true;
    case ONNX_UINT32:
      set(kDLUInt, 32);
      return true;
    case ONNX_UINT64:
      set(kDLUInt, 64);
      return true;
    case ONNX_BOOL:
      set(kDLBool, 8);
      return true;
    default:
      return false;
  }
}

// Inverse of TryOnnxToDL. Returns true and fills `*out` with the ONNX dtype
// code for a supported DLDataType (lanes must be 1); false otherwise.
inline bool TryDLToOnnx(DLDataType dt, int32_t* out) {
  if (dt.lanes != 1) return false;
  auto ok = [out](int32_t v) {
    *out = v;
    return true;
  };
  switch (dt.code) {
    case kDLFloat:
      if (dt.bits == 16) return ok(ONNX_FLOAT16);
      if (dt.bits == 32) return ok(ONNX_FLOAT);
      if (dt.bits == 64) return ok(ONNX_DOUBLE);
      return false;
    case kDLBfloat:
      if (dt.bits == 16) return ok(ONNX_BFLOAT16);
      return false;
    case kDLInt:
      if (dt.bits == 8) return ok(ONNX_INT8);
      if (dt.bits == 16) return ok(ONNX_INT16);
      if (dt.bits == 32) return ok(ONNX_INT32);
      if (dt.bits == 64) return ok(ONNX_INT64);
      return false;
    case kDLUInt:
      if (dt.bits == 8) return ok(ONNX_UINT8);
      if (dt.bits == 16) return ok(ONNX_UINT16);
      if (dt.bits == 32) return ok(ONNX_UINT32);
      if (dt.bits == 64) return ok(ONNX_UINT64);
      return false;
    case kDLBool:
      if (dt.bits == 8) return ok(ONNX_BOOL);
      return false;
    default:
      return false;
  }
}

}  // namespace dlpack
}  // namespace onnxsim

#endif  // ONNXSIM_DLPACK_DTYPE_H_
