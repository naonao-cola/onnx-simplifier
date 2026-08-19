/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure, dependency-free mapping between ONNX tensor element types and the
 * safetensors on-disk dtype strings
 * (https://github.com/huggingface/safetensors#format). Operates on the
 * *integer* ONNX dtype codes (the stable wire numbers of
 * onnx::TensorProto::DataType) rather than the onnx headers, so tensor_pool.h
 * / tensor_pool.cpp build and unit-test on their own -- including under
 * Emscripten and in the wheel build -- without configuring ONNX. Mirrors
 * dlpack_dtype.h's split (pure dtype mapping vs. the onnx::TensorProto glue
 * in tensor_pool_bridge.h) for the same reason.
 *
 * Supported set: BOOL, all signed/unsigned 8/16/32/64-bit ints, FLOAT16,
 * BFLOAT16, FLOAT, DOUBLE -- the dtypes with a fixed-width, raw, contiguous
 * layout AND a safetensors dtype string. safetensors has no encoding for
 * STRING, COMPLEX64/128, or UNDEFINED, so those are rejected here; a
 * TensorPool can't pool a STRING tensor.
 */
#ifndef ONNXSIM_TENSOR_POOL_DTYPE_H_
#define ONNXSIM_TENSOR_POOL_DTYPE_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace onnxsim {
namespace tensor_pool {

// Stable ONNX TensorProto::DataType wire numbers, mirrored so this header
// does not need the onnx protobuf headers. Kept in sync with onnx/onnx.proto3
// (see dlpack_dtype.h's identical OnnxDtype, which this duplicates rather
// than depends on so the two dtype mappings stay independently buildable).
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

// Bytes of one element of `onnx_dtype`. 0 for a dtype with no fixed-width
// element size, or one ToSafetensors rejects.
inline size_t ElementSize(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case ONNX_BOOL:
    case ONNX_INT8:
    case ONNX_UINT8:
      return 1;
    case ONNX_INT16:
    case ONNX_UINT16:
    case ONNX_FLOAT16:
    case ONNX_BFLOAT16:
      return 2;
    case ONNX_INT32:
    case ONNX_UINT32:
    case ONNX_FLOAT:
      return 4;
    case ONNX_INT64:
    case ONNX_UINT64:
    case ONNX_DOUBLE:
      return 8;
    default:
      return 0;
  }
}

// Map an ONNX dtype code to its safetensors header dtype string. Returns ""
// for a dtype safetensors cannot represent (STRING, COMPLEX64/128,
// UNDEFINED, or one of onnx's newer sub-byte/float8 encodings this mapping
// does not (yet) cover).
inline std::string_view ToSafetensors(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case ONNX_BOOL:
      return "BOOL";
    case ONNX_UINT8:
      return "U8";
    case ONNX_INT8:
      return "I8";
    case ONNX_INT16:
      return "I16";
    case ONNX_UINT16:
      return "U16";
    case ONNX_INT32:
      return "I32";
    case ONNX_UINT32:
      return "U32";
    case ONNX_INT64:
      return "I64";
    case ONNX_UINT64:
      return "U64";
    case ONNX_FLOAT16:
      return "F16";
    case ONNX_BFLOAT16:
      return "BF16";
    case ONNX_FLOAT:
      return "F32";
    case ONNX_DOUBLE:
      return "F64";
    default:
      return {};
  }
}

// Inverse of ToSafetensors. Returns ONNX_UNDEFINED for a name this mapping
// doesn't recognize -- including safetensors' F8_* family, which has no
// onnx raw-data equivalent this pool handles.
inline int32_t FromSafetensors(std::string_view dtype) {
  if (dtype == "BOOL") return ONNX_BOOL;
  if (dtype == "U8") return ONNX_UINT8;
  if (dtype == "I8") return ONNX_INT8;
  if (dtype == "I16") return ONNX_INT16;
  if (dtype == "U16") return ONNX_UINT16;
  if (dtype == "I32") return ONNX_INT32;
  if (dtype == "U32") return ONNX_UINT32;
  if (dtype == "I64") return ONNX_INT64;
  if (dtype == "U64") return ONNX_UINT64;
  if (dtype == "F16") return ONNX_FLOAT16;
  if (dtype == "BF16") return ONNX_BFLOAT16;
  if (dtype == "F32") return ONNX_FLOAT;
  if (dtype == "F64") return ONNX_DOUBLE;
  return ONNX_UNDEFINED;
}

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_DTYPE_H_
