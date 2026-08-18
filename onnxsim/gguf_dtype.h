/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure, dependency-free mapping between ONNX tensor element types and GGML
 * tensor types (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
 * plus the small set of GGUF wire-format constants (magic, version,
 * alignment, metadata value-type codes) needed to read/write the container
 * itself. Operates on the *integer* ONNX dtype codes and the *integer* GGML
 * type codes, not the onnx headers, so tensor_pool_gguf.cpp builds and unit-
 * tests on its own -- mirrors tensor_pool_dtype.h's split for safetensors,
 * for the same reason.
 *
 * IMPORTANT SCOPE NOTE: only GGML's plain, unquantized element types map to
 * an ONNX dtype (F32, F16, BF16, F64, I8, I16, I32, I64 -- storage where one
 * element is one fixed-width value, exactly what onnx::TensorProto::raw_data
 * and safetensors both assume). GGML's block-quantized types (Q4_0, Q4_K,
 * Q6_K, every IQ*_ variant, ...) -- which is what most real-world .gguf
 * files (quantized LLM checkpoints) actually store almost all of their
 * tensors as -- have NO onnx raw-data equivalent: each "element" is really a
 * shared block of packed sub-byte codes plus one or more scale/zero-point
 * values, decoded by dequantization math this mapping does not implement.
 * ToOnnx() rejects them (returns false) rather than guess; TensorPool's GGUF
 * loader (tensor_pool_gguf.cpp) skips such tensors and reports them, rather
 * than materializing garbage bytes as if they were raw floats. Practically,
 * loading a quantized LLM's .gguf through this pool will pool only its
 * unquantized tensors (typically norm scale/bias, RoPE frequencies, biases
 * -- a small minority of the file), not its weight matrices.
 */
#ifndef ONNXSIM_GGUF_DTYPE_H_
#define ONNXSIM_GGUF_DTYPE_H_

#include <cstddef>
#include <cstdint>

namespace onnxsim {
namespace tensor_pool {
namespace gguf {

// --- container format constants -------------------------------------------

inline constexpr uint32_t kMagic =
    0x46554747;  // "GGUF" read as a little-endian u32
// Only version 3 (the current format, and what every actively-maintained
// GGUF writer -- llama.cpp included -- has produced since 2023) is
// supported. Version 1 used 32-bit tensor/metadata counts and a different
// string encoding; version 2 is functionally version 3 minus a couple of
// later-added metadata conventions this reader doesn't depend on anyway, but
// is rare enough in the wild that it isn't worth the extra branch -- treated
// as unsupported alongside version 1.
inline constexpr uint32_t kSupportedVersion = 3;
inline constexpr uint64_t kDefaultAlignment = 32;

// gguf_metadata_value_type.
enum MetadataValueType : uint32_t {
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// ggml_type -- only the stable low IDs this mapping cares about are named;
// every other value (the quantized/k-quant/IQ* family, and any type newer
// than this list) is handled generically as "unsupported" by ToOnnx/IsRaw
// below without needing its own enumerator. GGML never reassigns an existing
// ID (only appends and occasionally retires one), so these are stable across
// ggml/llama.cpp versions.
enum GgmlType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_BF16 = 30,
};

// --- ONNX dtype <-> ggml_type (raw types only; see the file comment) ------

// Stable ONNX TensorProto::DataType wire numbers, duplicated (not shared via
// an #include) so this header stays independent of tensor_pool_dtype.h and
// buildable on its own, exactly like that header is independent of
// dlpack_dtype.h's copy of the same numbers.
enum OnnxDtype : int32_t {
  ONNX_UNDEFINED = 0,
  ONNX_FLOAT = 1,
  ONNX_INT8 = 3,
  ONNX_INT16 = 5,
  ONNX_INT32 = 6,
  ONNX_INT64 = 7,
  ONNX_FLOAT16 = 10,
  ONNX_DOUBLE = 11,
  ONNX_BFLOAT16 = 16,
};

// Bytes of one element. 0 for a ggml_type this mapping doesn't cover
// (quantized types report 0 here too -- computing their true per-block size
// isn't needed since they're never pooled, only skipped).
inline size_t ElementSize(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGML_TYPE_I8:
      return 1;
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
      return 2;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
      return 4;
    case GGML_TYPE_F64:
    case GGML_TYPE_I64:
      return 8;
    default:
      return 0;
  }
}

// True for a ggml_type with a fixed-width, unquantized, one-value-per-
// element layout -- the only kind TensorPool can pool (see the file
// comment). False for every quantized/k-quant/IQ* type and anything this
// mapping doesn't recognize.
inline bool IsRaw(uint32_t ggml_type) { return ElementSize(ggml_type) != 0; }

// Map a ggml_type to its ONNX dtype code. Returns false (leaving `*out`
// untouched) for a quantized type or one this mapping doesn't recognize.
inline bool ToOnnx(uint32_t ggml_type, int32_t* out) {
  switch (ggml_type) {
    case GGML_TYPE_F32:
      *out = ONNX_FLOAT;
      return true;
    case GGML_TYPE_F16:
      *out = ONNX_FLOAT16;
      return true;
    case GGML_TYPE_BF16:
      *out = ONNX_BFLOAT16;
      return true;
    case GGML_TYPE_F64:
      *out = ONNX_DOUBLE;
      return true;
    case GGML_TYPE_I8:
      *out = ONNX_INT8;
      return true;
    case GGML_TYPE_I16:
      *out = ONNX_INT16;
      return true;
    case GGML_TYPE_I32:
      *out = ONNX_INT32;
      return true;
    case GGML_TYPE_I64:
      *out = ONNX_INT64;
      return true;
    default:
      return false;
  }
}

// Inverse of ToOnnx. Returns false for an ONNX dtype with no raw ggml_type
// counterpart (STRING, COMPLEX64/128, UNDEFINED, the unsigned int family --
// ggml has no unsigned integer types at all, quantized or otherwise).
inline bool FromOnnx(int32_t onnx_dtype, uint32_t* out) {
  switch (onnx_dtype) {
    case ONNX_FLOAT:
      *out = GGML_TYPE_F32;
      return true;
    case ONNX_FLOAT16:
      *out = GGML_TYPE_F16;
      return true;
    case ONNX_BFLOAT16:
      *out = GGML_TYPE_BF16;
      return true;
    case ONNX_DOUBLE:
      *out = GGML_TYPE_F64;
      return true;
    case ONNX_INT8:
      *out = GGML_TYPE_I8;
      return true;
    case ONNX_INT16:
      *out = GGML_TYPE_I16;
      return true;
    case ONNX_INT32:
      *out = GGML_TYPE_I32;
      return true;
    case ONNX_INT64:
      *out = GGML_TYPE_I64;
      return true;
    default:
      return false;
  }
}

}  // namespace gguf
}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_GGUF_DTYPE_H_
