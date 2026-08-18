/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++17 tensor_pool_dtype_test.cpp -o t && ./t
 *
 * Dependency-free unit test for the ONNX<->safetensors dtype mapping. Like
 * dlpack_dtype_test.cpp, it builds and runs standalone (no ONNX / ONNX
 * Runtime), so it is wired into the ONNXSIM_TESTS target and runnable under
 * Emscripten.
 */
#include "tensor_pool_dtype.h"

#include <cstdio>
#include <string>

using namespace onnxsim::tensor_pool;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Row {
  int32_t onnx;
  const char* safetensors;
  size_t bytes;
};

}  // namespace

int main() {
  const Row rows[] = {
      {ONNX_BOOL, "BOOL", 1},   {ONNX_UINT8, "U8", 1},
      {ONNX_INT8, "I8", 1},     {ONNX_INT16, "I16", 2},
      {ONNX_UINT16, "U16", 2},  {ONNX_FLOAT16, "F16", 2},
      {ONNX_BFLOAT16, "BF16", 2}, {ONNX_INT32, "I32", 4},
      {ONNX_UINT32, "U32", 4}, {ONNX_FLOAT, "F32", 4},
      {ONNX_INT64, "I64", 8},  {ONNX_UINT64, "U64", 8},
      {ONNX_DOUBLE, "F64", 8},
  };

  for (const auto& row : rows) {
    std::string_view mapped = ToSafetensors(row.onnx);
    Check(mapped == row.safetensors,
         "ToSafetensors(" + std::to_string(row.onnx) +
             ") == " + row.safetensors);
    Check(ElementSize(row.onnx) == row.bytes,
         "ElementSize(" + std::to_string(row.onnx) +
             ") == " + std::to_string(row.bytes));
    // Round trip: onnx -> safetensors -> onnx recovers the same code.
    Check(FromSafetensors(mapped) == row.onnx,
         std::string("FromSafetensors(ToSafetensors(") +
             std::to_string(row.onnx) + ")) round-trips");
  }

  // Types safetensors cannot represent map to an empty string / UNDEFINED.
  Check(ToSafetensors(ONNX_STRING).empty(), "STRING has no safetensors dtype");
  Check(ToSafetensors(ONNX_COMPLEX64).empty(),
       "COMPLEX64 has no safetensors dtype");
  Check(ToSafetensors(ONNX_COMPLEX128).empty(),
       "COMPLEX128 has no safetensors dtype");
  Check(ToSafetensors(ONNX_UNDEFINED).empty(),
       "UNDEFINED has no safetensors dtype");
  Check(ElementSize(ONNX_STRING) == 0, "STRING has no fixed element size");

  Check(FromSafetensors("F8_E4M3") == ONNX_UNDEFINED,
       "unrecognized safetensors dtype maps to UNDEFINED");
  Check(FromSafetensors("bogus") == ONNX_UNDEFINED,
       "garbage safetensors dtype maps to UNDEFINED");

  if (g_failures == 0) {
    std::printf("tensor_pool_dtype_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_dtype_test: %d failure(s)\n", g_failures);
  return 1;
}
