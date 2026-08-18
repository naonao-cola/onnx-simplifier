/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++17 gguf_dtype_test.cpp -o t && ./t
 *
 * Dependency-free unit test for the ONNX<->GGML dtype mapping, mirroring
 * tensor_pool_dtype_test.cpp.
 */
#include "gguf_dtype.h"

#include <cstdio>
#include <string>

using namespace onnxsim::tensor_pool::gguf;

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
  uint32_t ggml;
  size_t bytes;
};

}  // namespace

int main() {
  const Row rows[] = {
      {ONNX_FLOAT, GGML_TYPE_F32, 4},     {ONNX_FLOAT16, GGML_TYPE_F16, 2},
      {ONNX_BFLOAT16, GGML_TYPE_BF16, 2}, {ONNX_DOUBLE, GGML_TYPE_F64, 8},
      {ONNX_INT8, GGML_TYPE_I8, 1},       {ONNX_INT16, GGML_TYPE_I16, 2},
      {ONNX_INT32, GGML_TYPE_I32, 4},     {ONNX_INT64, GGML_TYPE_I64, 8},
  };

  for (const auto& row : rows) {
    uint32_t ggml = 0xFFFFFFFF;
    Check(FromOnnx(row.onnx, &ggml) && ggml == row.ggml,
          "FromOnnx(" + std::to_string(row.onnx) +
              ") == " + std::to_string(row.ggml));
    Check(ElementSize(row.ggml) == row.bytes,
          "ElementSize(" + std::to_string(row.ggml) +
              ") == " + std::to_string(row.bytes));
    Check(IsRaw(row.ggml), "IsRaw(" + std::to_string(row.ggml) + ")");
    int32_t onnx = -1;
    Check(ToOnnx(row.ggml, &onnx) && onnx == row.onnx,
          "ToOnnx(FromOnnx(" + std::to_string(row.onnx) + "))) round-trips");
  }

  // Quantized / unrecognized ggml types are rejected, not guessed at.
  const uint32_t quantized[] = {2,  3,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 20, 29, 9999};
  for (uint32_t q : quantized) {
    int32_t onnx = -1;
    Check(!ToOnnx(q, &onnx),
          "ToOnnx rejects quantized/unknown type " + std::to_string(q));
    Check(!IsRaw(q),
          "IsRaw is false for quantized/unknown type " + std::to_string(q));
    Check(ElementSize(q) == 0,
          "ElementSize is 0 for quantized/unknown type " + std::to_string(q));
  }

  // ONNX dtypes with no raw ggml counterpart (unsigned ints, BOOL, STRING,
  // UNDEFINED -- ggml has no unsigned or bool tensor type at all).
  const int32_t unsupported_onnx[] = {2 /*UINT8*/, 4 /*UINT16*/, 9 /*BOOL*/,
                                      8 /*STRING*/, ONNX_UNDEFINED};
  for (int32_t d : unsupported_onnx) {
    uint32_t ggml = 0;
    Check(!FromOnnx(d, &ggml),
          "FromOnnx rejects unsupported ONNX dtype " + std::to_string(d));
  }

  if (g_failures == 0) {
    std::printf("gguf_dtype_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "gguf_dtype_test: %d failure(s)\n", g_failures);
  return 1;
}
