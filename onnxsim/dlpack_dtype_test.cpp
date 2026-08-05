/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dependency-free unit test for the ONNX<->DLPack dtype mapping. Like
 * sym_expr_test, it builds and runs standalone (no ONNX / ONNX Runtime), so it
 * is wired into the ONNXSIM_TESTS target and runnable under Emscripten.
 */
#include "dlpack_dtype.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace onnxsim::dlpack;

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
  uint8_t code;
  uint8_t bits;
  size_t bytes;
  const char* name;
};

}  // namespace

int main() {
  // Every supported ONNX dtype: forward map produces the expected DLDataType,
  // the element size is right, and the inverse map round-trips back to the same
  // ONNX code.
  const std::vector<Row> supported = {
      {ONNX_FLOAT, kDLFloat, 32, 4, "float32"},
      {ONNX_DOUBLE, kDLFloat, 64, 8, "float64"},
      {ONNX_FLOAT16, kDLFloat, 16, 2, "float16"},
      {ONNX_BFLOAT16, kDLBfloat, 16, 2, "bfloat16"},
      {ONNX_INT8, kDLInt, 8, 1, "int8"},
      {ONNX_INT16, kDLInt, 16, 2, "int16"},
      {ONNX_INT32, kDLInt, 32, 4, "int32"},
      {ONNX_INT64, kDLInt, 64, 8, "int64"},
      {ONNX_UINT8, kDLUInt, 8, 1, "uint8"},
      {ONNX_UINT16, kDLUInt, 16, 2, "uint16"},
      {ONNX_UINT32, kDLUInt, 32, 4, "uint32"},
      {ONNX_UINT64, kDLUInt, 64, 8, "uint64"},
      {ONNX_BOOL, kDLBool, 8, 1, "bool"},
  };

  for (const auto& r : supported) {
    DLDataType dt;
    Check(TryOnnxToDL(r.onnx, &dt), std::string("forward ") + r.name);
    Check(dt.code == r.code, std::string("code ") + r.name);
    Check(dt.bits == r.bits, std::string("bits ") + r.name);
    Check(dt.lanes == 1, std::string("lanes ") + r.name);
    Check(SizeOf(dt) == r.bytes, std::string("size ") + r.name);

    int32_t back = ONNX_UNDEFINED;
    Check(TryDLToOnnx(dt, &back), std::string("inverse ") + r.name);
    Check(back == r.onnx, std::string("round-trip ") + r.name);
  }

  // Unsupported ONNX dtypes must be rejected, not silently mismapped.
  const std::vector<int32_t> unsupported = {
      ONNX_UNDEFINED,      ONNX_STRING,  ONNX_COMPLEX64, ONNX_COMPLEX128,
      17 /*FLOAT8E4M3FN*/, 21 /*UINT4*/, 22 /*INT4*/,    999 /*garbage*/,
  };
  for (int32_t code : unsupported) {
    DLDataType dt;
    Check(!TryOnnxToDL(code, &dt), "reject onnx dtype " + std::to_string(code));
  }

  // Vector (lanes != 1) and unknown DLDataTypes must be rejected by the
  // inverse map.
  {
    int32_t out = 0;
    Check(!TryDLToOnnx(DLDataType{kDLFloat, 32, 4}, &out), "reject lanes=4");
    Check(!TryDLToOnnx(DLDataType{kDLInt, 4, 1}, &out), "reject int4");
    Check(!TryDLToOnnx(DLDataType{kDLComplex, 64, 1}, &out), "reject complex");
    Check(!TryDLToOnnx(DLDataType{kDLOpaqueHandle, 64, 1}, &out),
          "reject opaque");
  }

  // ------------------------------------------------------------------
  // SwapElementBytes: the raw_data <-> host-order conversion the DLPack
  // bridge applies on big-endian hosts. These checks are byte-order
  // independent -- they assert on the bytes, not on decoded scalars -- so
  // they mean the same thing whether this binary runs on x86_64 or s390x.
  // ------------------------------------------------------------------
  {
    // 1-byte elements are never touched.
    std::vector<uint8_t> one = {1, 2, 3};
    SwapElementBytes(one.data(), one.size(), 1);
    Check(one == std::vector<uint8_t>({1, 2, 3}),
          "swap elem_size=1 is a no-op");

    std::vector<uint8_t> two = {0x01, 0x02, 0x03, 0x04};
    SwapElementBytes(two.data(), two.size(), 2);
    Check(two == std::vector<uint8_t>({0x02, 0x01, 0x04, 0x03}),
          "swap elem_size=2 reverses each pair");

    std::vector<uint8_t> four = {1, 2, 3, 4, 5, 6, 7, 8};
    SwapElementBytes(four.data(), four.size(), 4);
    Check(four == std::vector<uint8_t>({4, 3, 2, 1, 8, 7, 6, 5}),
          "swap elem_size=4 reverses each quad");

    std::vector<uint8_t> eight = {1, 2, 3, 4, 5, 6, 7, 8};
    SwapElementBytes(eight.data(), eight.size(), 8);
    Check(eight == std::vector<uint8_t>({8, 7, 6, 5, 4, 3, 2, 1}),
          "swap elem_size=8 reverses the element");

    // Swapping twice restores the original, which is what makes one function
    // serve both the read and write direction.
    std::vector<uint8_t> rt = {9, 8, 7, 6, 5, 4, 3, 2};
    const std::vector<uint8_t> before = rt;
    SwapElementBytes(rt.data(), rt.size(), 4);
    SwapElementBytes(rt.data(), rt.size(), 4);
    Check(rt == before, "swap is its own inverse");

    // A trailing partial element is left alone rather than read out of bounds.
    std::vector<uint8_t> partial = {1, 2, 3, 4, 5, 6};
    SwapElementBytes(partial.data(), partial.size(), 4);
    Check(partial == std::vector<uint8_t>({4, 3, 2, 1, 5, 6}),
          "swap leaves a trailing partial element untouched");

    // Empty input must not walk off the (possibly null) pointer.
    SwapElementBytes(nullptr, 0, 8);
    Check(true, "swap tolerates an empty buffer");

    // The flag the bridge branches on must agree with the running host.
    const uint32_t probe = 1;
    const bool host_is_little = *reinterpret_cast<const uint8_t*>(&probe) == 1;
    Check(kRawDataIsHostOrder == host_is_little,
          "kRawDataIsHostOrder matches this host's byte order");
  }

  if (g_failures == 0) {
    std::printf("dlpack_dtype_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "dlpack_dtype_test: %d failure(s)\n", g_failures);
  return 1;
}
