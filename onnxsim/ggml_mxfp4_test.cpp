/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 ggml_mxfp4_test.cpp -o t && ./t
 *
 * Dependency-free unit test for ggml_mxfp4.h's GGML MXFP4 dequantization,
 * mirroring ggml_kquant_test.cpp's style.
 *
 * The block layout/formula this decodes was cross-checked against GGML's
 * own reference (ggml-common.h's block_mxfp4/kvalues_fp4, ggml-quants.c's
 * dequantize_row_mxfp4, ggml-impl.h's ggml_e8m0_to_fp32_half) by hand for
 * the exponent values used below. The cases here are small, hand-verifiable
 * vectors (E8M0 bytes chosen so the decoded scale is an exact power of two
 * with no float rounding to reason about), meant to catch a regression in
 * this file specifically, not to re-derive GGML's spec from scratch.
 */
#include "ggml_mxfp4.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace onnxsim::tensor_pool::gguf;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckNear(float got, float want, const std::string& what,
               float tol = 1e-6f) {
  Check(std::fabs(got - want) <= tol, what + " (got " + std::to_string(got) +
                                          ", want " + std::to_string(want) +
                                          ")");
}

// GgmlE8m0ToFloat32Half(x) is exactly a power of two for every x (see the
// function's own derivation), so every case here is bit-exact -- no
// tolerance needed beyond float rounding of the expected value itself.
void TestGgmlE8m0ToFloat32Half() {
  // GgmlE8m0ToFloat32Half(x) == 2^(x-128) for every x (see the function's
  // own derivation) -- checked against std::ldexp(1.0f, x - 128), an
  // independent standard-library computation of the same power of two.
  for (int x = 0; x <= 254; x += 17) {
    CheckNear(GgmlE8m0ToFloat32Half(static_cast<uint8_t>(x)),
              std::ldexp(1.0f, x - 128),
              "e8m0(" + std::to_string(x) + ") == 2^(" +
                  std::to_string(x - 128) + ")",
              0.0f);
  }
  // Named checkpoints for readability: x=127 -> 0.5, x=128 -> 1.0,
  // x=129 -> 2.0.
  CheckNear(GgmlE8m0ToFloat32Half(127), 0.5f, "e8m0(127) == 0.5", 0.0f);
  CheckNear(GgmlE8m0ToFloat32Half(128), 1.0f, "e8m0(128) == 1.0", 0.0f);
  CheckNear(GgmlE8m0ToFloat32Half(129), 2.0f, "e8m0(129) == 2.0", 0.0f);
}

// d = GgmlE8m0ToFloat32Half(128) = 1.0 exactly, so decoded values equal
// kMxfp4Values[code] exactly -- no scale-multiplication rounding to reason
// about. qs[0] = 0x21 packs code 1 (low nibble, element 0) and code 2 (high
// nibble, element 16); qs[1] = 0x9A packs code 10 (low nibble, element 1)
// and code 9 (high nibble, element 17). Every other qs byte is 0, so every
// other element decodes to kMxfp4Values[0] * d = 0.
void TestMxfp4Block() {
  std::vector<uint8_t> block(Mxfp4BlockBytes(GGML_TYPE_MXFP4), 0);
  Check(block.size() == 17, "MXFP4 block size is 17 bytes");
  block[0] = 128;   // e8m0 scale byte -> d = 1.0
  block[1] = 0x21;  // qs[0]: low nibble 1, high nibble 2
  block[2] = 0x9A;  // qs[1]: low nibble 0xA=10, high nibble 9

  float out[32];
  DequantizeMxfp4Block(block.data(), out);
  CheckNear(out[0], 1.0f, "MXFP4 element 0 (code 1)");
  CheckNear(out[16], 2.0f, "MXFP4 element 16 (code 2)");
  CheckNear(out[1], -2.0f, "MXFP4 element 1 (code 10 -> -2)");
  CheckNear(out[17], -1.0f, "MXFP4 element 17 (code 9 -> -1)");
  CheckNear(out[2], 0.0f, "MXFP4 element 2 (code 0)");
  CheckNear(out[31], 0.0f, "MXFP4 element 31 (code 0)");

  // Through the dispatcher too, for 2 concatenated blocks (64 elements).
  std::vector<uint8_t> two_blocks = block;
  two_blocks.insert(two_blocks.end(), block.begin(), block.end());
  std::vector<float> dispatched;
  Check(DequantizeGgmlMxfp4(two_blocks.data(), two_blocks.size(),
                            GGML_TYPE_MXFP4, 64, &dispatched),
        "DequantizeGgmlMxfp4(2 blocks) succeeds");
  Check(dispatched.size() == 64, "DequantizeGgmlMxfp4 output size");
  if (dispatched.size() == 64) {
    CheckNear(dispatched[0], 1.0f, "MXFP4 dispatcher block 0 elem 0");
    CheckNear(dispatched[32], 1.0f, "MXFP4 dispatcher block 1 elem 0");
    CheckNear(dispatched[33], -2.0f, "MXFP4 dispatcher block 1 elem 1");
  }
}

void TestDequantizeGgmlMxfp4RejectsBadInput() {
  std::vector<uint8_t> block(Mxfp4BlockBytes(GGML_TYPE_MXFP4), 0);
  std::vector<float> out;

  // Not a multiple of the block size.
  Check(!DequantizeGgmlMxfp4(block.data(), block.size(), GGML_TYPE_MXFP4, 33,
                             &out),
        "rejects non-block-aligned numel");

  // raw_size doesn't match numel's expected byte length.
  Check(!DequantizeGgmlMxfp4(block.data(), block.size() - 1, GGML_TYPE_MXFP4,
                             32, &out),
        "rejects mismatched raw_size");

  // Not MXFP4 at all (F32 -- a raw type, not a block-quantized one).
  Check(
      !DequantizeGgmlMxfp4(block.data(), block.size(), GGML_TYPE_F32, 32, &out),
      "rejects non-MXFP4 ggml_type");

  // Also not MXFP4: a K-quant type.
  Check(!DequantizeGgmlMxfp4(block.data(), block.size(), GGML_TYPE_Q8_0, 32,
                             &out),
        "rejects K-quant ggml_type");

  // Negative numel.
  Check(!DequantizeGgmlMxfp4(block.data(), block.size(), GGML_TYPE_MXFP4, -1,
                             &out),
        "rejects negative numel");

  Check(out.empty(), "no partial output written on any rejected call");
}

}  // namespace

int main() {
  TestGgmlE8m0ToFloat32Half();
  TestMxfp4Block();
  TestDequantizeGgmlMxfp4RejectsBadInput();

  if (g_failures == 0) {
    std::printf("ggml_mxfp4_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ggml_mxfp4_test: %d failure(s)\n", g_failures);
  return 1;
}
