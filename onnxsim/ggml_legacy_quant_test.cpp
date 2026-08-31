/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 ggml_legacy_quant_test.cpp -o t && ./t
 *
 * Dependency-free unit test for ggml_legacy_quant.h's GGML legacy quant
 * dequantization (Q4_0, Q4_1, Q5_0, Q5_1), mirroring ggml_kquant_test.cpp's
 * style.
 *
 * Every block layout/formula this decodes was cross-checked against GGML's
 * own reference (ggml-common.h's block_q4_0/q4_1/q5_0/q5_1, ggml-quants.c's
 * dequantize_row_q4_0/q4_1/q5_0/q5_1). The cases here are small,
 * hand-verifiable vectors (chosen so most of a block's bytes are zero and
 * its non-zero bytes' contribution can be checked by hand), meant to catch
 * a regression in this file specifically, not to re-derive GGML's spec from
 * scratch.
 */
#include "ggml_legacy_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
               float tol = 1e-4f) {
  Check(std::fabs(got - want) <= tol, what + " (got " + std::to_string(got) +
                                          ", want " + std::to_string(want) +
                                          ")");
}

// Encodes `f` as an IEEE754 half-precision bit pattern -- same reference
// encoder as ggml_kquant_test.cpp's EncodeF16 (round-to-nearest not
// implemented; every value used below is exactly representable in fp16),
// used only to BUILD test input.
uint16_t EncodeF16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

void WriteLE16(std::vector<uint8_t>& buf, uint16_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

// d = 2.0, qs[i] = i (so the low/high nibbles span the full 0-15 range this
// test cares about). expected[j] = (qs[j]&0xF - 8) * 2.0, expected[j+16] =
// (qs[j]>>4 - 8) * 2.0.
void TestQ4_0() {
  std::vector<uint8_t> block;
  WriteLE16(block, EncodeF16(2.0f));
  for (int i = 0; i < 16; ++i) block.push_back(static_cast<uint8_t>(i));
  Check(block.size() == LegacyQuantBlockBytes(GGML_TYPE_Q4_0), "Q4_0 size");

  float out[32];
  DequantizeQ4_0Block(block.data(), out);
  // qs[0] = 0x00: low nibble 0 -> (0-8)*2 = -16; high nibble 0 -> -16.
  CheckNear(out[0], -16.0f, "Q4_0 element 0");
  CheckNear(out[16], -16.0f, "Q4_0 element 16");
  // qs[15] = 0x0F: low nibble 15 -> (15-8)*2 = 14; high nibble 0 -> -16.
  CheckNear(out[15], 14.0f, "Q4_0 element 15");
  CheckNear(out[31], -16.0f, "Q4_0 element 31");

  std::vector<uint8_t> two_blocks = block;
  two_blocks.insert(two_blocks.end(), block.begin(), block.end());
  std::vector<float> dispatched;
  Check(DequantizeGgmlLegacyQuant(two_blocks.data(), two_blocks.size(),
                                  GGML_TYPE_Q4_0, 64, &dispatched),
        "DequantizeGgmlLegacyQuant(Q4_0, 2 blocks) succeeds");
  Check(dispatched.size() == 64, "DequantizeGgmlLegacyQuant(Q4_0) size");
  if (dispatched.size() == 64) {
    CheckNear(dispatched[0], -16.0f, "Q4_0 dispatcher block 0 elem 0");
    CheckNear(dispatched[32], -16.0f, "Q4_0 dispatcher block 1 elem 0");
  }
}

// d = 1.0, m = 0.5, qs[0] = 0x0A (low nibble 10, high nibble 0), rest zero.
// expected[0] = 10*1.0 + 0.5 = 10.5; expected[16] = 0*1.0 + 0.5 = 0.5.
void TestQ4_1() {
  std::vector<uint8_t> block(LegacyQuantBlockBytes(GGML_TYPE_Q4_1), 0);
  const uint16_t d_bits = EncodeF16(1.0f);
  const uint16_t m_bits = EncodeF16(0.5f);
  block[0] = static_cast<uint8_t>(d_bits & 0xFF);
  block[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
  block[2] = static_cast<uint8_t>(m_bits & 0xFF);
  block[3] = static_cast<uint8_t>((m_bits >> 8) & 0xFF);
  block[4] = 0x0A;  // qs[0]

  float out[32];
  DequantizeQ4_1Block(block.data(), out);
  CheckNear(out[0], 10.5f, "Q4_1 element 0");
  CheckNear(out[16], 0.5f, "Q4_1 element 16 (zeroed nibble)");
  CheckNear(out[1], 0.5f, "Q4_1 element 1 (zeroed nibble)");
}

// d = 2.0, qh selects the 5th bit for element 0 only (bit 0 -> xh_0 set for
// j=0), qs[0] low nibble = 3. expected[0] = ((3 | 16) - 16) * 2.0 = 3*2=6.0
// Wait: xh_0 for j=0 is bit (j+0)=0 of qh, shifted <<4 and masked 0x10, so a
// set bit 0 contributes 0x10 to the low nibble -> code = 3 | 0x10 = 19,
// 19-16=3, *d=2 -> 6.0. High nibble (element 16) uses xh_1 = bit (j+12)=12
// of qh; left at 0, so code = (qs[0]>>4) - 16 = 0-16 = -16, *2 = -32.0.
void TestQ5_0() {
  std::vector<uint8_t> block(LegacyQuantBlockBytes(GGML_TYPE_Q5_0), 0);
  const uint16_t d_bits = EncodeF16(2.0f);
  block[0] = static_cast<uint8_t>(d_bits & 0xFF);
  block[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
  // qh is block[2..5]; set bit 0.
  block[2] = 0x01;
  // qs starts at block[6]; qs[0] low nibble = 3.
  block[6] = 0x03;

  float out[32];
  DequantizeQ5_0Block(block.data(), out);
  CheckNear(out[0], 6.0f, "Q5_0 element 0 (5th bit set)");
  CheckNear(out[16], -32.0f, "Q5_0 element 16 (5th bit unset)");
  CheckNear(out[1], -32.0f, "Q5_0 element 1 (zeroed nibble, no 5th bit)");

  std::vector<uint8_t> two_blocks = block;
  two_blocks.insert(two_blocks.end(), block.begin(), block.end());
  std::vector<float> dispatched;
  Check(DequantizeGgmlLegacyQuant(two_blocks.data(), two_blocks.size(),
                                  GGML_TYPE_Q5_0, 64, &dispatched),
        "DequantizeGgmlLegacyQuant(Q5_0, 2 blocks) succeeds");
  Check(dispatched.size() == 64, "DequantizeGgmlLegacyQuant(Q5_0) size");
  if (dispatched.size() == 64) {
    CheckNear(dispatched[0], 6.0f, "Q5_0 dispatcher block 0 elem 0");
    CheckNear(dispatched[32], 6.0f, "Q5_0 dispatcher block 1 elem 0");
  }
}

// Same 5th-bit trick as Q5_0, but with an explicit min like Q4_1: d=1.0,
// m=0.5, qh bit 0 set (adds 16 to element 0's code), qs[0] low nibble = 3.
// expected[0] = (3|16)*1.0 + 0.5 = 19.5. expected[16] (xh_1 unset,
// qs[0]>>4=0): 0*1.0+0.5 = 0.5.
void TestQ5_1() {
  std::vector<uint8_t> block(LegacyQuantBlockBytes(GGML_TYPE_Q5_1), 0);
  const uint16_t d_bits = EncodeF16(1.0f);
  const uint16_t m_bits = EncodeF16(0.5f);
  block[0] = static_cast<uint8_t>(d_bits & 0xFF);
  block[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
  block[2] = static_cast<uint8_t>(m_bits & 0xFF);
  block[3] = static_cast<uint8_t>((m_bits >> 8) & 0xFF);
  // qh is block[4..7]; set bit 0.
  block[4] = 0x01;
  // qs starts at block[8]; qs[0] low nibble = 3.
  block[8] = 0x03;

  float out[32];
  DequantizeQ5_1Block(block.data(), out);
  CheckNear(out[0], 19.5f, "Q5_1 element 0 (5th bit set)");
  CheckNear(out[16], 0.5f, "Q5_1 element 16 (5th bit unset, zeroed nibble)");
}

void TestDequantizeGgmlLegacyQuantRejectsBadInput() {
  std::vector<uint8_t> q4_0_block(LegacyQuantBlockBytes(GGML_TYPE_Q4_0), 0);
  std::vector<float> out;

  Check(!DequantizeGgmlLegacyQuant(q4_0_block.data(), q4_0_block.size(),
                                   GGML_TYPE_Q4_0, 33, &out),
        "rejects non-block-aligned numel");
  Check(!DequantizeGgmlLegacyQuant(q4_0_block.data(), q4_0_block.size() - 1,
                                   GGML_TYPE_Q4_0, 32, &out),
        "rejects mismatched raw_size");
  Check(!DequantizeGgmlLegacyQuant(q4_0_block.data(), q4_0_block.size(),
                                   GGML_TYPE_F32, 32, &out),
        "rejects non-legacy-quant ggml_type");
  Check(!DequantizeGgmlLegacyQuant(q4_0_block.data(), q4_0_block.size(),
                                   GGML_TYPE_Q8_0, 32, &out),
        "rejects K-quant ggml_type");
  Check(!DequantizeGgmlLegacyQuant(q4_0_block.data(), q4_0_block.size(),
                                   GGML_TYPE_Q4_0, -1, &out),
        "rejects negative numel");
  Check(out.empty(), "no partial output written on any rejected call");
}

}  // namespace

int main() {
  TestQ4_0();
  TestQ4_1();
  TestQ5_0();
  TestQ5_1();
  TestDequantizeGgmlLegacyQuantRejectsBadInput();

  if (g_failures == 0) {
    std::printf("ggml_legacy_quant_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ggml_legacy_quant_test: %d failure(s)\n", g_failures);
  return 1;
}
