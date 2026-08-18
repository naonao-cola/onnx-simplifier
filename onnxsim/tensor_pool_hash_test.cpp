/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++17 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41
 *   -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 -DBLAKE3_USE_NEON=0 -I.. \
 *   tensor_pool_hash_test.cpp tensor_pool_hash.cpp \
 *   ../third_party/blake3/c/blake3.c \
 *   ../third_party/blake3/c/blake3_dispatch.c \
 *   ../third_party/blake3/c/blake3_portable.c -o t && ./t
 *
 * Checks both HashAlgorithm backends against their own published test
 * vectors (SHA-256's FIPS 180-4 examples; BLAKE3's official
 * test_vectors.json), plus ContentDigest's dtype/shape domain separation
 * and ParseHashAlgorithm/HashAlgorithmName's round trip.
 */
#include "tensor_pool_hash.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "tensor_pool_dtype.h"

using namespace onnxsim::tensor_pool;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// BLAKE3's official test vectors (test_vectors.json) generate each case's
// input as `input_len` bytes of the repeating sequence 0, 1, ..., 250, 0,
// 1, ... -- not e.g. all-zero -- so a wrong byte-order or off-by-one bug in
// the wrapper would very likely still be caught here.
std::string Blake3TestVectorInput(int len) {
  std::string s;
  s.reserve(static_cast<size_t>(len));
  for (int i = 0; i < len; ++i) {
    s.push_back(static_cast<char>(i % 251));
  }
  return s;
}

}  // namespace

int main() {
  // FIPS 180-4 / NIST SHA-256 example messages.
  struct ShaCase {
    const char* input;
    const char* hex;
  };
  const ShaCase kShaCases[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc",
       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
  };
  for (const auto& c : kShaCases) {
    std::string got = ToHex(RawDigest(HashAlgorithm::kSha256, c.input));
    Check(got == c.hex, std::string("sha256(\"") + c.input + "\") == " + c.hex +
                            " (got " + got + ")");
  }
  // The classic third NIST vector: one million repeated 'a', exercising
  // multi-block incremental Update() rather than a single call.
  {
    std::string got =
        ToHex(RawDigest(HashAlgorithm::kSha256, std::string(1000000, 'a')));
    const char* want =
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    Check(got == want,
          std::string("sha256(1e6 'a') == ") + want + " (got " + got + ")");
  }

  // BLAKE3's own published test vectors, input lengths 0/1/2 (32-byte
  // prefix of the (longer, XOF) published output).
  struct Blake3Case {
    int len;
    const char* hex;
  };
  const Blake3Case kBlake3Cases[] = {
      {0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
      {1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
      {2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"},
  };
  for (const auto& c : kBlake3Cases) {
    std::string got =
        ToHex(RawDigest(HashAlgorithm::kBlake3, Blake3TestVectorInput(c.len)));
    Check(got == c.hex, std::string("blake3(len=") + std::to_string(c.len) +
                            ") == " + c.hex + " (got " + got + ")");
  }

  // Both digests are 32 bytes (64 hex chars) for either algorithm.
  Check(RawDigest(HashAlgorithm::kBlake3, "x").size() == 32,
        "blake3 digest is 32 bytes");
  Check(RawDigest(HashAlgorithm::kSha256, "x").size() == 32,
        "sha256 digest is 32 bytes");

  // ContentDigest domain separation: same raw bytes, different dtype/shape
  // framing, must not collide.
  for (HashAlgorithm algo : {HashAlgorithm::kBlake3, HashAlgorithm::kSha256}) {
    std::string bytes(16, '\x07');
    std::string d_f32_2x2 = ContentDigest(algo, ONNX_FLOAT, {2, 2}, bytes);
    std::string d_i8_16 = ContentDigest(algo, ONNX_INT8, {16}, bytes);
    std::string d_f32_4 = ContentDigest(algo, ONNX_FLOAT, {4}, bytes);
    Check(d_f32_2x2 != d_i8_16,
          "ContentDigest distinguishes dtype for identical bytes");
    Check(d_f32_2x2 != d_f32_4,
          "ContentDigest distinguishes shape for identical bytes/dtype");
    // Same dtype/shape/bytes hashed twice is deterministic.
    Check(ContentDigest(algo, ONNX_FLOAT, {2, 2}, bytes) == d_f32_2x2,
          "ContentDigest is deterministic");
  }

  // The two algorithms must not silently agree with each other.
  Check(ContentDigest(HashAlgorithm::kBlake3, ONNX_FLOAT, {4}, "abcd") !=
            ContentDigest(HashAlgorithm::kSha256, ONNX_FLOAT, {4}, "abcd"),
        "blake3 and sha256 digests differ for the same tensor");

  // Name <-> enum round trip.
  Check(HashAlgorithmName(HashAlgorithm::kBlake3) == "blake3",
        "HashAlgorithmName(kBlake3) == \"blake3\"");
  Check(HashAlgorithmName(HashAlgorithm::kSha256) == "sha256",
        "HashAlgorithmName(kSha256) == \"sha256\"");
  Check(ParseHashAlgorithm("blake3") == HashAlgorithm::kBlake3,
        "ParseHashAlgorithm(\"blake3\") == kBlake3");
  Check(ParseHashAlgorithm("sha256") == HashAlgorithm::kSha256,
        "ParseHashAlgorithm(\"sha256\") == kSha256");
  {
    bool threw = false;
    try {
      ParseHashAlgorithm("md5");
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "ParseHashAlgorithm(\"md5\") throws invalid_argument");
  }

  Check(ToHex(std::string("\x00\x01\xab\xff", 4)) == "0001abff",
        "ToHex encodes lowercase, zero-padded byte pairs");

  if (g_failures == 0) {
    std::printf("tensor_pool_hash_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_hash_test: %d failure(s)\n", g_failures);
  return 1;
}
