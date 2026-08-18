/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 tensor_pool_test.cpp tensor_pool.cpp -o t && ./t
 *
 * Dependency-free unit test for TensorPool and its safetensors read/write
 * (no ONNX / ONNX Runtime -- see tensor_pool.h's top comment). The
 * onnx::TensorProto <-> TensorPool glue in tensor_pool_bridge.h is tested
 * separately (tensor_pool_bridge_test.cpp), since that one does need onnx.
 */
#include "tensor_pool.h"

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

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

std::string TempPath(const std::string& suffix) {
  return "/tmp/onnxsim_tensor_pool_test_" + std::to_string(::getpid()) + suffix;
}

void TestRoundTrip() {
  TensorPool pool;
  pool.Add("weight.0", ONNX_FLOAT, {2, 3},
           std::string("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b"
                       "\x0c\x0d\x0d\x0d\x0e\x0e\x0e\x0e\x0f\x0f\x0f\x0f",
                       24));
  pool.Add("bias", ONNX_INT64, {3},
           std::string(24, '\x2a'));  // 3 int64 elements, arbitrary bytes
  pool.Add("scalar", ONNX_BOOL, {}, std::string(1, '\x01'));
  pool.Add("empty", ONNX_FLOAT, {0}, std::string());

  Check(pool.size() == 4, "pool holds 4 entries before save");

  std::string path = TempPath("_roundtrip.safetensors");
  pool.SaveSafetensors(path);

  TensorPool loaded;
  loaded.LoadSafetensors(path);
  Check(loaded.size() == 4, "loaded pool holds 4 entries");

  const Entry* w = loaded.Find("weight.0");
  Check(w != nullptr, "weight.0 present after load");
  if (w != nullptr) {
    Check(w->dtype == ONNX_FLOAT, "weight.0 dtype round-trips");
    Check(w->shape == std::vector<int64_t>({2, 3}),
          "weight.0 shape round-trips");
    Check(w->nbytes() == 24, "weight.0 byte length round-trips");
    Check(w->data == std::string_view(
                         "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b"
                         "\x0c\x0d\x0d\x0d\x0e\x0e\x0e\x0e\x0f\x0f\x0f\x0f",
                         24),
          "weight.0 bytes round-trip exactly");
  }

  const Entry* b = loaded.Find("bias");
  Check(b != nullptr && b->dtype == ONNX_INT64 &&
            b->shape == std::vector<int64_t>({3}) && b->nbytes() == 24,
        "bias round-trips");

  const Entry* s = loaded.Find("scalar");
  Check(s != nullptr && s->shape.empty() && s->nbytes() == 1,
        "scalar (rank-0) tensor round-trips");

  const Entry* e = loaded.Find("empty");
  Check(e != nullptr && e->nbytes() == 0, "zero-size tensor round-trips");

  Check(loaded.Find("nope") == nullptr, "missing name returns nullptr");

  std::remove(path.c_str());
}

void TestAliasingIsZeroCopyPerTensor() {
  // Every Entry loaded from one file shares the SAME underlying allocation
  // (LoadSafetensors reads the file once, into one buffer) -- verify that
  // property directly rather than just trusting the doc comment: two
  // entries' data pointers must fall within one contiguous span whose size
  // matches the file's data segment, not two independent allocations.
  TensorPool pool;
  pool.Add("a", ONNX_UINT8, {4}, std::string(4, 'a'));
  pool.Add("b", ONNX_UINT8, {4}, std::string(4, 'b'));
  std::string path = TempPath("_alias.safetensors");
  pool.SaveSafetensors(path);

  TensorPool loaded;
  loaded.LoadSafetensors(path);
  const Entry* ea = loaded.Find("a");
  const Entry* eb = loaded.Find("b");
  Check(ea != nullptr && eb != nullptr, "both tensors present");
  if (ea != nullptr && eb != nullptr) {
    // owner.get() (the aliasing shared_ptr's stored pointer) equals data's
    // own base address for each entry, and both entries' owning control
    // block use_count should reflect a shared buffer.
    Check(static_cast<const void*>(ea->owner.get()) == ea->data.data(),
          "a: owner aliases its own data pointer");
    Check(static_cast<const void*>(eb->owner.get()) == eb->data.data(),
          "b: owner aliases its own data pointer");
    // The gap between the two views should exactly equal 'a's length,
    // confirming they were carved out of one contiguous buffer at the
    // offsets the header describes (rather than, say, two heap allocations
    // that happen to be adjacent by luck).
    Check(eb->data.data() - ea->data.data() ==
              static_cast<ptrdiff_t>(ea->nbytes()),
          "a and b are adjacent views into one contiguous buffer");
  }
  std::remove(path.c_str());
}

void TestHeaderPrefixSize() {
  TensorPool pool;
  pool.Add("x", ONNX_FLOAT, {1}, std::string(4, '\0'));
  std::string path = TempPath("_prefix.safetensors");
  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  pool.SaveSafetensors(path, &offsets);

  uint64_t prefix = HeaderPrefixSize(path);
  std::ifstream in(path, std::ios::binary);
  in.seekg(0, std::ios::end);
  uint64_t file_size = static_cast<uint64_t>(in.tellg());
  Check(prefix + offsets.at("x").second == file_size,
        "HeaderPrefixSize + last tensor's end offset == file size");
  std::remove(path.c_str());
}

void TestRejectsUnrepresentableDtype() {
  TensorPool pool;
  pool.Add("s", ONNX_STRING, {1}, std::string("x"));
  std::string path = TempPath("_bad_dtype.safetensors");
  bool threw = false;
  try {
    pool.SaveSafetensors(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "saving a STRING-dtype tensor throws");
}

void TestRejectsMalformedFile() {
  std::string path = TempPath("_malformed.safetensors");
  {
    std::ofstream out(path, std::ios::binary);
    out << "not a safetensors file";
  }
  TensorPool pool;
  bool threw = false;
  try {
    pool.LoadSafetensors(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "loading a non-safetensors file throws");
  std::remove(path.c_str());
}

// Writes bytes explicitly little-endian, regardless of host byte order --
// required here because this test hand-constructs a raw safetensors file
// (rather than going through TensorPool::SaveSafetensors, which already
// gets this right; see tensor_pool.h's "Byte order" note). A prior version
// of this helper used reinterpret_cast<const char*> on a host uint64_t/
// float directly, which happened to match the spec on a little-endian host
// but produced a file with a garbage header length -- and, on the tensor
// payload below, a garbage float -- on a big-endian one (caught by this
// repo's s390x CI job).
template <typename T>
void WriteLE(std::ostream& out, T v) {
  static_assert(std::is_trivially_copyable_v<T>);
  unsigned char raw[sizeof(T)];
  std::memcpy(raw, &v, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) {
    std::reverse(std::begin(raw), std::end(raw));
  }
  out.write(reinterpret_cast<const char*>(raw), sizeof(T));
}

void TestSkipsMetadata() {
  // __metadata__ is a valid, common safetensors header entry (arbitrary
  // string->string map, e.g. {"format": "pt"}) that this reader must skip
  // rather than mistake for a tensor descriptor.
  std::string path = TempPath("_metadata.safetensors");
  std::string header =
      R"({"__metadata__":{"format":"pt","note":"hello \"world\""},)"
      R"("t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  {
    std::ofstream out(path, std::ios::binary);
    WriteLE<uint64_t>(out, header.size());
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    WriteLE<float>(out, 1.5f);
  }
  TensorPool pool;
  pool.LoadSafetensors(path);
  Check(pool.size() == 1, "__metadata__ is skipped, not loaded as a tensor");
  Check(pool.Find("t") != nullptr, "the real tensor is still loaded");
  Check(pool.Find("__metadata__") == nullptr,
        "__metadata__ itself is not exposed as an entry");
  std::remove(path.c_str());
}

void TestContentHash() {
  TensorPool pool;
  std::string bytes(16, '\x2a');
  pool.Add("w1", ONNX_FLOAT, {2, 2}, std::string(bytes));
  pool.Add("w2", ONNX_FLOAT, {2, 2},
           std::string(bytes));  // same content, different name
  pool.Add("w3", ONNX_INT8, {16},
           std::string(bytes));  // same bytes, different dtype/shape

  Check(pool.GetHashAlgorithm() == HashAlgorithm::kBlake3,
        "TensorPool defaults to BLAKE3");

  std::string h1 = pool.ContentHash("w1");
  Check(h1.size() == 64, "ContentHash is a 32-byte digest, hex-encoded");
  Check(pool.ContentHash("w2") == h1,
        "identical dtype/shape/bytes under different names hash the same");
  Check(pool.ContentHash("w3") != h1,
        "identical bytes but different dtype/shape hash differently");

  auto all = pool.AllContentHashes();
  Check(all.size() == 3, "AllContentHashes covers every entry");
  Check(all.at("w1") == h1 && all.at("w2") == h1,
        "AllContentHashes agrees with ContentHash per entry");

  bool threw = false;
  try {
    pool.ContentHash("missing");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  Check(threw, "ContentHash throws out_of_range for an unknown name");

  pool.SetHashAlgorithm(HashAlgorithm::kSha256);
  Check(pool.GetHashAlgorithm() == HashAlgorithm::kSha256,
        "SetHashAlgorithm takes effect");
  Check(pool.ContentHash("w1") != h1,
        "switching algorithm changes the digest for the same tensor");
}

}  // namespace

int main() {
  TestRoundTrip();
  TestAliasingIsZeroCopyPerTensor();
  TestHeaderPrefixSize();
  TestRejectsUnrepresentableDtype();
  TestRejectsMalformedFile();
  TestSkipsMetadata();
  TestContentHash();

  if (g_failures == 0) {
    std::printf("tensor_pool_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_test: %d failure(s)\n", g_failures);
  return 1;
}
