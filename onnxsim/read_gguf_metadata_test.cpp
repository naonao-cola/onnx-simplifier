/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 read_gguf_metadata_test.cpp tensor_pool.cpp \
 *               tensor_pool_gguf.cpp -o t && ./t
 *
 * Dependency-free unit test for ReadGGUFMetadata (tensor_pool_gguf.cpp) --
 * the metadata-KV + tensor-info reader TensorPool::LoadGGUF itself does NOT
 * provide (LoadGGUF parses that very same header section but only ever
 * looks at general.alignment). Mirrors tensor_pool_gguf_test.cpp's
 * hand-built-bytes style, but exercises every GGUF metadata value type
 * (unsigned/signed ints of every width, both float widths, bool, string)
 * plus confirms ARRAY values are skipped correctly -- not surfaced, and not
 * mistaken for a truncated/misaligned read of whatever follows.
 */
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "gguf_dtype.h"
#include "tensor_pool.h"

using namespace onnxsim::tensor_pool;
using namespace onnxsim::tensor_pool::gguf;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string TempPath(const std::string& suffix) {
  return "/tmp/onnxsim_read_gguf_metadata_test_" + std::to_string(::getpid()) +
         suffix;
}

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

void WriteGGUFString(std::ostream& out, const std::string& s) {
  WriteLE<uint64_t>(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void WriteStringKV(std::ostream& out, const std::string& key,
                   const std::string& value) {
  WriteGGUFString(out, key);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_STRING);
  WriteGGUFString(out, value);
}

template <typename T>
void WriteScalarKV(std::ostream& out, const std::string& key,
                   uint32_t value_type, T value) {
  WriteGGUFString(out, key);
  WriteLE<uint32_t>(out, value_type);
  WriteLE<T>(out, value);
}

void WriteBoolKV(std::ostream& out, const std::string& key, bool value) {
  WriteGGUFString(out, key);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_BOOL);
  WriteLE<uint8_t>(out, value ? 1 : 0);
}

// A UINT32 array -- checks that ReadGGUFMetadata skips PAST an array's whole
// payload (not just its length prefix), so whatever comes after it in the
// file is read from the right offset.
void WriteUint32ArrayKV(std::ostream& out, const std::string& key,
                        const std::vector<uint32_t>& values) {
  WriteGGUFString(out, key);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_ARRAY);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_UINT32);  // element type
  WriteLE<uint64_t>(out, values.size());
  for (uint32_t v : values) WriteLE<uint32_t>(out, v);
}

// Same, but a STRING array -- a different skip path (per-element
// length-prefixed strings, not a fixed elem_size * len seek).
void WriteStringArrayKV(std::ostream& out, const std::string& key,
                        const std::vector<std::string>& values) {
  WriteGGUFString(out, key);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_ARRAY);
  WriteLE<uint32_t>(out, GGUF_METADATA_VALUE_TYPE_STRING);  // element type
  WriteLE<uint64_t>(out, values.size());
  for (const auto& s : values) WriteGGUFString(out, s);
}

// Every scalar type (including a negative signed case per width), both
// array-skip paths, and a two-entry tensor-info section -- with NO tensor
// data at all following it. A file this short would fail any attempt to
// read real tensor bytes, so successfully parsing to the end proves
// ReadGGUFMetadata truly never touches the data section.
void TestFullMetadataDecodeNoTensorDataNeeded() {
  std::string path = TempPath("_full.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 2);   // tensor_count
    WriteLE<uint64_t>(out, 14);  // metadata_kv_count

    WriteStringKV(out, "general.architecture", "llama");
    WriteScalarKV<uint8_t>(out, "u8", GGUF_METADATA_VALUE_TYPE_UINT8, 200);
    WriteScalarKV<uint16_t>(out, "u16", GGUF_METADATA_VALUE_TYPE_UINT16, 40000);
    WriteScalarKV<uint32_t>(out, "llama.block_count",
                            GGUF_METADATA_VALUE_TYPE_UINT32, 32);
    WriteScalarKV<uint64_t>(out, "u64", GGUF_METADATA_VALUE_TYPE_UINT64,
                            9876543210ULL);
    WriteScalarKV<int8_t>(out, "i8_neg", GGUF_METADATA_VALUE_TYPE_INT8, -5);
    WriteScalarKV<int16_t>(out, "i16_neg", GGUF_METADATA_VALUE_TYPE_INT16,
                           static_cast<int16_t>(-1000));
    WriteScalarKV<int32_t>(out, "i32_neg", GGUF_METADATA_VALUE_TYPE_INT32,
                           -70000);
    WriteScalarKV<int64_t>(out, "i64_neg", GGUF_METADATA_VALUE_TYPE_INT64,
                           -5000000000LL);
    WriteScalarKV<float>(out, "llama.rope.freq_base",
                         GGUF_METADATA_VALUE_TYPE_FLOAT32, 10000.0f);
    WriteScalarKV<double>(out, "f64", GGUF_METADATA_VALUE_TYPE_FLOAT64, -0.125);
    WriteBoolKV(out, "llama.attention.use_bias", true);
    WriteUint32ArrayKV(out, "some.int.array", {10, 20, 30});
    WriteStringArrayKV(out, "tokenizer.ggml.tokens", {"a", "bb", "ccc"});

    // Tensor 0: "tok_embeddings.weight", F32, ne=[8, 4] (ggml order --
    // reversed to onnx's [4, 8] on decode). Offset is 0 but never followed.
    WriteGGUFString(out, "tok_embeddings.weight");
    WriteLE<uint32_t>(out, 2);              // n_dimensions
    WriteLE<uint64_t>(out, 8);              // ne[0]
    WriteLE<uint64_t>(out, 4);              // ne[1]
    WriteLE<uint32_t>(out, GGML_TYPE_F32);  // type
    WriteLE<uint64_t>(out, 0);              // offset (never read)

    // Tensor 1: a K-quant tensor (Q4_K) -- proves ReadGGUFMetadata reports
    // its ggml_type as-is (12) rather than trying to interpret/skip it, and
    // proves the *previous* tensor's info was consumed at the right length
    // even though this one is a different element count/rank.
    WriteGGUFString(out, "blk.0.attn_q.weight");
    WriteLE<uint32_t>(out, 1);               // n_dimensions
    WriteLE<uint64_t>(out, 256);             // ne[0]
    WriteLE<uint32_t>(out, GGML_TYPE_Q4_K);  // type
    WriteLE<uint64_t>(out, 12345);           // offset (never read)

    // Deliberately NOT writing any tensor data -- ReadGGUFMetadata must
    // never seek/read past this point.
  }

  GGUFMetadata meta = ReadGGUFMetadata(path);

  Check(meta.kv.size() == 12,
        "12 scalar KV entries decoded (2 arrays correctly excluded)");
  Check(meta.kv.at("general.architecture").kind ==
                GGUFMetadataValue::Kind::kString &&
            meta.kv.at("general.architecture").string_value == "llama",
        "STRING value decodes");
  Check(meta.kv.at("u8").kind == GGUFMetadataValue::Kind::kInt &&
            meta.kv.at("u8").int_value == 200,
        "UINT8 value decodes");
  Check(meta.kv.at("u16").int_value == 40000, "UINT16 value decodes");
  Check(meta.kv.at("llama.block_count").int_value == 32,
        "UINT32 value decodes");
  Check(meta.kv.at("u64").int_value == 9876543210LL, "UINT64 value decodes");
  Check(meta.kv.at("i8_neg").int_value == -5,
        "INT8 negative value sign-extends correctly");
  Check(meta.kv.at("i16_neg").int_value == -1000,
        "INT16 negative value sign-extends correctly");
  Check(meta.kv.at("i32_neg").int_value == -70000,
        "INT32 negative value sign-extends correctly");
  Check(meta.kv.at("i64_neg").int_value == -5000000000LL,
        "INT64 negative value round-trips");
  Check(meta.kv.at("llama.rope.freq_base").kind ==
                GGUFMetadataValue::Kind::kFloat &&
            meta.kv.at("llama.rope.freq_base").float_value == 10000.0,
        "FLOAT32 value decodes exactly (10000.0 has an exact float32 repr)");
  Check(meta.kv.at("f64").float_value == -0.125,
        "FLOAT64 value decodes exactly (-0.125 has an exact float64 repr)");
  Check(meta.kv.at("llama.attention.use_bias").kind ==
                GGUFMetadataValue::Kind::kBool &&
            meta.kv.at("llama.attention.use_bias").bool_value == true,
        "BOOL value decodes");
  Check(meta.kv.find("some.int.array") == meta.kv.end(),
        "an int ARRAY value is omitted from kv, not surfaced");
  Check(meta.kv.find("tokenizer.ggml.tokens") == meta.kv.end(),
        "a string ARRAY value is omitted from kv, not surfaced");

  Check(meta.tensors.size() == 2, "both tensor-info entries decoded");
  if (meta.tensors.size() == 2) {
    Check(meta.tensors[0].name == "tok_embeddings.weight",
          "tensor 0 name decodes");
    Check(meta.tensors[0].shape == std::vector<int64_t>({4, 8}),
          "tensor 0 shape is reversed from ggml's ne[] to onnx order");
    Check(meta.tensors[0].ggml_type == GGML_TYPE_F32,
          "tensor 0 ggml_type decodes");
    Check(meta.tensors[1].name == "blk.0.attn_q.weight",
          "tensor 1 name decodes (correctly offset past tensor 0's info)");
    Check(meta.tensors[1].shape == std::vector<int64_t>({256}),
          "tensor 1 (rank 1) shape decodes");
    Check(meta.tensors[1].ggml_type == GGML_TYPE_Q4_K,
          "tensor 1's K-quant ggml_type is reported as-is, uninterpreted");
  }

  std::remove(path.c_str());
}

void TestEmptyMetadataAndNoTensors() {
  std::string path = TempPath("_empty.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 0);  // tensor_count
    WriteLE<uint64_t>(out, 0);  // metadata_kv_count
  }
  GGUFMetadata meta = ReadGGUFMetadata(path);
  Check(meta.kv.empty(), "no metadata entries");
  Check(meta.tensors.empty(), "no tensor entries");
  std::remove(path.c_str());
}

void TestRejectsBadMagic() {
  std::string path = TempPath("_badmagic.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    WriteLE<uint32_t>(out, 0xDEADBEEF);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 0);
    WriteLE<uint64_t>(out, 0);
  }
  bool threw = false;
  try {
    ReadGGUFMetadata(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "bad magic throws");
  std::remove(path.c_str());
}

void TestMissingFileThrows() {
  bool threw = false;
  try {
    ReadGGUFMetadata(TempPath("_does_not_exist.gguf"));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "a missing file throws rather than returning empty metadata");
}

}  // namespace

int main() {
  TestFullMetadataDecodeNoTensorDataNeeded();
  TestEmptyMetadataAndNoTensors();
  TestRejectsBadMagic();
  TestMissingFileThrows();

  if (g_failures == 0) {
    std::printf("read_gguf_metadata_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "read_gguf_metadata_test: %d failure(s)\n", g_failures);
  return 1;
}
