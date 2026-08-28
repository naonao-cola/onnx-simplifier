#include "tensor_pool_hash.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "blake3/c/blake3.h"

namespace onnxsim {
namespace tensor_pool {

namespace {

constexpr size_t kDigestBytes = 32;

// --- SHA-256 (FIPS 180-4), a small self-contained implementation ----------
//
// Not vendored (see tensor_pool_hash.h's file comment): this is the whole
// algorithm, sequential, block-buffered, with no external dependency.
class Sha256 {
 public:
  Sha256() { Reset(); }

  void Reset() {
    h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    buf_len_ = 0;
    total_bytes_ = 0;
  }

  void Update(std::string_view data) {
    total_bytes_ += data.size();
    size_t pos = 0;
    if (buf_len_ != 0) {
      size_t take = std::min(data.size(), size_t{64} - buf_len_);
      std::memcpy(buf_.data() + buf_len_, data.data(), take);
      buf_len_ += static_cast<uint32_t>(take);
      pos += take;
      if (buf_len_ == 64) {
        ProcessBlock(buf_.data());
        buf_len_ = 0;
      }
    }
    while (data.size() - pos >= 64) {
      ProcessBlock(data.data() + pos);
      pos += 64;
    }
    if (pos < data.size()) {
      size_t remaining = data.size() - pos;
      std::memcpy(buf_.data(), data.data() + pos, remaining);
      buf_len_ = static_cast<uint32_t>(remaining);
    }
  }

  std::string Finalize() {
    uint64_t total_bits = total_bytes_ * 8;
    // Append the mandatory 0x80 bit-marker byte, then zero-pad to a
    // 56-bytes-mod-64 boundary, leaving exactly 8 bytes for the trailing
    // big-endian bit-length field.
    uint8_t pad = 0x80;
    Update(std::string_view(reinterpret_cast<const char*>(&pad), 1));
    // Zero-pad to a 56-bytes-mod-64 boundary, leaving exactly 8 bytes for
    // the trailing bit-length field below. `need` is in [0, 63]; Update()
    // handles the case where buf_len_ + need crosses a 64-byte block
    // boundary on its own.
    static const std::array<char, 64> kZeros{};
    size_t need = (64 + 56 - buf_len_) % 64;
    Update(std::string_view(kZeros.data(), need));
    std::array<uint8_t, 8> len_be{};
    for (int i = 0; i < 8; ++i) {
      len_be[7 - i] = static_cast<uint8_t>(total_bits >> (8 * i));
    }
    Update(std::string_view(reinterpret_cast<const char*>(len_be.data()), 8));

    std::string digest(kDigestBytes, '\0');
    for (int i = 0; i < 8; ++i) {
      digest[4 * i + 0] = static_cast<char>((h_[i] >> 24) & 0xFF);
      digest[4 * i + 1] = static_cast<char>((h_[i] >> 16) & 0xFF);
      digest[4 * i + 2] = static_cast<char>((h_[i] >> 8) & 0xFF);
      digest[4 * i + 3] = static_cast<char>(h_[i] & 0xFF);
    }
    return digest;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void ProcessBlock(const char* block) {
    static constexpr uint32_t kK[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      auto b = reinterpret_cast<const uint8_t*>(block) + 4 * i;
      w[i] = (static_cast<uint32_t>(b[0]) << 24) |
             (static_cast<uint32_t>(b[1]) << 16) |
             (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t temp1 = h + s1 + ch + kK[i] + w[i];
      uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  std::array<uint32_t, 8> h_{};
  std::array<char, 64> buf_{};
  uint32_t buf_len_ = 0;
  uint64_t total_bytes_ = 0;
};

std::string Blake3Digest(std::string_view data) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, data.data(), data.size());
  std::string digest(kDigestBytes, '\0');
  blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t*>(digest.data()),
                         kDigestBytes);
  return digest;
}

std::string Sha256Digest(std::string_view data) {
  Sha256 sha;
  sha.Update(data);
  return sha.Finalize();
}

}  // namespace

std::string_view HashAlgorithmName(HashAlgorithm algo) {
  switch (algo) {
    case HashAlgorithm::kBlake3:
      return "blake3";
    case HashAlgorithm::kSha256:
      return "sha256";
  }
  throw std::invalid_argument("tensor_pool: unknown HashAlgorithm enumerator");
}

HashAlgorithm ParseHashAlgorithm(std::string_view name) {
  if (name == "blake3") return HashAlgorithm::kBlake3;
  if (name == "sha256") return HashAlgorithm::kSha256;
  throw std::invalid_argument("tensor_pool: unrecognized hash algorithm '" +
                              std::string(name) +
                              "' (expected 'blake3' or 'sha256')");
}

std::string RawDigest(HashAlgorithm algo, std::string_view data) {
  switch (algo) {
    case HashAlgorithm::kBlake3:
      return Blake3Digest(data);
    case HashAlgorithm::kSha256:
      return Sha256Digest(data);
  }
  throw std::invalid_argument("tensor_pool: unknown HashAlgorithm enumerator");
}

std::string ContentDigest(HashAlgorithm algo, int32_t dtype,
                          const std::vector<int64_t>& shape,
                          std::string_view data) {
  // Built as one contiguous framing buffer (rather than two incremental
  // Update() calls) so RawDigest -- and therefore both backends above --
  // stays the single, test-vector-checkable entry point; framing headers
  // are tiny (a handful of int64s) next to real tensor data, so the extra
  // copy this costs is negligible.
  std::string framed;
  framed.reserve(4 + 4 + shape.size() * 8 + data.size());
  auto append_le = [&framed](uint64_t v, int nbytes) {
    for (int i = 0; i < nbytes; ++i) {
      framed.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
  };
  append_le(static_cast<uint32_t>(dtype), 4);
  append_le(static_cast<uint32_t>(shape.size()), 4);
  for (int64_t dim : shape) {
    append_le(static_cast<uint64_t>(dim), 8);
  }
  framed.append(data.data(), data.size());
  return RawDigest(algo, framed);
}

std::string ToHex(std::string_view bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    out.push_back(kHex[c >> 4]);
    out.push_back(kHex[c & 0x0F]);
  }
  return out;
}

}  // namespace tensor_pool
}  // namespace onnxsim
