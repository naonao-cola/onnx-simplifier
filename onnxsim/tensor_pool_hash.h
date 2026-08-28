/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Content hashing for TensorPool entries: a name-independent digest of a
 * tensor's dtype + shape + raw bytes, usable both as an internal
 * dedup/lookup key (TensorPool::ContentHash) and, hex-encoded and exported
 * into a model's metadata_props (see tensor_pool_bridge.h's
 * AttachContentHashMetadata), as an external integrity value a downstream
 * consumer can recompute without onnxsim.
 *
 * Pluggable, not hardcoded to one algorithm, because those two uses are
 * different trust boundaries wearing the same digest: TensorPool's own
 * lookups never leave this process, so a fast non-cryptographic hash would
 * suffice there, but once a digest is exported into metadata_props for
 * another tool to verify, an adversary able to modify the file bytes can
 * also recompute a *non*-cryptographic hash to match -- integrity export
 * needs real (second-)preimage resistance. Rather than silently picking a
 * hash that's right for one use and wrong for the other, HashAlgorithm is
 * an explicit, caller-chosen knob on TensorPool (see
 * TensorPool::SetHashAlgorithm in tensor_pool.h), defaulting to a cryptographic
 * hash so the common case
 * ("call TensorPool::ContentHash, export it") is safe by default:
 *
 *   * kBlake3 (default) -- cryptographic, and (per BLAKE3's own published
 *     benchmarks: https://github.com/BLAKE3-team/BLAKE3#performance)
 *     several times faster than SHA-256 on typical hardware, so there is no
 *     throughput reason to reach for a weaker hash even for the
 *     process-internal pool lookups.
 *   * kSha256 -- for interoperability with a downstream verifier that only
 *     knows SHA-256 (an existing supply-chain/provenance pipeline, or a
 *     FIPS-validated-algorithm requirement), at the usual throughput cost
 *     SHA-256-vs-BLAKE3 benchmarks show. This repo's SHA-256 is its own
 *     straightforward FIPS 180-4 implementation (tensor_pool_hash.cpp), not
 *     vendored -- unlike BLAKE3, whose tree-mode construction is
 *     error-prone to reimplement from the spec, SHA-256 is a small, fully
 *     specified sequential algorithm with no room for a "close enough"
 *     reimplementation to silently diverge from the standard, so it doesn't
 *     carry the same correctness risk that would justify vendoring instead.
 *
 * Only these two are offered ("or other functions" was part of the original
 * ask): a hash exposed for external integrity verification should be a
 * well-vetted cryptographic hash, and BLAKE3 already dominates every faster
 * non-cryptographic hash (xxHash, CityHash, ...) in throughput while also
 * being safe to use as the default -- adding one of those would only be a
 * weaker option with no upside. SHA-256 stays purely for interop with
 * something else that has already standardized on it.
 */
#ifndef ONNXSIM_TENSOR_POOL_HASH_H_
#define ONNXSIM_TENSOR_POOL_HASH_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace onnxsim {
namespace tensor_pool {

enum class HashAlgorithm {
  kBlake3,
  kSha256,
};

// "blake3" / "sha256" -- also what ParseHashAlgorithm accepts.
std::string_view HashAlgorithmName(HashAlgorithm algo);

// Inverse of HashAlgorithmName. Throws std::invalid_argument for any other
// string (including a differently-cased spelling) rather than silently
// falling back to a default -- a typo'd algorithm name should fail loudly,
// not silently change what a caller's exported digest actually verifies.
HashAlgorithm ParseHashAlgorithm(std::string_view name);

// Raw digest bytes (32 for both algorithms) of `data`, with no framing.
// Exposed mainly so tensor_pool_hash_test.cpp can check this module's two
// backends against each algorithm's own published test vectors; prefer
// ContentDigest below for an actual tensor, which also domain-separates on
// dtype/shape.
std::string RawDigest(HashAlgorithm algo, std::string_view data);

// Digest of `dtype` + `shape` + `data`, domain-separated so two tensors
// with identical raw bytes but different dtype/shape never collide by
// construction (not just by the underlying hash's own collision
// resistance) -- e.g. a float32 [2,2] tensor and an int8 [16] tensor built
// from the same 16 bytes hash differently. The framing mixed in ahead of
// the raw bytes is `dtype` as a little-endian int32, then shape.size() as
// a little-endian uint32, then each shape[i] as a little-endian int64 --
// deliberately not onnx::TensorProto wire bytes, so this stays buildable
// without onnx (see tensor_pool.h's file comment for why that matters).
std::string ContentDigest(HashAlgorithm algo, int32_t dtype,
                          const std::vector<int64_t>& shape,
                          std::string_view data);

// Lowercase hex encoding of a digest (or any byte string).
std::string ToHex(std::string_view bytes);

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_HASH_H_
