/*
 * Vendored from https://github.com/BLAKE3-team/BLAKE3 (dual CC0-1.0 /
 * Apache-2.0; see LICENSE_CC0 / LICENSE_A2 in this directory), pinned at
 * commit 77b257eee7da5cd608eaf6be8343d3a4c9776af2 (v1.8.6).
 *
 * Only the five portable, dependency-free source files are vendored here
 * (blake3.h, blake3_impl.h, blake3.c, blake3_dispatch.c,
 * blake3_portable.c) -- none of the per-architecture SIMD implementations
 * (SSE2/SSE4.1/AVX2/AVX-512/NEON, each requiring its own intrinsics or
 * hand-written assembly). tensor_pool_hash.cpp compiles this subset with
 * -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512
 * -DBLAKE3_USE_NEON=0, which makes blake3_dispatch.c's runtime CPU-feature
 * dispatch fall through to blake3_portable.c unconditionally -- correct
 * and portable everywhere (matches this repo's other third_party/ code,
 * which also avoids per-arch SIMD source), just without BLAKE3's SIMD
 * speedup. Revisit if TensorPool content-hashing shows up as a real
 * bottleneck.
 */
#ifndef BLAKE3_H
#define BLAKE3_H

#include <stddef.h>
#include <stdint.h>

#if !defined(BLAKE3_API)
# if defined(_WIN32) || defined(__CYGWIN__)
#   if defined(BLAKE3_DLL)
#     if defined(BLAKE3_DLL_EXPORTS)
#       define BLAKE3_API __declspec(dllexport)
#     else
#       define BLAKE3_API __declspec(dllimport)
#     endif
#     define BLAKE3_PRIVATE
#   else
#     define BLAKE3_API
#     define BLAKE3_PRIVATE
#   endif
# elif __GNUC__ >= 4
#   define BLAKE3_API __attribute__((visibility("default")))
#   define BLAKE3_PRIVATE __attribute__((visibility("hidden")))
# else
#   define BLAKE3_API
#   define BLAKE3_PRIVATE
# endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE3_VERSION_STRING "1.8.6"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

// This struct is a private implementation detail. It has to be here because
// it's part of the blake3_hasher structure defined below.
typedef struct {
  uint32_t cv[8];
  uint64_t chunk_counter;
  uint8_t buf[BLAKE3_BLOCK_LEN];
  uint8_t buf_len;
  uint8_t blocks_compressed;
  uint8_t flags;
} blake3_chunk_state;

typedef struct {
  uint32_t key[8];
  blake3_chunk_state chunk;
  uint8_t cv_stack_len;
  // The stack size is MAX_DEPTH + 1 because we do lazy merging. For example,
  // with 7 chunks, we have 3 entries in the stack. Adding an 8th chunk
  // requires a 4th entry, rather than merging everything down to 1, because we
  // don't know whether more input is coming. This is different from how the
  // reference implementation does things.
  uint8_t cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
} blake3_hasher;

BLAKE3_API const char *blake3_version(void);
BLAKE3_API void blake3_hasher_init(blake3_hasher *self);
BLAKE3_API void blake3_hasher_init_keyed(blake3_hasher *self,
                                         const uint8_t key[BLAKE3_KEY_LEN]);
BLAKE3_API void blake3_hasher_init_derive_key(blake3_hasher *self, const char *context);
BLAKE3_API void blake3_hasher_init_derive_key_raw(blake3_hasher *self, const void *context,
                                                  size_t context_len);
BLAKE3_API void blake3_hasher_update(blake3_hasher *self, const void *input,
                                     size_t input_len);
#if defined(BLAKE3_USE_TBB)
BLAKE3_API void blake3_hasher_update_tbb(blake3_hasher *self, const void *input,
                                         size_t input_len);
#endif // BLAKE3_USE_TBB
BLAKE3_API void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out,
                                       size_t out_len);
BLAKE3_API void blake3_hasher_finalize_seek(const blake3_hasher *self, uint64_t seek,
                                            uint8_t *out, size_t out_len);
BLAKE3_API void blake3_hasher_reset(blake3_hasher *self);

#ifdef __cplusplus
}
#endif

#endif /* BLAKE3_H */
