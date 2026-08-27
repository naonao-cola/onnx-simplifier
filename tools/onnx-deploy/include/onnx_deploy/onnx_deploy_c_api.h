/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * A minimal, stable C ABI over onnx_deploy::KvCachePipeline (see
 * kv_cache_pipeline.h), mirroring onnxsim's own C ABI conventions
 * (onnxsim/capi/onnxsim_c_api.h): every fallible entry point returns an
 * OnnxDeployStatus and takes a nullable char** out_error for a freshly
 * allocated, NUL-terminated message on failure.
 *
 * The one thing this ABI adds beyond "call into KvCachePipeline": libort
 * (libonnxruntime.so/.dylib/.dll) is loaded and wired up at RUNTIME via
 * onnx_deploy_load_ort(), not linked at build time. This library builds and
 * links with zero dependency on any specific ONNX Runtime binary -- only its
 * headers, for the Ort::* C++ wrapper types kv_cache_pipeline.h uses -- so a
 * single compiled onnx_deploy_c.so/.dylib/.dll works against whichever
 * libonnxruntime build the caller points it at (CPU/GPU/version/EP mix),
 * swapped by passing a different path, no recompile. See ../../README.md.
 */
#ifndef ONNX_DEPLOY_C_API_H_
#define ONNX_DEPLOY_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(ONNX_DEPLOY_C_API_BUILD)
#define ONNX_DEPLOY_C_API __declspec(dllexport)
#else
#define ONNX_DEPLOY_C_API __declspec(dllimport)
#endif
#else
#define ONNX_DEPLOY_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for every fallible entry point. */
typedef enum OnnxDeployStatus {
  ONNX_DEPLOY_OK = 0,
  ONNX_DEPLOY_ERROR = 1,
} OnnxDeployStatus;

/*
 * Loads libort_path (a libonnxruntime shared library -- an official prebuilt
 * release's lib/libonnxruntime.so, a pip-installed onnxruntime's bundled
 * .so, a GPU/EP-specific build, whatever the caller wants to run against)
 * via dlopen/LoadLibrary, resolves its OrtGetApiBase export, and wires the
 * result up as the ONNX Runtime implementation every other onnx_deploy_*
 * call in this process uses from then on.
 *
 * Must be called exactly once, before any onnx_deploy_create call. Not
 * thread-safe with itself or with other onnx_deploy_* calls (call it once,
 * up front, before spawning any pipeline). Calling it a second time with a
 * different path re-points the process at a different libort -- existing
 * OnnxDeployPipeline handles created against the previous one become
 * invalid; this is "swappable" at the process level (restart-to-swap), not
 * a promise that two different libort builds can be live in the same
 * process at once.
 *
 * Returns ONNX_DEPLOY_OK on success. On ONNX_DEPLOY_ERROR, *out_error
 * receives a newly allocated, NUL-terminated message (if out_error is
 * non-NULL); release it with onnx_deploy_free_string.
 */
ONNX_DEPLOY_C_API OnnxDeployStatus onnx_deploy_load_ort(const char* libort_path, char** out_error);

/* Opaque handle over a loaded optimum-onnx export directory's sessions. */
typedef struct OnnxDeployPipeline OnnxDeployPipeline;

/*
 * Loads model_dir (see ../../README.md for the expected optimum-onnx
 * no_post_process=True export shape). onnx_deploy_load_ort must have
 * succeeded first. Returns NULL on failure, with *out_error set the same
 * way as onnx_deploy_load_ort (out_error may be NULL).
 */
ONNX_DEPLOY_C_API OnnxDeployPipeline* onnx_deploy_create(const char* model_dir, char** out_error);

/* Releases a pipeline created by onnx_deploy_create. NULL is ignored. */
ONNX_DEPLOY_C_API void onnx_deploy_destroy(OnnxDeployPipeline* pipeline);

/* Nonzero if `pipeline` loaded an encoder_model.onnx (seq2seq export). */
ONNX_DEPLOY_C_API int onnx_deploy_is_seq2seq(const OnnxDeployPipeline* pipeline);

/*
 * Greedy-decodes up to max_new_tokens token ids (batch size 1), stopping
 * early if a generated id equals eos_token_id (pass -1 to disable early
 * stop). `input_ids`/`num_input_ids` is the encoder input for a seq2seq
 * pipeline, or the decoder prompt for a decoder-only one.
 * decoder_start_token_id is only used for seq2seq pipelines.
 *
 * On success, returns ONNX_DEPLOY_OK and sets *out_ids to a freshly
 * allocated array of *out_count int64_t token ids (the newly generated
 * ids, not including the prompt); release it with onnx_deploy_free_ids.
 * On ONNX_DEPLOY_ERROR, *out_error is set the same way as
 * onnx_deploy_load_ort (out_error may be NULL); *out_ids/*out_count are
 * left untouched.
 */
ONNX_DEPLOY_C_API OnnxDeployStatus onnx_deploy_generate(OnnxDeployPipeline* pipeline, const int64_t* input_ids,
                                                         size_t num_input_ids, int64_t max_new_tokens,
                                                         int64_t eos_token_id, int64_t decoder_start_token_id,
                                                         int64_t** out_ids, size_t* out_count, char** out_error);

/* Free an array returned via an out_ids parameter. NULL is ignored. */
ONNX_DEPLOY_C_API void onnx_deploy_free_ids(int64_t* ids);

/* Free a string returned via an out_error parameter. NULL is ignored. */
ONNX_DEPLOY_C_API void onnx_deploy_free_string(char* data);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_DEPLOY_C_API_H_ */
