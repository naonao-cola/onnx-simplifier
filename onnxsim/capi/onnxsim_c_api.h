/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * A minimal, stable C ABI over the onnxsim C++ core. It exists so that other
 * languages (e.g. Rust, Go, C) can drive the simplifier without dealing with
 * C++ name mangling, exceptions, or the onnx::ModelProto type across the FFI
 * boundary. Models are exchanged as serialized ONNX ModelProto bytes, exactly
 * like the Python binding does.
 */
#ifndef ONNXSIM_C_API_H_
#define ONNXSIM_C_API_H_

#include <stddef.h>

#if defined(_WIN32)
#if defined(ONNXSIM_C_API_BUILD)
#define ONNXSIM_C_API __declspec(dllexport)
#else
#define ONNXSIM_C_API __declspec(dllimport)
#endif
#else
#define ONNXSIM_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for every fallible entry point. */
typedef enum OnnxsimStatus {
  ONNXSIM_OK = 0,
  ONNXSIM_ERROR = 1,
} OnnxsimStatus;

/*
 * Optional custom graph-rewriter callback.
 *
 * When supplied to onnxsim_simplify / onnxsim_simplify_path it is run inside
 * onnxsim's simplification fixed point, interleaved with the built-in
 * optimizer, shape inference and constant folding, so a rewrite can unlock
 * further simplification and vice versa. It mirrors the C++ `GraphRewriter` and
 * the Python `custom_rewriter` parameter, exchanging models as serialized ONNX
 * ModelProto bytes across the C boundary.
 *
 * On each round it is called with the current model in `in_model_data`
 * (`in_model_size` bytes) and must return:
 *   - a value > 0 : the model was rewritten.
 * `*out_model_data`/`*out_model_size` must be set to a newly allocated buffer
 * holding the rewritten serialized ModelProto. onnxsim parses it and then, if a
 * free callback was supplied, calls it with exactly that pointer and size to
 * release the buffer.
 *   - 0           : nothing was rewritten this round. The out-parameters are
 *     ignored and onnxsim keeps the model it already has (skipping a copy).
 *   - a value < 0 : the callback failed; simplification aborts with
 *     ONNXSIM_ERROR.
 *
 * `user_data` is passed through untouched on every call (both the rewrite and
 * the free callback). Passing a NULL rewrite callback disables the feature and
 * reproduces the previous behaviour exactly.
 */
typedef int (*OnnxsimRewriteFn)(void* user_data, const void* in_model_data,
                                size_t in_model_size, void** out_model_data,
                                size_t* out_model_size);

/*
 * Releases a buffer produced by an OnnxsimRewriteFn. Called with the same
 * `user_data` and with the `out_model_data`/`out_model_size` the rewrite
 * callback returned, once onnxsim has finished parsing it. May be NULL, in
 * which case onnxsim never frees the rewriter's output (the callback owns it).
 */
typedef void (*OnnxsimRewriteFreeFn)(void* user_data, void* model_data,
                                     size_t model_size);

/*
 * Simplify a model given as a serialized ONNX ModelProto.
 *
 * skip_optimizers semantics mirror the C++/Python API:
 *   - skip_optimizers_is_null != 0  => skip ALL optimizer passes (no graph
 *     optimization is performed); `skip_optimizers`/`num_skip_optimizers` are
 *     ignored.
 *   - skip_optimizers_is_null == 0  => run every fuse/elimination pass EXCEPT
 *     the `num_skip_optimizers` passes named in `skip_optimizers` (pass 0 to
 *     run all of them).
 *
 * constant_folding / shape_inference are treated as booleans (0 = false).
 * tensor_size_threshold bounds the byte size of tensors produced by constant
 * folding that are kept as initializers.
 *
 * target_opset_version, when > 0, converts the model to that opset version of
 * the default ONNX domain (using onnx's version converter) before simplifying;
 * a value <= 0 leaves the opset version unchanged.
 *
 * rewrite_fn, when non-NULL, is a custom graph rewriter run inside the
 * simplification fixed point (see OnnxsimRewriteFn). rewrite_free_fn releases
 * the buffers it produces (may be NULL). rewrite_user_data is passed through to
 * both callbacks untouched. Pass a NULL rewrite_fn to disable the feature.
 *
 * On ONNXSIM_OK, *out_data / *out_size receive a newly allocated buffer holding
 * the serialized simplified ModelProto; release it with onnxsim_free_buffer.
 * On ONNXSIM_ERROR, *out_error receives a newly allocated, NUL-terminated
 * message; release it with onnxsim_free_string. Either out_* pointer may be
 * NULL if the caller does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus
onnxsim_simplify(const void* model_data, size_t model_size,
                 const char* const* skip_optimizers, size_t num_skip_optimizers,
                 int skip_optimizers_is_null, int constant_folding,
                 int shape_inference, size_t tensor_size_threshold,
                 int target_opset_version, OnnxsimRewriteFn rewrite_fn,
                 OnnxsimRewriteFreeFn rewrite_free_fn, void* rewrite_user_data,
                 void** out_data, size_t* out_size, char** out_error);

/*
 * Same as onnxsim_simplify, but also applies a set of data-driven rewrite rules
 * inside the simplification fixed point. Each rule is a (pattern, replacement)
 * pair of serialized ONNX FunctionProto: the pattern's inputs are wildcards
 * binding to graph values, its body is the subgraph to match, and its outputs
 * are rewired to the replacement's outputs. This is the same
 * FunctionProto-based rewriter the Python and Rust bindings expose, so a rule
 * set authored once (e.g. via onnx.parser.parse_function) works from any
 * binding without depending on onnxscript.
 *
 * `pattern_data[i]`/`pattern_sizes[i]` and `replacement_data[i]`/
 * `replacement_sizes[i]` describe rule `i`, for `i` in [0, num_rules). Passing
 * num_rules == 0 behaves exactly like onnxsim_simplify. All other parameters
 * match onnxsim_simplify.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify_with_rules(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const void* const* pattern_data, const size_t* pattern_sizes,
    const void* const* replacement_data, const size_t* replacement_sizes,
    size_t num_rules, void** out_data, size_t* out_size, char** out_error);

/*
 * Same as onnxsim_simplify, but reads the input model from `in_path` and writes
 * the simplified model to `out_path`. On failure, *out_error receives a message
 * (free with onnxsim_free_string).
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify_path(
    const char* in_path, const char* out_path,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, char** out_error);

/*
 * Return the names of all available fuse/elimination optimizer passes as a
 * single NUL-terminated string with one pass name per line ('\n' separated).
 * Returns NULL on allocation failure. Release with onnxsim_free_string.
 */
ONNXSIM_C_API char* onnxsim_list_optimizers(void);

/* Free a buffer returned via an out_data parameter. NULL is ignored. */
ONNXSIM_C_API void onnxsim_free_buffer(void* data);

/* Free a string returned via an out_error parameter or onnxsim_list_optimizers.
 * NULL is ignored. */
ONNXSIM_C_API void onnxsim_free_string(char* data);

#ifdef __cplusplus
}
#endif

#endif /* ONNXSIM_C_API_H_ */
