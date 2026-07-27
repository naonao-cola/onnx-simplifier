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
 * On ONNXSIM_OK, *out_data / *out_size receive a newly allocated buffer holding
 * the serialized simplified ModelProto; release it with onnxsim_free_buffer.
 * On ONNXSIM_ERROR, *out_error receives a newly allocated, NUL-terminated
 * message; release it with onnxsim_free_string. Either out_* pointer may be
 * NULL if the caller does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version, void** out_data,
    size_t* out_size, char** out_error);

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
    size_t tensor_size_threshold, int target_opset_version, char** out_error);

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
