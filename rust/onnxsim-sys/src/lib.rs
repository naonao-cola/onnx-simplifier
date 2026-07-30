//! Low-level FFI bindings to the ONNX Simplifier (`onnxsim`) C API.
//!
//! These declarations mirror `onnxsim/capi/onnxsim_c_api.h` one-to-one. They are
//! `unsafe` and do no memory management on their own — every buffer or string
//! handed back through an out-parameter must be released with the matching
//! `onnxsim_free_*` function. Prefer the safe [`onnxsim`] crate unless you need
//! raw access.
//!
//! [`onnxsim`]: https://crates.io/crates/onnxsim
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// Return code shared by every fallible entry point.
///
/// Mirrors the `OnnxsimStatus` enum in the C header.
pub const ONNXSIM_OK: c_int = 0;
/// Failure return code; an error message is written to the `out_error` slot.
pub const ONNXSIM_ERROR: c_int = 1;

/// Custom graph-rewriter callback (mirrors `OnnxsimRewriteFn` in the C header).
///
/// Invoked once per fixed-point round with the current model as serialized
/// ModelProto bytes (`in_model_data`/`in_model_size`). Returns `> 0` after
/// setting `*out_model_data`/`*out_model_size` to a newly allocated buffer with
/// the rewritten model, `0` to report that nothing changed, or `< 0` on error.
pub type OnnxsimRewriteFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    in_model_data: *const c_void,
    in_model_size: usize,
    out_model_data: *mut *mut c_void,
    out_model_size: *mut usize,
) -> c_int;

/// Frees a buffer produced by an [`OnnxsimRewriteFn`] (mirrors
/// `OnnxsimRewriteFreeFn`). Called with the same `user_data` and the pointer and
/// size the rewrite callback returned, once onnxsim has parsed it.
pub type OnnxsimRewriteFreeFn =
    unsafe extern "C" fn(user_data: *mut c_void, model_data: *mut c_void, model_size: usize);

extern "C" {
    /// Simplify a serialized ONNX `ModelProto`.
    ///
    /// See `onnxsim_c_api.h` for the full contract. In brief:
    /// `skip_optimizers_is_null != 0` skips every optimizer pass; otherwise all
    /// passes run except the `num_skip_optimizers` names in `skip_optimizers`.
    /// On success (`ONNXSIM_OK`) `out_data`/`out_size` own a buffer to free with
    /// [`onnxsim_free_buffer`]; on failure (`ONNXSIM_ERROR`) `out_error` owns a
    /// string to free with [`onnxsim_free_string`].
    ///
    /// # Safety
    /// All non-null pointers must be valid; `model_data` must point to at least
    /// `model_size` bytes and `skip_optimizers` to `num_skip_optimizers`
    /// NUL-terminated strings when `skip_optimizers_is_null == 0`. When
    /// `rewrite_fn` is `Some`, it (and the matching `rewrite_free_fn`) must obey
    /// the callback contract in `onnxsim_c_api.h`.
    pub fn onnxsim_simplify(
        model_data: *const c_void,
        model_size: usize,
        skip_optimizers: *const *const c_char,
        num_skip_optimizers: usize,
        skip_optimizers_is_null: c_int,
        constant_folding: c_int,
        shape_inference: c_int,
        tensor_size_threshold: usize,
        target_opset_version: c_int,
        rewrite_fn: Option<OnnxsimRewriteFn>,
        rewrite_free_fn: Option<OnnxsimRewriteFreeFn>,
        rewrite_user_data: *mut c_void,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Simplify a serialized ONNX `ModelProto`, additionally applying a set of
    /// FunctionProto-based rewrite rules inside the simplification fixed point.
    ///
    /// See `onnxsim_c_api.h` for the full contract. Rule `i` is the
    /// (pattern, replacement) pair of serialized `FunctionProto` bytes at
    /// `pattern_data[i]`/`pattern_sizes[i]` and
    /// `replacement_data[i]`/`replacement_sizes[i]`. `num_rules == 0` behaves
    /// exactly like [`onnxsim_simplify`].
    ///
    /// # Safety
    /// All non-null pointers must be valid; the rule arrays must each hold
    /// `num_rules` entries, and every `*_data[i]` must point to at least the
    /// matching `*_sizes[i]` bytes. Otherwise as [`onnxsim_simplify`].
    pub fn onnxsim_simplify_with_rules(
        model_data: *const c_void,
        model_size: usize,
        skip_optimizers: *const *const c_char,
        num_skip_optimizers: usize,
        skip_optimizers_is_null: c_int,
        constant_folding: c_int,
        shape_inference: c_int,
        tensor_size_threshold: usize,
        target_opset_version: c_int,
        pattern_data: *const *const c_void,
        pattern_sizes: *const usize,
        replacement_data: *const *const c_void,
        replacement_sizes: *const usize,
        num_rules: usize,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Simplify a model read from `in_path`, writing the result to `out_path`.
    ///
    /// # Safety
    /// `in_path`/`out_path` must be valid NUL-terminated paths; the
    /// `skip_optimizers` and `rewrite_fn`/`rewrite_free_fn` rules match
    /// [`onnxsim_simplify`].
    pub fn onnxsim_simplify_path(
        in_path: *const c_char,
        out_path: *const c_char,
        skip_optimizers: *const *const c_char,
        num_skip_optimizers: usize,
        skip_optimizers_is_null: c_int,
        constant_folding: c_int,
        shape_inference: c_int,
        tensor_size_threshold: usize,
        target_opset_version: c_int,
        rewrite_fn: Option<OnnxsimRewriteFn>,
        rewrite_free_fn: Option<OnnxsimRewriteFreeFn>,
        rewrite_user_data: *mut c_void,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Return the available optimizer pass names as a newline-separated,
    /// NUL-terminated string (or null on allocation failure). Free with
    /// [`onnxsim_free_string`].
    ///
    /// # Safety
    /// Always safe to call; the returned pointer must be freed exactly once.
    pub fn onnxsim_list_optimizers() -> *mut c_char;

    /// Render a human-readable diff (op counts + model size) between an original
    /// and a simplified model, both serialized `ModelProto` bytes. On
    /// `ONNXSIM_OK`, `out_text` owns a NUL-terminated string to free with
    /// [`onnxsim_free_string`]; on `ONNXSIM_ERROR`, `out_error` owns a message.
    ///
    /// # Safety
    /// `original_data`/`simplified_data` must each point to at least the matching
    /// size in bytes; `out_text`/`out_error`, if non-null, receive owned strings
    /// that must be freed exactly once.
    pub fn onnxsim_model_info_diff(
        original_data: *const c_void,
        original_size: usize,
        simplified_data: *const c_void,
        simplified_size: usize,
        out_text: *mut *mut c_char,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Free a buffer produced by an `out_data` parameter. Null is ignored.
    ///
    /// # Safety
    /// `data` must have come from this library and not be freed twice.
    pub fn onnxsim_free_buffer(data: *mut c_void);

    /// Free a string produced by an `out_error` parameter or
    /// [`onnxsim_list_optimizers`]. Null is ignored.
    ///
    /// # Safety
    /// `data` must have come from this library and not be freed twice.
    pub fn onnxsim_free_string(data: *mut c_char);
}
