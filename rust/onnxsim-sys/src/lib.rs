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
    /// NUL-terminated strings when `skip_optimizers_is_null == 0`.
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
        out_data: *mut *mut c_void,
        out_size: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Simplify a model read from `in_path`, writing the result to `out_path`.
    ///
    /// # Safety
    /// `in_path`/`out_path` must be valid NUL-terminated paths; the
    /// `skip_optimizers` rules match [`onnxsim_simplify`].
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
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Return the available optimizer pass names as a newline-separated,
    /// NUL-terminated string (or null on allocation failure). Free with
    /// [`onnxsim_free_string`].
    ///
    /// # Safety
    /// Always safe to call; the returned pointer must be freed exactly once.
    pub fn onnxsim_list_optimizers() -> *mut c_char;

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
