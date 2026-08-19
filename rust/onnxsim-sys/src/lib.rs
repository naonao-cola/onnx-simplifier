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

// --- DLPack ABI ------------------------------------------------------------
//
// These structs mirror `third_party/dlpack/dlpack.h` and are the tensor
// exchange type at the constant-folding executor boundary (see
// `onnxsim_simplify_with_executor`). Only the plain (unversioned)
// `DLManagedTensor` is used across the C API. Layout must match the C header
// exactly; the `#[repr(C)]` attributes and field order are load-bearing.

/// DLPack device type. onnxsim's executor boundary only ever produces or
/// consumes [`DLDeviceType::kDLCPU`] tensors.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    /// CPU device.
    kDLCPU = 1,
    /// CUDA GPU device.
    kDLCUDA = 2,
    /// Pinned CUDA CPU memory.
    kDLCUDAHost = 3,
    /// OpenCL device.
    kDLOpenCL = 4,
    /// Vulkan buffer.
    kDLVulkan = 7,
    /// Metal (Apple GPU).
    kDLMetal = 8,
    /// Verilog simulator buffer.
    kDLVPI = 9,
    /// AMD ROCm GPU.
    kDLROCM = 10,
    /// Pinned ROCm CPU memory.
    kDLROCMHost = 11,
    /// Reserved extension device.
    kDLExtDev = 12,
    /// CUDA managed/unified memory.
    kDLCUDAManaged = 13,
    /// oneAPI unified shared memory.
    kDLOneAPI = 14,
    /// WebGPU buffer.
    kDLWebGPU = 15,
    /// Qualcomm Hexagon DSP.
    kDLHexagon = 16,
    /// Microsoft MAIA device.
    kDLMAIA = 17,
}

/// DLPack type-code family (`DLDataType::code`). onnxsim uses `kDLInt`,
/// `kDLUInt`, `kDLFloat`, `kDLBfloat`, and `kDLBool`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// Signed integer.
    kDLInt = 0,
    /// Unsigned integer.
    kDLUInt = 1,
    /// IEEE floating point.
    kDLFloat = 2,
    /// Opaque handle.
    kDLOpaqueHandle = 3,
    /// bfloat16.
    kDLBfloat = 4,
    /// Complex number.
    kDLComplex = 5,
    /// Boolean.
    kDLBool = 6,
}

/// A DLPack device (`DLDevice`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    /// Device family.
    pub device_type: DLDeviceType,
    /// Device index (0 for CPU).
    pub device_id: c_int,
}

/// A DLPack element type (`DLDataType`): `code` is a [`DLDataTypeCode`] value,
/// `bits` the element width, `lanes` the vector width (always 1 here).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type-code family; one of [`DLDataTypeCode`].
    pub code: u8,
    /// Number of bits in one element (e.g. 32 for float32).
    pub bits: u8,
    /// Number of lanes (1 for scalars).
    pub lanes: u16,
}

/// A non-owning DLPack tensor (`DLTensor`).
///
/// `shape` and `strides` point to `ndim` `i64` values; a null `strides` means
/// the tensor is row-major contiguous. Data begins at `data + byte_offset`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLTensor {
    /// Pointer to the element buffer.
    pub data: *mut c_void,
    /// Device the buffer lives on.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: c_int,
    /// Element data type.
    pub dtype: DLDataType,
    /// Pointer to `ndim` shape values.
    pub shape: *mut i64,
    /// Pointer to `ndim` stride values (elements, not bytes), or null if
    /// contiguous.
    pub strides: *mut i64,
    /// Offset in bytes from `data` to the first element.
    pub byte_offset: u64,
}

/// An owning, memory-managed DLPack tensor (`DLManagedTensor`).
///
/// The producer sets `deleter`; the consumer calls it exactly once when done,
/// which releases both `manager_ctx` and the `DLManagedTensor` itself.
#[repr(C)]
pub struct DLManagedTensor {
    /// The tensor being managed.
    pub dl_tensor: DLTensor,
    /// Producer-defined context reclaimed by `deleter`.
    pub manager_ctx: *mut c_void,
    /// Destructor; releases `manager_ctx` and `self`. May be null.
    pub deleter: Option<unsafe extern "C" fn(this: *mut DLManagedTensor)>,
}

/// Custom constant-folding executor callback (mirrors `OnnxsimExecuteFn`).
///
/// Replaces the built-in ONNX Runtime constant folder: for each fold group,
/// onnxsim hands over the sub-model (`model_data`/`model_size`, a serialized
/// `ModelProto`) and its input tensors (`inputs`/`num_inputs`, borrowed for the
/// call only), and the callback returns one owned [`DLManagedTensor`] per graph
/// output through `*out_outputs`/`*out_num_outputs`. Return `0` on success,
/// non-zero to abort. onnxsim releases each returned tensor via its own DLPack
/// `deleter`, then calls the paired [`OnnxsimExecuteFreeFn`] on the array.
pub type OnnxsimExecuteFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    model_data: *const c_void,
    model_size: usize,
    inputs: *const *const DLManagedTensor,
    num_inputs: usize,
    out_outputs: *mut *mut *mut DLManagedTensor,
    out_num_outputs: *mut usize,
) -> c_int;

/// Frees the output array produced by an [`OnnxsimExecuteFn`] (mirrors
/// `OnnxsimExecuteFreeFn`) — the array container only; each tensor is released
/// through its own deleter. Called with the same `user_data` and the exact
/// `outputs`/`num_outputs` the execute callback returned.
pub type OnnxsimExecuteFreeFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    outputs: *mut *mut DLManagedTensor,
    num_outputs: usize,
);

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

    /// Simplify a serialized ONNX `ModelProto`, driving constant folding through
    /// a host-provided executor callback instead of the built-in ONNX Runtime.
    ///
    /// This is the seam for embedding onnxsim in another ONNX-based stack. A
    /// `None` `execute_fn` falls back to the built-in executor, making this a
    /// drop-in superset of [`onnxsim_simplify`]. See `onnxsim_c_api.h` for the
    /// full contract; the rewrite parameters match [`onnxsim_simplify`].
    ///
    /// # Safety
    /// All non-null pointers must be valid; `model_data` must point to at least
    /// `model_size` bytes. When `execute_fn` is `Some`, it (and the matching
    /// `execute_free_fn`) must obey the [`OnnxsimExecuteFn`] contract: inputs are
    /// borrowed for the call, each returned [`DLManagedTensor`] is owned by
    /// onnxsim (released via its deleter), and all tensors are CPU, contiguous,
    /// and of a supported dtype. Otherwise as [`onnxsim_simplify`].
    pub fn onnxsim_simplify_with_executor(
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
        execute_fn: Option<OnnxsimExecuteFn>,
        execute_free_fn: Option<OnnxsimExecuteFreeFn>,
        execute_user_data: *mut c_void,
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

    /// Render a node- and value-level diff (matched by output tensor name)
    /// between an original and a simplified model, both serialized
    /// `ModelProto` bytes: which nodes/values were removed, added, or changed.
    /// Complementary to [`onnxsim_model_info_diff`], which reports
    /// op-count/size/MACs aggregates rather than the specific nodes and values
    /// involved. On `ONNXSIM_OK`, `out_text` owns a NUL-terminated string to
    /// free with [`onnxsim_free_string`]; on `ONNXSIM_ERROR`, `out_error` owns
    /// a message.
    ///
    /// # Safety
    /// `original_data`/`simplified_data` must each point to at least the matching
    /// size in bytes; `out_text`/`out_error`, if non-null, receive owned strings
    /// that must be freed exactly once.
    pub fn onnxsim_graph_diff(
        original_data: *const c_void,
        original_size: usize,
        simplified_data: *const c_void,
        simplified_size: usize,
        out_text: *mut *mut c_char,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Export a serialized ONNX `ModelProto` to a standalone safetensors
    /// archive at `out_path`: every initializer's bytes move into the archive
    /// with real, byte-accurate offsets (openable by the `safetensors` Python
    /// package / HF tooling with no onnxsim involved), and the graph itself is
    /// embedded alongside them, so `out_path` alone is both the model's
    /// weights and its graph. Reload it with [`onnxsim_import_safetensors`].
    /// On `ONNXSIM_ERROR`, `out_error` owns a message to free with
    /// [`onnxsim_free_string`].
    ///
    /// # Safety
    /// `model_data` must point to at least `model_size` bytes; `out_path` must
    /// be a valid NUL-terminated path; `out_error`, if non-null, receives an
    /// owned string that must be freed exactly once.
    pub fn onnxsim_export_safetensors(
        model_data: *const c_void,
        model_size: usize,
        out_path: *const c_char,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// Import a standalone safetensors archive at `in_path` (as produced by
    /// [`onnxsim_export_safetensors`], or any other tool following the same
    /// self-describing-archive convention) back into an ordinary, fully
    /// in-memory ONNX `ModelProto`. On `ONNXSIM_OK`, `out_data`/`out_size` own
    /// a buffer to free with [`onnxsim_free_buffer`]. Fails with
    /// `ONNXSIM_ERROR` if the archive has no embedded onnxsim model (e.g. a
    /// plain weights-only safetensors file with no graph to import).
    ///
    /// # Safety
    /// `in_path` must be a valid NUL-terminated path; `out_data`/`out_size`/
    /// `out_error`, if non-null, receive owned values that must be freed
    /// exactly once.
    pub fn onnxsim_import_safetensors(
        in_path: *const c_char,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// GGUF counterpart of [`onnxsim_export_safetensors`]. Same contract.
    ///
    /// # Safety
    /// Same as [`onnxsim_export_safetensors`].
    pub fn onnxsim_export_gguf(
        model_data: *const c_void,
        model_size: usize,
        out_path: *const c_char,
        out_error: *mut *mut c_char,
    ) -> c_int;

    /// GGUF counterpart of [`onnxsim_import_safetensors`]. Same contract,
    /// including failing with `ONNXSIM_ERROR` when the archive has no
    /// embedded model.
    ///
    /// # Safety
    /// Same as [`onnxsim_import_safetensors`].
    pub fn onnxsim_import_gguf(
        in_path: *const c_char,
        out_data: *mut *mut c_void,
        out_size: *mut usize,
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
