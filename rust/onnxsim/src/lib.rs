//! Safe Rust bindings to the [ONNX Simplifier](https://github.com/onnxsim/onnxsim)
//! (`onnxsim`).
//!
//! ONNX models exported from training frameworks often contain redundant
//! operators — dynamic shape gymnastics, no-op reshapes, constant subgraphs.
//! `onnxsim` runs shape inference and constant folding to replace those with
//! their computed results, producing a smaller, simpler, semantically identical
//! graph. This crate wraps the same C++ core the Python package and CLI use, so
//! Rust projects (e.g. model importers) can simplify models in-process instead
//! of shelling out.
//!
//! # Quick start
//!
//! Simplify a model already in memory as serialized `ModelProto` bytes:
//!
//! ```no_run
//! fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
//!     let model_bytes: Vec<u8> = std::fs::read("model.onnx")?;
//!     let simplified: Vec<u8> = onnxsim::simplify(&model_bytes)?;
//!     std::fs::write("model.opt.onnx", &simplified)?;
//!     Ok(())
//! }
//! ```
//!
//! Or operate directly on files (lets the C++ side stream models larger than
//! would be convenient to hold in a single buffer):
//!
//! ```no_run
//! fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
//!     onnxsim::simplify_path("model.onnx", "model.opt.onnx")?;
//!     Ok(())
//! }
//! ```
//!
//! # Tuning
//!
//! [`Options`] controls constant folding, shape inference, which optimizer
//! passes run, and the size threshold for folded tensors:
//!
//! ```no_run
//! fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
//!     let opts = onnxsim::Options::new()
//!         .shape_inference(false) // some models crash ONNX shape inference
//!         .skip_optimizer("eliminate_nop_transpose");
//!     let simplified = onnxsim::simplify_with(&std::fs::read("model.onnx")?, &opts)?;
//!     Ok(())
//! }
//! ```
//!
//! # Custom rewriter
//!
//! [`simplify_with_rewriter`] (and [`simplify_path_with_rewriter`]) run a
//! closure of yours inside the simplification fixed point — the Rust equivalent
//! of the Python `custom_rewriter` parameter. The closure receives the current
//! model as serialized `ModelProto` bytes each round and returns `Ok(None)`
//! (nothing changed), `Ok(Some(bytes))` (rewritten model), or `Err(..)` to
//! abort. Because it is interleaved with the built-in optimizer, shape inference
//! and constant folding, a rewrite can unlock further simplification and vice
//! versa.
//!
//! ```no_run
//! fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
//!     let model = std::fs::read("model.onnx")?;
//!     let simplified = onnxsim::simplify_with_rewriter(
//!         &model,
//!         &onnxsim::Options::new(),
//!         |bytes: &[u8]| {
//!             // Decode `bytes`, rewrite the graph with your library of choice,
//!             // and return the new bytes — or `Ok(None)` to change nothing.
//!             let _ = bytes;
//!             Ok::<_, onnxsim::Error>(None)
//!         },
//!     )?;
//!     let _ = simplified;
//!     Ok(())
//! }
//! ```
//!
//! # Custom executor
//!
//! [`simplify_with_executor`] drives constant folding through a closure of yours
//! instead of the built-in ONNX Runtime — the seam for embedding onnxsim on top
//! of another ONNX runtime. Each fold group calls the closure with the sub-model
//! as serialized `ModelProto` bytes and its input tensors as borrowed
//! [`TensorRef`]s; the closure evaluates it and returns one owned [`Tensor`] per
//! graph output. Tensors cross as DLPack under the hood, so nothing is
//! serialized to `TensorProto` on the way in or out.
//!
//! ```no_run
//! fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
//!     let model = std::fs::read("model.onnx")?;
//!     let simplified = onnxsim::simplify_with_executor(
//!         &model,
//!         &onnxsim::Options::new(),
//!         |submodel: &[u8], inputs: &[onnxsim::TensorRef]| {
//!             // Evaluate `submodel` with `inputs` in your runtime of choice and
//!             // return the outputs as `onnxsim::Tensor`s (in graph-output order).
//!             let _ = (submodel, inputs);
//!             Ok::<_, onnxsim::Error>(Vec::<onnxsim::Tensor>::new())
//!         },
//!     )?;
//!     let _ = simplified;
//!     Ok(())
//! }
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

/// Default upper bound on the byte size of a constant-folded tensor kept as an
/// initializer. Matches the Python package default of `1.5GB`.
pub const DEFAULT_TENSOR_SIZE_THRESHOLD: usize = 1_610_612_736; // 1.5 * 2^30

/// Errors returned by the simplifier.
#[derive(Debug)]
pub enum Error {
    /// The C++ core reported a failure. Carries its message.
    Simplify(String),
    /// An argument could not be converted for the FFI boundary (e.g. a path or
    /// optimizer name containing an interior NUL byte, or a non-UTF-8 path).
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Simplify(msg) => write!(f, "onnxsim failed: {msg}"),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Configuration for a simplification run.
///
/// Build with [`Options::new`] and the chained setters; every field has a sane
/// default so `Options::default()` reproduces the CLI's out-of-the-box
/// behaviour (all passes on, constant folding and shape inference enabled).
#[derive(Debug, Clone)]
pub struct Options {
    /// Optimizer-pass control.
    /// * `None` — skip **all** optimizer passes (no graph optimization).
    /// * `Some(list)` — run every fuse/elimination pass except the names in
    ///   `list` (an empty list runs them all).
    skip_optimizers: Option<Vec<String>>,
    constant_folding: bool,
    shape_inference: bool,
    tensor_size_threshold: usize,
    /// Target opset version (of the default ONNX domain) to convert the model
    /// to before simplifying. `None` leaves the opset version unchanged.
    target_opset_version: Option<i32>,
    /// FunctionProto-based rewrite rules applied inside the fixed point. Each
    /// entry is a `(pattern, replacement)` pair of serialized `FunctionProto`
    /// bytes. Empty (the default) means no rule-based rewriting.
    function_rewrite_rules: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            skip_optimizers: Some(Vec::new()),
            constant_folding: true,
            shape_inference: true,
            tensor_size_threshold: DEFAULT_TENSOR_SIZE_THRESHOLD,
            target_opset_version: None,
            function_rewrite_rules: Vec::new(),
        }
    }
}

impl Options {
    /// Create options with the default configuration.
    pub fn new() -> Self {
        Options::default()
    }

    /// Enable or disable constant folding (default: enabled).
    pub fn constant_folding(mut self, enabled: bool) -> Self {
        self.constant_folding = enabled;
        self
    }

    /// Enable or disable ONNX shape inference (default: enabled).
    ///
    /// Disabling can be a useful workaround: shape inference occasionally
    /// crashes or errors on unusual models.
    pub fn shape_inference(mut self, enabled: bool) -> Self {
        self.shape_inference = enabled;
        self
    }

    /// Set the maximum byte size of a constant-folded tensor that will be kept
    /// as an initializer (default: [`DEFAULT_TENSOR_SIZE_THRESHOLD`]).
    pub fn tensor_size_threshold(mut self, bytes: usize) -> Self {
        self.tensor_size_threshold = bytes;
        self
    }

    /// Convert the model to `version` (the opset version of the default ONNX
    /// domain) before simplifying, using onnx's version converter. Can be used
    /// to upgrade or downgrade the model's opset during simplification (default:
    /// leave the opset version unchanged).
    pub fn target_opset_version(mut self, version: i32) -> Self {
        self.target_opset_version = Some(version);
        self
    }

    /// Disable **all** optimizer passes. Constant folding and shape inference
    /// (if enabled) still run.
    pub fn without_optimizers(mut self) -> Self {
        self.skip_optimizers = None;
        self
    }

    /// Run every optimizer pass except the ones named here.
    ///
    /// Replaces any previously configured skip list and re-enables optimization
    /// if it had been turned off with [`Options::without_optimizers`]. Use
    /// [`list_optimizers`] to discover valid names.
    pub fn skip_optimizers<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.skip_optimizers = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Add a single optimizer pass to the skip list (enabling optimization if it
    /// was disabled).
    pub fn skip_optimizer<S: Into<String>>(mut self, name: S) -> Self {
        self.skip_optimizers
            .get_or_insert_with(Vec::new)
            .push(name.into());
        self
    }

    /// Add a FunctionProto-based rewrite rule, matched and applied by onnxsim's
    /// C++ core inside the simplification fixed point.
    ///
    /// `pattern` and `replacement` are serialized `onnx.FunctionProto` bytes:
    /// the pattern's inputs are wildcards binding to graph values, its body is
    /// the subgraph to match, and its outputs are rewired to the replacement's
    /// outputs. The same rules work from the Python and C bindings; this is the
    /// language-agnostic counterpart to onnxscript's Python rewriter. Only
    /// [`simplify`]/[`simplify_with`] apply the rules; the path-based variants
    /// reject a non-empty rule set.
    pub fn function_rewrite_rule(
        mut self,
        pattern: impl Into<Vec<u8>>,
        replacement: impl Into<Vec<u8>>,
    ) -> Self {
        self.function_rewrite_rules
            .push((pattern.into(), replacement.into()));
        self
    }
}

/// Simplify a serialized ONNX `ModelProto` with default options.
///
/// Returns the serialized simplified model.
pub fn simplify(model_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    simplify_with(model_bytes, &Options::default())
}

/// Simplify a serialized ONNX `ModelProto` with the given [`Options`].
///
/// Returns the serialized simplified model.
pub fn simplify_with(model_bytes: &[u8], options: &Options) -> Result<Vec<u8>, Error> {
    let ffi = FfiSkipOptimizers::new(options)?;

    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_size: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = if options.function_rewrite_rules.is_empty() {
        unsafe {
            onnxsim_sys::onnxsim_simplify(
                model_bytes.as_ptr() as *const c_void,
                model_bytes.len(),
                ffi.ptr(),
                ffi.len(),
                ffi.is_null_flag(),
                bool_to_c(options.constant_folding),
                bool_to_c(options.shape_inference),
                options.tensor_size_threshold,
                target_opset_to_c(options.target_opset_version),
                None,
                None,
                ptr::null_mut(),
                &mut out_data,
                &mut out_size,
                &mut out_error,
            )
        }
    } else {
        // Borrow the rule byte buffers into the pointer/size arrays the C API
        // expects; these locals stay alive across the call below.
        let rules = &options.function_rewrite_rules;
        let pattern_ptrs: Vec<*const c_void> = rules
            .iter()
            .map(|(p, _)| p.as_ptr() as *const c_void)
            .collect();
        let pattern_sizes: Vec<usize> = rules.iter().map(|(p, _)| p.len()).collect();
        let replacement_ptrs: Vec<*const c_void> = rules
            .iter()
            .map(|(_, r)| r.as_ptr() as *const c_void)
            .collect();
        let replacement_sizes: Vec<usize> = rules.iter().map(|(_, r)| r.len()).collect();
        unsafe {
            onnxsim_sys::onnxsim_simplify_with_rules(
                model_bytes.as_ptr() as *const c_void,
                model_bytes.len(),
                ffi.ptr(),
                ffi.len(),
                ffi.is_null_flag(),
                bool_to_c(options.constant_folding),
                bool_to_c(options.shape_inference),
                options.tensor_size_threshold,
                target_opset_to_c(options.target_opset_version),
                pattern_ptrs.as_ptr(),
                pattern_sizes.as_ptr(),
                replacement_ptrs.as_ptr(),
                replacement_sizes.as_ptr(),
                rules.len(),
                &mut out_data,
                &mut out_size,
                &mut out_error,
            )
        }
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        // Copy the C-owned buffer into a Vec, then release the original.
        let result =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        unsafe { onnxsim_sys::onnxsim_free_buffer(out_data) };
        Ok(result)
    } else {
        Err(take_error(out_error))
    }
}

/// Simplify the model at `input_path`, writing the result to `output_path`,
/// using default options.
pub fn simplify_path<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
) -> Result<(), Error> {
    simplify_path_with(input_path, output_path, &Options::default())
}

/// Simplify the model at `input_path`, writing the result to `output_path`,
/// using the given [`Options`].
pub fn simplify_path_with<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    options: &Options,
) -> Result<(), Error> {
    if !options.function_rewrite_rules.is_empty() {
        return Err(Error::InvalidArgument(
            "function_rewrite_rules are only supported by simplify/simplify_with, \
             not the path-based variants"
                .to_string(),
        ));
    }
    let in_c = path_to_cstring(input_path.as_ref())?;
    let out_c = path_to_cstring(output_path.as_ref())?;
    let ffi = FfiSkipOptimizers::new(options)?;

    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_simplify_path(
            in_c.as_ptr(),
            out_c.as_ptr(),
            ffi.ptr(),
            ffi.len(),
            ffi.is_null_flag(),
            bool_to_c(options.constant_folding),
            bool_to_c(options.shape_inference),
            options.tensor_size_threshold,
            target_opset_to_c(options.target_opset_version),
            None,
            None,
            ptr::null_mut(),
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        Ok(())
    } else {
        Err(take_error(out_error))
    }
}

/// Error a custom rewriter closure may return. Any type convertible into a
/// boxed [`std::error::Error`] works, including this crate's [`Error`].
pub type RewriterError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Type-erased rewriter closure: `Ok(None)` => nothing rewritten this round,
/// `Ok(Some(bytes))` => rewritten model, `Err` => abort simplification.
type RewriteCallback<'a> = dyn FnMut(&[u8]) -> Result<Option<Vec<u8>>, RewriterError> + 'a;

/// Holds the (type-erased) user closure and captures failures that cannot cross
/// the FFI boundary as a return value. A single, non-generic pair of trampolines
/// therefore serves every closure.
struct RewriterState<'a> {
    callback: &'a mut RewriteCallback<'a>,
    error: Option<RewriterError>,
    panicked: bool,
}

/// `extern "C"` shim matching [`onnxsim_sys::OnnxsimRewriteFn`]. Recovers the
/// `RewriterState` from `user_data`, runs the closure under `catch_unwind` (an
/// unwind across the C frames would be undefined behaviour), and translates its
/// outcome into the C return convention.
unsafe extern "C" fn rewrite_trampoline(
    user_data: *mut c_void,
    in_model_data: *const c_void,
    in_model_size: usize,
    out_model_data: *mut *mut c_void,
    out_model_size: *mut usize,
) -> c_int {
    let state = &mut *(user_data as *mut RewriterState);
    let input: &[u8] = if in_model_size == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(in_model_data as *const u8, in_model_size)
    };
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (state.callback)(input)));
    match outcome {
        Ok(Ok(None)) => 0,
        Ok(Ok(Some(bytes))) => {
            // Hand ownership of the buffer to the C core; it is returned to
            // `rewrite_free_trampoline` (with this exact length) after parsing.
            let boxed = bytes.into_boxed_slice();
            *out_model_size = boxed.len();
            *out_model_data = Box::into_raw(boxed) as *mut c_void;
            1
        }
        Ok(Err(e)) => {
            state.error = Some(e);
            -1
        }
        Err(_) => {
            state.panicked = true;
            -1
        }
    }
}

/// `extern "C"` shim matching [`onnxsim_sys::OnnxsimRewriteFreeFn`]. Reclaims a
/// buffer handed out by [`rewrite_trampoline`] using the length the C core
/// passes back.
unsafe extern "C" fn rewrite_free_trampoline(
    _user_data: *mut c_void,
    model_data: *mut c_void,
    model_size: usize,
) {
    if !model_data.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            model_data as *mut u8,
            model_size,
        )));
    }
}

// Map an FFI error, preferring a failure captured from the rewriter closure (it
// is more specific than the generic C-side message). `out_error`, if set, is
// always consumed so it cannot leak.
fn rewriter_error(state: &mut RewriterState, out_error: *mut c_char) -> Error {
    let c_message = if out_error.is_null() {
        None
    } else {
        match take_error(out_error) {
            Error::Simplify(msg) => Some(msg),
            other => Some(other.to_string()),
        }
    };
    if state.panicked {
        return Error::Simplify("the custom rewriter panicked".to_string());
    }
    if let Some(e) = state.error.take() {
        return Error::Simplify(format!("custom rewriter failed: {e}"));
    }
    Error::Simplify(c_message.unwrap_or_else(|| "unknown error (no message provided)".to_string()))
}

/// Simplify a serialized ONNX `ModelProto`, running a custom graph rewriter
/// inside the simplification fixed point.
///
/// This is the Rust counterpart of the Python `custom_rewriter` parameter and
/// the C API's rewrite callback. `rewriter` is called on each fixed-point round
/// with the current model as serialized `ModelProto` bytes and must return:
///
/// * `Ok(None)` — nothing was rewritten this round (onnxsim keeps the model it
///   already has and skips a copy);
/// * `Ok(Some(bytes))` — the rewritten model as serialized `ModelProto` bytes;
/// * `Err(e)` — abort simplification with this error.
///
/// Because the rewriter runs interleaved with the built-in optimizer, shape
/// inference and constant folding, a rewrite can unlock further simplification
/// and vice versa; the pipeline iterates until it converges.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = std::fs::read("model.onnx")?;
/// // A no-op rewriter that reports it changed nothing every round.
/// let simplified = onnxsim::simplify_with_rewriter(
///     &model,
///     &onnxsim::Options::new(),
///     |_bytes: &[u8]| Ok::<_, onnxsim::Error>(None),
/// )?;
/// # let _ = simplified;
/// # Ok(())
/// # }
/// ```
pub fn simplify_with_rewriter<F, E>(
    model_bytes: &[u8],
    options: &Options,
    mut rewriter: F,
) -> Result<Vec<u8>, Error>
where
    F: FnMut(&[u8]) -> Result<Option<Vec<u8>>, E>,
    E: Into<RewriterError>,
{
    let mut wrapped = move |bytes: &[u8]| rewriter(bytes).map_err(Into::into);
    let mut state = RewriterState {
        callback: &mut wrapped,
        error: None,
        panicked: false,
    };
    let ffi = FfiSkipOptimizers::new(options)?;

    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_size: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        onnxsim_sys::onnxsim_simplify(
            model_bytes.as_ptr() as *const c_void,
            model_bytes.len(),
            ffi.ptr(),
            ffi.len(),
            ffi.is_null_flag(),
            bool_to_c(options.constant_folding),
            bool_to_c(options.shape_inference),
            options.tensor_size_threshold,
            target_opset_to_c(options.target_opset_version),
            Some(rewrite_trampoline),
            Some(rewrite_free_trampoline),
            &mut state as *mut RewriterState as *mut c_void,
            &mut out_data,
            &mut out_size,
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        let result =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        unsafe { onnxsim_sys::onnxsim_free_buffer(out_data) };
        Ok(result)
    } else {
        Err(rewriter_error(&mut state, out_error))
    }
}

/// Simplify the model at `input_path`, writing the result to `output_path`,
/// running a custom graph rewriter inside the simplification fixed point.
///
/// See [`simplify_with_rewriter`] for the `rewriter` contract.
pub fn simplify_path_with_rewriter<P, Q, F, E>(
    input_path: P,
    output_path: Q,
    options: &Options,
    mut rewriter: F,
) -> Result<(), Error>
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
    F: FnMut(&[u8]) -> Result<Option<Vec<u8>>, E>,
    E: Into<RewriterError>,
{
    let in_c = path_to_cstring(input_path.as_ref())?;
    let out_c = path_to_cstring(output_path.as_ref())?;

    let mut wrapped = move |bytes: &[u8]| rewriter(bytes).map_err(Into::into);
    let mut state = RewriterState {
        callback: &mut wrapped,
        error: None,
        panicked: false,
    };
    let ffi = FfiSkipOptimizers::new(options)?;

    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_simplify_path(
            in_c.as_ptr(),
            out_c.as_ptr(),
            ffi.ptr(),
            ffi.len(),
            ffi.is_null_flag(),
            bool_to_c(options.constant_folding),
            bool_to_c(options.shape_inference),
            options.tensor_size_threshold,
            target_opset_to_c(options.target_opset_version),
            Some(rewrite_trampoline),
            Some(rewrite_free_trampoline),
            &mut state as *mut RewriterState as *mut c_void,
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        Ok(())
    } else {
        Err(rewriter_error(&mut state, out_error))
    }
}

/// Return the names of all available fuse/elimination optimizer passes.
///
/// These are the names accepted by [`Options::skip_optimizer`] and
/// [`Options::skip_optimizers`].
pub fn list_optimizers() -> Vec<String> {
    let raw = unsafe { onnxsim_sys::onnxsim_list_optimizers() };
    if raw.is_null() {
        return Vec::new();
    }
    let text = unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned();
    unsafe { onnxsim_sys::onnxsim_free_string(raw) };
    text.lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Render a human-readable diff between an `original` and a `simplified` model,
/// both serialized ONNX `ModelProto` bytes.
///
/// The report is an ASCII table comparing op counts and model size — the same
/// "here is the difference" summary the Python CLI prints after simplifying — so
/// a Rust caller can show a before/after view without decoding the models
/// itself. A metric that improved is flagged with a trailing `*`.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = std::fs::read("model.onnx")?;
/// let simplified = onnxsim::simplify(&model)?;
/// print!("{}", onnxsim::model_info_diff(&model, &simplified)?);
/// # Ok(())
/// # }
/// ```
pub fn model_info_diff(original: &[u8], simplified: &[u8]) -> Result<String, Error> {
    let mut out_text: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        onnxsim_sys::onnxsim_model_info_diff(
            original.as_ptr() as *const c_void,
            original.len(),
            simplified.as_ptr() as *const c_void,
            simplified.len(),
            &mut out_text,
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        if out_text.is_null() {
            return Ok(String::new());
        }
        let text = unsafe { CStr::from_ptr(out_text) }
            .to_string_lossy()
            .into_owned();
        unsafe { onnxsim_sys::onnxsim_free_string(out_text) };
        Ok(text)
    } else {
        Err(take_error(out_error))
    }
}

/// Render a node- and value-level diff between an `original` and a
/// `simplified` model, both serialized ONNX `ModelProto` bytes: which
/// nodes/values were removed, added, or changed (matched by output tensor
/// name), e.g. a Conv whose bias input got folded into its weight.
///
/// Complementary to [`model_info_diff`], which reports op-count/size/MACs
/// aggregates rather than the specific nodes and values involved.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = std::fs::read("model.onnx")?;
/// let simplified = onnxsim::simplify(&model)?;
/// print!("{}", onnxsim::graph_diff(&model, &simplified)?);
/// # Ok(())
/// # }
/// ```
pub fn graph_diff(original: &[u8], simplified: &[u8]) -> Result<String, Error> {
    let mut out_text: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        onnxsim_sys::onnxsim_graph_diff(
            original.as_ptr() as *const c_void,
            original.len(),
            simplified.as_ptr() as *const c_void,
            simplified.len(),
            &mut out_text,
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        if out_text.is_null() {
            return Ok(String::new());
        }
        let text = unsafe { CStr::from_ptr(out_text) }
            .to_string_lossy()
            .into_owned();
        unsafe { onnxsim_sys::onnxsim_free_string(out_text) };
        Ok(text)
    } else {
        Err(take_error(out_error))
    }
}

/// Export a serialized ONNX `ModelProto` to a standalone safetensors archive
/// at `out_path`.
///
/// Every initializer's bytes move into the archive with real, byte-accurate
/// offsets — openable by the `safetensors` Python package / HF tooling with
/// no onnxsim involved — and the graph itself is embedded alongside them, so
/// `out_path` alone is both the model's weights and its graph. Reload it with
/// [`import_safetensors`].
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = std::fs::read("model.onnx")?;
/// onnxsim::export_safetensors(&model, "model.onnx.safetensors")?;
/// # Ok(())
/// # }
/// ```
pub fn export_safetensors<P: AsRef<Path>>(model_bytes: &[u8], out_path: P) -> Result<(), Error> {
    let out_c = path_to_cstring(out_path.as_ref())?;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_export_safetensors(
            model_bytes.as_ptr() as *const c_void,
            model_bytes.len(),
            out_c.as_ptr(),
            &mut out_error,
        )
    };
    if status == onnxsim_sys::ONNXSIM_OK {
        Ok(())
    } else {
        Err(take_error(out_error))
    }
}

/// Import a standalone safetensors archive at `in_path` back into a
/// serialized ONNX `ModelProto`.
///
/// `in_path` must be an archive produced by [`export_safetensors`] (or any
/// other tool following the same self-describing-archive convention: an
/// embedded `model.onnx` entry alongside the tensors). Fails if the archive
/// has no embedded onnxsim model, e.g. a plain weights-only safetensors file
/// with no graph to import.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_bytes = onnxsim::import_safetensors("model.onnx.safetensors")?;
/// std::fs::write("model.onnx", &model_bytes)?;
/// # Ok(())
/// # }
/// ```
pub fn import_safetensors<P: AsRef<Path>>(in_path: P) -> Result<Vec<u8>, Error> {
    let in_c = path_to_cstring(in_path.as_ref())?;
    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_size: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_import_safetensors(
            in_c.as_ptr(),
            &mut out_data,
            &mut out_size,
            &mut out_error,
        )
    };
    if status == onnxsim_sys::ONNXSIM_OK {
        let result =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        unsafe { onnxsim_sys::onnxsim_free_buffer(out_data) };
        Ok(result)
    } else {
        Err(take_error(out_error))
    }
}

/// GGUF counterpart of [`export_safetensors`].
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = std::fs::read("model.onnx")?;
/// onnxsim::export_gguf(&model, "model.onnx.gguf")?;
/// # Ok(())
/// # }
/// ```
pub fn export_gguf<P: AsRef<Path>>(model_bytes: &[u8], out_path: P) -> Result<(), Error> {
    let out_c = path_to_cstring(out_path.as_ref())?;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_export_gguf(
            model_bytes.as_ptr() as *const c_void,
            model_bytes.len(),
            out_c.as_ptr(),
            &mut out_error,
        )
    };
    if status == onnxsim_sys::ONNXSIM_OK {
        Ok(())
    } else {
        Err(take_error(out_error))
    }
}

/// GGUF counterpart of [`import_safetensors`].
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_bytes = onnxsim::import_gguf("model.onnx.gguf")?;
/// std::fs::write("model.onnx", &model_bytes)?;
/// # Ok(())
/// # }
/// ```
pub fn import_gguf<P: AsRef<Path>>(in_path: P) -> Result<Vec<u8>, Error> {
    let in_c = path_to_cstring(in_path.as_ref())?;
    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_size: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        onnxsim_sys::onnxsim_import_gguf(
            in_c.as_ptr(),
            &mut out_data,
            &mut out_size,
            &mut out_error,
        )
    };
    if status == onnxsim_sys::ONNXSIM_OK {
        let result =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        unsafe { onnxsim_sys::onnxsim_free_buffer(out_data) };
        Ok(result)
    } else {
        Err(take_error(out_error))
    }
}

/// Owns the `CString`s and pointer array backing the `skip_optimizers` FFI
/// arguments, keeping them alive for the duration of a call.
struct FfiSkipOptimizers {
    // `_owned` keeps the C strings alive; `ptrs` points into them.
    _owned: Vec<CString>,
    ptrs: Vec<*const c_char>,
    is_null: bool,
}

impl FfiSkipOptimizers {
    fn new(options: &Options) -> Result<Self, Error> {
        match &options.skip_optimizers {
            None => Ok(FfiSkipOptimizers {
                _owned: Vec::new(),
                ptrs: Vec::new(),
                is_null: true,
            }),
            Some(names) => {
                let mut owned = Vec::with_capacity(names.len());
                for name in names {
                    let c = CString::new(name.as_str()).map_err(|_| {
                        Error::InvalidArgument(format!(
                            "optimizer name contains an interior NUL byte: {name:?}"
                        ))
                    })?;
                    owned.push(c);
                }
                let ptrs = owned.iter().map(|c| c.as_ptr()).collect();
                Ok(FfiSkipOptimizers {
                    _owned: owned,
                    ptrs,
                    is_null: false,
                })
            }
        }
    }

    fn ptr(&self) -> *const *const c_char {
        if self.ptrs.is_empty() {
            ptr::null()
        } else {
            self.ptrs.as_ptr()
        }
    }

    fn len(&self) -> usize {
        self.ptrs.len()
    }

    fn is_null_flag(&self) -> c_int {
        bool_to_c(self.is_null)
    }
}

// --- Custom executor -------------------------------------------------------
//
// The executor boundary lets you drive constant folding through your own ONNX
// runtime instead of the built-in ONNX Runtime. Tensors cross as DLPack
// DLManagedTensors; this module wraps that in a safe API where inputs arrive as
// borrowed [`TensorRef`]s and outputs are returned as owned [`Tensor`]s.

/// Element type of a tensor at the executor boundary.
///
/// This is the set onnxsim's constant folder exchanges (it matches the C++
/// dtype table); other ONNX dtypes (string, complex, float8, 4-bit) never cross
/// this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// IEEE half precision (float16).
    Float16,
    /// IEEE single precision (float32).
    Float32,
    /// IEEE double precision (float64).
    Float64,
    /// bfloat16.
    Bfloat16,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    Uint8,
    /// 16-bit unsigned integer.
    Uint16,
    /// 32-bit unsigned integer.
    Uint32,
    /// 64-bit unsigned integer.
    Uint64,
    /// Boolean (1 byte per element).
    Bool,
}

impl DataType {
    /// Size in bytes of a single element of this type.
    pub fn byte_size(self) -> usize {
        use DataType::*;
        match self {
            Bool | Int8 | Uint8 => 1,
            Float16 | Bfloat16 | Int16 | Uint16 => 2,
            Float32 | Int32 | Uint32 => 4,
            Float64 | Int64 | Uint64 => 8,
        }
    }

    /// The DLPack `(code, bits)` pair for this type (`lanes` is always 1).
    fn to_dl(self) -> (u8, u8) {
        use onnxsim_sys::DLDataTypeCode::*;
        use DataType::*;
        let (code, bits) = match self {
            Float16 => (kDLFloat, 16),
            Float32 => (kDLFloat, 32),
            Float64 => (kDLFloat, 64),
            Bfloat16 => (kDLBfloat, 16),
            Int8 => (kDLInt, 8),
            Int16 => (kDLInt, 16),
            Int32 => (kDLInt, 32),
            Int64 => (kDLInt, 64),
            Uint8 => (kDLUInt, 8),
            Uint16 => (kDLUInt, 16),
            Uint32 => (kDLUInt, 32),
            Uint64 => (kDLUInt, 64),
            Bool => (kDLBool, 8),
        };
        (code as u8, bits)
    }

    /// Recover a [`DataType`] from a DLPack `DLDataType`, or `None` if it is not
    /// one onnxsim's boundary supports (vectors, unknown code/width, etc.).
    fn from_dl(dt: onnxsim_sys::DLDataType) -> Option<DataType> {
        use onnxsim_sys::DLDataTypeCode;
        use DataType::*;
        if dt.lanes != 1 {
            return None;
        }
        // `dt.code` is a raw u8 from C; compare against the enum's discriminants.
        let code = dt.code;
        let float = DLDataTypeCode::kDLFloat as u8;
        let bfloat = DLDataTypeCode::kDLBfloat as u8;
        let int = DLDataTypeCode::kDLInt as u8;
        let uint = DLDataTypeCode::kDLUInt as u8;
        let boolean = DLDataTypeCode::kDLBool as u8;
        match (code, dt.bits) {
            (c, 16) if c == float => Some(Float16),
            (c, 32) if c == float => Some(Float32),
            (c, 64) if c == float => Some(Float64),
            (c, 16) if c == bfloat => Some(Bfloat16),
            (c, 8) if c == int => Some(Int8),
            (c, 16) if c == int => Some(Int16),
            (c, 32) if c == int => Some(Int32),
            (c, 64) if c == int => Some(Int64),
            (c, 8) if c == uint => Some(Uint8),
            (c, 16) if c == uint => Some(Uint16),
            (c, 32) if c == uint => Some(Uint32),
            (c, 64) if c == uint => Some(Uint64),
            (c, 8) if c == boolean => Some(Bool),
            _ => None,
        }
    }
}

/// An owned CPU tensor your executor returns for a graph output: raw
/// little-endian element bytes plus dtype and shape.
///
/// `data.len()` must equal `shape.iter().product::<i64>() as usize *
/// dtype.byte_size()`; [`simplify_with_executor`] aborts the run with an error
/// if it does not.
#[derive(Debug, Clone)]
pub struct Tensor {
    dtype: DataType,
    shape: Vec<i64>,
    data: Vec<u8>,
}

impl Tensor {
    /// Build an output tensor from its dtype, shape, and raw little-endian bytes.
    pub fn new(dtype: DataType, shape: Vec<i64>, data: Vec<u8>) -> Self {
        Tensor { dtype, shape, data }
    }

    /// Element type.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Shape (dimension sizes).
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Raw little-endian element bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Number of elements (product of the shape; 1 for a scalar).
    fn num_elements(&self) -> i64 {
        self.shape.iter().product()
    }

    /// Move into a heap-allocated DLManagedTensor owned by the C side. The
    /// returned pointer is released by [`output_deleter`] (invoked by onnxsim
    /// through the tensor's DLPack `deleter`).
    fn into_managed(self) -> *mut onnxsim_sys::DLManagedTensor {
        use onnxsim_sys::{DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};
        let (code, bits) = self.dtype.to_dl();
        // Keep shape and data alive in a context box; the DLTensor points into
        // the Vecs' heap buffers, which stay put when the box is leaked.
        let mut ctx = Box::new(OutputCtx {
            shape: self.shape,
            data: self.data,
        });
        let ndim = ctx.shape.len() as c_int;
        let shape_ptr = ctx.shape.as_mut_ptr();
        let data_ptr = ctx.data.as_mut_ptr() as *mut c_void;
        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim,
                dtype: DLDataType {
                    code,
                    bits,
                    lanes: 1,
                },
                shape: shape_ptr,
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(output_deleter),
        });
        Box::into_raw(managed)
    }
}

/// Backing store for a [`Tensor`] handed to the C side as a DLManagedTensor.
struct OutputCtx {
    shape: Vec<i64>,
    data: Vec<u8>,
}

/// DLPack `deleter` for tensors produced by [`Tensor::into_managed`]. Reclaims
/// both the context box (shape + data) and the DLManagedTensor box.
unsafe extern "C" fn output_deleter(this: *mut onnxsim_sys::DLManagedTensor) {
    if this.is_null() {
        return;
    }
    let managed = Box::from_raw(this);
    if !managed.manager_ctx.is_null() {
        drop(Box::from_raw(managed.manager_ctx as *mut OutputCtx));
    }
    // `managed` (the DLManagedTensor box) is dropped here.
}

/// A borrowed view of an input tensor onnxsim feeds to the executor. Valid only
/// for the duration of the executor call; copy out anything you need to keep.
pub struct TensorRef<'a> {
    dtype: DataType,
    shape: &'a [i64],
    data: &'a [u8],
}

impl<'a> TensorRef<'a> {
    /// Element type.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Shape (dimension sizes).
    pub fn shape(&self) -> &'a [i64] {
        self.shape
    }

    /// Raw little-endian element bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Borrow the fields of a DLManagedTensor onnxsim passed in. Rejects tensors
    /// that are not CPU, not contiguous, or of an unsupported dtype.
    ///
    /// # Safety
    /// `mt` must be a valid DLManagedTensor whose buffers outlive `'a`.
    unsafe fn from_managed(mt: &'a onnxsim_sys::DLManagedTensor) -> Result<TensorRef<'a>, Error> {
        use onnxsim_sys::DLDeviceType;
        let t = &mt.dl_tensor;
        if t.device.device_type != DLDeviceType::kDLCPU {
            return Err(Error::Simplify(
                "executor input tensor is not on the CPU device".to_string(),
            ));
        }
        if !t.strides.is_null() {
            return Err(Error::Simplify(
                "executor input tensor is not contiguous (non-null strides)".to_string(),
            ));
        }
        let dtype = DataType::from_dl(t.dtype).ok_or_else(|| {
            Error::Simplify(format!(
                "executor input tensor has an unsupported dtype (code={}, bits={})",
                t.dtype.code, t.dtype.bits
            ))
        })?;
        let ndim = t.ndim.max(0) as usize;
        let shape = std::slice::from_raw_parts(t.shape, ndim);
        let count: i64 = shape.iter().product();
        let nbytes = count.max(0) as usize * dtype.byte_size();
        let base = (t.data as *const u8).add(t.byte_offset as usize);
        let data = std::slice::from_raw_parts(base, nbytes);
        Ok(TensorRef { dtype, shape, data })
    }
}

/// Type-erased executor closure: `(sub-model bytes, input tensors) -> outputs`.
type ExecuteCallback<'a> =
    dyn for<'b> FnMut(&'b [u8], &'b [TensorRef<'b>]) -> Result<Vec<Tensor>, RewriterError> + 'a;

/// Holds the user closure and captures failures that cannot cross the FFI as a
/// return value (a closure error, a panic, or an invalid input tensor).
struct ExecutorState<'a> {
    callback: &'a mut ExecuteCallback<'a>,
    error: Option<RewriterError>,
    panicked: bool,
}

/// `extern "C"` shim matching [`onnxsim_sys::OnnxsimExecuteFn`]. Recovers the
/// `ExecutorState`, borrows the input tensors, runs the closure under
/// `catch_unwind`, and hands the outputs back as an owned DLManagedTensor array.
unsafe extern "C" fn execute_trampoline(
    user_data: *mut c_void,
    model_data: *const c_void,
    model_size: usize,
    inputs: *const *const onnxsim_sys::DLManagedTensor,
    num_inputs: usize,
    out_outputs: *mut *mut *mut onnxsim_sys::DLManagedTensor,
    out_num_outputs: *mut usize,
) -> c_int {
    let state = &mut *(user_data as *mut ExecutorState);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model: &[u8] = if model_size == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(model_data as *const u8, model_size)
        };
        let mut refs: Vec<TensorRef> = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let mt = *inputs.add(i);
            if mt.is_null() {
                return Err(RewriterError::from(Box::new(Error::Simplify(
                    "executor received a null input tensor".to_string(),
                ))));
            }
            refs.push(TensorRef::from_managed(&*mt)?);
        }
        (state.callback)(model, &refs)
    }));
    match outcome {
        Ok(Ok(outputs)) => {
            // Validate each output's byte length before building DLManagedTensors,
            // so a malformed tensor is a clean error rather than an out-of-bounds
            // read on the C side.
            for (i, t) in outputs.iter().enumerate() {
                let expected = t.num_elements().max(0) as usize * t.dtype.byte_size();
                if t.data.len() != expected {
                    state.error = Some(RewriterError::from(Box::new(Error::Simplify(format!(
                        "executor output {i} has {} bytes but dtype {:?} and shape {:?} \
                         imply {expected}",
                        t.data.len(),
                        t.dtype,
                        t.shape
                    )))));
                    return -1;
                }
            }
            let mut boxed: Box<[*mut onnxsim_sys::DLManagedTensor]> = outputs
                .into_iter()
                .map(Tensor::into_managed)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let len = boxed.len();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed); // ownership passes to execute_free_trampoline
            *out_outputs = ptr;
            *out_num_outputs = len;
            0
        }
        Ok(Err(e)) => {
            state.error = Some(e);
            -1
        }
        Err(_) => {
            state.panicked = true;
            -1
        }
    }
}

/// `extern "C"` shim matching [`onnxsim_sys::OnnxsimExecuteFreeFn`]. Reclaims the
/// output-array container allocated in [`execute_trampoline`]. The tensors
/// themselves are released by onnxsim through their own deleters, so this must
/// not touch them.
unsafe extern "C" fn execute_free_trampoline(
    _user_data: *mut c_void,
    outputs: *mut *mut onnxsim_sys::DLManagedTensor,
    num_outputs: usize,
) {
    if outputs.is_null() {
        return;
    }
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        outputs,
        num_outputs,
    )));
}

/// Map an FFI failure, preferring an error captured from the executor closure.
fn executor_error(state: &mut ExecutorState, out_error: *mut c_char) -> Error {
    let c_message = if out_error.is_null() {
        None
    } else {
        match take_error(out_error) {
            Error::Simplify(msg) => Some(msg),
            other => Some(other.to_string()),
        }
    };
    if state.panicked {
        return Error::Simplify("the custom executor panicked".to_string());
    }
    if let Some(e) = state.error.take() {
        return Error::Simplify(format!("custom executor failed: {e}"));
    }
    Error::Simplify(c_message.unwrap_or_else(|| "unknown error (no message provided)".to_string()))
}

/// Simplify a serialized ONNX `ModelProto`, driving constant folding through a
/// custom executor instead of the built-in ONNX Runtime.
///
/// This is the Rust counterpart of the C API's executor callback — the seam for
/// running onnxsim on top of another ONNX runtime. `executor` is called once per
/// fold group with the sub-model as serialized `ModelProto` bytes and its input
/// tensors (borrowed [`TensorRef`]s, positional to `graph.input`); it must
/// evaluate the sub-model and return one owned [`Tensor`] per graph output, in
/// order.
///
/// `function_rewrite_rules` on the options are not supported here (this entry
/// point takes an executor, not a rule set) and cause an error if set.
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
/// let model = std::fs::read("model.onnx")?;
/// let simplified = onnxsim::simplify_with_executor(
///     &model,
///     &onnxsim::Options::new(),
///     |_submodel: &[u8], _inputs: &[onnxsim::TensorRef]| {
///         // Run `_submodel` in your runtime with `_inputs`, then return the
///         // output tensors. (A real executor must actually evaluate it.)
///         Ok::<_, onnxsim::Error>(Vec::<onnxsim::Tensor>::new())
///     },
/// )?;
/// # let _ = simplified;
/// # Ok(())
/// # }
/// ```
pub fn simplify_with_executor<F, E>(
    model_bytes: &[u8],
    options: &Options,
    mut executor: F,
) -> Result<Vec<u8>, Error>
where
    F: for<'b> FnMut(&'b [u8], &'b [TensorRef<'b>]) -> Result<Vec<Tensor>, E>,
    E: Into<RewriterError>,
{
    if !options.function_rewrite_rules.is_empty() {
        return Err(Error::InvalidArgument(
            "function_rewrite_rules are only supported by simplify/simplify_with, \
             not simplify_with_executor"
                .to_string(),
        ));
    }

    let mut wrapped =
        move |model: &[u8], inputs: &[TensorRef]| executor(model, inputs).map_err(Into::into);
    let mut state = ExecutorState {
        callback: &mut wrapped,
        error: None,
        panicked: false,
    };
    let ffi = FfiSkipOptimizers::new(options)?;

    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_size: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();

    let status = unsafe {
        onnxsim_sys::onnxsim_simplify_with_executor(
            model_bytes.as_ptr() as *const c_void,
            model_bytes.len(),
            ffi.ptr(),
            ffi.len(),
            ffi.is_null_flag(),
            bool_to_c(options.constant_folding),
            bool_to_c(options.shape_inference),
            options.tensor_size_threshold,
            target_opset_to_c(options.target_opset_version),
            None,
            None,
            ptr::null_mut(),
            Some(execute_trampoline),
            Some(execute_free_trampoline),
            &mut state as *mut ExecutorState as *mut c_void,
            &mut out_data,
            &mut out_size,
            &mut out_error,
        )
    };

    if status == onnxsim_sys::ONNXSIM_OK {
        let result =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        unsafe { onnxsim_sys::onnxsim_free_buffer(out_data) };
        Ok(result)
    } else {
        Err(executor_error(&mut state, out_error))
    }
}

fn bool_to_c(value: bool) -> c_int {
    if value {
        1
    } else {
        0
    }
}

/// Encode the optional target opset version for the C ABI: `None` (leave the
/// opset unchanged) maps to 0, which the C API treats as "no change".
fn target_opset_to_c(version: Option<i32>) -> c_int {
    version.unwrap_or(0) as c_int
}

/// Consume a C-owned error string, returning a Rust `Error` and freeing the
/// original allocation.
fn take_error(out_error: *mut c_char) -> Error {
    if out_error.is_null() {
        return Error::Simplify("unknown error (no message provided)".to_string());
    }
    let message = unsafe { CStr::from_ptr(out_error) }
        .to_string_lossy()
        .into_owned();
    unsafe { onnxsim_sys::onnxsim_free_string(out_error) };
    Error::Simplify(message)
}

#[cfg(unix)]
fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        Error::InvalidArgument(format!("path contains an interior NUL byte: {path:?}"))
    })
}

#[cfg(not(unix))]
fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    let s = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("path is not valid UTF-8: {path:?}")))?;
    CString::new(s).map_err(|_| {
        Error::InvalidArgument(format!("path contains an interior NUL byte: {path:?}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests exercise pure-Rust logic and do not touch the FFI, so they
    // run under `cargo test` even without the native library linked.

    #[test]
    fn default_options() {
        let opts = Options::default();
        assert!(opts.constant_folding);
        assert!(opts.shape_inference);
        assert_eq!(opts.skip_optimizers, Some(Vec::new()));
        assert_eq!(opts.tensor_size_threshold, DEFAULT_TENSOR_SIZE_THRESHOLD);
        assert_eq!(opts.target_opset_version, None);
    }

    #[test]
    fn target_opset_version_setter() {
        let opts = Options::new().target_opset_version(18);
        assert_eq!(opts.target_opset_version, Some(18));
        // None encodes to 0 (the C API's "no change" sentinel); Some passes through.
        assert_eq!(target_opset_to_c(None), 0);
        assert_eq!(target_opset_to_c(Some(18)), 18);
    }

    #[test]
    fn without_optimizers_sets_null() {
        let opts = Options::new().without_optimizers();
        let ffi = FfiSkipOptimizers::new(&opts).unwrap();
        assert!(ffi.is_null);
        assert_eq!(ffi.is_null_flag(), 1);
        assert!(ffi.ptr().is_null());
    }

    #[test]
    fn skip_optimizer_reenables_optimization() {
        let opts = Options::new()
            .without_optimizers()
            .skip_optimizer("eliminate_nop_transpose");
        assert_eq!(
            opts.skip_optimizers,
            Some(vec!["eliminate_nop_transpose".to_string()])
        );
    }

    #[test]
    fn ffi_skip_optimizers_layout() {
        let opts = Options::new().skip_optimizers(["a", "bb"]);
        let ffi = FfiSkipOptimizers::new(&opts).unwrap();
        assert!(!ffi.is_null);
        assert_eq!(ffi.len(), 2);
        assert!(!ffi.ptr().is_null());
    }

    #[test]
    fn interior_nul_is_rejected() {
        let opts = Options::new().skip_optimizer("bad\0name");
        assert!(matches!(
            FfiSkipOptimizers::new(&opts),
            Err(Error::InvalidArgument(_))
        ));
    }

    // The rewriter trampolines are pure Rust (no FFI into the native library),
    // so their translation of a closure's outcome to the C return convention —
    // and the buffer hand-off/reclaim — can be exercised directly.

    unsafe fn call_rewrite(
        state: &mut RewriterState,
        input: &[u8],
        out_data: &mut *mut c_void,
        out_size: &mut usize,
    ) -> c_int {
        rewrite_trampoline(
            state as *mut RewriterState as *mut c_void,
            input.as_ptr() as *const c_void,
            input.len(),
            out_data,
            out_size,
        )
    }

    #[test]
    fn rewrite_trampoline_reports_unchanged() {
        let mut cb = |_: &[u8]| Ok::<_, RewriterError>(None);
        let mut state = RewriterState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_data: *mut c_void = ptr::null_mut();
        let mut out_size = 0usize;
        let rc = unsafe { call_rewrite(&mut state, &[1, 2, 3], &mut out_data, &mut out_size) };
        assert_eq!(rc, 0);
        assert!(out_data.is_null());
    }

    #[test]
    fn rewrite_trampoline_roundtrips_rewritten_buffer() {
        let payload = vec![9u8, 8, 7, 6];
        let expected = payload.clone();
        let mut cb = move |_: &[u8]| Ok::<_, RewriterError>(Some(payload.clone()));
        let mut state = RewriterState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_data: *mut c_void = ptr::null_mut();
        let mut out_size = 0usize;
        let rc = unsafe { call_rewrite(&mut state, &[0], &mut out_data, &mut out_size) };
        assert_eq!(rc, 1);
        assert_eq!(out_size, expected.len());
        let returned =
            unsafe { std::slice::from_raw_parts(out_data as *const u8, out_size) }.to_vec();
        assert_eq!(returned, expected);
        // Reclaim through the free trampoline exactly as the C core would.
        unsafe { rewrite_free_trampoline(ptr::null_mut(), out_data, out_size) };
    }

    #[test]
    fn rewrite_trampoline_captures_error() {
        let mut cb = |_: &[u8]| Err::<Option<Vec<u8>>, RewriterError>("boom".into());
        let mut state = RewriterState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_data: *mut c_void = ptr::null_mut();
        let mut out_size = 0usize;
        let rc = unsafe { call_rewrite(&mut state, &[0], &mut out_data, &mut out_size) };
        assert_eq!(rc, -1);
        let err = rewriter_error(&mut state, ptr::null_mut());
        assert!(matches!(&err, Error::Simplify(m) if m.contains("boom")));
    }

    #[test]
    fn rewrite_trampoline_captures_panic() {
        let mut cb = |_: &[u8]| -> Result<Option<Vec<u8>>, RewriterError> { panic!("kaboom") };
        let mut state = RewriterState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_data: *mut c_void = ptr::null_mut();
        let mut out_size = 0usize;
        // Silence the default panic hook's backtrace print during the test.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rc = unsafe { call_rewrite(&mut state, &[0], &mut out_data, &mut out_size) };
        std::panic::set_hook(prev);
        assert_eq!(rc, -1);
        assert!(state.panicked);
        let err = rewriter_error(&mut state, ptr::null_mut());
        assert!(matches!(&err, Error::Simplify(m) if m.contains("panicked")));
    }

    // --- Executor boundary -------------------------------------------------
    //
    // Like the rewriter trampolines, the executor trampolines and the
    // DLManagedTensor conversions are pure Rust, so the whole input-view ->
    // closure -> output-tensor -> free round trip can be driven directly without
    // the native library.

    #[test]
    fn datatype_byte_size_and_dl_roundtrip() {
        use DataType::*;
        let all = [
            Float16, Float32, Float64, Bfloat16, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32,
            Uint64, Bool,
        ];
        for dt in all {
            let (code, bits) = dt.to_dl();
            assert_eq!(bits as usize, dt.byte_size() * 8);
            let recovered = DataType::from_dl(onnxsim_sys::DLDataType {
                code,
                bits,
                lanes: 1,
            });
            assert_eq!(recovered, Some(dt));
        }
        // Vectors and unknown widths are rejected.
        assert_eq!(
            DataType::from_dl(onnxsim_sys::DLDataType {
                code: onnxsim_sys::DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 4,
            }),
            None
        );
    }

    /// Build a borrowed input DLManagedTensor (as onnxsim would) over caller-owned
    /// shape/data buffers, feed it through the executor trampoline, and return the
    /// closure's outcome plus the reconstructed outputs.
    #[test]
    fn execute_trampoline_roundtrips_tensors() {
        // Input: int32 tensor shape [2] = [1, 2]; the closure echoes it back as
        // an output, exercising both TensorRef::from_managed and into_managed.
        let mut in_shape: [i64; 1] = [2];
        let mut in_data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0];
        let mut in_tensor = onnxsim_sys::DLManagedTensor {
            dl_tensor: onnxsim_sys::DLTensor {
                data: in_data.as_mut_ptr() as *mut c_void,
                device: onnxsim_sys::DLDevice {
                    device_type: onnxsim_sys::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: 1,
                dtype: onnxsim_sys::DLDataType {
                    code: onnxsim_sys::DLDataTypeCode::kDLInt as u8,
                    bits: 32,
                    lanes: 1,
                },
                shape: in_shape.as_mut_ptr(),
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        let input_ptrs: [*const onnxsim_sys::DLManagedTensor; 1] = [&in_tensor];

        let mut seen_shape: Vec<i64> = Vec::new();
        let mut seen_bytes: Vec<u8> = Vec::new();
        let mut cb = |model: &[u8], inputs: &[TensorRef]| {
            assert_eq!(model, b"model-bytes");
            assert_eq!(inputs.len(), 1);
            seen_shape = inputs[0].shape().to_vec();
            seen_bytes = inputs[0].data().to_vec();
            assert_eq!(inputs[0].dtype(), DataType::Int32);
            Ok::<_, RewriterError>(vec![Tensor::new(
                DataType::Int32,
                inputs[0].shape().to_vec(),
                inputs[0].data().to_vec(),
            )])
        };
        let mut state = ExecutorState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };

        let model = b"model-bytes";
        let mut out_outputs: *mut *mut onnxsim_sys::DLManagedTensor = ptr::null_mut();
        let mut out_num: usize = 0;
        let rc = unsafe {
            execute_trampoline(
                &mut state as *mut ExecutorState as *mut c_void,
                model.as_ptr() as *const c_void,
                model.len(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                &mut out_outputs,
                &mut out_num,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(seen_shape, vec![2]);
        assert_eq!(seen_bytes, vec![1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(out_num, 1);

        // Inspect the produced output tensor, then release it exactly as onnxsim
        // would: each element via its deleter, then the array via the free shim.
        unsafe {
            let out0 = *out_outputs;
            assert!(!out0.is_null());
            let t = &(*out0).dl_tensor;
            assert_eq!(t.ndim, 1);
            assert_eq!(std::slice::from_raw_parts(t.shape, 1), &[2]);
            let bytes = std::slice::from_raw_parts(t.data as *const u8, 8);
            assert_eq!(bytes, &[1, 0, 0, 0, 2, 0, 0, 0]);
            let deleter = (*out0).deleter.expect("output tensor must carry a deleter");
            deleter(out0);
            execute_free_trampoline(ptr::null_mut(), out_outputs, out_num);
        }
        // Keep the borrowed input buffers alive until here.
        let _ = (&in_shape, &in_data, &mut in_tensor);
    }

    #[test]
    fn execute_trampoline_rejects_wrong_output_length() {
        let mut cb = |_: &[u8], _: &[TensorRef]| {
            // Claims shape [4] int32 (16 bytes) but supplies only 4 bytes.
            Ok::<_, RewriterError>(vec![Tensor::new(
                DataType::Int32,
                vec![4],
                vec![0, 0, 0, 0],
            )])
        };
        let mut state = ExecutorState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_outputs: *mut *mut onnxsim_sys::DLManagedTensor = ptr::null_mut();
        let mut out_num: usize = 0;
        let rc = unsafe {
            execute_trampoline(
                &mut state as *mut ExecutorState as *mut c_void,
                ptr::null(),
                0,
                ptr::null(),
                0,
                &mut out_outputs,
                &mut out_num,
            )
        };
        assert_eq!(rc, -1);
        let err = executor_error(&mut state, ptr::null_mut());
        assert!(matches!(&err, Error::Simplify(m) if m.contains("bytes")));
    }

    #[test]
    fn execute_trampoline_captures_panic() {
        let mut cb =
            |_: &[u8], _: &[TensorRef]| -> Result<Vec<Tensor>, RewriterError> { panic!("kaboom") };
        let mut state = ExecutorState {
            callback: &mut cb,
            error: None,
            panicked: false,
        };
        let mut out_outputs: *mut *mut onnxsim_sys::DLManagedTensor = ptr::null_mut();
        let mut out_num: usize = 0;
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rc = unsafe {
            execute_trampoline(
                &mut state as *mut ExecutorState as *mut c_void,
                ptr::null(),
                0,
                ptr::null(),
                0,
                &mut out_outputs,
                &mut out_num,
            )
        };
        std::panic::set_hook(prev);
        assert_eq!(rc, -1);
        assert!(state.panicked);
        let err = executor_error(&mut state, ptr::null_mut());
        assert!(matches!(&err, Error::Simplify(m) if m.contains("panicked")));
    }
}
