# Custom WebGPU kernels from model metadata (experimental)

**Status: experimental.** Attaching a kernel (hand-written or
tinygrad-generated) to a model and dispatching it against raw GPU buffers
works today and is tested end to end (Python -> `.onnx` bytes -> real WebGPU
device). Splicing that dispatch into a real `onnxruntime-web` pipeline --
running `onnxruntime-web` up to a flagged node, executing the custom program
via this metadata, and resuming `onnxruntime-web` past it, with no CPU
round-trip -- also works today, for the single-flagged-node case (see
`onnxsim/webgpu_custom_kernel_runtime.py` below). What's not built yet is
automatically chaining several such splices across a whole model. This
document describes what exists and where the boundary is.

## What this is

Six pieces, one per language/language-boundary:

- **`onnxsim/webgpu_kernel_metadata.py`** -- attaches a custom WebGPU
  *program* (one or more WGSL kernel `steps`, each with its own entry point,
  a static `[x, y, z]` dispatch size, and a list of `@group`/`@binding`
  bindings -- a binding names a node tensor, a named scratch
  `intermediate`, or an inline `constant`) to a specific node, as one
  JSON-valued `metadata_props` entry on that `NodeProto`, keyed
  `"onnxsim.webgpu_kernel"`. See that module's own docstring for the exact
  schema and why it's steps (plural), and
  `attach_webgpu_kernel`/`read_webgpu_kernel`/`list_webgpu_kernels`.
- **`onnxsim/webgpu_tinygrad_codegen.py`** -- generates a
  `WebgpuKernelSpec` automatically for specific gaps `onnxsim.webgpu_target`
  flags (`Conv` with a 3-D spatial rank, `Resize` align_corners
  downsampling), by building the equivalent computation as a real
  [tinygrad](https://github.com/tinygrad/tinygrad) `Tensor` graph and
  rendering tinygrad's own scheduled UOp IR to WGSL with its `WGSLRenderer`
  -- entirely offline, no real GPU needed. `tinygrad` is an optional
  dependency (`pip install onnxsim[webgpu-codegen]`), only imported if one
  of this module's `generate_*` functions is actually called. See that
  module's own docstring for exactly how the lowering works, what's covered,
  and what's deliberately left out (`ConvTranspose`, Attention).
- **`scripts/convertmodel/onnx_node_metadata.mjs`** -- reads that metadata
  back out of raw ONNX `ModelProto` bytes in the browser/Node, via a small
  hand-rolled protobuf reader rather than a full protobuf runtime or
  onnx.js (neither of which onnxruntime-web ships, and onnxruntime-web's own
  API doesn't expose arbitrary `metadata_props`). The field numbers it reads
  are not guessed -- they were pulled directly off the installed `onnx`
  Python package's own protobuf descriptors, which protobuf's own
  backward-compatibility rules guarantee never change.
- **`scripts/convertmodel/webgpu_kernel_dispatcher.mjs`** -- compiles and
  dispatches every step of a program in order, on one command encoder:
  `dispatchWebgpuProgram(device, spec, buffersByTensor)` creates the bind
  group layout(s)/pipeline for each step, allocates and frees any
  `intermediate`/`constant` buffers the program needs for its own run, and
  always awaits completion before returning. Also has small
  `createStorageBuffer`/`readBackFloat32Buffer` helpers for
  uploading/downloading a `Float32Array`.
- **`onnxsim/webgpu_custom_kernel_runtime.py`** -- `split_around_node(model,
  node_name)` splits a model into a `(pre, post)` pair with the named node
  physically excised from both, reusing `onnxsim.vitisai_target.split_model`
  (built for a different EP-placement problem, but exactly the right "cut a
  graph at a tensor boundary" primitive) twice -- once at the node's own
  inputs, once at its outputs. Physically removing the node (rather than
  leaving it in place and letting `onnxruntime-web` fail on it) matters: see
  the module's own docstring for why `onnxruntime-web`'s WebGPU partitioner
  commits a node to WebGPU by op type alone, so an unsupported *variant* of
  an otherwise-supported op is only caught once its kernel actually runs,
  failing the whole session.
- **`scripts/convertmodel/webgpu_custom_kernel_runtime.mjs`** --
  `runOnnxModelWithCustomKernel(...)` runs `pre` and `post` as ordinary
  `onnxruntime-web` WebGPU sessions and dispatches the excised node's own
  program between them via `dispatchWebgpuProgram`, using
  `preferredOutputLocation: 'gpu-buffer'` and `Tensor.fromGpuBuffer` (the
  same GPU-buffer interop `onnxruntime-web` itself uses for chaining
  sessions) so the split point never touches the CPU.

## What this does not do (yet)

The single-flagged-node splice above is real and tested end to end, but
there is no *automatic* multi-node splicer: chaining several flagged nodes
in one model, or picking split points itself from
`onnxsim.webgpu_target.estimate_webgpu_islands`, is still up to the caller
-- call `split_around_node` once per flagged node yourself.

Also out of scope here: dispatch sizes that depend on a dynamic input shape
(each step's `dispatch` is a fixed `[x, y, z]` triple, not a formula), and
any kind of automatic kernel *tuning* (trying several workgroup
sizes/variants and picking the fastest) -- this only executes the program a
caller (or `webgpu_tinygrad_codegen`) attaches. `webgpu_custom_kernel_runtime.mjs`
additionally requires `pre` to exist (a node whose inputs are at least
partly produced by another node) -- the case where the flagged node
consumes only the model's own top-level inputs (`split_around_node`'s `pre`
is `None` then) isn't wired up on the JS side yet.

`webgpu_tinygrad_codegen` itself has its own, narrower boundary: only
`Conv` (not `ConvTranspose`) at any spatial rank, and only 4-D `Resize` in
`"linear"`/`align_corners` mode with a constant `scales` input whose ratios
divide each spatial dimension to an exact integer size (ONNX's and
tinygrad's own align_corners coordinate formulas only agree in that case --
see the module's docstring). Attention (`com.microsoft::Attention`
`mask_index`) is flagged by `onnxsim.webgpu_target` but not yet generated
here, since the op has several mutually incompatible `mask_index` shapes and
picking the wrong one would silently produce a working-but-wrong kernel.

## Testing

- `tests/test_webgpu_kernel_metadata.py` -- attach/read/list round trips and
  validation errors for the schema itself, pure Python, no browser.
- `tests/test_webgpu_tinygrad_codegen.py` -- for each `generate_*` function,
  runs the same tinygrad `Tensor` graph on tinygrad's own CPU device and
  checks it against `onnx.reference.ReferenceEvaluator` running the actual
  ONNX node -- this verifies the ONNX -> tinygrad translation (attribute
  handling, padding convention, axis order), independent of whether the WGSL
  rendering/dispatch is correct. Skipped when `tinygrad` isn't installed.
- `scripts/convertmodel/test/onnx_node_metadata.test.mjs` -- proves the JS
  reader agrees with a real `.onnx` file the Python side wrote (not just
  that each side round-trips its own data). Plain Node, no browser, part of
  `npm run test:all` (`test:onnx-node-metadata`).
- `tests/test_webgpu_custom_kernel_runtime.py` -- checks `split_around_node`
  purely as graph surgery (no browser, no `onnxruntime`): recomposes `pre`
  -> the excised node (run standalone via `onnx.reference.ReferenceEvaluator`,
  standing in for a real dispatch) -> `post` and compares against
  `ReferenceEvaluator` running the original, unsplit graph. Covers the
  "node consumes only top-level graph inputs" (`pre is None`) case and a
  side input that bypasses both `pre` and the excised node.
- `scripts/convertmodel/test/webgpu_kernel_dispatcher.test.mjs` -- reads a
  hand-written WGSL elementwise-add kernel out of a real `.onnx` fixture's
  node metadata and runs it on a real WebGPU device (Playwright/Chromium,
  same requirement and SwiftShader caveat as
  `webgpu_attention_placement.test.mjs`), checking the GPU output against a
  plain-JS reference. Runs in
  `.github/workflows/convertmodel-webgpu-kernel-dispatcher.yml`.
- `scripts/convertmodel/test/webgpu_tinygrad_codegen.test.mjs` -- same idea,
  but for the actual WGSL `webgpu_tinygrad_codegen` generates (Conv3D,
  Resize align_corners) rather than a hand-written kernel -- the first real
  GPU execution of a tinygrad-rendered kernel, closing the loop with
  `tests/test_webgpu_tinygrad_codegen.py`'s CPU-only numeric check. Runs in
  the same workflow.
- `scripts/convertmodel/test/webgpu_custom_kernel_runtime.test.mjs` -- the
  full pipeline: a real `onnxruntime-web` WebGPU session runs `pre`, its
  GPU-buffer output is spliced into a tinygrad-generated Conv3D program, and
  that result is spliced into a second real `onnxruntime-web` WebGPU
  session (`post`) -- checked against `onnx.reference.ReferenceEvaluator`
  running the *original*, unsplit model. Runs in the same workflow.
- `scripts/convertmodel/test/make_webgpu_kernel_fixture.py`,
  `make_webgpu_tinygrad_codegen_fixture.py`, and
  `make_webgpu_custom_kernel_runtime_fixture.py` regenerate the fixtures the
  `.test.mjs` files above read; each loads the `onnxsim` module(s) it needs
  directly by file path rather than `import onnxsim`, so regenerating needs
  only `onnx`/`numpy` (plus `tinygrad` for the latter two) rather than a
  built onnxsim wheel -- same convention and reasoning as
  `scripts/convertmodel/test/make_ep_placement_fixtures.py`.
