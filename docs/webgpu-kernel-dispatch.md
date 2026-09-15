# Custom WebGPU kernels from model metadata (experimental)

**Status: experimental, standalone primitive.** Attaching a kernel to a model
and dispatching it against raw GPU buffers works today and is tested end to
end (Python -> `.onnx` bytes -> real WebGPU device). Splicing a dispatch like
this into an actual `onnxruntime-web` session -- so a node onnxsim flags as
unsupported (`onnxsim.webgpu_target`) runs through here instead of crashing
-- is **not built yet**. This document describes what exists and where the
boundary is.

## What this is

Four pieces, one per language/language-boundary:

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

## What this does not do (yet)

There is no graph splitter: something that would run an `onnxruntime-web`
session up to a flagged node, execute the WGSL kernel via this metadata, and
resume another session past it -- handing GPU buffers between the two
without a CPU round-trip, most likely via `onnxruntime-web`'s GPU-buffer IO
binding. That "runtime on top of ort-web" is future work; what's here is the
metadata schema and a standalone way to run one kernel against arbitrary
buffers, which that runtime would build on.

Also out of scope here: dispatch sizes that depend on a dynamic input shape
(each step's `dispatch` is a fixed `[x, y, z]` triple, not a formula), and
any kind of automatic kernel *tuning* (trying several workgroup
sizes/variants and picking the fastest) -- this only executes the program a
caller (or `webgpu_tinygrad_codegen`) attaches.

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
- `scripts/convertmodel/test/webgpu_kernel_dispatcher.test.mjs` -- reads a
  hand-written WGSL elementwise-add kernel out of a real `.onnx` fixture's
  node metadata and runs it on a real WebGPU device (Playwright/Chromium,
  same requirement and SwiftShader caveat as
  `webgpu_attention_placement.test.mjs`), checking the GPU output against a
  plain-JS reference. Runs in
  `.github/workflows/convertmodel-webgpu-kernel-dispatcher.yml`.
- `scripts/convertmodel/test/make_webgpu_kernel_fixture.py` regenerates the
  fixture (`webgpu_kernel_add.onnx` + `webgpu_kernel_fixture.json`) both
  `.test.mjs` files above read; it loads
  `onnxsim/webgpu_kernel_metadata.py` directly by file path rather than
  `import onnxsim`, so it needs only the `onnx` package, not a built onnxsim
  wheel -- same convention and reasoning as
  `scripts/convertmodel/test/make_ep_placement_fixtures.py`.
