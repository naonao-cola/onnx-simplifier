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
  `dispatchWebgpuProgram(device, spec, buffersByTensor, {profile})` creates
  the bind group layout(s)/pipeline for each step, allocates and frees any
  `intermediate`/`constant` buffers the program needs for its own run, and
  always awaits completion before returning. Also has small
  `createStorageBuffer`/`readBackFloat32Buffer` helpers for
  uploading/downloading a `Float32Array`, and `supportsWebgpuProfiling(device)`
  / the `profile: true` option itself for per-step GPU timing -- see
  "Profiling" below.
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
  sessions) so the split point never touches the CPU. Its own `profile: true`
  option forwards straight through to `dispatchWebgpuProgram`, returning
  `{outputs, profiling}` instead of just `outputs`.

## Profiling

`dispatchWebgpuProgram(device, spec, buffersByTensor, {profile: true})`
times each step on the GPU itself and returns
`{timings: [{index, entryPoint, durationNs}, ...] | null}` -- `null` when
`device` can't do it (check `supportsWebgpuProfiling(device)` to tell
"didn't ask" apart from "device can't"). It uses whichever of two real
WebGPU timestamp-query mechanisms `device` was actually created with --
verified directly against a real WebGPU device, not assumed from the spec:

- The standard `"timestamp-query"` feature: `beginComputePass({timestampWrites})`.
- Chromium's own `"chromium-experimental-timestamp-query-inside-passes"`:
  explicit `pass.writeTimestamp(querySet, index)` calls just inside the pass.

Both are the *same* two primitives `onnxruntime-web`'s own WebGPU EP
profiling picks between, so the durations this reports are directly
comparable to `onnxruntime-web`'s own per-kernel numbers for the node a
generated kernel replaced. The Chromium fallback matters in practice, not
just in theory: verified against a real (SwiftShader) WebGPU device that
`onnxruntime-web`'s WebGPU backend requests the Chromium feature over the
standard one whenever the adapter offers both (checked directly against the
installed `onnxruntime-web` bundle's own source) -- so the device
`webgpu_custom_kernel_runtime.mjs` shares with `onnxruntime-web` normally
has *only* the Chromium feature, not the standard one, and profiling a
kernel spliced into a real `onnxruntime-web` session would silently report
no timings at all without this fallback.

## Could the codegen itself run client-side (Pyodide)?

`onnxsim.webgpu_tinygrad_codegen`'s `generate_*` functions run server-side
today (Python, via a CLI or offline script). Whether tinygrad's own
Tensor -> UOp -> WGSL pipeline *could* instead run inside the WASM converter
UI itself, via [Pyodide](https://pyodide.org/), was an open question --
`scripts/convertmodel/test/pyodide_tinygrad_codegen.test.mjs` answers it:
yes, verified directly, for both an elementwise case and Conv3D. It does
this in plain Node (Pyodide runs standalone via its own bundled WASM build,
no browser or GPU needed) by fetching tinygrad's wheel straight from PyPI
and unpacking it into Pyodide's own filesystem -- deliberately skipping
`micropip`/Pyodide's own package CDN, which numpy would otherwise need
(numpy has no pure-Python wheel on PyPI; it needs a real wasm32 build only
Pyodide's own index carries). Building each `Tensor` from a plain (possibly
nested) Python list instead of a numpy array, and never calling
`.numpy()`/`.realize()`, keeps numpy out of `sys.modules` for the whole
codegen path -- checked directly, not assumed, for both cases.

This is **not** the same claim as "`webgpu_tinygrad_codegen.py` runs
unmodified inside Pyodide": that module still imports `numpy` at module
level for its own convenience (random dummy data for scheduling, reading a
real initializer's bytes via `onnx.numpy_helper`), and hasn't itself been
run inside Pyodide. What's proven is that the underlying tinygrad machinery
those functions build on has no fundamental barrier to running client-side.
Turning that into an actual in-browser "generate a kernel for this model,
live" feature would still need either a numpy-free rewrite of the
ONNX-tensor-reading parts (raw `initializer.raw_data` bytes are plain
packed floats, readable with the stdlib `struct`/`array` modules) or numpy
loaded the normal way (which works fine in a real browser's Pyodide --
Pyodide's package CDN is only unreachable in network-restricted sandboxes
like the one this was first verified in, not in general), plus a JS port of
`onnxsim.webgpu_target`'s gap-detection to decide *when* to invoke it. Both
are un-built follow-ups, not done here.

### One flagged node, generated live, verified end to end

`scripts/convertmodel/test/pyodide_webgpu_single_node_codegen.test.mjs`
takes the above further, for the single simplest real case (a `Conv` node
with every attribute at its ONNX default -- the same
`webgpu_tinygrad_conv3d.onnx` fixture `webgpu_tinygrad_codegen.test.mjs`
already dispatches): it regenerates the *exact same kernel*
`onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel` already produced for
that node offline, entirely client-side, with no `onnx`-the-Python-package
and no numpy anywhere in the browser-side path:

1. `scripts/convertmodel/onnx_conv_node_reader.mjs` -- a new hand-rolled
   protobuf reader, sibling to `onnx_node_metadata.mjs` -- reads the node's
   own shapes (from a graph input or an initializer) and attributes
   (`kernel_shape`, `strides`, `dilations`, `group`, `pads`, `auto_pad`,
   with the same defaults `generate_conv_kernel` itself applies) straight
   out of the raw `.onnx` bytes, in JS.
2. That shape/attribute info -- never any tensor *values*, which kernel
   generation doesn't depend on -- drives a numpy-free, onnx-free
   translation of `generate_conv_kernel` + its own `_lower_tensor_program`
   helper, run inside Pyodide, producing a
   `onnxsim.webgpu_kernel_metadata.WebgpuKernelSpec`-shaped JSON program.
3. That live-generated spec is asserted to `deepEqual` the spec
   `generate_conv_kernel` already attached to the fixture offline -- same
   WGSL text, entry point, dispatch, bindings -- proving this isn't just "a
   kernel that happens to run" but the identical one.
4. The live spec (not the fixture's own) is then dispatched via
   `webgpu_kernel_dispatcher.mjs` against a real WebGPU device, fed the
   fixture's real concrete `x`/`w` values, and checked against
   `onnx.reference.ReferenceEvaluator`'s own output for the real node --
   the same numeric bar every other kernel test here holds itself to.

Scope: this is one op (`Conv`), one node, attributes read from a graph
input/initializer only (no shape-inference fallback for an upstream node's
output shape -- see `onnx_conv_node_reader.mjs`'s own docstring). Getting
from here to "flag a gap and generate its kernel live for any node in any
real model" still needs the same two follow-ups named above (a JS port of
`onnxsim.webgpu_target`'s gap-detection to pick *which* node, and either a
numpy-free `initializer.raw_data` reader or a real browser's numpy for
`generate_resize_kernel`'s case, which -- unlike Conv -- does depend on a
constant input's actual values). What this does settle: the remaining gap
is that decision-making and data-plumbing, not tinygrad's own codegen
machinery, which this proves reproduces the server-side result exactly.

### A whole graph, any op tinygrad supports -- via tinygrad's own ONNX importer

The single-node prototype above still needed a bespoke, per-op JS reader
(`onnx_conv_node_reader.mjs`) mirroring `generate_conv_kernel`'s own
hand-written translation, because `onnxsim.webgpu_tinygrad_codegen` only
covers two ops. tinygrad itself ships a much broader answer:
`tinygrad.nn.onnx.OnnxRunner` (`tinygrad/nn/onnx.py`) is a *generic* ONNX
interpreter with its own hand-rolled protobuf parser (no dependency on the
`onnx` Python package at all) and a large op table (`get_onnx_ops()`)
covering well over 100 ops, executing a graph node by node as ordinary
tinygrad `Tensor` calls.

`scripts/convertmodel/test/pyodide_webgpu_onnxrunner_codegen.test.mjs`
proves this runs inside Pyodide and produces real, correct WebGPU kernels
for a **whole multi-node graph** -- not tuned per op by onnxsim at all:

1. The raw `.onnx` model bytes (a two-node `Conv3D -> Relu` graph --
   `Relu` has no bespoke onnxsim codegen at all) are written straight into
   Pyodide's own virtual filesystem; `OnnxRunner` parses and runs the whole
   thing itself, so unlike the single-node case there is **no JS-side
   shape/attribute reader at all**.
2. `OnnxRunner(path)` is constructed, and the graph is run with random
   dummy input data (again: kernel generation depends only on shapes, never
   values), entirely inside `with Context(DEV="WEBGPU"):` -- every tensor
   this produces, including every initializer, is tagged for the `WEBGPU`
   device with no real device backend ever touched.
3. The same `_lower_tensor_program` lowering `generate_conv_kernel` itself
   uses (schedule, find each real compute kernel, render each to WGSL via
   `WGSLRenderer`) turns the resulting output tensor into a
   `WebgpuKernelSpec`-shaped program -- tinygrad's own scheduler fuses the
   Conv3D and Relu into a **single** kernel here, with no fusion logic of
   onnxsim's own involved.
4. That live spec is checked, field for field, against an offline "ground
   truth" spec (`make_webgpu_onnxrunner_fixture.py`) generated the exact
   same way but with *different* random dummy data -- an exact match here
   is real evidence the result is deterministic from shapes alone, not a
   coincidence of matching inputs -- and then dispatched against a real
   WebGPU device with the fixture's real weights, checked against
   `onnx.reference.ReferenceEvaluator`.

Two non-obvious things surfaced building this, both specific to using
`OnnxRunner` rather than hand-building the `Tensor` graph:

- **`tinygrad.nn.onnx` must be imported inside `Context(DEV="WEBGPU")`,
  not just the tensors a caller builds.** That module evaluates a stray
  `Tensor(0)` default argument (for `ConvInteger`) at *module import* time,
  which resolves tinygrad's device default like any other tensor
  construction -- harmless wherever some other usable device exists (a real
  Python process normally has a working CPU backend), but Pyodide has *no*
  usable tinygrad device at all outside that context (no GPU, no native
  compiler for a CPU backend), so the bare import fails outright with "no
  usable devices" otherwise.
- **An initializer must be constructed directly on the target device, not
  moved there afterward.** `OnnxPBParser` parses initializers on tinygrad's
  *default* device; calling `OnnxRunner(...).to("WEBGPU")` afterward (the
  single-node prototype never needed this, since its hand-built leaf
  tensors were already constructed with `device="WEBGPU"`) inserts a
  same-thread device-transfer op that, once scheduled, materializes into a
  *new* buffer distinct from the one captured before scheduling -- so
  matching an initializer's own buffer identity against the buffer the
  compiled kernel actually reads silently fails. Building everything
  (including the `OnnxRunner(...)` construction itself) inside
  `Context(DEV="WEBGPU")` avoids the transfer, and therefore the mismatch,
  entirely.

Scope: still a single-output graph (multi-output would need iterating
`OnnxRunner.graph_outputs`), and correctness of `OnnxRunner`'s own op
*translations* is checked separately and natively -- in
`make_webgpu_onnxrunner_fixture.py`, on tinygrad's ordinary CPU-ish device,
against `ReferenceEvaluator` -- since Pyodide itself has no way to execute
anything (no real device backend at all, same limitation
`pyodide_tinygrad_codegen.test.mjs` already documented). What this does
prove: a path to covering *any* op tinygrad's own ONNX importer already
implements, for an *entire graph* at once, without onnxsim writing a single
line of per-op translation -- a materially larger step than the single
hand-picked `Conv` node above.

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
  plain-JS reference, and (requesting `"timestamp-query"` explicitly at
  device creation, so the check is meaningful rather than skipped) that
  `profile: true` reports a real non-negative GPU duration. Runs in
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
  running the *original*, unsplit model, with `profile: true` passed through
  end to end (this is the test that actually exercises the Chromium
  timestamp-query fallback described above, since it shares
  `onnxruntime-web`'s own device rather than creating one itself). Runs in
  the same workflow.
- `scripts/convertmodel/test/make_webgpu_kernel_fixture.py`,
  `make_webgpu_tinygrad_codegen_fixture.py`, and
  `make_webgpu_custom_kernel_runtime_fixture.py` regenerate the fixtures the
  `.test.mjs` files above read; each loads the `onnxsim` module(s) it needs
  directly by file path rather than `import onnxsim`, so regenerating needs
  only `onnx`/`numpy` (plus `tinygrad` for the latter two) rather than a
  built onnxsim wheel -- same convention and reasoning as
  `scripts/convertmodel/test/make_ep_placement_fixtures.py`.
