# Tuning a tinygrad-generated WebGPU kernel via real browser execution

**Status: experimental.** `onnxsim.webgpu_kernel_tuning.generate_kernel_candidates`
generates several *alternative* WebGPU kernels for the exact same
computation -- differently tiled/upcast/unrolled variants of the same
kernel `onnxsim.webgpu_tinygrad_codegen` would otherwise render exactly
once -- so a caller can dispatch every candidate on a **real** WebGPU
device and keep whichever is actually fastest there, via
`webgpu_kernel_dispatcher.mjs`'s own `profile: true` GPU timing (see
`docs/webgpu-kernel-dispatch.md`'s own "Profiling" section).

## Why this needed its own module, not just `BEAM=N`

tinygrad already has an autotuner. Setting `BEAM=N` makes
`tinygrad.codegen.to_program` call `tinygrad.codegen.opt.search.beam_search`,
which tries several `Opt`-tuned kernel variants and picks the fastest --
**by actually compiling and running each one itself**, via `dev =
Device[s.ren.target.device]` (`tinygrad/codegen/opt/search.py`, read
directly off the installed 0.14.0 source). That line is exactly the wall
this whole `webgpu_tinygrad_codegen` module family exists to route around:
it needs a real, natively-loaded `Device["WEBGPU"]` (the actual
`dawn`/`wgpu-native` shared library), which frequently isn't available
wherever kernels are *generated* (a CI runner, a server, this repo's own
dev sandboxes) even though it's exactly what an end user's real browser
always has.

`onnxsim.webgpu_kernel_tuning` reuses only the **device-free half** of
tinygrad's own autotuner -- `tinygrad.codegen.opt.postrange.Scheduler` and
`tinygrad.codegen.opt.search.get_kernel_actions`, which enumerate candidate
`Opt` combinations via plain Python schedule manipulation
(`Scheduler.copy()` + `.apply_opt()`, no compilation, no device at all).
What tinygrad's own `beam_search` does *next* -- compile, run, time, pick
-- is the caller's job instead, against a real WebGPU device reached from
JS.

## What it does

`generate_kernel_candidates(named_tensors, output_name, max_candidates=32)`
has the same contract as
`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` (the same
leaf-tensor/output-name inputs), but for each real kernel tinygrad
schedules, returns a `KernelCandidates` (one `WebgpuKernelStep` per
candidate `get_kernel_actions` finds, capped at `max_candidates`) instead
of only tinygrad's own default rendering. A kernel's tuning options change
its loop/tiling structure -- how many work-items iterate, how much stays in
registers/local memory -- never *which* buffers it reads or writes, so
every candidate shares identical bindings; only `wgsl`/`entry_point`/
`dispatch` differ. `KernelCandidates.spec_for(index, intermediates)` builds
the single-step `WebgpuKernelSpec` `dispatchWebgpuProgram` actually
consumes for whichever candidate a caller wants to run.

## Verified end to end, with a real speed difference

`scripts/convertmodel/test/webgpu_kernel_tuning.test.mjs` dispatches every
candidate from `make_webgpu_kernel_tuning_fixture.py`'s fixture (8
candidates for a real `Conv2D`) on a real WebGPU device (Playwright/
Chromium), with `profile: true`, and checks:

- Every single candidate -- not just whichever turns out fastest --
  computes the numerically correct output against
  `onnx.reference.ReferenceEvaluator`. Tuning options that changed
  correctness would be a tinygrad bug, not something this module tries to
  re-verify in general, but checking it here costs nothing and catches a
  wiring mistake on this module's own side.
- Real per-candidate GPU durations come back, and picking the minimum
  actually mattered: verified by hand against this exact fixture, the
  *untuned* baseline (`applied_opts == []`) was the **slowest** of the 8
  candidates -- about **3x slower** than the fastest tuned one
  (`~2.9ms` vs `~0.94ms`). Real, not hypothetical: tinygrad's own default
  (non-BEAM) rendering is not a good kernel for this shape on this device.

## Scope

Like `webgpu_tinygrad_codegen` itself, this only ever produces candidates
-- generation, dispatch, timing, and picking a winner are three separate
steps, and only the first happens in Python. A model with more than one
scheduled kernel call gets candidates enumerated independently per call;
this module does not attempt to jointly tune across calls, and there is no
persistence/caching layer yet for a picked winner (e.g. keyed by GPU
vendor/browser) -- a caller re-runs the whole dispatch-and-compare loop
every time today.

## How much could tinygrad's tuned kernel outperform WebNN?

`scripts/convertmodel/test/webgpu_kernel_tuning_vs_webnn.test.mjs` answers
this directly, on the *same* Conv2D as the fixture above: it dispatches every
tinygrad-tuned candidate (fastest wins) and, separately, runs
`webgpu_kernel_tuning_fixture.onnx` (the same Conv2D, saved as an ordinary
standalone model by `make_webgpu_kernel_tuning_fixture.py`) through
onnxruntime-web's WebNN execution provider -- both timed the same way
(wall-clock median over several warmed-up runs), since WebNN has no
GPU-timestamp-query equivalent exposed through onnxruntime-web the way
`dispatchWebgpuProgram`'s own `profile: true` does.

Like `webnn_reshape_placement.test.mjs` (`docs/webnn.md`,
`onnxsim/webnn_target.py`), this is **attempted and reported, not required**:
WebNN's browser support is still experimental. Concretely, in this repo's own
dev sandbox (headless Linux Chromium), `navigator.ml` is absent under plain
`--enable-unsafe-webgpu`, but *does* appear -- and its `"gpu"` device type
context actually builds and runs -- once
`--enable-features=WebMachineLearningNeuralNetwork` is also passed. So
whether the comparison runs at all depends on that flag and the runner's
browser, not the platform alone; when no WebNN device is reachable, the test
still reports tinygrad's own fastest candidate (useful on its own) and skips
only the comparison-specific checks.

**A measured result** from that sandbox: WebNN's `"gpu"` device type came out
**~1.7-2.4x *faster*** than tinygrad's own best-tuned candidate across
repeated runs -- the opposite direction from what the tuning work above might
suggest. Take that with real caution, though: neither side is running on real
hardware there. WebGPU goes through SwiftShader's software rasterizer (see
`webgpu_hf_demo.test.mjs`'s own comment), and Chromium's WebNN `"gpu"` device
type falls back to its own software ML backend when there's no real GPU/NPU
init path available in a headless Linux container. So this result says
neither backend is a safe default assumption in a software-emulated sandbox
-- it does not say which one wins on an end user's actual GPU or NPU. Treat
the *magnitude* (WebNN and a hand-tuned custom kernel can land within a small
constant factor of each other on the same op) as the finding, and the
*direction* as unconfirmed pending a run on real hardware (a real macOS/
Windows CI runner, or a developer's own machine with
`ORT_REQUIRE_WEBNN=1 npm run test:webgpu-kernel-tuning-vs-webnn`).

This is also why offloading conv/matmul/gemm to WebNN wholesale (the other
half of the question that motivated this work) isn't a clear win to chase
blindly: where WebNN is actually reachable, it's already competitive with a
hand-tuned custom kernel on at least this op, without onnxsim needing to
generate or maintain any kernel at all -- but `onnxsim.webnn_target`'s own
gaps (non-constant `Reshape`/`Expand` shapes, INT64 graph boundaries) mean
"reachable" is model-dependent, and this comparison only covers the one op
it measures, not the fusion patterns (conv+activation) the original question
also asked about.

## Running the tuner from the actual converter page

Everything above ran offline (a Python script) or through a Node+Playwright
test -- useful for proving the idea, but not something a visitor to the
actual converter page (`scripts/convertmodel/index.html`) could ever trigger
themselves. `scripts/convertmodel/webgpu_kernel_tuner.mjs` closes that gap: an
opt-in **"Tune this kernel…"** button on a `Conv` node's own entry in the
existing "Custom WebGPU kernels" panel
(`webgpu_kernel_annotations_view.mjs`) runs the *entire* loop above live, in
the browser, no server involved:

1. **Generation, live, via Pyodide.** `onnx_conv_node_reader.mjs` reads the
   target node's shapes/attributes straight out of the model bytes already in
   the page (no re-upload). `webgpu_kernel_tuner.mjs` then boots
   [Pyodide](https://pyodide.org/) (loaded from a CDN on first use, not
   bundled into the page) and fetches tinygrad's wheel straight from PyPI --
   the same numpy-free, onnx-free technique
   `pyodide_webgpu_single_node_codegen.test.mjs` already proved works for a
   single kernel, extended here to `onnxsim.webgpu_kernel_tuning`'s own
   `Scheduler`/`get_kernel_actions` approach so *several* candidates come
   back, not just one. Both downloads are real, multi-second, multi-megabyte
   fetches -- nothing loads until the button is clicked, and a second tuning
   run (same node or a different one) reuses the already-booted runtime.
2. **Dispatch, live, on a real device.** Every candidate is dispatched via
   the page's own `webgpu_kernel_dispatcher.mjs` against random data matching
   the node's real shapes (see `webgpu_kernel_tuner.mjs`'s own docstring for
   why random data is exactly as valid here as real pipeline data would be --
   picking a winner is about speed, and kernel generation never depends on
   concrete values), timed the same way `webgpu_kernel_tuning_vs_webnn.test.mjs`
   times its own candidates, ranked, fastest first.
3. **Export.** Clicking **"Export tuned model"** attaches the winning
   candidate's `WebgpuKernelSpec` onto that exact node via
   `onnx_metadata_writer.mjs`'s `attachWebgpuKernelSpec` -- the write-side
   counterpart to `onnx_node_metadata.mjs`'s read-only
   `readWebgpuKernelSpecs`, and the JS equivalent of
   `onnxsim.webgpu_kernel_metadata.attach_webgpu_kernel` -- and downloads the
   result. This is a from-scratch, hand-rolled protobuf field editor (see
   that module's own docstring): it never assumes it understands the whole
   `ModelProto` schema, only the one path from the top-level message down to
   the target node's `metadata_props`, so every other field at every level
   (initializers, other nodes, opset imports, ...) round-trips untouched,
   verified byte-for-byte in `test/onnx_metadata_writer.test.mjs`. Nothing
   about Simplify/Optimize themselves changes -- a tuned export is a separate
   download, on top of whatever model bytes are already in the page.

Scope matches `onnx_conv_node_reader.mjs`'s own: a `Conv` node (any spatial
rank, including what this repo calls "Conv3D") with fully static shapes, and
only when tinygrad schedules it to exactly one kernel call. Clicking Tune on
anything else surfaces `readConvNodeInfo`'s own clear error rather than
silently doing nothing.

**Verified end to end** in `test/webgpu_kernel_tuner_ui.test.mjs`, driving the
*real* page (not a synthetic harness): load a fixture, click Tune, wait for
several real candidates with real per-candidate timings, click Export, and
confirm the downloaded model's attached kernel both matches the chosen
winner and still computes the right answer against the same ground truth
`webgpu_tinygrad_codegen.test.mjs` itself checks against. The one piece that
sandbox couldn't verify against the real, public CDN is noted in that test's
own comment (an internal network-policy restriction specific to this repo's
dev sandbox, not a property of a real visitor's browser or of the CI runner
this test runs on) -- everything else, including the tinygrad-wheel-from-PyPI
fetch, the real dispatch loop, and the export/re-verify round trip, was
proven working, unmodified, end to end.
