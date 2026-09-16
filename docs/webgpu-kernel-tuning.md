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
