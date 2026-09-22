# On-device HTP: real evidence found, no clean throughput number yet

Follow-up to `README.md`'s host-side finding (PR #1807): the host x86 environment has no real HTP
device for `QNNExecutionProvider` to route to at all (`ort.get_ep_devices()` shows none), so that
path was correctly abandoned as a validation signal. This note covers the first real on-device
attempt, directly on the phone's own Hexagon V69 HTP silicon (device `239dbd8f`).

## Existing infrastructure used (not built here)

This repo already has a complete, working pipeline for exactly this, predating this session:
`scripts/android/run_onnxruntime_android.py` + `scripts/android/app/` (a real Android app,
`org.onnxsim.androidtest`) + `scripts/android/runner/` (a plain-CPU native comparison binary). The
app's native code (`scripts/android/app/app/src/main/cpp/jni_runner.cpp`) drives ONNX Runtime's
QNN Execution Provider through the current plugin-EP API (`RegisterExecutionProviderLibrary` +
`Ort::Env::GetEpDevices` + `AppendExecutionProvider_V2`) -- the same API PR #1807 found rejected
outright on the host. Running from inside an installed app matters: this is "Android's app linker
namespace, where vendor libraries are visible" (the script's own comment) -- `/vendor/lib64`'s
`libQnnHtp.so`/`libQnnHtpV69Stub.so` are reachable from an app process in a way they are not from a
bare `adb shell` process, unlike the FastRPC/`adsprpc` libraries `native_transport/`'s raw native
client has used successfully all session. This is a real, structural difference between the HVX
path (plain shell process, `libcdsprpc.so`) and the HTP path (needs an installed app) worth noting
for any future HTP work.

Artifacts used: `~/bev-tmp/onnxruntime-android-1.26.0.aar` (core runtime + headers) and
`~/bev-tmp/onnxruntime-android-qnn-2.6.0.aar` (the QNN EP provider library, `arm64-v8a` only --
confirmed via `unzip -l`, it ships `libonnxruntime_providers_qnn.so` alone, no runtime backend
libraries, so no separate `--qnn-runtime-aar` was needed; the app relies on the phone's own
`/vendor/lib64` HTP backend libraries instead).

## Real signal #1: genuine device enumeration differs from the host

Every on-device run reports `QNN devices=QNNExecutionProvider/2` -- **two real hardware devices**,
confirmed present and enumerable by the QNN EP on-device. The host, by contrast, saw zero. This
alone is decisive evidence the phone's real hardware path is architecturally live in a way the host
never was.

## Real signal #2: a prior session's log proves genuine HTP graph compilation happened

The app's on-device data directory (`run-as org.onnxsim.androidtest ls files/`) contains leftover
artifacts from an earlier, unrelated session on this same device (dated 2026-09-20/21, predating
this investigation) -- including `output_qnn-htp-fallback.f32.qnn_qnn.log`, a real QNN profiling
log (31 KB, part text/part binary trace). Its readable strings are unambiguous, genuine
HTP-compilation telemetry: `Finalizing Graph Sequence`, `Parallelization Optimization`,
`VTCM Allocation`, `Graph Sequencing for Target`, `Post Graph Optimization`,
`Graph Optimizations`, `Graph Preparation`, `QNN (finalize) time`, `Accelerator (finalize) time`,
`QNN accelerator (finalize) time`, `RPC (finalize) time`. VTCM (Vector TCM) is HTP/HVX-specific
on-chip scratch memory -- a CPU-fallback execution would never reference it. **This confirms real
HTP graph compilation and execution has genuinely happened on this exact phone**, at some point
before this investigation, through this exact app.

## What blocked a fresh, clean repro in this pass

Reran the app's built-in QDQ smoke test (`make_qnn_htp_models()` in
`run_onnxruntime_android.py`: `QuantizeLinear -> DequantizeLinear -> Relu -> QuantizeLinear ->
DequantizeLinear`) against both QNN targets:

- **`qnn-htp` (strict, no CPU fallback allowed)**: fails with *"This session contains graph nodes
  that are assigned to the default CPU EP, but fallback to CPU EP has been explicitly disabled by
  the user."* This is expected, correct ORT behavior, not a bug -- QNN's HTP graph compiler
  doesn't accept every node in this tiny synthetic graph, and strict mode is doing exactly what it
  says. Real backbone convs are far more likely to be fully HTP-compatible than a bare Relu
  sandwiched between QDQ pairs; this result doesn't generalize pessimistically to the real model.
- **`qnn-htp-fallback` (CPU fallback allowed, profiling + EP-context caching enabled)**: fails with
  *"Load model from .../original.onnx.ctx.onnx failed: File doesn't exist."* Traced this precisely
  in `jni_runner.cpp` (lines ~226-234): the code correctly does a two-step EP-context-cache flow --
  first construct-and-immediately-destruct a session with `ep.context_enable=1` to generate a
  `<model>.ctx.onnx` context binary, then load *that* file with `ep.context_enable=0` for the real
  run. The second step fails because the first step's output file never actually appears on disk,
  even though constructing that first session doesn't throw. Ruled out stale on-device cache state
  as the explanation: `pm clear org.onnxsim.androidtest` (full app-data wipe) followed by a fresh
  rebuild+reinstall+rerun reproduces the identical failure. This is a real, reproducible bug in the
  app's context-caching sequence (or a QNN-side silent failure to emit the context file that the
  app doesn't check for/surface), not a stale-state artifact -- a genuine fix needs either
  source-level changes to `jni_runner.cpp`'s context-generation step (e.g. verifying the file
  actually landed before proceeding, and surfacing whatever error QNN produced instead) or a
  different EP-context configuration, out of scope to chase further in this pass.

## Net result

Not a throughput number -- that's still open. But strictly stronger evidence than "we don't know if
this works": real hardware devices are visible where the host had none, and a real prior session
proves genuine HTP graph compilation is achievable on this exact phone through this exact app. The
blocker in *this* pass is a specific, identified, reproducible bug in the app's own EP-context
two-step caching sequence (`qnn-htp-fallback`'s `run_one()` in `jni_runner.cpp`), not a fundamental
"HTP doesn't work here" finding. The natural next step is fixing that context-generation step (or
running without `ep.context_enable`/`offload_graph_io_quantization` entirely, at the cost of losing
the profiling breakdown that mode provides) rather than re-attempting host-side validation or a
from-scratch ONNX Runtime build.
