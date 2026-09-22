# The EP-context-cache bug is fixed; the next blocker is a hard OEM platform restriction

Follow-up to `ondevice_findings.md` (PR #1808), which found `qnn-htp-fallback`'s two-step
EP-context-cache generation in `scripts/android/app/app/src/main/cpp/jni_runner.cpp` silently
produced no `.ctx.onnx` file, with the session construction that was supposed to write it not
throwing any error.

## The bug, and the fix

`jni_runner.cpp` used the older, session-config-entry-based EP-context mechanism
(`options.AddConfigEntry("ep.context_enable", "1")` etc.) together with a plain
`Ort::Session` construction to trigger context generation as a side effect. That mechanism
predates ORT's newer plugin-EP registration API (`RegisterExecutionProviderLibrary` +
`Ort::Env::GetEpDevices` + `AppendExecutionProvider_V2`, the API this app already uses to reach
QNN). In ORT 1.26 (the exact version this app builds against, confirmed by extracting
`headers/onnxruntime_cxx_api.h` directly from `~/bev-tmp/onnxruntime-android-1.26.0.aar`), that old
config entry is silently ignored for a plugin-registered EP: the session constructs successfully
as an ordinary (non-context-generating) session, no error, no file.

ORT 1.22+ ships a dedicated, documented replacement: `OrtCompileApi` / `Ort::CompileModel` +
`Ort::ModelCompilationOptions` (`CreateModelCompilationOptionsFromSessionOptions`, explicitly
designed to take an existing `SessionOptions` with EPs already appended -- exactly this app's
existing flow). Verified present in the exact AAR this app links against before touching anything.

Fixed by replacing the construct-and-discard-a-session trick with the real Compile API:

```cpp
Ort::ModelCompilationOptions compile_options(ort_env, run_options);
compile_options.SetInputModelPath(model.c_str());
compile_options.SetOutputModelPath(context_path.c_str());
compile_options.SetEpContextEmbedMode(false);
Ort::Status compile_status = Ort::CompileModel(ort_env, compile_options);
if (!compile_status.IsOK()) throw std::runtime_error("CompileModel failed: " + compile_status.GetErrorMessage());
session = std::make_unique<Ort::Session>(ort_env, context_path.c_str(), run_options);
```

(and removed the now-dead `ep.context_enable`/`ep.context_embed_mode` config entries from the
earlier, unconditional setup code -- they did nothing useful with this API either).

**Confirmed fixed**: rebuilt, reinstalled, reran `qnn-htp-fallback` on real hardware (device
`239dbd8f`). The `.ctx.onnx` file now gets written and loads successfully -- the run reaches real
inference and profiling output (`PASS qnn-htp-fallback ... profile=.../output_qnn-htp-fallback...json`),
no more "File doesn't exist". This is a real fix, not a workaround.

## The next blocker, found immediately after: a hard OEM platform restriction, not a bug

The now-working run's own logcat excerpt shows a new, different failure at the QNN backend load
step itself:

```
QNN SetupBackend failed Unable to load backend, error:  dlopen failed: library "libQnnHtp.so" not found
```

...even though that exact file is physically present at `/vendor/lib64/libQnnHtp.so` (confirmed in
PR #1808). Checked the obvious hypothesis first: `AndroidManifest.xml` declared
`<uses-native-library android:name="libcdsprpc.so" ...>` (needed for the FastRPC/HVX work this
session's `native_transport/` has relied on) but never declared `libQnnHtp.so`. Added it (plus
`libQnnHtpV69Stub.so`) and rebuilt -- **identical failure, byte-for-byte the same log line**. So a
missing manifest declaration wasn't the (sole) cause either.

The real cause: Android's vendor-native-library linker namespace additionally requires the
*vendor partition itself* to allowlist a library as app-visible, via
`/vendor/etc/public.libraries.txt`. Read it directly on-device:

```
$ adb shell cat /vendor/etc/public.libraries.txt
libqti-perfd-client.so
libadsprpc.so
libcdsprpc.so
libsdsprpc.so
libfastcvopt.so
```

The FastRPC libraries this whole project's HVX path has used successfully all session
(`libadsprpc.so`, `libcdsprpc.so`, `libsdsprpc.so`) are explicitly present. **`libQnnHtp.so` is
not.** No manifest declaration on the app's side, and no ORT/QNN configuration change, can make an
app-visible-namespace `dlopen()` succeed against a vendor library this list excludes -- this is the
OEM's own policy, enforced by the platform linker, not an application bug. The only paths around
it (a rooted device to edit `/vendor/etc/public.libraries.txt`, or running as a privileged/OEM
system app) are both out of scope, matching this project's established stance on invasive
device modification (see `../tinygrad_hexagon_bridge/README.md`'s own "rooting is out of scope"
note for the exact same phone).

## Net result

The bug this task was scoped to fix -- the EP-context-cache generation failure -- is genuinely
fixed and verified working. It was not, however, the last thing blocking a real HTP throughput
number on this specific device: immediately behind it sits a hard, unfixable-from-userspace OEM
library-visibility restriction. A real on-device HTP number for this Mask R-CNN backbone is not
reachable on this exact phone through this app-based approach without root or a signed/privileged
system app, both out of scope here. This is a precise, evidenced stopping point, not an
unexplored one -- documented the same way this thread's two prior findings (PR #1807's host
fallback, PR #1808's cache-write bug) were, rather than forced further.
