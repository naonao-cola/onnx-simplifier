# Mask R-CNN live demo app (Android, Hexagon HTP + HVX)

An Android app that runs the full Mask R-CNN (ONNX model zoo `MaskRCNN-12-qdq`) on the phone,
frame by frame, with boxes, labels, instance masks and an FPS / latency counter. It runs exactly
the pipeline `../e2e_pipeline/` measured (PR #1841): `native/maskrcnn_engine.cpp` `#include`s
`../e2e_pipeline/e2e_run.cpp` unchanged and only swaps the image source and the output path.

| region | engine |
|---|---|
| backbone (optimized per PR #1833), box head, mask head | HTP via ORT + QNN EP (`com.qualcomm.qti:qnn-runtime` 2.50.0, bundled) |
| RPN post-processing (TopK, decode, NMS, merge) | HVX DSP, our FastRPC skel (`../tinygrad_hexagon_bridge/rpn_fused`) |
| RoiAlign | HVX DSP, our FastRPC skel (`../tinygrad_hexagon_bridge/roialign_fast`) |
| everything else | ORT CPU, 4 threads |

## Does this work from an app, not just an adb shell? Yes.

Everything before this ran as an `adb shell` native process. PR #1810 had found that an app can't
`dlopen` the *vendor's* `/vendor/lib64/libQnnHtp.so` (it isn't in `/vendor/etc/public.libraries.txt`).
Bundling Qualcomm's own runtime from Maven (as PR #1829 did for the shell) avoids that. Measured in
this app, an `untrusted_app_27` process on device `239dbd8f` (Xiaomi 12S, Snapdragon 8+ Gen 1,
Hexagon V69), no root, no system changes:

- **Bundled QNN runtime loads from the APK's `nativeLibraryDir`** (`useLegacyPackaging true`, so the
  libraries are real files): `QNN EP registered, 1 NPU device(s)`, and every HTP session is created
  strict (`session.disable_cpu_ep_fallback=1`).
- **DSP-side skels load from the app's lib dir.** `ADSP_LIBRARY_PATH` is set in-process (`setenv`,
  before the first FastRPC call) to the app's `nativeLibraryDir` first. `libQnnHtpV69Skel.so`,
  `librpn_rpc.so` and `libroialign_rpc.so` (our skels, renamed `lib*.so` so the package manager
  extracts them) all load: `rpn skel open`, `roialign skel open`.
- **Unsigned PD is allowed** for the app: `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`
  returns 0, and the unsigned skels run.
- **`libcdsprpc.so` loads** (it is in `public.libraries.txt`; declared with `<uses-native-library>`).
- **SELinux denials seen, all harmless:** `untrusted_app_27` is denied `search`/`getattr` on
  `/vendor/dsp` (`adsprpcd_file`), which is only FastRPC's default skel search path, and `open` on
  `/dev/adsprpc-smd` / `adsprpc-smd-secure`. `libcdsprpc` then falls back to the CDSP device node the
  app *is* allowed, and every skel above loads and runs.

One app-context gotcha: files `adb` creates under `/sdcard/Android/data/<pkg>` are owned by the
shell user, so the app can't enter them (`chdir ... models` failed). `deploy.sh` instead copies the
models into the app's **internal** files dir with `run-as` (the APK is debuggable).

## Build and run

```bash
# 1. models: ../e2e_pipeline/README.md "Reproduce" (build_models.py). If ../e2e_pipeline/build.sh
#    already ran on this phone, deploy.sh copies them on-device from /data/local/tmp/e2e.
# 2. build (heavy step; cap it on a shared machine)
systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0 \
  -E HEXAGON_SDK_ROOT=... -E HEXAGON_TOOLCHAIN=... -E ANDROID_HOME=~/android-sdk ./build_app.sh
# 3. install + models + test images
IMGS="cats.jpg img_000000000139.jpg ..." ./deploy.sh        # or MODELS=<build_models.py --out dir>
# 4. run: camera mode (default) or a loop over the test images
adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity --es mode images
adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity --es pipe pipe_e_opt_ctx.txt
```

Toolchain used: Android SDK platform 34 + build-tools 34, NDK 27.2, AGP 8.5.2, Gradle 8.7 (offline),
JDK 21. No CameraX: the app uses the framework Camera2 API, so it needs no extra Maven dependencies.
