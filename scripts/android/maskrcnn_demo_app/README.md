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

## Measured on the phone

Steady state, in the app (UI drawing and, in camera mode, the live camera running). "Latency" is one
`nativeRun` call, from the RGBA bitmap to the four output tensors. FPS is how often a processed frame
reaches the screen: latency plus JPEG decode or preview grab, scaling, and drawing.

| mode | pipeline | FPS | latency per frame | startup (session setup) |
|---|---|---:|---:|---:|
| test images (6 COCO images, looped) | `pipe_e_opt.txt` (HTP graphs compiled at startup) | **5.5-6.2** | 127-171 ms (varies per image) | 6.5-7.8 s |
| test images | `pipe_e_opt_ctx.txt` (HTP sessions from EP-context models) | 5.0-5.6 | 136-186 ms | **0.75 s** |
| camera, 1440x1080 back camera | `pipe_e_opt.txt` | 4.5-5.8 | 150-190 ms | 6.5 s |

For comparison, the one-process `adb shell` driver (`../e2e_pipeline`, same pipeline, no UI, no
camera) measured 130-158 ms per image JIT and 156-188 ms with EP-context models. The app matches it
per image. Its FPS is lower than 1000/latency because frames aren't pipelined: decoding or grabbing
the next frame waits for the previous inference.

- **EP-context vs JIT:** the same tradeoff `../e2e_pipeline` found. EP-context models cut startup
  from ~7 s to 0.75 s but slow the box head (by ~20 ms here). A demo that starts often should use
  `pipe_e_opt_ctx.txt`; a long-running one should use `pipe_e_opt.txt`.
- **Camera mode is slower** than test-image mode and drifts down over a few minutes (5.8 -> 4.6 FPS
  within ~40 s here). The preview grab runs on the UI thread and the camera pipeline competes with
  ORT's 4 CPU threads; thermal throttling is likely too. Not investigated further.
- **Preprocessing:** the app quantizes the RGBA bitmap straight to the backbone's uint8 NHWC input
  (same float math as the pipeline's `quant_in` step on `eval_common.canvas`), so no fp32 image is
  ever built. That takes 2.5-15 ms depending on CPU contention, against ~15 ms for the e2e driver's
  fp32 -> uint8 loop.
- **Accuracy:** same pipeline and models as `../e2e_pipeline` (58/61 matched vs all-ONNX-Runtime on
  its 6 images). Checked visually here: `cats.jpg` gives cat 0.99, cat 0.98, remote 0.82, and COCO
  000000000724 gives stop sign 1.00 and truck 0.79. That matches the e2e run's 3 and 2 detections on
  those images. The app resizes with Android's bilinear `createScaledBitmap` instead of PIL's, so
  scores can differ slightly from the e2e numbers.

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

## Two bugs found building it

- **Wrong labels:** the first version used torchvision's 91-id COCO list, and the cats came out as
  "bird". This model (maskrcnn-benchmark lineage) uses **81 contiguous classes** (cat = 16,
  remote = 66); `Coco.java` now has that list.
- **Every box had the last detection's mask:** the overlay reused one mutable 28x28 bitmap for all
  detections. A hardware-accelerated canvas records draw calls and uploads bitmap contents only at
  render time, so all masks came out as the last one drawn. It now creates one bitmap per detection.

## Known limits

- **No frame pipelining:** capture, inference and drawing run one after the other. Overlapping the
  next frame's capture and preprocessing with the current inference would bring FPS closer to
  1000/latency.
- **The displayed image is the processed frame**, not the live preview. That keeps boxes aligned
  with what was inferred; a small live camera thumbnail sits in the corner.
- **The APK doesn't bundle the models** (about 100 MB); `deploy.sh` copies them in with `run-as`, so
  this needs a debuggable build. A release build would download them into internal storage.
- **Fixed 800x1088 input**, like the rest of this pipeline: frames are scaled to fit, top-left
  aligned.
