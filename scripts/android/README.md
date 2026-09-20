# ONNX Runtime Android integration smoke test

This smoke test runs an ONNX model and its `onnxsim` result with ONNX Runtime's
CPU execution provider on a connected Android arm64 device. When given the QNN
provider AAR, it also tries QNN HTP, QNN GPU, and Android NNAPI with NNAPI's CPU
device disabled. It also builds a tiny NNAPI RELU model and compiles it directly
for `qti-dsp`, checking that the DSP driver accepts and executes the program.
The QNN sessions disable ORT CPU fallback; a pass means QNN accepted the
complete graph. The automatic NNAPI run records available device names/types
and disables NNAPI's CPU device.

The CPU and NNAPI probes keep a float32 `Relu` node and remove redundant
`Identity` nodes around it. The QNN HTP probe uses a separate quantize/dequantize
`Relu` graph because HTP expects quantized models. Each run checks both graph
outputs against the expected values with CPU fallback disabled for hardware
providers.

## Requirements

- A connected arm64 Android phone on API 29+ with USB debugging enabled and `adb` authorized.
- Android SDK platform tools, API 26 or newer, and NDK 27.2 (or a compatible NDK).
- CMake and Python packages `onnx`, `numpy`, and this repository's `onnxsim`.
- An ONNX Runtime Android AAR, such as `onnxruntime-android-1.30.0.aar`.
- For DSP/GPU runs, a QNN provider AAR built for the same ONNX Runtime version.

## Run

```bash
python scripts/android/run_onnxruntime_android.py \
  --runtime-aar /path/to/onnxruntime-android.aar \
  --qnn-aar /path/to/onnxruntime-android-qnn.aar \
  --android-sdk "$ANDROID_HOME" \
  --ndk-version 27.2.12479018
```

Use `--adb /path/to/adb` or `--serial SERIAL` to select a specific ADB client or
device. QNN HTP and GPU runs are attempted when `--qnn-aar` is provided; a
backend that cannot load or execute reports `SKIP`. Use `--require-htp` and/or
`--require-gpu` to make the matching QNN backend mandatory. Use
`--require-nnapi-hw` to require NNAPI execution with its CPU device disabled.
Use `--require-nnapi-dsp` to require the explicit `qti-dsp` NNAPI compile and run.
Android chooses among its available NNAPI hardware devices, so this check does
not claim a specific GPU driver was selected. CPU is always required. Host build
artifacts use a temporary directory unless `--work-dir` is provided; phone
staging files are removed after the run. The debug test APK remains installed.

The runtime and QNN AARs are not vendored. The script extracts only the arm64
runtime library, headers, and QNN provider needed to build the smoke-test
executable.

## Xiaomi 12S probe result

The CPU and NNAPI float32 probes pass, including direct compilation on
`qti-dsp`. Android reports `qti-gpu` and `qti-dsp` hardware devices; the NNAPI
execution provider chooses the device automatically, while the direct DSP probe
explicitly selects `qti-dsp`. The QNN EP exposes an NPU device, but the
quantized QDQ `Relu` graph still leaves nodes assigned to the default CPU EP
with fallback disabled. It also does not expose a separate QNN GPU device on
this phone, so QNN HTP/GPU remain unconfirmed.
