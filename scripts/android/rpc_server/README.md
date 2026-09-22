# Native onnxsim RPC server for Android

A C++ implementation of the `onnxsim.rpc` wire protocol (see `docs/rpc.md`), linked against
onnxruntime's C++ API. Deploys as a single ~2 MB ARM64 binary via `adb push` -- no Python, no
Termux -- and speaks the exact same protocol as `python -m onnxsim.rpc server`, so the existing
`onnxsim.rpc` Python client (`connect`, `upload`, `load_model`, `run`, `time_evaluator`) works
against it unmodified.

## Build

```bash
# Extract an onnxruntime-android AAR (jni/arm64-v8a/libonnxruntime.so + headers/)
unzip onnxruntime-android-1.30.0.aar -d ort

cmake -S scripts/android/rpc_server -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DORT_INCLUDE_DIR=ort/headers -DORT_LIBRARY=ort/jni/arm64-v8a/libonnxruntime.so
cmake --build build
```

## Deploy and run

```bash
adb push build/onnxsim_rpc_server ort/jni/arm64-v8a/libonnxruntime.so /data/local/tmp/onnxsim_rpc/
adb shell "cd /data/local/tmp/onnxsim_rpc && chmod 755 onnxsim_rpc_server && \
  LD_LIBRARY_PATH=. ./onnxsim_rpc_server --host 127.0.0.1 --port 9090 --key phone \
  --workspace /data/local/tmp/onnxsim_rpc/ws &"
adb forward tcp:9090 tcp:9090
```

```python
import onnxsim.rpc as rpc
session = rpc.connect("127.0.0.1", 9090, key="phone")
print(session.info)  # platform, onnxruntime version, available providers (CPU/NNAPI/XNNPACK/...)
model = session.load_model("model.onnx", providers=["NnapiExecutionProvider", "CPUExecutionProvider"])
outputs = model.run({"x": x})
timing = model.time_evaluator({"x": x}, number=5, repeat=3)
```

## Status

Verified end-to-end against a real device (Xiaomi 12S, Android 13, onnxruntime-android 1.30.0):
`hello`/key handshake, `info`, `upload`, `load_model` (bytes, uploaded name, `providers`), `run`,
`time_evaluator`, and wrong-key rejection, all through the unmodified `onnxsim.rpc` Python client.
Execution providers exercised: `CPUExecutionProvider` (0.65 s median for the Mask R-CNN backbone
at 800x1088, matching desktop ONNX Runtime CPU bit-for-bit) and `NnapiExecutionProvider` (2.4 s;
NNAPI's own numerics differ from CPU by up to 0.09 mean-abs on the FPN feature maps, which is
NNAPI's fp16-ish internal precision on this device, not an integration bug).

Not implemented: the `runtime="tinygrad"` extension the Python server has (would need an
embedded interpreter; native code generation is a much larger undertaking; the phone's DSP is
reached through TVM's own Hexagon RPC path instead, see `../../../docs/tvm-hexagon-conv-transpose-handoff.md`),
TLS, and the tracker (the Python tracker works with a native server -- `register_with_tracker`'s
wire messages are the same JSON frames).

## Security

Same model as the Python server (see `docs/rpc.md`): binds to loopback by default, the key is an
identifier and not authentication, and it executes whatever ONNX model a client sends. `upload`
sanitises file names to stay inside the workspace directory.
