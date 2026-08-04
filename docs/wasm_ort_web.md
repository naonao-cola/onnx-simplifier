# WASM constant folding via onnxruntime-web (experimental)

**Status: experimental / work in progress.** Opt-in, `OFF` by default. The
default WebAssembly build is unchanged. This document describes the design and
how to build and test the variant.

## What this is

onnxsim's constant folding runs sub-models through a `ModelExecutor` abstraction
(`onnxsim/onnxsim.h`). There are two implementations upstream:

- `CppModelExecutor` — calls a statically linked ONNX Runtime C++ library. This
  is what the default WASM build uses (`ONNXSIM_BUILTIN_ORT=ON`), which compiles
  ONNX Runtime from source into onnxsim's own `.wasm`.
- `PyModelExecutor` — the Python "trampoline": `Run` calls back into the pip
  `onnxruntime` package, so the wheel links no ONNX Runtime C++
  (`NO_BUILTIN_ORT`).

The `ModelExecutor` boundary exchanges tensors as DLPack `DLManagedTensor`
(`onnxsim.h` / `onnxsim/dlpack_bridge.h`, see `docs/dlpack-executor.md`), so no
tensor is serialized to `TensorProto` across it.

This variant adds the WebAssembly analogue of the Python trampoline: a
`JsModelExecutor` whose `Run` delegates each fold group to the page's
**onnxruntime-web** module. Built this way, the onnxsim WASM module links **no**
ONNX Runtime:

- The slow from-source ONNX Runtime compile disappears from the build.
- The module shrinks — it no longer carries a second copy of ORT alongside the
  `onnxruntime-web` the converter page already loads for its inference panel.

## How it works

```
Simplify → RunOps → executor.Run(subModel, DLManagedTensor feeds)  (C++, onnxsim.cpp)
                        │
                        │  JsModelExecutor (js_model_executor.cpp)
                        │   - concat all feed bytes + a flat meta array (one copy each)
                        │   - Module.onnxsimOrtWebRun(modelBytes, inputsData, inputsMeta)
                        │   - .await() the returned Promise  ← needs Asyncify
                        ▼
        onnxsimOrtWebRun (ort_executor.mjs, makeOrtRunner)  (JS)
                        │   - ort.InferenceSession.create(modelBytes)
                        │   - session.run(feeds)
                        ▼
                 onnxruntime-web (wasm/WebGPU EP)
```

The C++↔JS contract (see `js_model_executor.cpp` and `ort_executor.mjs`) is
**batched**: all of a fold group's tensors cross in a single concatenated byte
blob plus one flat metadata array, in both directions, so the number of embind
round trips is O(1) rather than O(tensors × fields):

- **Input**: `onnxsimOrtWebRun(modelBytes, inputsData, inputsMeta)`, where
  `inputsData` is every feed's raw little-endian bytes concatenated and
  `inputsMeta` is a `Float64Array` of `[dtype, ndim, dims...]` per feed
  (`dtype` = ONNX `TensorProto.DataType`).
- **Output**: `{ data: Uint8Array, meta: Float64Array }` in the same layout.
- Tensors are **positional** — no names cross. Input i binds to
  `session.inputNames[i]`; outputs are emitted in `session.outputNames` order.
  Both equal the sub-model's graph input/output order, which is how the built-in
  executor maps feeds and how `RunOps` names returned tensors.

Supported dtypes match `CppModelExecutor`: FLOAT, DOUBLE, INT64, UINT64, INT32,
UINT8, INT8, UINT16, INT16, BOOL.

### The one hard part: sync C++ ↔ async JS

The Python trampoline is straightforward because `onnxruntime`'s `run()` is
synchronous. `onnxruntime-web` is asynchronous (`await create`, `await run`),
but `ModelExecutor::Run` is a synchronous call deep inside `Simplify`. The
bridge is **Asyncify** (`-sASYNCIFY`): `emscripten::val::await()` unwinds the
wasm stack at the `Run` call, lets the JS Promise settle, then rewinds. A
consequence is that the exported `onnxsimplify_export` becomes async (returns a
Promise); `worker.js` awaits it when `onnxsim_needs_ort_web()` is true.

Because the wasm heap can move/grow while suspended (`ALLOW_MEMORY_GROWTH=1`),
`Run` copies the model bytes and the batched input blob + metadata into JS-owned
buffers *before* the await, so nothing points into the wasm heap across the
suspend.

## Building

```sh
# default build (unchanged): compiles ONNX Runtime into the module
./build_wasm.sh

# experimental ORT-web variant: no ONNX Runtime compiled/linked
ORT_WEB=ON ./build_wasm.sh
```

`ORT_WEB=ON` skips the ONNX Runtime source download, configures with
`-DONNXSIM_WASM_ORT_WEB=ON`, and builds into `build-wasm-node-OFF-ortweb/`. The
CMake option forces `ONNXSIM_BUILTIN_ORT=OFF`, links `onnx` directly (ORT no
longer provides it transitively), compiles `js_model_executor.cpp`, and adds
the Asyncify link flags.

Deploy the resulting `onnxsim.js` / `onnxsim.wasm` next to the page as usual.
`worker.js` detects the variant at runtime via `onnxsim_needs_ort_web()`, loads
onnxruntime-web from the CDN, and registers `Module.onnxsimOrtWebRun` before the
first conversion. The built-in-ORT build reports `false` and the worker path is
byte-for-byte the old behavior.

## Testing

- The deployed convertmodel demo IS the ORT-web variant:
  `.github/workflows/static.yml` builds with `ORT_WEB=ON` and ships the module to
  GitHub Pages (production). Open that page, convert a model with constant folding
  enabled, and the folding runs through the page's onnxruntime-web. To get the same
  page built from a pull request, comment `/preview` on it — the workflow then
  deploys that PR's head to Cloudflare Pages and replies with the URL.
- Manual/local: build with `ORT_WEB=ON`, serve `scripts/convertmodel/`, convert a
  model with constant folding enabled, and confirm the output matches the
  built-in-ORT build for the same model.

## protobuf for wasm (resolved via ONNX_BUILD_CUSTOM_PROTOBUF)

The first `ORT_WEB=ON` build failed while compiling onnx's *own* libraries
(`onnx_proto`, `onnx`) with `onnx-ml.pb.h: unknown type name
'PROTOBUF_NAMESPACE_OPEN'`. The default WASM build relies on **ONNX Runtime to
build protobuf for the wasm target and hand it to onnx** (`build_ort.cmake` sets
`onnxruntime_USE_FULL_PROTOBUF` and `ONNX_TARGET_NAME onnxruntime_webassembly`).
Removing ORT (`ONNXSIM_BUILTIN_ORT=OFF`) also removes that wasm protobuf, so
onnx's generated `.pb.*` code fails to compile; `build_wasm.sh` only builds a
**host** protoc (for codegen), not a wasm protobuf runtime.

Fix: the `ONNXSIM_WASM_ORT_WEB` path sets `ONNX_BUILD_CUSTOM_PROTOBUF=ON`, so
onnx builds its own bundled protobuf cross-compiled for the wasm target. onnx
fetches a **pinned protobuf version** (25.1, per onnx's `sbom.cdx.json`) and
still uses a **host** protoc for codegen via `ONNX_CUSTOM_PROTOC_EXECUTABLE`
(passed by `build_wasm.sh` as `which protoc`). The host protoc must match that
version, or the generated `.pb.*` code is incompatible with the runtime headers
("generated by an older version of protoc"). The deploy workflow
(`static.yml`) therefore installs **protoc 25.1** rather than the distro's older
`protobuf-compiler`. When building locally, put a protoc matching onnx's pinned
protobuf on `PATH` before running `ORT_WEB=ON ./build_wasm.sh`.

## Not done yet / follow-ups

- Per-fold-group `InferenceSession.create` may dominate runtime; session reuse or
  coarser batching is a likely optimization.
- Only the dtypes above are bridged (same as the built-in executor); others throw
  a clear error.
- Not wired for the `ONNXSIM_WASM_NODE` (NODERAWFS) build or a Node smoke test yet.
- The `JsModelExecutor` C++ / `ort_executor.mjs` bridge compiles but has not been
  exercised at runtime yet; it needs a browser/Node folding test with
  onnxruntime-web loaded (the JS runner registered on the Module).
