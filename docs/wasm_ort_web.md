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
- `PyModelExecutor` — the Python "trampoline": `_Run` calls back into the pip
  `onnxruntime` package, so the wheel links no ONNX Runtime C++
  (`NO_BUILTIN_ORT`).

This variant adds the WebAssembly analogue of the Python trampoline: a
`JsModelExecutor` whose `_Run` delegates each fold group to the page's
**onnxruntime-web** module. Built this way, the onnxsim WASM module links **no**
ONNX Runtime:

- The slow from-source ONNX Runtime compile disappears from the build.
- The module shrinks — it no longer carries a second copy of ORT alongside the
  `onnxruntime-web` the converter page already loads for its inference panel.

## How it works

```
Simplify → RunOps → executor._Run(subModel, inputs)      (C++, onnxsim.cpp)
                        │
                        │  JsModelExecutor (js_model_executor.cpp)
                        │   - serialize subModel + inputs to JS-owned buffers
                        │   - Module.onnxsimOrtWebRun(modelBytes, inputs)
                        │   - .await() the returned Promise  ← needs Asyncify
                        ▼
        onnxsimOrtWebRun (ort_executor.mjs, makeOrtRunner)  (JS)
                        │   - ort.InferenceSession.create(modelBytes)
                        │   - session.run(feeds)
                        ▼
                 onnxruntime-web (wasm/WebGPU EP)
```

The C++↔JS contract (see `js_model_executor.cpp` and `ort_executor.mjs`):

- **Input** `{ name, dataType, dims: number[], data: Uint8Array }` per feed,
  where `dataType` is the ONNX `TensorProto.DataType` enum value and `data` is
  raw little-endian element bytes.
- **Output** is a map `{ [name]: { dataType, dims, data } }`. `JsModelExecutor`
  reads it back in `graph().output()` order, because `RunOps` names the returned
  tensors positionally in that order.

Supported dtypes match `CppModelExecutor`: FLOAT, DOUBLE, INT64, UINT64, INT32,
UINT8, INT8, UINT16, INT16, BOOL.

### The one hard part: sync C++ ↔ async JS

The Python trampoline is straightforward because `onnxruntime`'s `run()` is
synchronous. `onnxruntime-web` is asynchronous (`await create`, `await run`),
but `ModelExecutor::_Run` is a synchronous call deep inside `Simplify`. The
bridge is **Asyncify** (`-sASYNCIFY`): `emscripten::val::await()` unwinds the
wasm stack at the `_Run` call, lets the JS Promise settle, then rewinds. A
consequence is that the exported `onnxsimplify_export` becomes async (returns a
Promise); `worker.js` awaits it when `onnxsim_needs_ort_web()` is true.

Because the wasm heap can move/grow while suspended (`ALLOW_MEMORY_GROWTH=1`),
`_Run` copies the model bytes and every input into JS-owned `Uint8Array`s
*before* the await, so nothing points into the wasm heap across the suspend.

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

- The default build and its CI (`.github/workflows/static.yml`,
  `convertmodel-inference.yml`) are unaffected — this variant is off by default.
- Manual: build with `ORT_WEB=ON`, serve `scripts/convertmodel/`, convert a
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
onnx builds its own bundled protobuf cross-compiled for the wasm target. The
host protoc is still used for codegen via `ONNX_CUSTOM_PROTOC_EXECUTABLE` (passed
by `build_wasm.sh`), so the generated code and the runtime protobuf come from the
matching version.

## Not done yet / follow-ups

- Per-fold-group `InferenceSession.create` may dominate runtime; session reuse or
  coarser batching is a likely optimization.
- Only the dtypes above are bridged (same as the built-in executor); others throw
  a clear error.
- Not wired for the `ONNXSIM_WASM_NODE` (NODERAWFS) build or a Node smoke test yet.
- The `JsModelExecutor` C++ / `ort_executor.mjs` bridge has not been exercised
  yet because the build fails earlier at the protobuf step above.
