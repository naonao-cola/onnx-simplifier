# onnx-finetune-wasm

Same training loop as `../src/main.cpp` (the native CLI), compiled to
WebAssembly and exposed to JS via Embind instead of argv, so it can run
fine-tuning entirely client-side in a browser tab.

## Status

**The C++/Embind code and the JS-facing API are complete and self-contained.**
The build wiring (`CMakeLists.txt`) has **not been build-verified** -- see the
comment at the top of that file for why: it's a structural port of onnxsim's
own working wasm build (`../../CMakeLists.txt`'s `EMSCRIPTEN` branch,
`../../build_wasm.sh`) with `onnxruntime_ENABLE_TRAINING_APIS=ON` added, but
onnxsim's own wasm build has never turned that flag on, and upstream ONNX
Runtime's *own* wasm+training JS bindings were removed from its source tree
after v1.19.2. Whether `onnxruntime_webassembly` still comes out the other
end with training symbols intact when that flag is set is the one open
question -- it needs an actual build attempt to confirm, which is a
substantial compile (same ballpark as building ORT for native, likely more
given wasm's slower codegen).

Why still worth doing this way: the alternative -- reviving ORT's deleted
`onnxruntime-web/training` JS/TS layer -- means resurrecting and maintaining
someone else's abandoned code. This instead reuses only what's still alive
and maintained: the C++ training API (used natively, working, verified
elsewhere in this repo) and onnxsim's own proven Emscripten build recipe.
If the training+wasm CMake combination turns out not to build cleanly, the
fix is scoped to this one file, not a resurrection project.

## Design

Everything that can avoid touching a virtual filesystem does:
`CheckpointState::LoadCheckpointFromBuffer` and the `std::vector<uint8_t>`
overload of the `TrainingSession` constructor take the four artifact files
directly as bytes, so the JS side just needs `fetch()` + `ArrayBuffer` --
no `FS.writeFile` staging. The one exception is `ExportModelForInferencing`,
which only has a path-based signature upstream in this ORT version; that
single write goes through Emscripten's in-memory MEMFS (never a real disk)
and gets read straight back into a `Uint8Array` to hand to JS.

`FinetuneSession` (Embind class, `src/onnx_finetune_wasm.cpp`) mirrors the
native CLI's loop one-for-one:

| native CLI | wasm binding |
|---|---|
| load artifacts from `--artifacts-dir` | `new Module.FinetuneSession(checkpointBytes, trainingModelBytes, evalModelBytes, optimizerModelBytes)` |
| `--lr` | `session.setLearningRate(lr)` |
| `TrainStep` + loss print | `session.trainStep(inputFloat32Array, targetFloat32Array, batch, inputDim, targetDim)` -> loss |
| `OptimizerStep` | `session.optimizerStep()` |
| `LazyResetGrad` | `session.lazyResetGrad()` |
| `ExportModelForInferencing` + `--output-names` | `session.exportModel(['name1', ...])` -> `Uint8Array` |

Batch construction, shuffling, and the epoch loop live in JS
(`example/app.js`) rather than C++, same division of responsibility as the
native CLI (C++ owns the training step, the caller owns the data loop).

## Building

Requires `emcmake`/`emcc` (from emsdk) and a host `protoc` binary (protobuf's
codegen runs on the host during a cross build, same requirement as
`../../build_wasm.sh` has for onnxsim itself):

```sh
# from an onnxruntime source checkout with training support (same one used
# for the native build, see ../README.md)
PROTOC=$(which protoc)  # or build one, see ../../build_wasm.sh

emcmake cmake -B build \
  -DORT_SOURCE_DIR=/path/to/onnxruntime \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -t onnx-finetune-wasm
```

Output: `build/onnx-finetune-wasm.js` + `build/onnx-finetune-wasm.wasm`.
Copy both into `example/` (or adjust the `<script src=...>` path in
`example/index.html`) to run the demo.

## Trying the example

Generate the same toy artifacts the native CLI's example uses, alongside
`example/`:

```sh
cd example
python3 ../../scripts/make_toy_model.py -o toy_model.onnx
python3 ../../scripts/make_synthetic_data.py --num-samples 2048
PYTHONPATH=/path/to/onnxruntime/build/Release/build/lib \
  python3 ../../scripts/generate_artifacts.py toy_model.onnx -o artifacts
```

Then serve `example/` with any static file server (needs no special
headers -- this build has wasm threads off) and open `index.html`. It
should show the same loss curve as the native CLI's toy example (roughly
5 -> under 0.001 over 20 epochs) and offer a `finetuned.onnx` download at
the end.

## Memory

wasm32's linear memory is capped at 4 GiB (see the `MAXIMUM_MEMORY` comment
in `../../CMakeLists.txt` for onnxsim's own reasoning on this same limit).
Measurements against the native build earlier in this project's history put
full AdamW fine-tuning at roughly 4x a model's raw fp32 weight size in RSS --
so this comfortably covers small-to-medium models (a few hundred million
parameters) but rules out anything approaching billion-parameter scale
in-browser.
