# Notes for Claude

## The Python wheel build does NOT build ONNX Runtime

Don't get confused by `ONNXSIM_BUILTIN_ORT` when working on the wheel build.

- `CMakeLists.txt` defaults `ONNXSIM_BUILTIN_ORT` to **ON**, which builds ONNX Runtime
  (from `third_party/onnxruntime-1.28.0`, via `cmake/build_ort.cmake`, fully out-of-tree
  -- see that file) and makes it available as a constant-folding backend
  (`GetBuiltinModelExecutor()`). That default is for the standalone C++/WASM builds,
  **not** the Python wheel.
- `setup.py` explicitly passes **`-DONNXSIM_BUILTIN_ORT=OFF`** when building the wheel, so
  ONNX Runtime is **never compiled** as part of `pip install` / wheel builds. The
  extension is compiled without `ONNXSIM_HAS_ORT`, which `#ifdef`s out all the `Ort::`
  C++ code (`GetBuiltinModelExecutor()`, `dlpack_bridge.h`'s `Ort::Value` glue).
- onnxsim's own optimizer/shape-inference pipeline (its `onnx`/`onnx-optimizer` fork
  under `third_party/onnx`) is **always** built, regardless of `ONNXSIM_BUILTIN_ORT` --
  it is not part of what that flag controls. When `ONNXSIM_BUILTIN_ORT=ON`, ONNX Runtime
  is built/linked as a fully separate, out-of-tree artifact specifically so its own
  (differently-versioned, vendored) onnx copy never enters onnxsim's own CMake target
  graph -- onnx's own `CMakeLists.txt` hardcodes its target names, so two onnx copies
  cannot coexist in one CMake project.
- At runtime, `onnxruntime` is only an **optional** Python dependency
  (`[project.optional-dependencies]` in `pyproject.toml`). onnxsim uses the pip-installed
  `onnxruntime` package for constant folding / correctness checking when present, and
  falls back to onnx's reference evaluator when it isn't.

So: **building ONNX Runtime is not required to build, test, or ship the Python wheel.**
If you see long ONNX Runtime C++ compilation, that's the `ONNXSIM_BUILTIN_ORT=ON` path
(standalone C++/WASM), not the wheel path.
