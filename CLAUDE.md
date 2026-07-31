# Notes for Claude

## The Python wheel build does NOT build ONNX Runtime

Don't get confused by `ONNXSIM_BUILTIN_ORT` when working on the wheel build.

- `CMakeLists.txt` defaults `ONNXSIM_BUILTIN_ORT` to **ON**, which would compile the
  vendored ONNX Runtime C++ source under `third_party/onnxruntime-1.28.0` (via
  `cmake/build_ort.cmake`). That default is for the standalone C++/WASM builds, **not**
  the Python wheel.
- `setup.py` explicitly passes **`-DONNXSIM_BUILTIN_ORT=OFF`** when building the wheel, so
  the vendored ONNX Runtime is **never compiled** as part of `pip install` / wheel builds.
  The extension is compiled with the `NO_BUILTIN_ORT` define, which `#ifdef`s out all the
  `Ort::` C++ code in `onnxsim/onnxsim.cpp` (and elsewhere).
- At runtime, `onnxruntime` is only an **optional** Python dependency
  (`[project.optional-dependencies]` in `pyproject.toml`). onnxsim uses the pip-installed
  `onnxruntime` package for constant folding / correctness checking when present, and
  falls back to onnx's reference evaluator when it isn't.

So: **building or vendoring the ONNX Runtime C++ library is not required to build, test, or
ship the Python wheel.** If you see long ONNX Runtime C++ compilation, that's the
`ONNXSIM_BUILTIN_ORT=ON` path (standalone C++/WASM), not the wheel path.
