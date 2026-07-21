# Cross-compiling Windows wheels on Linux

This directory builds an onnxsim **Windows** (x86_64, MSVC ABI) wheel from a
**Linux** host, so the expensive C++ compile can run on a cheap Ubuntu runner.
The wheel is then handed to a Windows runner purely for `pytest`.

`.github/workflows/windows-cross.yml` wires the two halves together:

```
build-on-linux (ubuntu-24.04)   ->   test-on-windows (windows-2025)
  clang-cl + xwin cross-build          pip install the wheel + run tests
```

## Why not cibuildwheel?

cibuildwheel only builds Windows wheels **on a Windows host** — it will not
cross-build them on Linux. So the Ubuntu job is a hand-written cross-compile
instead of a cibuildwheel invocation. The Windows job is a plain test runner.

## How the cross-build works (`build_windows_wheel.sh`)

The extension is MSVC-ABI, produced with `clang-cl` + `lld-link` targeting
`x86_64-pc-windows-msvc`, so it loads under the official python.org CPython.

1. **MSVC CRT + Windows SDK** are downloaded with
   [`xwin`](https://github.com/Jake-Shadle/xwin) — no Visual Studio needed.
2. **Target CPython** headers + import libraries come from
   [python-build-standalone](https://github.com/astral-sh/python-build-standalone);
   abi3 wheels link `python3.lib`, version-specific wheels link `pythonXY.lib`.
3. **Host `protoc`** (a Linux protobuf build) runs ONNX's code generation. ONNX
   supports this via `ONNX_CUSTOM_PROTOC_EXECUTABLE`; a cross-built `protoc.exe`
   could not run on the Linux host.
4. **Target abseil + protobuf** are cross-built so ONNX has something to link
   against. Versions are read from ONNX's SBOM so the host `protoc` and the
   target `libprotobuf` always match.
5. **onnxsim's nanobind extension** is cross-built against all of the above.
6. **`assemble_wheel.py`** packs the `.pyd` into a correctly tagged wheel
   (setup.py cannot drive packaging when cross-compiling, as it assumes the
   host's extension suffix and build layout).

`windows-clang-cl.toolchain.cmake` is the CMake toolchain that points clang-cl
at the xwin sysroot; it is reused for every CMake sub-build.

## Scope / status

* Currently a **proof of concept** limited to CPython **3.12 (abi3)**; that
  wheel also serves 3.13+. `build-and-test.yml` still builds the release
  Windows wheels natively — this path runs alongside it, it does not replace it.
* To extend to 3.10 / 3.11, run the script with `PYVER=3.10 ABI3=0` etc. and
  add matching test-matrix entries.
* onnxruntime is **not** compiled in (`-DONNXSIM_BUILTIN_ORT=OFF`, matching the
  pip build), which keeps the cross-compile surface to onnx + protobuf +
  nanobind.

## Running it locally

Requires clang-cl/lld-link (LLVM), cmake, ninja, `pip install nanobind`, and
`xwin` on `PATH`. Note `xwin` downloads the MSVC SDK from Microsoft's servers,
so it needs unrestricted network access.

```sh
PYVER=3.12 ABI3=1 bash scripts/cross/build_windows_wheel.sh
# -> wheelhouse/onnxsim-<ver>-cp312-abi3-win_amd64.whl
```
