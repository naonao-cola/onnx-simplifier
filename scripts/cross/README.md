# Cross-compiling Windows wheels on Linux

This directory builds an onnxsim **Windows** (x86_64) wheel from a **Linux**
host, so the expensive C++ compile can run on a cheap Ubuntu runner. The wheel
is then handed to a Windows runner purely for `pytest`.

`.github/workflows/windows-cross.yml` wires the two halves together:

```
build-on-linux (ubuntu-24.04)   ->   test-on-windows (windows-2025)
  llvm-mingw cross-build               pip install the wheel + run tests
```

## Why not cibuildwheel?

cibuildwheel only builds Windows wheels **on a Windows host** — it will not
cross-build them on Linux. So the Ubuntu job is a hand-written cross-compile
instead of a cibuildwheel invocation. The Windows job is a plain test runner.

## Toolchains (`BACKEND`)

Two toolchains are supported; the workflow uses **llvm-mingw**.

* **`llvm-mingw`** (default) — clang + lld + a self-contained UCRT/libc++
  sysroot ([llvm-mingw](https://github.com/mstorsjo/llvm-mingw)). No Microsoft
  SDK download; the C++ runtime is statically linked so the wheel ships no
  extra DLLs. Targets the UCRT, matching modern python.org CPython. The C++ ABI
  differs from MSVC, but that is invisible to CPython — only the extern-`"C"`
  `PyInit_*` entry point crosses the boundary, and every dependency (onnx,
  protobuf, abseil, nanobind) is built with this same toolchain. nanobind
  officially supports MinGW-w64.
* **`clang-cl`** — clang in MSVC mode against an MSVC CRT + Windows SDK fetched
  with [`xwin`](https://github.com/Jake-Shadle/xwin). Produces a genuine
  MSVC-ABI binary; kept as an alternative.

Select with `BACKEND=llvm-mingw` / `BACKEND=clang-cl`. Each has a matching
`windows-<backend>.toolchain.cmake`.

## How the cross-build works (`build_windows_wheel.sh`)

1. **Toolchain** — download llvm-mingw (or the MSVC SDK via xwin).
2. **Target CPython** headers + import libraries from the official CPython
   [NuGet package](https://www.nuget.org/packages/python). Its `python3.lib`
   imports the stable-ABI `python3.dll` (python-build-standalone's imports the
   versioned `pythonXY.dll` instead), so the abi3 build links `python3.lib` and
   produces a wheel that loads on any CPython >= 3.12.
3. **Host `protoc`** (a Linux protobuf build) runs ONNX's code generation via
   `ONNX_CUSTOM_PROTOC_EXECUTABLE`; a cross-built `protoc.exe` could not run on
   the Linux host. Built with the host compiler — the target toolchain is kept
   off `PATH` here, since llvm-mingw's `clang` defaults to a Windows target.
4. **Target abseil + protobuf** cross-built so ONNX has something to link.
   Versions come from ONNX's SBOM so the host `protoc` and target `libprotobuf`
   always match.
5. **onnxsim's nanobind extension** cross-built against all of the above. A
   small portability patch (`onnx-msvc-portability.patch`) is applied to the
   vendored onnx first: onnx's Windows code uses a reinterpret-cast pointer as a
   non-type template argument and a case-sensitive `<Windows.h>`, both of which
   only MSVC's `cl.exe` accepts — Clang (clang-cl and mingw alike) rejects them.
   The patch is applied idempotently at build time and reverts cleanly; it is a
   candidate for upstreaming to onnx.
6. **`assemble_wheel.py`** packs the `.pyd` into a correctly tagged wheel
   (setup.py cannot drive packaging when cross-compiling, as it assumes the
   host's extension suffix and build layout).

When `sccache` is on `PATH`, it is used as the CMake compiler launcher for every
stage, so object files (host protoc, target abseil/protobuf, onnx/onnxsim) are
cached across runs. The workflow provides it via `mozilla-actions/sccache-action`
with the GitHub Actions cache backend, matching `build-and-test.yml`; the first
run is a cold cache, later runs reuse it.

## Scope / status

* **Proof of concept** building a single **cp312-abi3** wheel and testing it on
  CPython **3.12 and 3.13** (one wheel, matrixed test). `build-and-test.yml`
  still builds the release Windows wheels natively — this path runs alongside it.
* The wheel is a genuine **abi3** (limited-API) module: nanobind sets
  `Py_LIMITED_API` and links nuget's `python3.lib`, so the `.pyd` imports
  `python3.dll` and loads on any CPython >= 3.12. Build a version-specific wheel
  instead with `ABI3=0` (links `pythonXY.dll`, tag `cp3XX-cp3XX`).
* onnxruntime is **not** compiled in (`-DONNXSIM_BUILTIN_ORT=OFF`, matching the
  pip build), keeping the cross-compile surface to onnx + protobuf + nanobind.

## Running it locally

Requires cmake, ninja, a host C/C++ compiler, `pip install nanobind`, and
network access (downloads llvm-mingw + the target CPython). For `clang-cl` you
additionally need `xwin` on `PATH`, which downloads the MSVC SDK from Microsoft.

```sh
BACKEND=llvm-mingw PYVER=3.12 ABI3=1 bash scripts/cross/build_windows_wheel.sh
# -> wheelhouse/onnxsim-<ver>-cp312-abi3-win_amd64.whl
```
