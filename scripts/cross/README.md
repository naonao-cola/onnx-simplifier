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
2. **Target CPython** headers + import libraries from
   [python-build-standalone](https://github.com/astral-sh/python-build-standalone);
   abi3 wheels link `python3.lib`, version-specific wheels link `pythonXY.lib`.
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

## Scope / status

* **Proof of concept** covering CPython **3.12 and 3.13**, one version-specific
  wheel each (matrixed in the workflow). `build-and-test.yml` still builds the
  release Windows wheels natively — this path runs alongside it.
* Wheels are **version-specific, not abi3**: python-build-standalone's
  `python3.lib` imports the versioned `pythonXY.dll` rather than the stable-ABI
  `python3.dll`, so a single abi3 wheel cannot span versions with these import
  libraries. To add 3.10 / 3.11, add matrix entries (the script already accepts
  any `PYVER`).
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
