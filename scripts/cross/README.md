# Cross-compiling Windows wheels on Linux

This directory builds an onnxsim **Windows** (x86_64) wheel from a **Linux**
host, so the expensive C++ compile can run on a cheap Ubuntu runner. The wheel
is then handed to a Windows runner purely for `pytest`.

`.github/workflows/build-and-test.yml` wires the two halves together (the
`build_wheels_windows_cross` and `test_wheels_windows_cross` jobs), producing
the Windows release wheels in place of a native Windows build:

```
build_wheels_windows_cross (ubuntu-24.04)  ->  test_wheels_windows_cross (windows-2025)
  llvm-mingw cross-build                         uv install the wheel + run tests
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

* Covers the Windows CPython versions shipped by `build-and-test.yml`:
  **3.10, 3.11, 3.12, 3.13**. This cross-build **is** the Windows release path —
  it replaces the native Windows build, and its wheels are published to PyPI by
  the `upload_pypi` job alongside the natively-built Linux/macOS wheels.
* **3.12 and 3.13 share one `cp312-abi3` wheel**: a genuine limited-API module
  (nanobind sets `Py_LIMITED_API` and links nuget's `python3.lib`, so the `.pyd`
  imports `python3.dll`). **3.10 and 3.11** are below nanobind's abi3 floor
  (3.12), so they get **version-specific** wheels (`ABI3=0`, tag `cp3XX-cp3XX`,
  linking `pythonXY.dll`).
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

---

# Cross-building for s390x (big endian)

The second cross-build in this directory targets **s390x Linux**, for a
different reason than the Windows one: not to ship wheels, but to run onnxsim's
test suite on a **big-endian** CPU. ONNX defines `TensorProto.raw_data` as
little-endian on every host, so any code that reinterprets those bytes in host
order is a latent bug that only x86/ARM's byte order hides. s390x is the only
mainstream big-endian Linux target a major distro still supports, and Ubuntu
ships `python3-onnx` / `python3-numpy` / `python3-pytest` for it — which is what
makes this practical, since PyPI has no s390x wheels.

Everything is compiled by the **host** toolchain (`s390x-linux-gnu-g++`), so the
build runs at native speed; only the resulting binaries execute under
`qemu-s390x-static`. An emulated build of the onnx + protobuf stack would take
hours.

## Procedure

```sh
# 1. Ubuntu s390x rootfs + binfmt_misc -> qemu-s390x-static  (root; once)
sudo bash scripts/cross/bootstrap_s390x_rootfs.sh

# 2. Cross-build the extension (host protoc, target abseil/protobuf, onnxsim)
bash scripts/cross/build_s390x_extension.sh

# 3. Install it into the rootfs and run pytest there, big endian
sudo bash scripts/cross/run_s390x_tests.sh
```

## The little-endian control

A big-endian test run on its own is hard to read: a failure may be an
endianness bug, or just an artifact of this unusual environment (no
onnxruntime, no torch, an older distro onnx). So build the **same commit** for
x86_64 and run the **same suite** in an amd64 rootfs with the **same package
versions**. Then byte order is the only variable and the two failure sets can be
diffed directly:

```sh
sudo ARCH=amd64 bash scripts/cross/bootstrap_s390x_rootfs.sh
bash scripts/cross/build_native_control.sh
sudo SYSROOT=/rootfs-amd64 BUILD=$PWD/.native-build-control/onnxsim-build \
     bash scripts/cross/run_s390x_tests.sh
```

## Notes / gotchas

* **binfmt magic must mask `EI_OSABI`.** glibc's own binaries (notably
  `ldconfig.real`) are tagged `ELFOSABI_GNU`, so a handler that pins byte 7 to
  SYSV makes exactly those fail with `Exec format error` part-way through
  `debootstrap`, corrupting the dpkg status file. The mask also has to accept
  `ET_DYN` for static-pie binaries.
* **The host interpreter must match the target CPython's X.Y.** CMake's
  `FindPython` validates the interpreter against the target headers and refuses
  a mismatch, even though the interpreter only runs ONNX's codegen.
* **The distro's `python3-onnx` is too old** for onnxsim's vendored onnx and
  fails a large part of the suite for reasons unrelated to byte order.
  `run_s390x_tests.sh` therefore installs the vendored onnx (assembled from the
  same cross-build) ahead of it on `PYTHONPATH`. protobuf has a pure-Python
  wheel, so it needs no s390x build; `ml_dtypes` does not, and is compiled once
  inside the rootfs under emulation by the bootstrap script.
* onnxruntime, torch and timm have no s390x builds, so the test modules that
  import them at module scope cannot be collected. onnxsim falls back to onnx's
  reference evaluator, which is the supported no-onnxruntime path.

## In CI

`.github/workflows/big-endian.yml` runs all three steps plus `ctest`. It is
weekly, on demand, and on pull requests touching the harness or the files that
read and write `raw_data` — a cold run is ~40 minutes, too much for every PR.
sccache carries the host protoc and target abseil/protobuf object files across
runs, and the one step the host toolchain cannot take over (compiling
`ml_dtypes` inside the rootfs under emulation) has its wheel cached separately
via `WHEELHOUSE`.

See `docs/big-endian.md` for what these runs found, and for the one known
failure the job deselects by name.
