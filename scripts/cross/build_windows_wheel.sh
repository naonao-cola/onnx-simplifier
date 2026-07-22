#!/usr/bin/env bash
#
# Cross-compile an onnxsim Windows (x86_64) wheel from Linux.
#
# This produces a wheel that is binary-compatible with the official python.org
# CPython on Windows, without ever running a Windows machine.  It exists so the
# expensive C++ build can run on a cheap Linux runner; the wheel is then handed
# to a Windows runner purely for `pytest` (see .github/workflows/windows-cross.yml).
#
# Two toolchains are supported via BACKEND:
#   llvm-mingw  clang + lld + a self-contained UCRT/libc++ sysroot (default).
#               No Microsoft SDK download; statically links the C++ runtime.
#   clang-cl    clang in MSVC mode against an MSVC CRT + Windows SDK from xwin.
#
# High-level stages:
#   1. Set up the target toolchain (llvm-mingw download, or xwin MSVC SDK).
#   2. Download the *target* Windows CPython (for its headers + python3.lib).
#   3. Build a *host* protoc (Linux) so ONNX can run code generation.
#   4. Cross-build abseil + protobuf for the *target* (Windows) so ONNX can link.
#   5. Cross-build onnxsim's nanobind extension against all of the above.
#   6. Pack the resulting .pyd into a correctly tagged wheel.
#
# Dependency versions are pinned to whatever the vendored ONNX pins (see the
# SBOM in third_party/onnx-optimizer/third_party/onnx/sbom.cdx.json); a mismatch
# between the host protoc and the target libprotobuf would break the build.
#
# Required environment:
#   PYVER          Target CPython version, e.g. "3.12".
#   ABI3           "1" for a limited-API (abi3) wheel (>=3.12 only), else "0".
# Optional environment:
#   BACKEND        "llvm-mingw" (default) or "clang-cl".
#   WORK           Scratch directory (default: $PWD/.cross-build).
#   PROTOBUF_VER / ABSL_VER / NANOBIND_VER   Override versions (default: SBOM).
#   JOBS           Parallel build jobs (default: nproc).
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
: "${PYVER:?set PYVER, e.g. 3.12}"
: "${ABI3:=0}"
: "${BACKEND:=llvm-mingw}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}/.cross-build}"
JOBS="${JOBS:-$(nproc)}"
ARCH="x86_64"

ONNX_DIR="${REPO_ROOT}/third_party/onnx-optimizer/third_party/onnx"
SBOM="${ONNX_DIR}/sbom.cdx.json"

sbom_ver() {  # sbom_ver <component-name>
  python3 - "$SBOM" "$1" <<'PY'
import json, sys
sbom, name = sys.argv[1], sys.argv[2]
data = json.load(open(sbom))
for c in data.get("components", []):
    if c.get("name") == name:
        print(c["version"]); break
else:
    sys.exit(f"component {name} not found in SBOM")
PY
}

PROTOBUF_VER="${PROTOBUF_VER:-$(sbom_ver protobuf)}"
ABSL_VER="${ABSL_VER:-$(sbom_ver abseil-cpp)}"
NANOBIND_VER="${NANOBIND_VER:-$(sbom_ver nanobind)}"

# Target Windows CPython headers + import libraries come from the official
# CPython NuGet package. Crucially its python3.lib imports the stable-ABI
# python3.dll (unlike python-build-standalone, whose python3.lib imports the
# versioned pythonXY.dll) -- required for a genuine abi3 wheel. Pin per version.
declare -A NUGET_PYFULL_MAP=(
  ["3.10"]="3.10.11"
  ["3.11"]="3.11.9"
  ["3.12"]="3.12.10"
  ["3.13"]="3.13.14"
)
NUGET_PYFULL="${NUGET_PYFULL:-${NUGET_PYFULL_MAP[$PYVER]:?no nuget pin for $PYVER}}"

# llvm-mingw release (used when BACKEND=llvm-mingw).
LLVM_MINGW_REL="${LLVM_MINGW_REL:-20250709}"

echo "== onnxsim Windows cross-build =="
echo "   backend       : ${BACKEND}"
echo "   target python : ${PYVER} (nuget ${NUGET_PYFULL}), abi3=${ABI3}"
echo "   protobuf      : ${PROTOBUF_VER}"
echo "   abseil-cpp    : ${ABSL_VER}"
echo "   nanobind      : ${NANOBIND_VER}"
echo "   work dir      : ${WORK}"

XWIN_DIR="${WORK}/xwin"
LLVM_MINGW_DIR="${WORK}/llvm-mingw"
DEPS_HOST="${WORK}/deps-host"     # host (Linux) protobuf install -> protoc
DEPS_TARGET="${WORK}/deps-target" # target (Windows) absl + protobuf install
WINPY="${WORK}/winpy"             # target Windows CPython
DL="${WORK}/dl"
mkdir -p "${WORK}" "${DL}"

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ ! -f "${dest}" ]]; then
    echo ">> downloading ${url}"
    curl -fSL --retry 4 --retry-delay 2 -o "${dest}.tmp" "${url}"
    mv "${dest}.tmp" "${dest}"
  fi
}

# Use sccache as the compiler launcher for every stage when available, so the
# host protoc, target abseil/protobuf, and onnx/onnxsim object files are cached
# across runs (matching build-and-test.yml). The workflow provides sccache via
# mozilla-actions/sccache-action with the GitHub Actions cache backend.
SCCACHE_ARGS=()
if command -v sccache >/dev/null; then
  SCCACHE_ARGS=( -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache )
  sccache --start-server >/dev/null 2>&1 || true
  echo "sccache enabled: $(command -v sccache)"
fi

# ---------------------------------------------------------------------------
# 1. Target toolchain
#
# TOOLCHAIN_ARGS carries the backend-specific -D flags handed to every target
# CMake configure (deps in stage 4 and onnxsim in stage 5).
# ---------------------------------------------------------------------------
TOOLCHAIN_ARGS=( -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/scripts/cross/windows-${BACKEND}.toolchain.cmake" )

case "${BACKEND}" in
  clang-cl)
    if [[ ! -d "${XWIN_DIR}/crt" ]]; then
      echo "== [1/6] downloading MSVC CRT + Windows SDK with xwin =="
      command -v xwin >/dev/null || { echo "xwin not on PATH"; exit 1; }
      xwin --accept-license --arch x86_64 splat --include-debug-libs --output "${XWIN_DIR}"
    fi
    TOOLCHAIN_ARGS+=( -DXWIN_DIR="${XWIN_DIR}" )
    ;;
  llvm-mingw)
    if [[ ! -d "${LLVM_MINGW_DIR}/bin" ]]; then
      echo "== [1/6] downloading llvm-mingw ${LLVM_MINGW_REL} =="
      # ucrt = link the Universal CRT, matching modern python.org CPython.
      mm_url="https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_REL}/llvm-mingw-${LLVM_MINGW_REL}-ucrt-ubuntu-22.04-${ARCH}.tar.xz"
      fetch "${mm_url}" "${DL}/llvm-mingw.tar.xz"
      mkdir -p "${LLVM_MINGW_DIR}"
      tar -C "${LLVM_MINGW_DIR}" --strip-components=1 -xf "${DL}/llvm-mingw.tar.xz"
    fi
    # Reference the toolchain compilers by absolute path; do NOT put this bin
    # dir on PATH before the host protoc build in stage 3, because llvm-mingw's
    # `clang` defaults to a Windows target and would miscompile the host tool.
    TOOLCHAIN_ARGS+=(
      -DMINGW_C_COMPILER="${LLVM_MINGW_DIR}/bin/clang"
      -DMINGW_CXX_COMPILER="${LLVM_MINGW_DIR}/bin/clang++"
      -DMINGW_RC="${LLVM_MINGW_DIR}/bin/llvm-rc"
      -DMINGW_AR="${LLVM_MINGW_DIR}/bin/llvm-ar"
      -DMINGW_RANLIB="${LLVM_MINGW_DIR}/bin/llvm-ranlib"
    )
    ;;
  *)
    echo "unknown BACKEND '${BACKEND}' (expected llvm-mingw or clang-cl)"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 2. Target Windows CPython (headers + python3.lib / pythonXY.lib)
# ---------------------------------------------------------------------------
if [[ ! -d "${WINPY}/tools" ]]; then
  echo "== [2/6] fetching target Windows CPython ${NUGET_PYFULL} (nuget) =="
  fetch "https://api.nuget.org/v3-flatcontainer/python/${NUGET_PYFULL}/python.${NUGET_PYFULL}.nupkg" "${DL}/winpy.nupkg"
  mkdir -p "${WINPY}"
  ( cd "${WINPY}" && unzip -oq "${DL}/winpy.nupkg" 'tools/include/*' 'tools/libs/*' )
fi
WINPY_INC="${WINPY}/tools/include"
WINPY_LIBDIR="${WINPY}/tools/libs"
PYVER_NODOT="${PYVER//./}"
# Development.Module needs the versioned import lib (FindPython validates the
# library version against the interpreter). Development.SABIModule uses
# python3.lib. For an abi3 build nanobind links only the SABI lib, and nuget's
# python3.lib imports python3.dll -- so the .pyd is a genuine abi3 module that
# loads on any CPython >= 3.12 (the nanobind Py_LIMITED_API floor).
WINPY_MODULE_LIB="${WINPY_LIBDIR}/python${PYVER_NODOT}.lib"  # Development.Module
WINPY_SABI_LIB="${WINPY_LIBDIR}/python3.lib"                 # Development.SABIModule
for f in "${WINPY_MODULE_LIB}" "${WINPY_SABI_LIB}"; do
  [[ -f "${f}" ]] || { echo "missing ${f}"; ls -la "${WINPY_LIBDIR}"; exit 1; }
done

# ---------------------------------------------------------------------------
# 3. Host protoc (Linux) -- ONNX runs this during code generation.
#    Built with the host compiler; the target toolchain is NOT on PATH here.
# ---------------------------------------------------------------------------
HOST_PROTOC="${DEPS_HOST}/bin/protoc"
if [[ ! -x "${HOST_PROTOC}" ]]; then
  echo "== [3/6] building host protobuf ${PROTOBUF_VER} (for protoc) =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-host-src" "${WORK}/protobuf-host-src"
  mkdir -p "${WORK}/absl-host-src" "${WORK}/protobuf-host-src"
  tar -C "${WORK}/absl-host-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-host-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  cmake -S "${WORK}/absl-host-src" -B "${WORK}/absl-host-build" -G Ninja \
    "${SCCACHE_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/absl-host-build" --target install -j "${JOBS}"

  cmake -S "${WORK}/protobuf-host-src" -B "${WORK}/protobuf-host-build" -G Ninja \
    "${SCCACHE_ARGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="${DEPS_HOST}" -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/protobuf-host-build" --target install -j "${JOBS}"
fi
"${HOST_PROTOC}" --version

# From here on the target toolchain may be on PATH (needed so `-fuse-ld=lld`
# and llvm-rc/llvm-ar resolve for llvm-mingw target builds).
if [[ "${BACKEND}" == "llvm-mingw" ]]; then
  export PATH="${LLVM_MINGW_DIR}/bin:${PATH}"
fi

# ---------------------------------------------------------------------------
# 4. Target abseil + protobuf (Windows) -- ONNX links these
# ---------------------------------------------------------------------------
if [[ ! -f "${DEPS_TARGET}/.done" ]]; then
  echo "== [4/6] cross-building abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} for Windows =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  mkdir -p "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  tar -C "${WORK}/absl-tgt-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-tgt-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  common_target_args=(
    -G Ninja
    "${TOOLCHAIN_ARGS[@]}"
    "${SCCACHE_ARGS[@]}"
    -DCMAKE_BUILD_TYPE=Release
    # ONNX links protobuf statically (onnxsim CI: ONNX_USE_PROTOBUF_SHARED_LIBS=OFF).
    -DBUILD_SHARED_LIBS=OFF
  )

  cmake -S "${WORK}/absl-tgt-src" -B "${WORK}/absl-tgt-build" "${common_target_args[@]}" \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${DEPS_TARGET}"
  cmake --build "${WORK}/absl-tgt-build" --target install -j "${JOBS}"

  cmake -S "${WORK}/protobuf-tgt-src" -B "${WORK}/protobuf-tgt-build" "${common_target_args[@]}" \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -DCMAKE_PREFIX_PATH="${DEPS_TARGET}" -DCMAKE_INSTALL_PREFIX="${DEPS_TARGET}"
  cmake --build "${WORK}/protobuf-tgt-build" --target install -j "${JOBS}"
  touch "${DEPS_TARGET}/.done"
fi

# ---------------------------------------------------------------------------
# 5. Cross-build the onnxsim extension
# ---------------------------------------------------------------------------
echo "== [5/6] cross-building onnxsim extension =="

# Apply the onnx MSVC-portability patch: onnx's Windows code uses constructs only
# cl.exe accepts (a reinterpret-cast pointer as a non-type template argument, and
# a case-sensitive <Windows.h>), which Clang -- clang-cl and mingw alike --
# rejects when cross-compiling. Applied idempotently so local re-runs are safe.
ONNX_PATCH="${REPO_ROOT}/scripts/cross/onnx-msvc-portability.patch"
if git -C "${ONNX_DIR}" apply --reverse --check "${ONNX_PATCH}" 2>/dev/null; then
  echo "onnx portability patch already applied"
elif git -C "${ONNX_DIR}" apply --check "${ONNX_PATCH}" 2>/dev/null; then
  git -C "${ONNX_DIR}" apply "${ONNX_PATCH}"
  echo "applied onnx portability patch"
else
  echo "ERROR: ${ONNX_PATCH} does not apply cleanly (onnx version drift?)"; exit 1
fi

NANOBIND_CMAKE_DIR="$(python3 -m nanobind --cmake_dir)"

BUILD_DIR="${WORK}/onnxsim-build-${PYVER}"
if [[ "${ABI3}" == "1" ]]; then BUILD_DIR="${BUILD_DIR}-abi3"; fi
rm -rf "${BUILD_DIR}"

HOSTPY="$(command -v python3)"
# onnx splits the Python roles when cross-compiling: the Python3 package supplies
# the target dev libraries (Development.Module + Development.SABIModule) while the
# Python package supplies the host interpreter. onnxsim/onnx-optimizer also use
# the Python package for target dev libs. Provide hints for both namespaces:
# host interpreter + target include + target module lib (pythonXY.lib) + target
# SABI lib (python3.lib).
python_args=(
  -DPython_EXECUTABLE="${HOSTPY}"
  -DPython_INCLUDE_DIR="${WINPY_INC}"
  -DPython_LIBRARY="${WINPY_MODULE_LIB}"
  -DPython_SABI_LIBRARY="${WINPY_SABI_LIB}"
  -DPython3_EXECUTABLE="${HOSTPY}"
  -DPython3_INCLUDE_DIR="${WINPY_INC}"
  -DPython3_LIBRARY="${WINPY_MODULE_LIB}"
  -DPython3_SABI_LIBRARY="${WINPY_SABI_LIB}"
)

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
  "${TOOLCHAIN_ARGS[@]}" \
  "${SCCACHE_ARGS[@]}" \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="${REPO_ROOT}/scripts/cross/find_python_early.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_BUILD_PYTHON=ON \
  -DONNX_INSTALL=OFF \
  -DONNXSIM_PYTHON=ON \
  -DONNXSIM_BUILTIN_ORT=OFF \
  -DONNX_USE_LITE_PROTO=OFF \
  -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="${HOST_PROTOC}" \
  -DProtobuf_PROTOC_EXECUTABLE="${HOST_PROTOC}" \
  -DCMAKE_PREFIX_PATH="${DEPS_TARGET};${NANOBIND_CMAKE_DIR}" \
  -Dnanobind_DIR="${NANOBIND_CMAKE_DIR}" \
  "${python_args[@]}"

cmake --build "${BUILD_DIR}" --target onnxsim_cpp2py_export -j "${JOBS}"

# Locate the built extension. abi3 modules are named .pyd, but for a
# version-specific (non-abi3) cross build nanobind derives the suffix from the
# HOST interpreter (e.g. .cpython-310-x86_64-linux-gnu.so) even though the file
# is a Windows PE built by the mingw toolchain. Accept either; assemble_wheel
# renames it to onnxsim_cpp2py_export.pyd.
PYD="$(find "${BUILD_DIR}" \( -name 'onnxsim_cpp2py_export*.pyd' -o -name 'onnxsim_cpp2py_export*.so' \) -print -quit)"
[[ -n "${PYD}" ]] || { echo "no extension module produced"; find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*'; exit 1; }
echo "built: ${PYD}"

# Report which DLLs the module imports (expects pythonXY.dll for the target
# version, the UCRT api-ms-win-crt-* set, and system DLLs -- no libstdc++/libgcc).
if [[ "${BACKEND}" == "llvm-mingw" ]]; then
  echo "== ${PYD##*/} DLL imports =="
  "${LLVM_MINGW_DIR}/bin/llvm-readobj" --coff-imports "${PYD}" 2>/dev/null | grep -iE "Name:.*\.dll" || true
fi

# ---------------------------------------------------------------------------
# 6. Pack the wheel
# ---------------------------------------------------------------------------
echo "== [6/6] packing wheel =="
python3 "${REPO_ROOT}/scripts/cross/assemble_wheel.py" \
  --repo-root "${REPO_ROOT}" \
  --pyd "${PYD}" \
  --pyver "${PYVER}" \
  --abi3 "${ABI3}" \
  --outdir "${REPO_ROOT}/wheelhouse"

if command -v sccache >/dev/null; then
  echo "== sccache stats =="
  sccache --show-stats || true
fi

echo "== done =="
ls -la "${REPO_ROOT}/wheelhouse"
