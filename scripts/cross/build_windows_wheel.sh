#!/usr/bin/env bash
#
# Cross-compile an onnxsim Windows (x86_64, MSVC ABI) wheel from Linux.
#
# This produces a wheel that is binary-compatible with the official python.org
# CPython on Windows, without ever running a Windows machine.  It exists so the
# expensive C++ build can run on a cheap Linux runner; the wheel is then handed
# to a Windows runner purely for `pytest` (see .github/workflows/windows-cross.yml).
#
# High-level stages:
#   1. Download the MSVC CRT + Windows SDK with `xwin`.
#   2. Download the *target* Windows CPython (for its headers + python3.lib).
#   3. Build a *host* protoc (Linux) so ONNX can run code generation.
#   4. Cross-build abseil + protobuf for the *target* (Windows) so ONNX can link.
#   5. Cross-build onnxsim's nanobind extension against all of the above.
#   6. Pack the resulting .pyd into a correctly tagged wheel.
#
# The dependency versions are pinned to whatever the vendored ONNX pins (see the
# SBOM in third_party/onnx-optimizer/third_party/onnx/sbom.cdx.json); a mismatch
# between the host protoc and the target libprotobuf would break the build.
#
# Required environment:
#   PYVER          Target CPython version, e.g. "3.12".
#   ABI3           "1" to build a limited-API (abi3) wheel (only valid for >=3.12),
#                  "0" for a version-specific wheel.
# Optional environment:
#   WORK           Scratch directory (default: $PWD/.cross-build).
#   PROTOBUF_VER   Override protobuf version (default: read from SBOM).
#   ABSL_VER       Override abseil version   (default: read from SBOM).
#   NANOBIND_VER   Override nanobind version (default: read from SBOM).
#   JOBS           Parallel build jobs (default: nproc).
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
: "${PYVER:?set PYVER, e.g. 3.12}"
: "${ABI3:=0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}/.cross-build}"
JOBS="${JOBS:-$(nproc)}"
ARCH="x86_64"
TRIPLE="x86_64-pc-windows-msvc"

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

# python-build-standalone release providing the target Windows interpreter.
# Only the headers and import libraries are used; the version tag is pinned so
# the build is reproducible.  Override PBS_RELEASE / PBS_PYFULL to bump.
PBS_RELEASE="${PBS_RELEASE:-20250723}"
declare -A PBS_PYFULL_MAP=(
  ["3.10"]="3.10.18"
  ["3.11"]="3.11.13"
  ["3.12"]="3.12.11"
  ["3.13"]="3.13.5"
)
PBS_PYFULL="${PBS_PYFULL:-${PBS_PYFULL_MAP[$PYVER]:?no python-build-standalone pin for $PYVER}}"

echo "== onnxsim Windows cross-build =="
echo "   target python : ${PYVER} (${PBS_PYFULL}), abi3=${ABI3}"
echo "   protobuf      : ${PROTOBUF_VER}"
echo "   abseil-cpp    : ${ABSL_VER}"
echo "   nanobind      : ${NANOBIND_VER}"
echo "   work dir      : ${WORK}"

XWIN_DIR="${WORK}/xwin"
DEPS_HOST="${WORK}/deps-host"     # host (Linux) protobuf install -> protoc
DEPS_TARGET="${WORK}/deps-target" # target (Windows) absl + protobuf install
WINPY="${WORK}/winpy"             # target Windows CPython
DL="${WORK}/dl"
mkdir -p "${WORK}" "${DL}"

TOOLCHAIN="${REPO_ROOT}/scripts/cross/windows-clang-cl.toolchain.cmake"

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ ! -f "${dest}" ]]; then
    echo ">> downloading ${url}"
    curl -fSL --retry 4 --retry-delay 2 -o "${dest}.tmp" "${url}"
    mv "${dest}.tmp" "${dest}"
  fi
}

# ---------------------------------------------------------------------------
# 1. MSVC CRT + Windows SDK via xwin
# ---------------------------------------------------------------------------
if [[ ! -d "${XWIN_DIR}/crt" ]]; then
  echo "== [1/6] downloading MSVC CRT + Windows SDK with xwin =="
  command -v xwin >/dev/null || { echo "xwin not on PATH"; exit 1; }
  xwin --accept-license --arch x86_64 splat --include-debug-libs --output "${XWIN_DIR}"
fi

# ---------------------------------------------------------------------------
# 2. Target Windows CPython (headers + python3.lib / pythonXY.lib)
# ---------------------------------------------------------------------------
if [[ ! -d "${WINPY}/python" ]]; then
  echo "== [2/6] fetching target Windows CPython ${PBS_PYFULL} =="
  pbs_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PBS_PYFULL}+${PBS_RELEASE}-${ARCH}-pc-windows-msvc-install_only.tar.gz"
  fetch "${pbs_url}" "${DL}/winpy.tar.gz"
  mkdir -p "${WINPY}"
  tar -C "${WINPY}" -xf "${DL}/winpy.tar.gz"   # extracts a top-level "python/" dir
fi
WINPY_INC="${WINPY}/python/include"
WINPY_LIBDIR="${WINPY}/python/libs"
PYVER_NODOT="${PYVER//./}"
if [[ "${ABI3}" == "1" ]]; then
  WINPY_LIB="${WINPY_LIBDIR}/python3.lib"          # limited API import library
else
  WINPY_LIB="${WINPY_LIBDIR}/python${PYVER_NODOT}.lib"
fi
[[ -f "${WINPY_LIB}" ]] || { echo "missing ${WINPY_LIB}"; ls -la "${WINPY_LIBDIR}"; exit 1; }

# ---------------------------------------------------------------------------
# 3. Host protoc (Linux) -- ONNX runs this during code generation
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
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/absl-host-build" --target install -j "${JOBS}"

  cmake -S "${WORK}/protobuf-host-src" -B "${WORK}/protobuf-host-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="${DEPS_HOST}" -DCMAKE_INSTALL_PREFIX="${DEPS_HOST}"
  cmake --build "${WORK}/protobuf-host-build" --target install -j "${JOBS}"
fi
"${HOST_PROTOC}" --version

# ---------------------------------------------------------------------------
# 4. Target abseil + protobuf (Windows) -- ONNX links these
# ---------------------------------------------------------------------------
if [[ ! -f "${DEPS_TARGET}/lib/libprotobuf.lib" && ! -f "${DEPS_TARGET}/lib/protobuf.lib" ]]; then
  echo "== [4/6] cross-building abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} for Windows =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  mkdir -p "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  tar -C "${WORK}/absl-tgt-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-tgt-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  common_target_args=(
    -G Ninja
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
    -DXWIN_DIR="${XWIN_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    # ONNX links protobuf statically with the static CRT off; match onnxsim CI
    # which sets ONNX_USE_PROTOBUF_SHARED_LIBS=OFF.
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
fi

# ---------------------------------------------------------------------------
# 5. Cross-build the onnxsim extension
# ---------------------------------------------------------------------------
echo "== [5/6] cross-building onnxsim extension =="
NANOBIND_CMAKE_DIR="$(python3 -m nanobind --cmake_dir)"

BUILD_DIR="${WORK}/onnxsim-build-${PYVER}$([[ "${ABI3}" == "1" ]] && echo -abi3)"
rm -rf "${BUILD_DIR}"

sabi_args=()
if [[ "${ABI3}" == "1" ]]; then
  # nanobind STABLE_ABI needs the SABI import library (python3.lib).
  sabi_args+=( -DPython_SABI_LIBRARY="${WINPY_LIBDIR}/python3.lib" )
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DXWIN_DIR="${XWIN_DIR}" \
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
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DPython_INCLUDE_DIR="${WINPY_INC}" \
  -DPython_LIBRARY="${WINPY_LIB}" \
  "${sabi_args[@]}"

cmake --build "${BUILD_DIR}" --target onnxsim_cpp2py_export -j "${JOBS}"

PYD="$(find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*.pyd' -print -quit)"
[[ -n "${PYD}" ]] || { echo "no .pyd produced"; find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*'; exit 1; }
echo "built: ${PYD}"

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

echo "== done =="
ls -la "${REPO_ROOT}/wheelhouse"
