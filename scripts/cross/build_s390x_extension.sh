#!/usr/bin/env bash
# Cross-build the onnxsim CPython extension for s390x (big endian) Linux from an
# x86_64 host, so the test suite can be run big-endian under qemu-user.
#
# Same shape as build_windows_wheel.sh: a host protoc runs ONNX's code
# generation, abseil + protobuf are cross-built for the target, and the
# extension links against them. The target sysroot is an Ubuntu s390x rootfs
# (see scripts/cross/README.md) which also supplies CPython, numpy and onnx for
# the actual test run.
#
# Everything is compiled by the *host* toolchain (s390x-linux-gnu-g++), so the
# build runs at native speed; only the resulting binaries execute under qemu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}/.cross-build-s390x}"
SYSROOT="${SYSROOT:-/rootfs-s390x}"
JOBS="${JOBS:-$(nproc)}"
PYVER="${PYVER:-3.12}"

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

DEPS_HOST="${WORK}/deps-host"      # host protobuf install -> protoc
DEPS_TARGET="${WORK}/deps-target"  # target (s390x) absl + protobuf install
DL="${WORK}/dl"
mkdir -p "${DL}"

fetch() {  # fetch <url> <dest>
  [[ -s "$2" ]] && return 0
  echo "-- fetching $1"
  curl -fsSL --retry 3 -o "$2" "$1"
}

TOOLCHAIN_ARGS=(
  -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/scripts/cross/linux-s390x.toolchain.cmake"
  -DONNXSIM_S390X_SYSROOT="${SYSROOT}"
)

# ---------------------------------------------------------------------------
# 1. Host protoc -- ONNX runs this during code generation. A cross-built
#    s390x protoc could not execute on this host.
# ---------------------------------------------------------------------------
HOST_PROTOC="${DEPS_HOST}/bin/protoc"
if [[ ! -x "${HOST_PROTOC}" ]]; then
  echo "== [1/3] building host abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} (for protoc) =="
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
# 2. Target abseil + protobuf -- ONNX links these
# ---------------------------------------------------------------------------
if [[ ! -f "${DEPS_TARGET}/.done" ]]; then
  echo "== [2/3] cross-building abseil ${ABSL_VER} + protobuf ${PROTOBUF_VER} for s390x =="
  fetch "https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz" "${DL}/absl.tar.gz"
  fetch "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz" "${DL}/protobuf.tar.gz"

  rm -rf "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  mkdir -p "${WORK}/absl-tgt-src" "${WORK}/protobuf-tgt-src"
  tar -C "${WORK}/absl-tgt-src"     --strip-components=1 -xf "${DL}/absl.tar.gz"
  tar -C "${WORK}/protobuf-tgt-src" --strip-components=1 -xf "${DL}/protobuf.tar.gz"

  common_target_args=(
    -G Ninja
    "${TOOLCHAIN_ARGS[@]}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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
# 3. Cross-build the extension (and the dependency-free C++ unit tests)
# ---------------------------------------------------------------------------
echo "== [3/3] cross-building onnxsim for s390x =="

NANOBIND_CMAKE_DIR="$(python3 -m nanobind --cmake_dir)"
BUILD_DIR="${WORK}/onnxsim-build"
# CMake's FindPython validates the interpreter against the target headers and
# refuses a version mismatch, so the host interpreter must be the same X.Y as
# the target CPython in the sysroot (it only runs ONNX's codegen scripts).
HOSTPY="${HOSTPY:-$(command -v "python${PYVER}")}"
[[ -x "${HOSTPY}" ]] || { echo "need a host python${PYVER} to match the target"; exit 1; }

TGT_PY_INC="${SYSROOT}/usr/include/python${PYVER}"
TGT_PY_LIB="${SYSROOT}/usr/lib/s390x-linux-gnu/libpython${PYVER}.so"
TGT_PY_SABI="${SYSROOT}/usr/lib/s390x-linux-gnu/libpython3.so"

# onnx splits the Python roles when cross-compiling (target dev libs land in the
# Python3 namespace, the host interpreter in Python); onnxsim and onnx-optimizer
# use the Python namespace for target dev libs. Provide hints for both, exactly
# as the Windows cross-build does.
python_args=(
  -DPython_EXECUTABLE="${HOSTPY}"
  -DPython_INCLUDE_DIR="${TGT_PY_INC}"
  -DPython_LIBRARY="${TGT_PY_LIB}"
  -DPython_SABI_LIBRARY="${TGT_PY_SABI}"
  -DPython3_EXECUTABLE="${HOSTPY}"
  -DPython3_INCLUDE_DIR="${TGT_PY_INC}"
  -DPython3_LIBRARY="${TGT_PY_LIB}"
  -DPython3_SABI_LIBRARY="${TGT_PY_SABI}"
)

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja -Wno-dev -Wdeprecated \
  "${TOOLCHAIN_ARGS[@]}" \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="${REPO_ROOT}/scripts/cross/find_python_early.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_BUILD_PYTHON=ON \
  -DONNX_INSTALL=OFF \
  -DONNXSIM_PYTHON=ON \
  -DONNXSIM_BUILTIN_ORT=OFF \
  -DONNXSIM_TESTS=ON \
  -DONNX_USE_LITE_PROTO=OFF \
  -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="${HOST_PROTOC}" \
  -DCMAKE_PREFIX_PATH="${DEPS_TARGET};${NANOBIND_CMAKE_DIR}" \
  -Dnanobind_DIR="${NANOBIND_CMAKE_DIR}" \
  "${python_args[@]}"

cmake --build "${BUILD_DIR}" --target onnxsim_cpp2py_export -j "${JOBS}"

SO="$(find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*.so' -print -quit)"
[[ -n "${SO}" ]] || { echo "no extension module produced"; exit 1; }
echo "built: ${SO}"
file "${SO}"
