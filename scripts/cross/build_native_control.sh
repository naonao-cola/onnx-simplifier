#!/usr/bin/env bash
# Build the onnxsim extension natively for x86_64 with the SAME CMake settings
# the s390x cross-build uses, so the two can be compared with byte order as the
# only difference. Used as the little-endian control for endianness testing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${REPO_ROOT}/.cross-build-s390x}"   # reuse the fetched sources
NATIVE_WORK="${REPO_ROOT}/.native-build-control"
JOBS="${JOBS:-$(nproc)}"
PYVER="${PYVER:-3.12}"

HOSTPY="$(command -v "python${PYVER}")"
NANOBIND_CMAKE_DIR="$(python3 -m nanobind --cmake_dir)"
BUILD_DIR="${NATIVE_WORK}/onnxsim-build"

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja -Wno-dev -Wdeprecated \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_BUILD_PYTHON=ON \
  -DONNX_INSTALL=OFF \
  -DONNXSIM_PYTHON=ON \
  -DONNXSIM_BUILTIN_ORT=OFF \
  -DONNXSIM_TESTS=ON \
  -DONNX_USE_LITE_PROTO=ON \
  -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
  -DCMAKE_PREFIX_PATH="${NANOBIND_CMAKE_DIR}" \
  -Dnanobind_DIR="${NANOBIND_CMAKE_DIR}" \
  -DPython_EXECUTABLE="${HOSTPY}" \
  -DPython3_EXECUTABLE="${HOSTPY}"

cmake --build "${BUILD_DIR}" --target onnxsim_cpp2py_export -j "${JOBS}"

SO="$(find "${BUILD_DIR}" -name 'onnxsim_cpp2py_export*.so' -print -quit)"
echo "built: ${SO}"
file "${SO}"
