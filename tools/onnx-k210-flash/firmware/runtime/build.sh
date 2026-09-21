#!/usr/bin/env bash
# Build prebuilt/onnx-k210-runtime.bin from source.
#
#   ./build.sh [WORKDIR]      (default WORKDIR: ./build-work)
#
# Downloads the Kendryte RISC-V toolchain and kendryte-standalone-sdk, swaps
# the SDK's bundled nncase runtime (1.0.0, 2021) for nncase's v1.9.0 K210
# runtime -- it has to match the nncase==1.9.0 compiler that
# ../../scripts/onnx_to_kmodel.py uses, or run() fails with a
# `runtime_module.cpp shape_reg` assertion -- and builds this directory's
# src/. Result: $WORKDIR/onnx-k210-runtime.bin
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mkdir -p "${1:-$HERE/build-work}" && cd "${1:-$HERE/build-work}" && pwd)"
PROJ="${PROJ:-onnx_k210_runtime}"
SRC_DIR="${SRC_DIR:-$HERE/src}"
EXTRA_SRC="${EXTRA_SRC:-}"
OUT_NAME="${OUT_NAME:-onnx-k210-runtime}"
TOOLCHAIN_URL=https://github.com/kendryte/kendryte-gnu-toolchain/releases/download/v8.2.0-20190409/kendryte-toolchain-ubuntu-amd64-8.2.0-20190409.tar.xz
NNCASE_RT_URL=https://github.com/kendryte/nncase/releases/download/v1.9.0/nncaseruntime-riscv64-none-k210.zip

cd "$WORK"
[ -d kendryte-toolchain ] || { curl -fsSL "$TOOLCHAIN_URL" | tar xJ; }
[ -d kendryte-standalone-sdk ] || git clone -q --depth 1 https://github.com/kendryte/kendryte-standalone-sdk.git

SDK="$WORK/kendryte-standalone-sdk"
V1="$SDK/lib/nncase/v1"
if [ ! -f "$V1/.nncase-1.9.0" ]; then
  curl -fsSL -o nncaseruntime.zip "$NNCASE_RT_URL"
  rm -rf nncaseruntime && mkdir nncaseruntime && (cd nncaseruntime && unzip -q ../nncaseruntime.zip)
  rm -rf "$V1/lib" "$V1/include"
  cp -r nncaseruntime/lib nncaseruntime/include "$V1/"
  rm -f "$V1/lib/libkendryte.a" # the SDK builds its own; see sdk-patch/nncase_v1_CMakeLists.txt
  cp "$HERE/sdk-patch/nncase_v1_CMakeLists.txt" "$V1/CMakeLists.txt"
  touch "$V1/.nncase-1.9.0"
fi

rm -rf "$SDK/src/$PROJ" && mkdir -p "$SDK/src/$PROJ"
cp "$SRC_DIR"/* "$SDK/src/$PROJ/"
for f in $EXTRA_SRC; do cp "$f" "$SDK/src/$PROJ/"; done

rm -rf "$SDK/build" && mkdir "$SDK/build" && cd "$SDK/build"
# CMAKE_POLICY_VERSION_MINIMUM: the SDK declares cmake_minimum_required < 3.5, which CMake 4.x rejects.
cmake .. -DPROJ="$PROJ" -DTOOLCHAIN="$WORK/kendryte-toolchain/bin" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >cmake.log
make -j"$(nproc)" >make.log 2>&1 || { tail -30 make.log; exit 1; }
cp "$PROJ.bin" "$WORK/$OUT_NAME.bin"
ls -l "$WORK/$OUT_NAME.bin"
