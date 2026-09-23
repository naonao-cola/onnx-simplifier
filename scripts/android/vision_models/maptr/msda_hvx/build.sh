#!/bin/bash
# Build MapTR's phone chain on top of the generic HVX MSDA core (../../../msda_hvx/): the core's skel +
# stub (its build.sh, BUILD_ONLY), then map_run against ../../../htp_exploration/qnn_shell's ORT (run
# its fetch_libs.sh once).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
SRC="$(cd "$(dirname "$0")" && pwd)"
CORE="$SRC/../../../msda_hvx"
QS="$SRC/../../../htp_exploration/qnn_shell"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT"
OUT="$OUT" BUILD_ONLY=1 "$CORE/build.sh"
cd "$OUT"
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$QS/headers")
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ "${INC[@]}" -o map_run "$SRC/map_run.cpp" msda_stub.o -L "$QS/libs" -lonnxruntime \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
