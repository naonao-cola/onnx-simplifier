#!/bin/bash
# Build BEVFormer's phone pieces on top of the generic HVX MSDA core (../../../msda_hvx/): the core's
# skel + stub (its build.sh, BUILD_ONLY), then enc_run (the encoder chain), frame_run (the whole frame) and qnn_run_multi (for the
# backbone/decoder pieces) against ../../../htp_exploration/qnn_shell's ORT (run its fetch_libs.sh).
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
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ "${INC[@]}" -o enc_run "$SRC/enc_run.cpp" msda_stub.o -L "$QS/libs" -lonnxruntime \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ "${INC[@]}" -I "$SRC" -o frame_run "$SRC/frame_run.cpp" msda_stub.o -L "$QS/libs" -lonnxruntime \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$QS/headers" -o qnn_run_multi "$QS/qnn_run_multi.cpp" -L "$QS/libs" -lonnxruntime
