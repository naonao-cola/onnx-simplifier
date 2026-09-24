#!/bin/bash
# Build the msda skel (../../../msda_hvx, BUILD_ONLY) and dec_run (this directory) into $OUT.
#   needs HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN, ../../../htp_exploration/qnn_shell/{libs,headers}
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HOME/.cache/onnxsim-rtdetr/msda_build}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
QS="$SRC/../../../htp_exploration/qnn_shell"
[ -d "$QS/libs" ] || "$QS/fetch_libs.sh"
OUT="$OUT" BUILD_ONLY=1 "$SRC/../../../msda_hvx/build.sh"
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$OUT" -I "$SRC/../../../msda_hvx" \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -I "$QS/headers" \
  -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o "$OUT/dec_run" "$SRC/dec_run.cpp" "$OUT/msda_stub.o" \
  -L "$QS/libs" -lonnxruntime -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $OUT/dec_run $OUT/msda_rpc.so"
