#!/bin/bash
# Build the phone runtime: pipe_run (this dir) and qnn_run_multi (../../htp_exploration/qnn_shell),
# against the ORT + QNN libraries ../../htp_exploration/qnn_shell/fetch_libs.sh downloads.
#   PIPE_DSP_KERNELS=1 also links the rpn/roialign FastRPC stubs (Mask R-CNN's HVX kernels); that
#   needs HEXAGON_SDK_ROOT/HEXAGON_TOOLCHAIN and builds their skels exactly as
#   ../../e2e_pipeline/build.sh does (BUILD_ONLY=1 there), then links against its build dir.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
QS="$HERE/../../htp_exploration/qnn_shell"
NDK="${NDK:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin}"
SDK="${HEXAGON_SDK_ROOT:-$HOME/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2}"
B="$HERE/build"
mkdir -p "$B"
[ -d "$QS/libs" ] || "$QS/fetch_libs.sh"
CXX="$NDK/aarch64-linux-android29-clang++"
INC=(-I "$QS/headers" -I "$SDK/incs" -I "$SDK/incs/stddef" -I "$SDK/ipc/fastrpc/rpcmem/inc")
LIBS=(-L "$QS/libs" -lonnxruntime -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc)
if [ -n "${PIPE_DSP_KERNELS:-}" ]; then
  E="$HERE/../../e2e_pipeline"
  EB="${E2E_BUILD:-$E/build}"
  [ -f "$EB/rpn_stub.o" ] || BUILD="$EB" BUILD_ONLY=1 OUT=/dev/null IMGS=/dev/null "$E/build.sh"
  out="$B/pipe_run_dsp"
  [ "$out" -nt "$HERE/pipe_run.cpp" ] || "$CXX" -O2 -std=c++17 -static-libstdc++ -DPIPE_DSP_KERNELS "${INC[@]}" \
    -I "$EB/rpn_fused" -I "$EB/roi" -o "$out" "$HERE/pipe_run.cpp" "$EB/rpn_glue.o" "$EB/rpn_stub.o" "$EB/roi_stub.o" "${LIBS[@]}"
else
  [ "$B/pipe_run" -nt "$HERE/pipe_run.cpp" ] ||
    "$CXX" -O2 -std=c++17 -static-libstdc++ "${INC[@]}" -o "$B/pipe_run" "$HERE/pipe_run.cpp" "${LIBS[@]}"
fi
[ "$B/qnn_run_multi" -nt "$QS/qnn_run_multi.cpp" ] ||
  "$CXX" -O2 -std=c++17 -static-libstdc++ -I "$QS/headers" -o "$B/qnn_run_multi" "$QS/qnn_run_multi.cpp" -L "$QS/libs" -lonnxruntime
