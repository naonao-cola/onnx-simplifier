#!/bin/bash
# Build the DFA skel (dfa_rpc.idl / dfa_impl.c over ../../../msda_hvx/msda_kernel.h), the s4d_run
# one-frame chain runner and the s4d_scene scene runner (instance bank on the phone) into $OUT.
#   HEXAGON_SDK_ROOT=.../Hexagon_SDK/6.4.0.2 HEXAGON_TOOLCHAIN=.../HEXAGON_Tools/19.0.04/Tools ./build.sh
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
NDK_CC="${NDK_CC:-${NDK_CXX%++}}"
HEX_ARCH="${HEX_ARCH:-v69}"
SRC="$(cd "$(dirname "$0")" && pwd)"
MSDA="$SRC/../../../msda_hvx"
Q="$SRC/../../../htp_exploration/qnn_shell"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/dfa_rpc.idl "$SRC"/dfa_impl.c "$SRC"/dfa_core.h "$SRC"/s4d_common.h "$SRC"/s4d_run.cpp "$SRC"/s4d_scene.cpp \
  "$MSDA"/msda_kernel.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" dfa_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o dfa_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o dfa_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o dfa_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CC" -O2 -c "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o dfa_stub.o dfa_rpc_stub.c
for prog in s4d_run s4d_scene; do
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$Q/headers" \
    -o $prog $prog.cpp dfa_stub.o -L "$Q/libs" -lonnxruntime \
    -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
done
echo "built $OUT/dfa_rpc.so $OUT/s4d_run $OUT/s4d_scene"
