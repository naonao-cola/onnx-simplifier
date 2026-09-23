#!/bin/bash
# Build the phone runner (fastbev_run.cpp + csrc/postproc.c + the fbgather FastRPC stub) and the
# fbgather skel. Output in $OUT (default ~/.cache/fastbev/runtime_build).
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
FB="$SRC/.."
SDK="${HEXAGON_SDK_ROOT:-/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2}"
TC="${HEXAGON_TOOLCHAIN:-$SDK/tools/HEXAGON_Tools/19.0.04/Tools}"
HEX_ARCH="${HEX_ARCH:-v69}"
Q="$FB/../../htp_exploration/qnn_shell"
NDK_BIN="${NDK_BIN:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin}"
OUT="${OUT:-$HOME/.cache/fastbev/runtime_build}"
mkdir -p "$OUT" && cd "$OUT"
[ -d "$Q/libs" ] || "$Q/fetch_libs.sh"
cp "$FB"/dsp/fbgather_rpc.idl "$FB"/dsp/fbgather_impl.c "$FB"/dsp/fbgather_kernel.h .
"$SDK/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$SDK/incs" -I "$SDK/incs/stddef" fbgather_rpc.idl
INC=(-I . -I "$SDK/incs" -I "$SDK/incs/stddef")
QURT_INC=(-I "$SDK/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$SDK/rtos/qurt/compute$HEX_ARCH/include/posix")
"$TC/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o fbgather_rpc_skel.c
"$TC/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o fbgather_impl.c
"$TC/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o fbgather_rpc.so skel.o impl.o "$TC/target/hexagon/lib/$HEX_ARCH/G0/pic/libgcc.so"
CC="$NDK_BIN/aarch64-linux-android29-clang"
"$CC" -O2 -c "${INC[@]}" -I "$SDK/ipc/fastrpc/rpcmem/inc" fbgather_rpc_stub.c -o stub.o
"$CC" -O2 -ffp-contract=off -c "$FB/csrc/postproc.c" -o postproc.o
"$NDK_BIN/aarch64-linux-android29-clang++" -O2 -ffp-contract=off -std=c++17 -static-libstdc++ "${INC[@]}" \
  -I "$SDK/ipc/fastrpc/rpcmem/inc" -I "$Q/headers" -o fastbev_run "$SRC/fastbev_run.cpp" stub.o postproc.o \
  -L "$Q/libs" -lonnxruntime -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
echo "built $OUT/fastbev_run $OUT/fbgather_rpc.so"
