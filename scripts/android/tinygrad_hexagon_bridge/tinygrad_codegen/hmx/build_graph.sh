#!/bin/bash
# Build the FastRPC skel + client for a graph emitted by qdq_net.py (k<n>.c, graph.h, blob.bin in $1):
#   HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... ./build_graph.sh <graph dir>
# The client's CASE mode then runs it: case/a.bin = the input, case/b.bin = blob.bin, case/ref.bin = the expected output.
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
HEX_ARCH=v69; SRC="$(cd "$(dirname "$0")" && pwd)"; HMX_GEMM="$SRC/../../../hmx_gemm"
cd "$1"
cp "$SRC"/tg_hmx_rpc.idl "$SRC"/tg_graph_impl.c "$SRC"/tg_hmx_client.c "$HMX_GEMM"/hmx_runtime.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" tg_hmx_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
HC=("$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH -mhvx=$HEX_ARCH -mhvx-length=128b -mhmx)
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon$HEX_ARCH "${INC[@]}" -o skel.o tg_hmx_rpc_skel.c
"${HC[@]}" "${INC[@]}" "${QURT_INC[@]}" -o impl.o tg_graph_impl.c
ls k*.c | xargs -P 4 -I{} sh -c '"$@" -Wno-deprecated-non-prototype -o "$(basename {} .c).o" {}' _ "${HC[@]}"
objs=(k*.o)
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o tg_hmx_rpc.so skel.o impl.o "${objs[@]}" "$LIBPATH/pic/libgcc.a" "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o tg_hmx_client tg_hmx_client.c tg_hmx_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
echo "built $1/tg_hmx_rpc.so $1/tg_hmx_client"
