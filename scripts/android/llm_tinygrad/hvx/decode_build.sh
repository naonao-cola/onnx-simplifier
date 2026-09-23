#!/bin/bash
# Build the whole-model decode skel (decode_impl.c + tinygrad kernels.h) and client, push them with the weights, run.
#   KERNELS=<dir with llm_kernels.py --ops b576,b1536,b576x4,b1536x4 --variants tcp output>
#   BLOBS=<dir with decode_ref.py --export output>  D=/data/local/tmp/<yours>
# Run the phone part under ~/.cache/android-phone/phone-run (shared phone).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${KERNELS:?}" "${BLOBS:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
HEX_ARCH="${HEX_ARCH:-v69}"
OUT="${OUT:-$BLOBS/build}"
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/decode_rpc.idl "$SRC"/decode_impl.c "$SRC"/decode_client.c "$KERNELS"/kernels.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" decode_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o decode_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o decode_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o decode_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o decode_client decode_client.c decode_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
