#!/bin/bash
# Build the LLM GEMV skel + client and run it on the phone's CDSP (TVM-free FastRPC, like
# ../../tinygrad_hexagon_bridge/tinygrad_codegen/build.sh). $DATA holds llm_kernels.py's output.
# Run under ~/.cache/android-phone/phone-run (shared phone).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?directory with kernels.h and the .bin files}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
HEX_ARCH="${HEX_ARCH:-v69}"
REPS="${REPS:-15}"
TURBO="${TURBO:-0}"
OUT="${OUT:-$DATA/build}"
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/llm_rpc.idl "$SRC"/llm_impl.c "$SRC"/llm_client.c "$DATA"/kernels.h .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" llm_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o llm_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o llm_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o llm_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o llm_client llm_client.c llm_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
[ -n "${BUILD_ONLY:-}" ] && exit 0
D=/data/local/tmp/codex-android-llm-tinygrad/hvx
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push llm_client llm_rpc.so "$DATA"/*.bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/llm_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./llm_client 'file:///llm_rpc.so?llm_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO"
