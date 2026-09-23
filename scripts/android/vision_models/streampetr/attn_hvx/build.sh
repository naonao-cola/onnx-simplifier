#!/bin/bash
# Build the StreamPETR cross-attention skel (attn_rpc.idl / attn_impl.c / attn_kernel.h) + its phone
# bench client into $OUT (also attn_stub.o for other clients), then -- unless BUILD_ONLY=1 -- push them
# with emulate.py case directories to D and run the bench. Wrap the phone part in the host's phone lock.
#   CASES: case dirs, FLAGS: comma list (threads | 256 * rows/4), REPS, TURBO, D: phone directory.
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
HEX_ARCH="${HEX_ARCH:-v69}"
REPS="${REPS:-10}" TURBO="${TURBO:-0}" FLAGS="${FLAGS:-4}"
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/attn_rpc.idl "$SRC"/attn_impl.c "$SRC"/attn_kernel.h "$SRC"/attn_io.h "$SRC"/attn_client.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" attn_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o attn_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  -Wno-unused-function "${INC[@]}" "${QURT_INC[@]}" -o impl.o attn_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o attn_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 -c "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o attn_stub.o attn_rpc_stub.c
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o attn_client attn_client.c attn_stub.o \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
[ -n "${BUILD_ONLY:-}" ] && exit 0
: "${CASES:?emulate.py case dirs}"
D="${D:-/data/local/tmp/attn_hvx-${USER:-user}}"
adb -s "$DEVICE_SERIAL" shell "mkdir -p $D"
adb -s "$DEVICE_SERIAL" push attn_client attn_rpc.so $D/ >/dev/null
names=()
for c in $CASES; do
  n=$(basename "$c"); names+=("$n")
  adb -s "$DEVICE_SERIAL" shell "[ -d $D/$n ]" 2>/dev/null || adb -s "$DEVICE_SERIAL" push "$c" "$D/$n" >/dev/null
done
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/attn_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  timeout 120 ./attn_client 'file:///attn_rpc.so?attn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO $FLAGS ${names[*]}"
