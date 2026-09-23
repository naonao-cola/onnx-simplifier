#!/bin/bash
# Build + check the fbgather kernel (fbgather_kernel.h).
#   ./build_dsp.sh qemu  DATA   HVX body vs plain C vs the expected volume under qemu-hexagon-static
#   ./build_dsp.sh phone DATA   FastRPC skel + client on the phone's CDSP (under the phone lock)
# DATA: a dsp_inputs.py directory. Phone dir /data/local/tmp/codex-android-fastbev/dsp.
set -euo pipefail
MODE="$1"; DATA="$(cd "$2" && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)"
SDK="${HEXAGON_SDK_ROOT:-/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2}"
TC="${HEXAGON_TOOLCHAIN:-$SDK/tools/HEXAGON_Tools/19.0.04/Tools}"
HEX_ARCH="${HEX_ARCH:-v69}"  # SM8475
OUT="${OUT:-$HOME/.cache/fastbev/dsp_build}"
mkdir -p "$OUT" && cd "$OUT"
if [ "$MODE" = qemu ]; then
  clang-19 --target=hexagon -mcpu=hexagonv68 -mhvx=v68 -mhvx-length=128b -O2 -static -nostdlib -ffreestanding \
    -fuse-ld=lld -I "$TC/target/hexagon/include" -o fbgq "$SRC/fbgather_qemu.c"
  qemu-hexagon-static ./fbgq 67584 "$DATA"/t{0,1,2,3}.bin "$DATA"/l{0,1,2,3}.bin "$DATA/vol_ref.bin"
  exit
fi
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
cp "$SRC"/fbgather_rpc.idl "$SRC"/fbgather_impl.c "$SRC"/fbgather_kernel.h "$SRC"/fbgather_client.c .
"$SDK/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$SDK/incs" -I "$SDK/incs/stddef" fbgather_rpc.idl
INC=(-I . -I "$SDK/incs" -I "$SDK/incs/stddef")
QURT_INC=(-I "$SDK/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$SDK/rtos/qurt/compute$HEX_ARCH/include/posix")
"$TC/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o fbgather_rpc_skel.c
"$TC/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  "${INC[@]}" "${QURT_INC[@]}" -o impl.o fbgather_impl.c
"$TC/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o fbgather_rpc.so skel.o impl.o "$TC/target/hexagon/lib/$HEX_ARCH/G0/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$SDK/ipc/fastrpc/rpcmem/inc" -o fbgather_client fbgather_client.c fbgather_rpc_stub.c \
  -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
D=/data/local/tmp/codex-android-fastbev/dsp
S="${DEVICE_SERIAL:-239dbd8f}"
PHONE_LOCK_OWNER=codex/android-fastbev "$HOME/.cache/android-phone/phone-run" bash -c "
  adb -s $S shell mkdir -p $D
  adb -s $S push fbgather_client fbgather_rpc.so $DATA/t0.bin $DATA/t1.bin $DATA/t2.bin $DATA/t3.bin \
    $DATA/l0.bin $DATA/l1.bin $DATA/l2.bin $DATA/l3.bin $DATA/vol_ref.bin $D/ >/dev/null
  adb -s $S shell \"chmod 755 $D/fbgather_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
    ./fbgather_client 'file:///fbgather_rpc.so?fbgather_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' ${REPS:-10} ${TURBO:-0} ${CONFIGS:-1,2,4,6}\""
