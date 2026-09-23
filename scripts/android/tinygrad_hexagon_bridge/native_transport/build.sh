#!/bin/bash
# Builds and runs the from-scratch, TVM-free Hexagon FastRPC transport PoC end to end:
#   mini_rpc.idl --(qaic)--> mini_rpc_{skel,stub}.c, mini_rpc.h
#   mini_rpc_skel.c + mini_rpc_impl.c --(hexagon-clang + hexagon-link)--> mini_rpc.so   (DSP side)
#   client_main.c + mini_rpc_stub.c   --(Android NDK aarch64 clang)-->    mini_client  (host side)
#   adb push both to the device, run mini_client, which drives mini_rpc.so purely via
#   libcdsprpc.so's remote_handle64_open/invoke/close -- no tvm.rpc, no tvm.contrib.hexagon,
#   no MinRPC, no libhexagon_rpc_skel.so anywhere in the loop. See ../README.md for the
#   two real bugs this hit and how they were root-caused.
#
# Requires: HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN (Hexagon v73 clang/link), the Android NDK
# (aarch64-linux-android<api>-clang on PATH or NDK_CLANG set), and an adb-reachable device
# (DEVICE_SERIAL, default 239dbd8f).
set -euo pipefail

: "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to the Hexagon SDK root (contains incs/, tools/)}"
: "${HEXAGON_TOOLCHAIN:?set HEXAGON_TOOLCHAIN to .../tools/HEXAGON_Tools/<ver>/Tools}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"

cd "$(dirname "$0")"

QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
"$QAIC" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" mini_rpc.idl

"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -Wall \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -o mini_rpc_skel.o mini_rpc_skel.c
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
  -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -o mini_rpc_impl.o mini_rpc_impl.c

LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic \
  -o mini_rpc.so mini_rpc_skel.o mini_rpc_impl.o "$LIBPATH/pic/libgcc.so"

"$NDK_CLANG" -O2 -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -o mini_client client_main.c mini_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc

adb -s "$DEVICE_SERIAL" shell "mkdir -p /data/local/tmp/native_transport"
adb -s "$DEVICE_SERIAL" push mini_client mini_rpc.so /data/local/tmp/native_transport/
adb -s "$DEVICE_SERIAL" shell "chmod 755 /data/local/tmp/native_transport/mini_client"
# If gen_gemm_test_data.py's output is present, push it too so client_main.c's real-kernel test
# (cin=64,cout=256,m=54400, hex_gemm_kernel.py's own default shape) runs instead of being skipped.
if [ -f gemm_a.bin ] && [ -f gemm_bp.bin ]; then
  adb -s "$DEVICE_SERIAL" push gemm_a.bin gemm_bp.bin /data/local/tmp/native_transport/
fi
if [ -f boxhead_a.bin ] && [ -f boxhead_bp.bin ]; then
  adb -s "$DEVICE_SERIAL" push boxhead_a.bin boxhead_bp.bin /data/local/tmp/native_transport/
fi
adb -s "$DEVICE_SERIAL" shell "cd /data/local/tmp/native_transport && \
  LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp/native_transport \
  ./mini_client 'file:///mini_rpc.so?mini_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp'"
