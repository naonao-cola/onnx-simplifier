#!/bin/bash
# Build + run qfsim.py's captured plain-tinygrad float kernels (and hand_blend.c) on the phone's CDSP through a
# dedicated TVM-free FastRPC skel, then pull back <tag>_out.bin for analyze.py. Kernels are compiled for HEX_ARCH=v69:
# v73 code would use IEEE HVX float, which this V69 phone doesn't have (results come back as 0).
# $DATA holds `qfsim.py <op> <n> --dump $DATA <tag>` output (<tag>.c, <tag>.meta, <tag>_buf*.bin).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"; HEX_ARCH="${HEX_ARCH:-v69}"; REPS="${REPS:-9}"
OUT="${OUT:-$(mktemp -d)}"; SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC"/qf_rpc.idl "$SRC"/qf_impl.c "$SRC"/qf_client.c "$SRC"/hand_blend.c .
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" qf_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
HC=("$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b)
# dispatch table: every captured kernel, plus hand_blend on blend_qf's buffers if that capture exists
: > calls.txt; k=0; objs=(); { echo "#pragma once"; } > qf_dispatch.h; cases=""
add() { local tag=$1 dtag=$2 meta; meta=$(cut -d' ' -f2- "$DATA/$dtag.meta"); local nb=${meta%% *}
  echo "$k $tag $dtag $meta" >> calls.txt
  echo "extern void $tag($(printf 'void*,%.0s' $(seq $nb) | sed 's/,$//'));" >> qf_dispatch.h
  cases+="    case $k: $tag($(for i in $(seq 0 $((nb-1))); do printf 'b[%d],' $i; done | sed 's/,$//')); break;"$'\n'; k=$((k+1)); }
for m in "$DATA"/*.meta; do t=$(basename "$m" .meta); cp "$DATA/$t.c" .; "${HC[@]}" "${INC[@]}" -o "$t.o" "$t.c"; objs+=("$t.o"); add "$t" "$t"; done
if [ -f "$DATA/blend_qf.meta" ]; then "${HC[@]}" "${INC[@]}" -o hand_blend.o hand_blend.c; objs+=(hand_blend.o); add hand_blend blend_qf; fi
{ echo "#define QF_NKERN $k"; echo "static void qf_call(int k, void** b) { switch (k) {"; printf "%s" "$cases"; echo "  } }"; } >> qf_dispatch.h
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o qf_rpc_skel.c
"${HC[@]}" "${INC[@]}" "${QURT_INC[@]}" -o impl.o qf_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o qf_rpc.so skel.o impl.o "${objs[@]}" "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o qf_client qf_client.c qf_rpc_stub.c \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
D=/data/local/tmp/qf_codegen
adb -s "$DEVICE_SERIAL" shell "rm -rf $D && mkdir -p $D"
adb -s "$DEVICE_SERIAL" push qf_client qf_rpc.so calls.txt "$DATA"/*_buf[1-5].bin $D/ >/dev/null
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/qf_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./qf_client 'file:///qf_rpc.so?qf_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS"
for t in $(cut -d' ' -f2 calls.txt); do adb -s "$DEVICE_SERIAL" pull "$D/${t}_out.bin" "$DATA/${t}_out.bin" >/dev/null; done
adb -s "$DEVICE_SERIAL" shell "rm -rf $D"
