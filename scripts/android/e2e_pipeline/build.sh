#!/bin/bash
# Build the one-process driver + the three DSP skels (sources from ../tinygrad_hexagon_bridge, used
# unchanged), push everything to the phone, and run the given stages.
#   OUT=<build_models.py --out dir> IMGS=<prepare_inputs.py dir> ./build.sh <stage>...
# Results: $RES/<stage>/ (final outputs pulled back per image) and $RES/<stage>.log.
# Libraries come from ../htp_exploration/qnn_shell/fetch_libs.sh (Maven Central, not committed).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${OUT:?models dir}" "${IMGS:?images dir}"
NDK="${NDK:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
WARMUP="${WARMUP:-2}"
REPS="${REPS:-7}"
RES="${RES:-$PWD/results}"
D=/data/local/tmp/e2e
HERE="$(cd "$(dirname "$0")" && pwd)"
TG="$HERE/../tinygrad_hexagon_bridge"
QS="$HERE/../htp_exploration/qnn_shell"
[ -d "$QS/libs" ] || "$QS/fetch_libs.sh"
B="${BUILD:-$HERE/build}"
mkdir -p "$B" "$RES"
A=(adb -s "$DEVICE_SERIAL")
QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
INC=(-I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
HC="$HEXAGON_TOOLCHAIN/bin/hexagon-clang"

# DSP skels, built exactly as their own build.sh do (same flags)
mkdir -p "$B/rpn_fused" "$B/proposal_decode" "$B/topk" "$B/nms" "$B/roi" "$B/roiu8"
cp "$TG"/rpn_fused/*.h "$TG"/rpn_fused/rpn_impl.c "$TG"/rpn_fused/rpn_rpc.idl "$B/rpn_fused/"
for d in proposal_decode topk nms; do cp "$TG/$d/"*_kernel.h "$B/$d/"; done
cp "$TG"/roialign_fast/roialign_rpc.idl "$TG"/roialign_fast/roialign_impl.c "$TG"/roialign_fast/roialign_kernel.h "$B/roi/"
cp "$TG"/roialign_fast/roialign_u8_rpc.idl "$TG"/roialign_fast/roialign_u8_impl.c "$TG"/roialign_fast/roialign_u8_kernel.h "$B/roiu8/"
(cd "$B/rpn_fused" && "$QAIC" "${INC[@]}" rpn_rpc.idl &&
  "$HC" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -I . "${INC[@]}" -o skel.o rpn_rpc_skel.c &&
  "$HC" -c -O2 -fPIC -ffp-contract=off -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall \
    -Wno-unused-function -I . "${INC[@]}" "${QURT_INC[@]}" -o impl.o rpn_impl.c &&
  "$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o rpn_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so")
(cd "$B/roi" && "$QAIC" "${INC[@]}" roialign_rpc.idl &&
  "$HC" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -I . "${INC[@]}" -o skel.o roialign_rpc_skel.c &&
  "$HC" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall -I . "${INC[@]}" "${QURT_INC[@]}" \
    -o impl.o roialign_impl.c &&
  "$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o roialign_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so")
# the merged uint8 RoiAlign (roialign_fast/build_u8.sh's flags)
(cd "$B/roiu8" && "$QAIC" "${INC[@]}" roialign_u8_rpc.idl &&
  "$HC" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -I . "${INC[@]}" -o skel.o roialign_u8_rpc_skel.c &&
  "$HC" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" -mhvx-length=128b -Wall -I . "${INC[@]}" "${QURT_INC[@]}" \
    -o impl.o roialign_u8_impl.c &&
  "$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o roialign_u8_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so")

# the driver (ARM64): ORT C++ API + the three qaic stubs + rpn_glue.c (rpn_model_io.h, unchanged)
CC="$NDK/aarch64-linux-android29-clang"
CXX="$NDK/aarch64-linux-android29-clang++"
RPC=(-I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$B/rpn_fused" -I "$B/roi" -I "$B/roiu8")
"$CC" -O2 -ffp-contract=off -Wno-unused-function "${INC[@]}" "${RPC[@]}" -c -o "$B/rpn_glue.o" "$HERE/rpn_glue.c"
"$CC" -O2 "${INC[@]}" "${RPC[@]}" -c -o "$B/rpn_stub.o" "$B/rpn_fused/rpn_rpc_stub.c"
"$CC" -O2 "${INC[@]}" "${RPC[@]}" -c -o "$B/roi_stub.o" "$B/roi/roialign_rpc_stub.c"
"$CC" -O2 "${INC[@]}" "${RPC[@]}" -c -o "$B/roiu8_stub.o" "$B/roiu8/roialign_u8_rpc_stub.c"
"$CXX" -O2 -std=c++17 -static-libstdc++ -I "$QS/headers" "${INC[@]}" "${RPC[@]}" -o "$B/e2e_run" "$HERE/e2e_run.cpp" \
  "$B/rpn_glue.o" "$B/rpn_stub.o" "$B/roi_stub.o" "$B/roiu8_stub.o" -L "$QS/libs" -lonnxruntime \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc

[ -n "${BUILD_ONLY:-}" ] && exit 0

# push (only what changed: models and libs are large)
"${A[@]}" shell "mkdir -p $D/imgs"
push() {  # push if size differs
  for f in "$@"; do
    sz=$(stat -c %s "$f")
    dsz=$("${A[@]}" shell "stat -c %s $D/${DEST:-}$(basename "$f") 2>/dev/null" | tr -d '\r' || true)
    [ "$sz" = "$dsz" ] || "${A[@]}" push -q "$f" "$D/${DEST:-}"
  done
}
push "$QS"/libs/* "$B/e2e_run" "$B/rpn_fused/rpn_rpc.so" "$B/roi/roialign_rpc.so" "$B/roiu8/roialign_u8_rpc.so" \
  "$OUT"/*.onnx "$OUT"/pipe_*.txt "$OUT"/levels.txt "$OUT"/model.txt "$OUT"/l*_anchors.bin
DEST=imgs/ push "$IMGS"/*.bin
"${A[@]}" shell "chmod 755 $D/e2e_run"

IMG_LIST=$(cd "$IMGS" && ls *.bin | sed "s#^#imgs/#" | tr '\n' ' ')
for st in "$@"; do
  "${A[@]}" shell "rm -rf $D/res_$st && mkdir -p $D/res_$st"
  "${A[@]}" shell "cd $D && ${EXTRA_ENV:-} LD_LIBRARY_PATH=$D:/vendor/lib64 \
    ADSP_LIBRARY_PATH='$D;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
    ./e2e_run pipe_$st.txt $WARMUP $REPS res_$st $IMG_LIST" 2>&1 | tee "$RES/$st.log"
  rm -rf "$RES/$st" && mkdir -p "$RES/$st"
  "${A[@]}" pull -q "$D/res_$st/." "$RES/$st/" >/dev/null
done
