#!/bin/bash
# Build + run the fused RPN span (TopK -> decode -> filter -> NMS -> merge -> TopK -> gather) on the
# phone's CDSP through a dedicated, TVM-free FastRPC skel -- same pattern as ../topk/build.sh.
# DATA = capture_rpn_fused.py's output after rpn_host_check has run on it once (that writes
# DATA/lN_anchors.bin); it must also contain levels.txt (copy it from the proposal-decode capture).
# With ORT_AAR (an extracted onnxruntime-android AAR: headers/ + jni/arm64-v8a/), also runs the
# phone-CPU ORT baseline on the same 604-node span.
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}" "${DATA:?capture_rpn_fused.py output dir}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
HEX_ARCH="${HEX_ARCH:-v73}"
REPS="${REPS:-21}"
TURBO="${TURBO:-0}"
IMAGES="${IMAGES:-$(cd "$DATA" && ls -d img_* | tr "\n" " ")}"
OUT="${OUT:-$(mktemp -d)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
mkdir -p rpn_fused && cp "$SRC"/*.h "$SRC"/*.c "$SRC"/rpn_rpc.idl rpn_fused/
for d in proposal_decode topk nms; do mkdir -p $d && cp "$SRC/../$d/"*_kernel.h $d/; done
cd rpn_fused
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" rpn_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix")
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -mcpu=hexagon"$HEX_ARCH" "${INC[@]}" -o skel.o rpn_rpc_skel.c
# -ffp-contract=off: decode and NMS are exact only with the graph's separate mul/add roundings
"$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -c -O2 -fPIC -ffp-contract=off -mcpu=hexagon"$HEX_ARCH" -mhvx="$HEX_ARCH" \
  -mhvx-length=128b -Wall -Wno-unused-function "${INC[@]}" "${QURT_INC[@]}" -o impl.o rpn_impl.c
LIBPATH="$HEXAGON_TOOLCHAIN/target/hexagon/lib/$HEX_ARCH/G0"
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o rpn_rpc.so skel.o impl.o "$LIBPATH/pic/libgcc.so"
"$NDK_CLANG" -O2 -ffp-contract=off -Wno-unused-function "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" \
  -o rpn_client rpn_client.c rpn_rpc_stub.c -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc
D=/data/local/tmp/rpn_fused
adb -s "$DEVICE_SERIAL" shell "rm -rf $D && mkdir -p $D"
adb -s "$DEVICE_SERIAL" push rpn_client rpn_rpc.so "$DATA"/levels.txt "$DATA"/model.txt "$DATA"/l*_anchors.bin $D/ >/dev/null
for im in $IMAGES; do
  adb -s "$DEVICE_SERIAL" shell "mkdir -p $D/$im"
  adb -s "$DEVICE_SERIAL" push "$DATA/$im"/l*_scores.bin "$DATA/$im"/l*_deltas.bin "$DATA/$im"/l*_nchw_q.bin \
    "$DATA/$im"/l*_nms_sel.bin "$DATA/$im"/proposals.bin $D/$im/ >/dev/null
done
adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/rpn_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ${PHASES:+PHASES=1} ./rpn_client 'file:///rpn_rpc.so?rpn_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $REPS $TURBO $IMAGES"
if [ -n "${ORT_AAR:-}" ]; then
  "$NDK_CLANG" -O2 -I "$ORT_AAR/headers" -o ort_rpn_bench ort_rpn_bench.c -L "$ORT_AAR/jni/arm64-v8a" -lonnxruntime
  adb -s "$DEVICE_SERIAL" push ort_rpn_bench "$ORT_AAR/jni/arm64-v8a/libonnxruntime.so" "$DATA"/rpn_region.onnx \
    "$DATA"/rpn_region_io.txt $D/ >/dev/null
  adb -s "$DEVICE_SERIAL" shell "chmod 755 $D/ort_rpn_bench && cd $D && LD_LIBRARY_PATH=$D ./ort_rpn_bench $IMAGES"
fi
adb -s "$DEVICE_SERIAL" shell "rm -rf $D"
