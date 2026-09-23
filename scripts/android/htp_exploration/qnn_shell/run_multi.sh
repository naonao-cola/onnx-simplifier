#!/bin/bash
# Like run.sh, but for multi-input models via qnn_run_multi.
#   ./run_multi.sh <model.onnx> <manifest.txt> <mode: cpu|htp|htp-fallback> <iters> [ctx.onnx]
# The manifest's data files (third column) are pushed alongside; give them as host paths and the
# manifest is rewritten to their basenames on the device. Env passed through: QNN_PERF,
# ORT_THREADS, ORT_LOG, QNN_EXTRA, ORT_PROFILE. Device dir: $R (default /data/local/tmp/qnn_rest).
set -euo pipefail
cd "$(dirname "$0")"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
R="${R:-/data/local/tmp/qnn_rest}"
A=(adb -s "$DEVICE_SERIAL")
[ -d libs ] || ./fetch_libs.sh
[ qnn_run_multi -nt qnn_run_multi.cpp ] ||
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I headers -o qnn_run_multi qnn_run_multi.cpp -L libs -lonnxruntime
"${A[@]}" shell "mkdir -p $R"
# push libs/binary only when missing or changed (the model data can be large; libs are ~100 MB)
for f in qnn_run_multi libs/*; do
  sz=$(stat -c %s "$f")
  dsz=$("${A[@]}" shell "stat -c %s $R/$(basename "$f") 2>/dev/null" | tr -d '\r' || true)
  [ "$sz" = "$dsz" ] || "${A[@]}" push -q "$f" $R/
done
"${A[@]}" push -q "$1" $R/
man=$(mktemp)
while read -r name dt file dims; do
  [ -z "${name:-}" ] && continue
  "${A[@]}" push -q "$file" $R/
  echo "$name $dt $(basename "$file") $dims" >> "$man"
done < "$2"
"${A[@]}" push -q "$man" $R/manifest.txt
rm -f "$man"
ENVS=""
for v in QNN_PERF ORT_THREADS ORT_LOG QNN_EXTRA ORT_PROFILE; do [ -n "${!v:-}" ] && ENVS="$ENVS $v=${!v}"; done
"${A[@]}" shell "cd $R && $ENVS LD_LIBRARY_PATH=$R \
  ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ./qnn_run_multi $(basename "$1") manifest.txt $3 $4 out_$3 ${5:-}"
