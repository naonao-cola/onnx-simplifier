#!/bin/bash
# Run a model once under ORT + QNN EP on the phone's HTP with CPU fallback allowed and verbose
# logging, and summarize which nodes QNN took and why it rejected the rest.
#   ./partition_report.sh <model.onnx> <manifest.txt> [iters]
# Needs ../htp_exploration/qnn_shell/{libs,headers} (run its fetch_libs.sh once) and an
# adb-reachable phone. Manifest format: see ../htp_exploration/qnn_shell/qnn_run_multi.cpp.
# Output: <model>.partition.log (full ORT log) and a printed summary.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
Q="$here/../htp_exploration/qnn_shell"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
R="${R:-/data/local/tmp/vmprobe}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
A=(adb -s "$DEVICE_SERIAL")
[ -d "$Q/libs" ] || "$Q/fetch_libs.sh"
bin="$here/.qnn_run_multi"
[ "$bin" -nt "$Q/qnn_run_multi.cpp" ] ||
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$Q/headers" -o "$bin" "$Q/qnn_run_multi.cpp" -L "$Q/libs" -lonnxruntime
"${A[@]}" shell "mkdir -p $R"
for f in "$bin" "$Q"/libs/*; do
  d=$(basename "$f"); [ "$d" = .qnn_run_multi ] && d=qnn_run_multi
  sz=$(stat -c %s "$f"); dsz=$("${A[@]}" shell "stat -c %s $R/$d 2>/dev/null" | tr -d '\r' || true)
  [ "$sz" = "$dsz" ] || "${A[@]}" push -q "$f" "$R/$d"
done
"${A[@]}" push -q "$1" $R/
man=$(mktemp)
while read -r name dt file dims; do
  [ -z "${name:-}" ] && continue
  "${A[@]}" push -q "$file" $R/
  echo "$name $dt $(basename "$file") $dims" >>"$man"
done <"$2"
"${A[@]}" push -q "$man" $R/manifest.txt && rm -f "$man"
log="${1%.onnx}.partition.log"
"${A[@]}" shell "cd $R && ORT_LOG=1 LD_LIBRARY_PATH=$R \
  ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ./qnn_run_multi $(basename "$1") manifest.txt htp-fallback ${3:-3} out 2>&1" >"$log" || true
python3 "$here/summarize_partition.py" "$log"
