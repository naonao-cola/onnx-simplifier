#!/bin/bash
# Run a model under ORT + QNN EP on the phone's HTP and report what QNN rejected and why.
#   ./partition_report.sh <model.onnx> <manifest.txt> [iters]
# Runs twice: `htp-fallback` (CPU takes whatever QNN rejects; ORT's verbose log goes to logcat,
# which is where the per-op "Failed to validate op X with error 0x..." lines come from), then
# strict `htp` (fails loudly if anything would land on the CPU). Needs
# ../htp_exploration/qnn_shell/{libs,headers} (run its fetch_libs.sh once) and an adb-reachable
# phone. Manifest format: "<name> <f32|i64|i32|u8> <file.bin> <d0,d1,...>" per input.
# Output: <model>.logcat, <model>.htp-fallback.out, <model>.htp.out and a printed summary.
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
base="${1%.onnx}"
model="$(basename "$1")"
run() {  # $1 = mode, $2 = ORT log level, $3 = iterations
  "${A[@]}" shell "cd $R && ORT_LOG=$2 QNN_PERF=${QNN_PERF:-burst} LD_LIBRARY_PATH=$R \
    ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
    ./qnn_run_multi $model manifest.txt $1 $3 out_$1 2>&1" || true
}
"${A[@]}" logcat -c
run htp-fallback 0 "${3:-3}" >"$base.htp-fallback.out"
"${A[@]}" logcat -d >"$base.logcat"
run htp 2 "${3:-10}" >"$base.htp.out"
python3 "$here/summarize_partition.py" "$1" "$base.logcat" "$base.htp-fallback.out" "$base.htp.out"
