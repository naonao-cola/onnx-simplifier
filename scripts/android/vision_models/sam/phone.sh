#!/bin/bash
# phone.sh <model.onnx> <cpu|htp|htp-fallback> <iters> <local dir> <name> <manifest>...
# Runs one model on the phone (ORT + QNN EP, ../../htp_exploration/qnn_shell/qnn_run_multi.cpp)
# once per manifest in <local dir>: the first manifest `iters` times (the timed run), the rest once.
# HTP runs compile an EP-context model once (<name>.ctx.onnx on the phone, redone when the model
# changes). Outputs come back as <local dir>/out<k>_o<i>.bin; stdout has "=== set <k>" + the log.
# Called by sam.py under ~/.cache/android-phone/phone-run (the shared phone lock).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
Q="$here/../../htp_exploration/qnn_shell"
model="$1"; mode="$2"; iters="$3"; loc="$4"; name="$5"; shift 5
R="${R:-/data/local/tmp/codex-android-sam-variants}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
A=(adb -s "${ANDROID_SERIAL:-239dbd8f}")
[ -d "$Q/libs" ] || "$Q/fetch_libs.sh"
bin="$here/.qnn_run_multi"
[ "$bin" -nt "$Q/qnn_run_multi.cpp" ] ||
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$Q/headers" -o "$bin" "$Q/qnn_run_multi.cpp" -L "$Q/libs" -lonnxruntime
"${A[@]}" shell "mkdir -p $R"
push() {  # push if the phone copy differs (md5, not size: a re-quantized model keeps its size)
  local src="$1" dst="$2"
  local a b
  a=$(md5sum "$src" | cut -d' ' -f1)
  b=$("${A[@]}" shell "md5sum $dst 2>/dev/null" | cut -d' ' -f1 | tr -d '\r' || true)
  if [ "$a" != "$b" ]; then "${A[@]}" push -q "$src" "$dst"; return 0; fi
  return 1
}
push "$bin" "$R/qnn_run_multi" || true
for f in "$Q"/libs/*; do push "$f" "$R/$(basename "$f")" || true; done
m="$name.onnx"
if push "$model" "$R/$m"; then "${A[@]}" shell "rm -f $R/$name.ctx.onnx"; fi
ctx=""; [ "$mode" = cpu ] || ctx="$name.ctx.onnx"
k=0
for man in "$@"; do
  while read -r n dt f dims; do
    [ -z "${n:-}" ] && continue
    "${A[@]}" push -q "$loc/$f" "$R/$f"
  done <"$loc/$man"
  "${A[@]}" push -q "$loc/$man" "$R/$man"
  it=1; [ "$k" = 0 ] && it="$iters"
  echo "=== set $k"
  log=$("${A[@]}" shell "cd $R && ${EXTRA_ENV:-} QNN_PERF=${QNN_PERF:-burst} LD_LIBRARY_PATH=$R \
    ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
    ./qnn_run_multi $m $man $mode $it out$k $ctx 2>&1" || true)
  echo "$log"
  if ! grep -q PASS <<<"$log"; then  # never keep a context from a failed compile; stop early
    "${A[@]}" shell "rm -f $R/$name.ctx.onnx"
    break
  fi
  nout=$(grep -c "^out " <<<"$log" || true)
  for ((i = 0; i < nout; i++)); do
    "${A[@]}" pull -q "$R/out${k}_o$i.bin" "$loc/out${k}_o$i.bin" 2>/dev/null || true
  done
  "${A[@]}" shell "rm -f $R/out${k}_o*.bin" || true
  k=$((k + 1))
done
