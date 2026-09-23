#!/bin/bash
# Phone step of bisect_llm.py: run W/bisect/<graph>.dbg.onnx strict on the HTP with qnn_run_multi and
# pull every output back as W/bisect/out_htp_o<i>.bin. Run under ~/.cache/android-phone/phone-run.
#   ./bisect_llm.sh W dec_prefill.fp16.onnx
set -euo pipefail
cd "$(dirname "$0")"
W=$1 G=${2%.onnx}.dbg.onnx
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
R=/data/local/tmp/codex-android-llm-tinygrad
QS=../htp_exploration/qnn_shell
A=(adb -s "$DEVICE_SERIAL")
BIN=$W/qnn_run_multi
[ "$BIN" -nt $QS/qnn_run_multi.cpp ] ||
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I $QS/headers -o "$BIN" $QS/qnn_run_multi.cpp -L $QS/libs -lonnxruntime
"${A[@]}" shell "mkdir -p $R/bisect && rm -f $R/bisect/out_htp_o*.bin"
"${A[@]}" push -q "$BIN" $R/ >/dev/null
"${A[@]}" push -q "$W/bisect/$G" "$W"/bisect/in_*.bin $R/bisect/ >/dev/null
sed "s|$W/bisect/||" "$W/bisect/manifest.txt" > "$W/bisect/manifest.dev.txt"
"${A[@]}" push -q "$W/bisect/manifest.dev.txt" $R/bisect/manifest.txt >/dev/null
"${A[@]}" shell "cd $R/bisect && QNN_PERF=burst LD_LIBRARY_PATH=$R \
  ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ../qnn_run_multi $G manifest.txt ${MODE:-htp-fallback} 3 out_htp" | grep -E "median|FAIL|PASS"
"${A[@]}" pull -q $R/bisect/. "$W/bisect/" >/dev/null
