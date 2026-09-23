#!/bin/bash
# Push llm_run + ORT/QNN libs + models + inputs to the phone, run, pull the outputs back.
#   ./run_phone.sh enc <model.onnx> <mode> <iters> <indir> <nsets> <host_outdir>
#   ./run_phone.sh dec <prefill.onnx> <decode.onnx> <mode> <ngen> <indir> <nprompts> <host_outdir>
# mode: cpu | htp | htp-fallback. Env passed through: ORT_THREADS QNN_PERF QNN_EXTRA ORT_LOG
# PREFILL_ITERS SAVE_LOGITS. HTP runs compile an EP-context cache named after the model's md5 once.
# The phone is shared: run this under ~/.cache/android-phone/phone-run.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE_SERIAL="${DEVICE_SERIAL:-${ANDROID_SERIAL:-239dbd8f}}"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
R="${R:-/data/local/tmp/codex-android-llm-tinygrad}"
QS=../htp_exploration/qnn_shell
A=(adb -s "$DEVICE_SERIAL")
[ -d $QS/libs ] || $QS/fetch_libs.sh
BIN=${BIN:-$HOME/.cache/llm-work/llm_run}
[ "$BIN" -nt llm_run.cpp ] ||
  "$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I $QS/headers -o "$BIN" llm_run.cpp -L $QS/libs -lonnxruntime
"${A[@]}" shell "mkdir -p $R"
push() {  # push if the phone's copy differs (md5); echoes the device path
  local f=$1 d=$R/$(basename "$1")
  local h; h=$(md5sum "$f" | cut -d' ' -f1)
  local g; g=$("${A[@]}" shell "md5sum $d 2>/dev/null" | cut -d' ' -f1 | tr -d '\r' || true)
  [ "$h" = "$g" ] || "${A[@]}" push -q "$f" "$d" >&2
  echo "$d"
}
ctx() { echo "$R/ctx_$(md5sum "$1" | cut -c1-12).onnx"; }
for f in "$BIN" $QS/libs/*; do push "$f" >/dev/null; done
ENVS=""
for v in ORT_THREADS QNN_PERF QNN_EXTRA ORT_LOG PREFILL_ITERS SAVE_LOGITS; do [ -n "${!v:-}" ] && ENVS="$ENVS $v=${!v}"; done
RUN="cd $R && $ENVS LD_LIBRARY_PATH=$R ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp'"
cmd=$1; shift
tag=out_$$
if [ "$cmd" = enc ]; then
  model=$1 mode=$2 iters=$3 indir=$4 nsets=$5 out=$6
  m=$(push "$model")
  "${A[@]}" shell "rm -rf $R/in_enc && mkdir -p $R/in_enc $R/$tag"
  "${A[@]}" push -q "$indir"/. "$R/in_enc/" >/dev/null
  "${A[@]}" shell "$RUN ./llm_run enc $m $mode $iters $nsets in_enc $tag $([ $mode = cpu ] || ctx "$model")"
else
  pre=$1 dec=$2 mode=$3 ngen=$4 indir=$5 np=$6 out=$7
  p=$(push "$pre"); d=$(push "$dec")
  "${A[@]}" shell "rm -rf $R/in_dec && mkdir -p $R/in_dec $R/$tag"
  "${A[@]}" push -q "$indir"/. "$R/in_dec/" >/dev/null
  cx=""
  [ "$mode" = cpu ] || cx="$(ctx "$pre") $(ctx "$dec")"
  "${A[@]}" shell "$RUN ./llm_run dec $p $d $mode $ngen $np in_dec $tag $cx"
fi
mkdir -p "$out"
"${A[@]}" pull -q "$R/$tag/." "$out/" >/dev/null
"${A[@]}" shell "rm -rf $R/$tag"
