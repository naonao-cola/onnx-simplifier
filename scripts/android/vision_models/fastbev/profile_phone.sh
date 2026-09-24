#!/bin/bash
# QNN detailed per-op profile of one piece on the HTP (under the phone lock), summarized by
# ../bevformer_tiny/profile_ops.py.   ./profile_phone.sh <work> <piece> [suffix, default .sim]
# Detailed profiling serializes ops: shares are relative, scale by run_phone.sh's latency.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
W="$1"; P="$2"; S="${3-.sim}"
export R="${R:-/data/local/tmp/codex-android-fastbev}"
IN="$W/$P$S.in"; [ -d "$IN" ] || IN="$W/$P.in"  # the int8 pieces have their own (uint8) inputs
A=(adb -s "${DEVICE_SERIAL:-239dbd8f}")
PHONE_LOCK_OWNER="${PHONE_LOCK_OWNER:-codex/android-fastbev}" "$HOME/.cache/android-phone/phone-run" bash -c '
  set -e
  ITERS=1 "$1/../../vision_models_probe/partition_report.sh" "$2" "$3" 1 >/dev/null
  "${@:5}" shell "cd $R && QNN_EXTRA=profiling_level=detailed,profiling_file_path=$R/prof.csv QNN_PERF=burst \
    LD_LIBRARY_PATH=$R ADSP_LIBRARY_PATH=\"$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" \
    ./qnn_run_multi $(basename "$2") manifest.txt htp 3 out_prof >/dev/null 2>&1"
  "${@:5}" pull -q $R/prof.csv "$4"' _ "$here" "$W/$P$S.onnx" "$IN/manifest.txt" "$W/prof_$P$S.csv" "${A[@]}"
python3 "$here/../bevformer_tiny/profile_ops.py" "$W/$P$S.onnx" "$W/prof_$P$S.csv"
