#!/bin/bash
# Run backbone.onnx on the HTP (strict, burst) with QNN's own profiling on, and pull the CSV.
#   ./profile_backbone.sh <backbone.onnx> <input.bin> <outdir> [level: basic|detailed|optrace] [extra k=v,...]
# Assumes run_ceiling.sh already pushed qnn_run + libs to /data/local/tmp/qnn_ceiling.
set -euo pipefail
DEV="${DEVICE_SERIAL:-239dbd8f}"
R=/data/local/tmp/qnn_ceiling
MODEL="$1"; IN="$2"; OUT="$3"; LEVEL="${4:-detailed}"; EXTRA="${5:-}"
mkdir -p "$OUT"
adb -s "$DEV" push -q "$MODEL" "$IN" $R/
M=$(basename "$MODEL"); I=$(basename "$IN")
tag="$LEVEL${EXTRA:+_$(echo "$EXTRA" | tr ',=' '__')}"
X="profiling_level=$LEVEL,profiling_file_path=$R/prof_$tag.csv${EXTRA:+,$EXTRA}"
adb -s "$DEV" shell "cd $R && rm -f prof_$tag.csv && QNN_PERF=burst QNN_EXTRA='$X' LD_LIBRARY_PATH=$R \
  ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ./qnn_run $M $I htp ${ITERS:-8} o_prof" > "$OUT/run_$tag.log" 2>&1 || true
adb -s "$DEV" pull "$R/prof_$tag.csv" "$OUT/" >/dev/null 2>&1 || echo "no profiling CSV produced"
adb -s "$DEV" shell "cd $R && rm -f o_prof_*.bin prof_$tag.csv"
grep -E "^(run|PASS|FAIL|session)" "$OUT/run_$tag.log" | tail -12
