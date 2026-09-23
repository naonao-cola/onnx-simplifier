#!/bin/bash
# Run one exported piece on the phone's HTP (ORT + QNN EP) under the shared phone lock: partition
# report (what QNN refuses, htp-fallback latency), then strict all-HTP; pull the outputs and compare
# with export.py's torch fp32 reference.   ./run_phone.sh <work> <piece> [model suffix, default .sim]
# Phone dir: /data/local/tmp/codex-android-fastbev (R). ITERS (default 10) timed iterations.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
W="$1"; P="$2"; S="${3-.sim}"
export R="${R:-/data/local/tmp/codex-android-fastbev}"
LOCK="${PHONE_LOCK:-$HOME/.cache/android-phone/phone-run}"
[ -x "$LOCK" ] || LOCK=env
base="${P%.u8}"
PHONE_LOCK_OWNER=codex/android-fastbev "$LOCK" bash -c '
  "$1/../../vision_models_probe/partition_report.sh" "$2" "$3" "$4"
  for mode in htp-fallback htp; do
    grep -q PASS "$5.$mode.out" || continue
    n=$(grep -c "^out " "$5.$mode.out" || true)
    for ((i = 0; i < n; i++)); do
      adb -s "${DEVICE_SERIAL:-239dbd8f}" pull -q "$R/out_${mode}_o$i.bin" "$5.$mode.o$i.bin" 2>/dev/null || true
    done
  done' _ "$here" "$W/$P$S.onnx" "$W/$P.in/manifest.txt" "${ITERS:-10}" "$W/$P$S"
for mode in htp-fallback htp; do
  grep -q PASS "$W/$P$S.$mode.out" 2>/dev/null || continue
  if [[ "$P" == *.q8 ]]; then python3 "$here/compare_q8.py" "$W" "$P$S" "$mode" || true
  else python3 "$here/../bevformer_tiny/compare_out.py" "$W" "$P" "$S" "$mode" || true; fi
done
