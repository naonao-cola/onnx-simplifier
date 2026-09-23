#!/bin/bash
# Run one exported piece on the phone's HTP (ORT + QNN EP): partition report (what QNN refuses,
# htp-fallback latency), then strict all-HTP; pull each mode's outputs and compare with the torch
# fp32 reference export.py saved.   ./run_phone.sh <work> <piece> [model suffix, default .sim]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
W="$1"; P="$2"; S="${3-.sim}"
R="${R:-/data/local/tmp/codex-android-maptr}"
export R
"$here/../../vision_models_probe/partition_report.sh" "$W/$P$S.onnx" "$W/$P.in/manifest.txt" "${ITERS:-10}"
for mode in htp-fallback htp; do
  grep -q PASS "$W/$P$S.$mode.out" || continue
  for i in 0 1; do
    adb -s "${DEVICE_SERIAL:-239dbd8f}" pull -q "$R/out_${mode}_o$i.bin" "$W/$P$S.$mode.o$i.bin" 2>/dev/null || true
  done
  python3 "$here/compare_out.py" "$W" "$P" "$S" "$mode"
done
