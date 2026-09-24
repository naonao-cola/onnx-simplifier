#!/bin/bash
# ./bisect_run.sh <work> <piece> [substr...]: expose those intermediates, run strict HTP, report.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
W="$1"; P="$2"; shift 2
export R="${R:-/data/local/tmp/bevformer_tiny}"
find "$W/$P.in" -name 'dbgref_*' -delete
python3 "$here/bisect_precision.py" make "$W" "$P" "$@"
"$here/../../vision_models_probe/partition_report.sh" "$W/$P.dbg.onnx" "$W/$P.in/manifest.txt" 1 >/dev/null 2>&1 || true
grep -h "PASS\|FAIL" "$W/$P.dbg.htp.out"
python3 "$here/bisect_precision.py" report "$W" "$P"
