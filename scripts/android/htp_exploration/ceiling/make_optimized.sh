#!/bin/bash
# Apply the backbone rewrites from ../ceiling_findings.md in order, keeping every stage:
#   <out>/1_qres    residual Adds as int8 QDQ units (quantized shortcut; the only numeric change)
#   <out>/2_qout    uint8 graph outputs (lossless)
#   <out>/3_qraw    raw RPN conv outputs, no per-anchor Reshape/Transpose/Sigmoid (lossless*)
#   <out>/4_in      uint8 NHWC image input (lossless)
#   <out>/5_final   NHWC FPN outputs (lossless)
#   (* scores become logits; the consumer applies sigmoid in float, within half a uint8 step)
#   ./make_optimized.sh <backbone.onnx> <outdir> [python]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IN="$1"; OUT="$2"; PY="${3:-python3}"
mkdir -p "$OUT"/{1_qres,2_qout,3_qraw,4_in,5_final}
"$PY" "$HERE/quantize_residuals.py" "$IN" "$OUT/1_qres/backbone.onnx"
"$PY" "$HERE/quantized_outputs.py" "$OUT/1_qres/backbone.onnx" "$OUT/2_qout/backbone.onnx"
"$PY" "$HERE/raw_rpn_outputs.py" "$OUT/2_qout/backbone.onnx" "$OUT/3_qraw/backbone.onnx" >/dev/null
"$PY" "$HERE/quantized_input.py" "$OUT/3_qraw/backbone.onnx" "$OUT/4_in/backbone.onnx" nhwc
"$PY" "$HERE/nhwc_fpn_outputs.py" "$OUT/4_in/backbone.onnx" "$OUT/5_final/backbone.onnx"
echo "final model: $OUT/5_final/backbone.onnx (input qparams: $OUT/4_in/input_qparams.json)"
