#!/bin/bash
# Download the pre-exported ONNX models this plan probes into ./models/ (not committed).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
get() { [ -s "models/$2" ] || curl -sfL --retry 3 -o "models/$2" "https://huggingface.co/$1/resolve/main/$3"; }
# YOLO11n (Ultralytics, 640x640, fp32, exported with NMS left out)
get aaurelions/yolo11n.onnx yolo11n.onnx yolo11n.onnx
# RT-DETR r18vd (onnx-community export of PekingU/rtdetr_r18vd, fp32 and dynamic-int8)
get onnx-community/rtdetr_r18vd rtdetr_r18vd.onnx onnx/model.onnx
get onnx-community/rtdetr_r18vd rtdetr_r18vd_int8.onnx onnx/model_int8.onnx
# Depth Anything V2 Small (onnx-community, fp32 and dynamic-int8)
get onnx-community/depth-anything-v2-small depth_anything_v2_small.onnx onnx/model.onnx
get onnx-community/depth-anything-v2-small depth_anything_v2_small_int8.onnx onnx/model_int8.onnx
ls -la models
