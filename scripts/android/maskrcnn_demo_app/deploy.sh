#!/bin/bash
# Install the demo APK and give it its models and test images.
#   ./deploy.sh                    models copied on-device from ../e2e_pipeline's /data/local/tmp/e2e
#   MODELS=<build_models.py --out dir> ./deploy.sh      models pushed from the host instead
#   IMGS="a.jpg b.jpg ..." ./deploy.sh                  JPEGs for the "images" test mode
# Then:  adb shell am start -n org.onnxsim.maskrcnndemo/.MainActivity [--es mode images] [--es pipe pipe_e_opt.txt]
#
# Files go to the app's *internal* files dir through `run-as` (the APK is debuggable): files adb
# puts under /sdcard/Android/data/<pkg> are owned by the shell user and unreadable by the app.
set -euo pipefail
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
A=(adb -s "$DEVICE_SERIAL")
PKG=org.onnxsim.maskrcnndemo
HERE="$(cd "$(dirname "$0")" && pwd)"
APK="$HERE/app/build/outputs/apk/debug/app-debug.apk"
STAGE=/data/local/tmp/maskrcnn_demo_stage
"${A[@]}" install -r -g "$APK"            # -g grants CAMERA
"${A[@]}" shell am force-stop $PKG
RA() { "${A[@]}" shell "run-as $PKG sh -c '$1'"; }
RA "mkdir -p files/models files/imgs"
FILES="pipe_e_u8_ctx.txt pipe_e_u8.txt pipe_e_opt.txt pipe_e_opt_ctx.txt levels.txt model.txt l0_anchors.bin l1_anchors.bin l2_anchors.bin l3_anchors.bin
l4_anchors.bin backbone_opt.onnx adapt_scores.onnx seg_e_opt_1.onnx seg_e_opt_2.onnx seg_e_opt_3.onnx seg_e_opt_4.onnx
seg_e_opt_5.onnx box_head_1000_nhwc.onnx mask_head_32_nhwc.onnx mask_head_100_nhwc.onnx
box_head_1000_u8.onnx mask_head_32_u8.onnx mask_head_100_u8.onnx"
if [ -n "${MODELS:-}" ]; then
  SRC=$STAGE/models
  "${A[@]}" shell "mkdir -p $SRC"
  for f in $FILES; do "${A[@]}" push -q "$MODELS/$f" "$SRC/"; done
else
  SRC="${DEVICE_MODELS:-/data/local/tmp/e2e}"
fi
for f in $FILES; do RA "cmp -s $SRC/$f files/models/$f || cp $SRC/$f files/models/"; done
# EP-context models compiled earlier on this phone (e2e_run writes them on its first ctx run); if
# absent, the app compiles them itself on its first launch with that pipe (JIT cost, once) and
# reuses them after. NO_CTX=1 skips the copy (to measure that first launch).
if [ -z "${NO_CTX:-}" ]; then
  for f in backbone_opt box_head_1000_nhwc mask_head_32_nhwc mask_head_100_nhwc; do
    RA "[ ! -f $SRC/$f.ctx.onnx ] || cmp -s $SRC/$f.ctx.onnx files/models/$f.ctx.onnx || cp $SRC/$f.ctx.onnx files/models/"
  done
  for f in backbone_opt.ctx0.onnx backbone_opt.ctx0_qnn.bin box_head_1000_u8.ctx0.onnx box_head_1000_u8.ctx0_qnn.bin \
           mask_head_32_u8.ctx0.onnx mask_head_32_u8.ctx0_qnn.bin mask_head_100_u8.ctx0.onnx mask_head_100_u8.ctx0_qnn.bin; do
    RA "[ ! -f $SRC/$f ] || cmp -s $SRC/$f files/models/$f || cp $SRC/$f files/models/"
  done
fi
if [ -n "${IMGS:-}" ]; then
  "${A[@]}" shell "mkdir -p $STAGE/imgs"
  for i in $IMGS; do "${A[@]}" push -q "$i" "$STAGE/imgs/"; done
  RA "cp $STAGE/imgs/*.jpg files/imgs/"
fi
RA "ls -la files/models files/imgs"
