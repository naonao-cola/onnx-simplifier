#!/usr/bin/env bash
# Build the camera demo: BOARD=m5stickv|cube ./build.sh [WORKDIR]
# (default BOARD=m5stickv, default WORKDIR ./build-work)
# -> WORKDIR/onnx-k210-camera-$BOARD.bin. Same SDK/toolchain/nncase 1.9.0 setup
# as ../runtime/build.sh (which does the work).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BOARD="${BOARD:-m5stickv}"
case "$BOARD" in m5stickv|cube) ;; *) echo "BOARD must be m5stickv or cube" >&2; exit 1 ;; esac
export PROJ=onnx_k210_camera
export SRC_DIR="$HERE/src"
export EXTRA_SRC="$HERE/../runtime/src/w25qxx.c $HERE/../runtime/src/w25qxx.h"
export OUT_NAME="onnx-k210-camera-$BOARD"
exec "$HERE/../runtime/build.sh" "${1:-$HERE/build-work}"
