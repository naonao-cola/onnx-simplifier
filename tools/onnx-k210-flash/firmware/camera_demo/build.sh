#!/usr/bin/env bash
# Build the M5StickV camera demo: ./build.sh [WORKDIR]  (default ./build-work)
# -> WORKDIR/onnx-k210-camera.bin. Same SDK/toolchain/nncase 1.9.0 setup as
# ../runtime/build.sh (which does the work).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ=onnx_k210_camera
export SRC_DIR="$HERE/src"
export EXTRA_SRC="$HERE/../runtime/src/w25qxx.c $HERE/../runtime/src/w25qxx.h"
export OUT_NAME=onnx-k210-camera
exec "$HERE/../runtime/build.sh" "${1:-$HERE/build-work}"
