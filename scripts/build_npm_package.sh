#!/usr/bin/env bash
# Build the ORT-web WASM variant from source, then stage it into npm/onnxsim/
# (see stage_npm_package.sh). For local testing (`npm/onnxsim && npm pack`)
# and as the CI fallback when no matching static.yml artifact is available to
# reuse (see .github/workflows/npm-publish.yml).
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

cd "$ROOT_DIR"
ORT_WEB=ON "$ROOT_DIR/build_wasm.sh"
"$SCRIPT_DIR/stage_npm_package.sh"
