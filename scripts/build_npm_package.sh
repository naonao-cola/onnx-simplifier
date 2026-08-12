#!/usr/bin/env bash
# Build the ORT-web WASM variant from source, then stage it into npm/onnxsim/
# (see stage_npm_package.sh). This is the local-dev equivalent of what
# .github/workflows/static.yml does in CI (build_wasm.sh + stage_npm_package.sh
# as part of its existing wasm build, rather than a separate workflow
# rebuilding the module again); use it for local testing (`npm/onnxsim && npm
# pack`).
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

cd "$ROOT_DIR"
ORT_WEB=ON "$ROOT_DIR/build_wasm.sh"
"$SCRIPT_DIR/stage_npm_package.sh"
