#!/usr/bin/env bash
# Stage a built ORT-web wasm module (onnxsim.js/onnxsim.wasm) plus the shared
# JS executor into npm/onnxsim/, so `npm pack` / `npm publish` run from that
# directory produce a self-contained package.
#
# Split out from build_npm_package.sh so .github/workflows/static.yml can call
# it directly right after its own wasm build (build_wasm.sh) without
# rebuilding the module a second time in a separate workflow.
#
# Usage: stage_npm_package.sh [dir containing onnxsim.js and onnxsim.wasm]
# Defaults to build-wasm-node-OFF-ortweb/ (build_wasm.sh's ORT_WEB=ON output).
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)
PKG_DIR="$ROOT_DIR/npm/onnxsim"
BUILD_DIR="${1:-$ROOT_DIR/build-wasm-node-OFF-ortweb}"

# Emscripten's MODULARIZE output here is a CommonJS module (`module.exports =
# create_onnxsim`); npm/onnxsim/package.json declares "type": "module", so a
# plain .js file would be parsed as ESM and fail on that `module.exports`.
# Renaming to .cjs keeps Node's CommonJS loader (and __dirname, which the
# module uses to locate the sibling .wasm file) regardless of the package's
# module type.
cp "$BUILD_DIR/onnxsim.js" "$PKG_DIR/onnxsim.cjs"
cp "$BUILD_DIR/onnxsim.wasm" "$PKG_DIR/onnxsim.wasm"
cp "$ROOT_DIR/scripts/convertmodel/ort_executor.mjs" "$PKG_DIR/ort_executor.mjs"

echo "npm package staged in $PKG_DIR"
