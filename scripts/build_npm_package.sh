#!/usr/bin/env bash
# Build the ORT-web WASM variant and stage its output (plus the shared JS
# executor) inside npm/onnxsim/, so `npm pack` / `npm publish` run from that
# directory produce a self-contained package.
#
# The generated artifacts are build products, not checked into git (see
# npm/onnxsim/.gitignore); this script is what recreates them, for both local
# testing (`npm/onnxsim && npm pack`) and the npm-publish CI workflow.
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)
PKG_DIR="$ROOT_DIR/npm/onnxsim"
BUILD_DIR="$ROOT_DIR/build-wasm-node-OFF-ortweb"

cd "$ROOT_DIR"
ORT_WEB=ON "$ROOT_DIR/build_wasm.sh"

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
