#!/usr/bin/env bash
# Verify that every binding's manifest declares the same version as a release
# tag, so e.g. tagging v0.8.0 actually matches what npm/Cargo report instead
# of drifting silently -- see the "Bump npm/onnxsim to 0.7.1 -- 0.7.0 is
# already published" fix, where npm/onnxsim/package.json had fallen behind.
#
# This only checks; it never writes. The Python wheel needs no entry here --
# setup.py already derives its version from `git describe --tags` at build
# time (falling back to the committed VERSION file only when git tags aren't
# available, e.g. building from an sdist).
#
# Checks:
#   - npm/onnxsim/package.json   "version"
#   - rust/Cargo.toml            [workspace.package] version
#   - VERSION                    (root file; the Python sdist/wheel fallback
#                                 and the source CMakeLists.txt reads to bake
#                                 the version into the WASM module)
#
# Usage: check_version_sync.sh [expected-version]
#   expected-version: e.g. "0.8.0" (no leading "v"). Defaults to the latest
#   tag reachable from HEAD (git describe --tags --abbrev=0).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

if [ "$#" -ge 1 ] && [ -n "$1" ]; then
  EXPECTED="$1"
else
  EXPECTED=$(git -C "$ROOT_DIR" describe --tags --abbrev=0)
fi
EXPECTED="${EXPECTED#v}"

echo "Expected version: $EXPECTED"

NPM_VERSION=$(python3 -c "import json; print(json.load(open('$ROOT_DIR/npm/onnxsim/package.json'))['version'])")
RUST_VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT_DIR/rust/Cargo.toml" | head -n1)
FILE_VERSION=$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")

status=0

check() {
  local name="$1" actual="$2"
  if [ "$actual" != "$EXPECTED" ]; then
    echo "MISMATCH: $name is '$actual', expected '$EXPECTED'"
    status=1
  else
    echo "OK: $name is '$actual'"
  fi
}

check "npm/onnxsim/package.json" "$NPM_VERSION"
check "rust/Cargo.toml [workspace.package] version" "$RUST_VERSION"
check "VERSION" "$FILE_VERSION"

exit "$status"
