#!/usr/bin/env bash
# Write a release version into every binding manifest that
# scripts/check_version_sync.sh verifies. Pure file edits only -- no git
# add/commit/push here -- so .github/workflows/version-sync.yml's "bump" job
# (and any maintainer testing this locally) decides what to do with the
# resulting diff.
#
# Updates:
#   - npm/onnxsim/package.json   via `npm version` (keeps npm's own
#                                 formatting/lockfile handling, same as a
#                                 maintainer running it by hand)
#   - rust/Cargo.toml            the `version = "..."` line under
#                                 [workspace.package]; member crates use
#                                 version.workspace = true and pick it up
#                                 automatically
#   - VERSION                    root file; Python sdist/wheel fallback and
#                                 the WASM build's baked-in version string
#                                 (see CMakeLists.txt)
#
# Usage: bump_binding_versions.sh <version>
#   version: e.g. "0.8.0" (no leading "v").
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

if [ "$#" -lt 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi
VERSION="${1#v}"

echo "Bumping binding manifests to $VERSION"

(
  cd "$ROOT_DIR/npm/onnxsim"
  npm version "$VERSION" --no-git-tag-version --allow-same-version
)

sed -i "s/^version = \".*\"/version = \"$VERSION\"/" "$ROOT_DIR/rust/Cargo.toml"

echo "$VERSION" > "$ROOT_DIR/VERSION"

echo "Done. Run scripts/check_version_sync.sh $VERSION to verify."
