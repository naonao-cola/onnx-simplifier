#!/usr/bin/env bash
# Fetches a pinned, prebuilt llama.cpp release (llama-cli + the ggml CPU
# backend) for use as an *external reference oracle* -- e.g. running a real
# GGUF checkpoint through actual llama.cpp inference to check logits against
# whatever onnxsim reconstructs from the same file.
#
# This is deliberately NOT how onnxsim gets llama.cpp: there is no
# third_party/llama.cpp submodule and nothing here is a build dependency of
# the CMake project or the Python wheel (see CLAUDE.md's ONNXSIM_BUILTIN_ORT
# note for the same philosophy re: ONNX Runtime -- optional tooling stays
# out-of-tree and is fetched on demand, never compiled into onnxsim itself).
# Test code that wants this oracle should invoke this script (or call
# llama_cpp_bin_dir() below) and skip gracefully if it exits non-zero.
#
# llama.cpp has no semver -- every commit to master bumps a monotonic build
# tag ("b12345"). LLAMA_CPP_VERSION pins us to one specific, previously
# verified tag; bump it deliberately (and re-verify), never silently track
# "latest".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-b10642}"
CACHE_DIR="${LLAMA_CPP_CACHE_DIR:-${REPO_ROOT}/.cache/llama-cpp}"
DEST_DIR="${CACHE_DIR}/${LLAMA_CPP_VERSION}"

# Exit code 2 means "no prebuilt asset for this platform" -- callers (tests)
# should treat that as skip-the-oracle, not a hard failure. Everything else
# non-zero is a real error (download/extract failure).
EXIT_UNSUPPORTED_PLATFORM=2

detect_asset() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "${os}" in
    Linux)
      case "${arch}" in
        x86_64) echo "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz" ;;
        aarch64 | arm64) echo "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-arm64.tar.gz" ;;
        s390x) echo "llama-${LLAMA_CPP_VERSION}-bin-ubuntu-s390x.tar.gz" ;;
        *) return 1 ;;
      esac
      ;;
    Darwin)
      case "${arch}" in
        arm64) echo "llama-${LLAMA_CPP_VERSION}-bin-macos-arm64.tar.gz" ;;
        x86_64) echo "llama-${LLAMA_CPP_VERSION}-bin-macos-x64.tar.gz" ;;
        *) return 1 ;;
      esac
      ;;
    *)
      # Windows (MSYS/Git Bash/etc.) ships as .zip, not .tar.gz, and this
      # script only knows how to unpack tar.gz below -- treat as unsupported
      # here rather than half-handling it.
      return 1
      ;;
  esac
}

# Prints the directory containing llama-cli (and friends) on stdout,
# downloading/extracting first if needed. Exits EXIT_UNSUPPORTED_PLATFORM if
# this host has no matching prebuilt asset.
llama_cpp_bin_dir() {
  local asset
  if ! asset="$(detect_asset)"; then
    echo "fetch_llama_cpp.sh: no prebuilt ${LLAMA_CPP_VERSION} asset for" \
      "$(uname -s)/$(uname -m) -- skip the llama.cpp reference oracle" >&2
    return "${EXIT_UNSUPPORTED_PLATFORM}"
  fi

  local bin_dir="${DEST_DIR}/build/bin"
  if [[ -x "${bin_dir}/llama-cli" ]]; then
    echo "${bin_dir}"
    return 0
  fi

  local url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${asset}"
  mkdir -p "${DEST_DIR}"
  local archive="${DEST_DIR}/${asset}"

  echo "fetch_llama_cpp.sh: downloading ${url}" >&2
  curl -fL --retry 3 --retry-connrefused -o "${archive}.part" "${url}"
  mv "${archive}.part" "${archive}"

  echo "fetch_llama_cpp.sh: extracting ${archive}" >&2
  tar -xzf "${archive}" -C "${DEST_DIR}"
  rm -f "${archive}"

  # The release tarball's top-level entries are named llama-<tag>/... (see
  # ggml-org/llama.cpp's release workflow, e.g. "-s ,^\.,llama-<tag>," in its
  # `tar` invocation), not "build/bin/..." -- normalize so callers always
  # find the binaries at the same DEST_DIR/build/bin path regardless of the
  # exact archive layout for this release.
  if [[ ! -x "${bin_dir}/llama-cli" ]]; then
    local extracted
    extracted="$(find "${DEST_DIR}" -maxdepth 2 -name 'llama-cli' -print -quit)"
    if [[ -z "${extracted}" ]]; then
      echo "fetch_llama_cpp.sh: llama-cli not found after extracting ${archive}" >&2
      return 1
    fi
    mkdir -p "${bin_dir}"
    ln -sf "$(dirname "${extracted}")"/* "${bin_dir}/"
  fi

  echo "${bin_dir}"
}

# Only run as a driver when executed directly (`./fetch_llama_cpp.sh`), not
# when sourced by test code that just wants the llama_cpp_bin_dir function.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  llama_cpp_bin_dir
fi
