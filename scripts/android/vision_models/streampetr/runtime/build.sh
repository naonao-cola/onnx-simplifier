#!/bin/bash
# Build petr_run against ../../../htp_exploration/qnn_shell's ORT (run its fetch_libs.sh first).
set -euo pipefail
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
SRC="$(cd "$(dirname "$0")" && pwd)"
QS="$SRC/../../../htp_exploration/qnn_shell"
OUT="${OUT:-$SRC/build}"
mkdir -p "$OUT"
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$QS/headers" -o "$OUT/petr_run" "$SRC/petr_run.cpp" -L "$QS/libs" -lonnxruntime
