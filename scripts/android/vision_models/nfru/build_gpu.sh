#!/usr/bin/env bash
# Cross-compiles nfru_run (OpenCL via dlopen of the vendor libOpenCL.so, ../nss/cl_dl.h + ORT/QNN EP) for
# arm64 Android. Needs the Android NDK, ../../htp_exploration/qnn_shell/{libs,headers} (fetch_libs.sh) and
# the Khronos OpenCL headers (CL_HEADERS, default /usr/include -- only CL/*.h is used).
set -euo pipefail
cd "$(dirname "$0")"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
QNN=../../htp_exploration/qnn_shell
[ -d "$QNN/libs" ] || "$QNN/fetch_libs.sh"
CL_HEADERS="${CL_HEADERS:-/usr/include}"
OUT="${OUT:-build}"
mkdir -p "$OUT"
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I ../nss -I "$QNN/headers" -I "$CL_HEADERS" -o "$OUT/nfru_run" \
  nfru_run.cpp -L "$QNN/libs" -lonnxruntime -ldl
echo "$OUT/nfru_run"
