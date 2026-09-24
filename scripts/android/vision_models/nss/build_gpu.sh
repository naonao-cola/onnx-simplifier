#!/usr/bin/env bash
# Cross-compiles nss_run (OpenCL via dlopen of the vendor libOpenCL.so + ORT/QNN EP) and cl_bench (one
# OpenCL kernel timed on its own) for arm64 Android.
# Needs: the Android NDK, ../../htp_exploration/qnn_shell/{libs,headers} (fetch_libs.sh), and the Khronos
# OpenCL headers (CL_HEADERS, default /usr/include -- only CL/*.h is used).
set -euo pipefail
cd "$(dirname "$0")"
NDK_CXX="${NDK_CXX:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++}"
QNN=../../htp_exploration/qnn_shell
[ -d "$QNN/libs" ] || "$QNN/fetch_libs.sh"
CL_HEADERS="${CL_HEADERS:-/usr/include}"
OUT="${OUT:-build}"
mkdir -p "$OUT"
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$QNN/headers" -I "$CL_HEADERS" -o "$OUT/nss_run" nss_run.cpp \
  -L "$QNN/libs" -lonnxruntime -ldl
"$NDK_CXX" -O2 -std=c++17 -static-libstdc++ -I "$CL_HEADERS" -o "$OUT/cl_bench" cl_bench.cpp -ldl
echo "$OUT/nss_run $OUT/cl_bench"
