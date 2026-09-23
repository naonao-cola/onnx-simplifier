#!/bin/bash
# Build the Mask R-CNN demo APK.
#   HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... ANDROID_HOME=~/android-sdk ./build_app.sh
# 1. ORT + QNN EP + Qualcomm QNN runtime libs (Maven Central) via ../htp_exploration/qnn_shell/fetch_libs.sh
# 2. the two Hexagon FastRPC skels + their ARM stubs, built by ../e2e_pipeline/build.sh (BUILD_ONLY)
# 3. native/maskrcnn_engine.cpp (includes ../e2e_pipeline/e2e_run.cpp) -> libmaskrcnn_demo.so
# 4. everything into app/src/main/jniLibs/arm64-v8a, then gradle assembleDebug (offline).
set -euo pipefail
: "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
ANDROID_HOME="${ANDROID_HOME:-$HOME/android-sdk}"
NDK="${NDK:-$ANDROID_HOME/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin}"
HERE="$(cd "$(dirname "$0")" && pwd)"
QS="$HERE/../htp_exploration/qnn_shell"
E2E="$HERE/../e2e_pipeline"
B="$HERE/native/out"
J="$HERE/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$B" "$J"
[ -f "$QS/libs/libQnnHtp.so" ] || "$QS/fetch_libs.sh"
# skels + stubs (e2e build.sh builds its own driver too; OUT/IMGS are unused with BUILD_ONLY)
BUILD="$B/e2e" BUILD_ONLY=1 OUT=/nonexistent IMGS=/nonexistent NDK="$NDK" "$E2E/build.sh"
INC=(-I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc"
     -I "$B/e2e/rpn_fused" -I "$B/e2e/roi")
"$NDK/aarch64-linux-android29-clang++" -O2 -std=c++17 -shared -fPIC -static-libstdc++ -I "$QS/headers" "${INC[@]}" \
  -o "$J/libmaskrcnn_demo.so" "$HERE/native/maskrcnn_engine.cpp" \
  "$B/e2e/rpn_glue.o" "$B/e2e/rpn_stub.o" "$B/e2e/roi_stub.o" \
  -L "$QS/libs" -lonnxruntime -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
  -ljnigraphics -llog -Wl,--no-undefined
cp "$QS"/libs/*.so "$J/"
cp "$B/e2e/rpn_fused/rpn_rpc.so" "$J/librpn_rpc.so"      # jniLibs must be lib*.so to be extracted
cp "$B/e2e/roi/roialign_rpc.so" "$J/libroialign_rpc.so"
printf 'sdk.dir=%s\n' "$ANDROID_HOME" > "$HERE/local.properties"
GRADLE="${GRADLE:-$(ls -d "$HOME"/.gradle/wrapper/dists/gradle-*-bin/*/gradle-*/bin/gradle 2>/dev/null | tail -1)}"
ANDROID_HOME="$ANDROID_HOME" "${GRADLE:-gradle}" --offline -q -p "$HERE" assembleDebug
ls -la "$HERE/app/build/outputs/apk/debug/"*.apk
