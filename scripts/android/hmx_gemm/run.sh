#!/bin/bash
# Push + run the HMX GEMM skel on the phone, with the known-good msda health check:
#   ./run.sh setup | ./run.sh gemm <mode> <M> <K> <N> [iters] [bias] | ./run.sh health
# Run only under the host phone lock: PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"
D="${D:-/data/local/tmp/codex-android-hmx-gemm}"
B="${OUT:-$(cd "$(dirname "$0")" && pwd)/build}"
MSDA_BUILD="${MSDA_BUILD:-$HOME/.cache/msda_generic_build}"
MSDA_CASE="${MSDA_CASE:-$HOME/.cache/msda_generic_cases/bevformer_tsa}"
A=(adb -s "$S")
case "$1" in
  setup)
    "${A[@]}" shell "mkdir -p $D"
    "${A[@]}" push "$B/hmx_gemm_client" "$B/hmx_gemm_rpc.so" "$MSDA_BUILD/msda_client" "$MSDA_BUILD/msda_rpc.so" $D/ >/dev/null
    "${A[@]}" push "$MSDA_CASE" "$D/case" >/dev/null
    "${A[@]}" shell "chmod 755 $D/hmx_gemm_client $D/msda_client" ;;
  gemm)
    shift
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout ${GEMM_TIMEOUT:-30} ./hmx_gemm_client \
      'file:///hmx_gemm_rpc.so?hmx_gemm_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $*; echo exit=\$?" ;;
  health)
    sleep 2
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./msda_client \
      'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' 3 0 4 case 2>&1 | tail -1; echo exit=\$?" ;;
  clean) "${A[@]}" shell "rm -rf $D" ;;
esac
