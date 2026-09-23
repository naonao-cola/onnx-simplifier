#!/bin/bash
# Push the probe (and, for health checks, a known-good msda_hvx build + case) and run one probe:
#   ./run.sh setup                         push everything
#   ./run.sh probe <hmx_client args after uri...>
#   ./run.sh health                        known-good skel must still pass (after every probe)
# Run under the host's phone lock: PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh ...
set -euo pipefail
S="${ANDROID_SERIAL:-239dbd8f}"
D="${D:-/data/local/tmp/codex-android-hmx-probe}"
B="${OUT:-$(cd "$(dirname "$0")" && pwd)/build}"
MSDA_BUILD="${MSDA_BUILD:-$HOME/.cache/msda_generic_build}"
MSDA_CASE="${MSDA_CASE:-$HOME/.cache/msda_generic_cases/bevformer_tsa}"
A=(adb -s "$S")
case "$1" in
  setup)
    "${A[@]}" shell "mkdir -p $D"
    "${A[@]}" push "$B/hmx_client" "$B/hmx_rpc.so" "$MSDA_BUILD/msda_client" "$MSDA_BUILD/msda_rpc.so" $D/ >/dev/null
    "${A[@]}" push "$MSDA_CASE" "$D/case" >/dev/null
    for f in "${@:2}"; do "${A[@]}" push "$f" $D/ >/dev/null; done
    "${A[@]}" shell "chmod 755 $D/hmx_client $D/msda_client"
    ;;
  push) shift; for f in "$@"; do "${A[@]}" push "$f" $D/ >/dev/null; done ;;
  probe)
    shift
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./hmx_client \
      'file:///hmx_rpc.so?hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' $*; echo exit=\$?"
    ;;
  pull) "${A[@]}" pull "$D/$2" "$3" >/dev/null ;;
  health)
    sleep 4
    "${A[@]}" shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 60 ./msda_client \
      'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' 3 0 4 case 2>&1 | tail -2; echo exit=\$?"
    ;;
esac
