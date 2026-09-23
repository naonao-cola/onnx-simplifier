#!/bin/bash
# Push map_run + the MSDA skel + a piece dir + frame dirs to the phone, run the chain, pull the outputs.
#   ./phone.sh <build dir> <piece dir> <frames root> <warmup> <reps> <frame index>...
# Run it under the host's phone lock (PHONE_LOCK_OWNER=... ~/.cache/android-phone/phone-run ./phone.sh ...).
# env: R (phone dir), DEC (htp|split), MSDA_FLAGS, QNN_PERF (default burst)
set -euo pipefail
B="$1"; P="$2"; F="$3"; WU="$4"; REPS="$5"; shift 5
R="${R:-/data/local/tmp/codex-android-maptr}"
A=(adb -s "${DEVICE_SERIAL:-239dbd8f}")
push() {  # push if the size differs
  local sz dsz; sz=$(stat -L -c %s "$1"); dsz=$("${A[@]}" shell "stat -c %s $2 2>/dev/null" | tr -d '\r' || true)
  [ "$sz" = "$dsz" ] || "${A[@]}" push -q "$1" "$2"
}
"${A[@]}" shell "mkdir -p $R/pieces"
"${A[@]}" push -q "$B/map_run" "$B/msda_rpc.so" "$R/"  # always: a rebuild can keep the size
"${A[@]}" shell "rm -f $R/pieces/*"
for f in "$P"/*; do "${A[@]}" push -q "$(readlink -f "$f")" "$R/pieces/$(basename "$f")"; done
dirs=()
for i in "$@"; do
  "${A[@]}" shell "mkdir -p $R/frames/$i"
  for f in "$F/$i"/*; do case "$(basename "$f")" in ref_bev*|ref_cls*|ref_pts*|*.out.f32) ;; *) push "$f" "$R/frames/$i/$(basename "$f")";; esac; done
  dirs+=("frames/$i")
done
"${A[@]}" shell "chmod 755 $R/map_run && cd $R && DEC=${DEC:-htp} MSDA_FLAGS=${MSDA_FLAGS:-4} QNN_PERF=${QNN_PERF:-burst} \
  LD_LIBRARY_PATH=$R ADSP_LIBRARY_PATH='$R;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' \
  ./map_run pieces $WU $REPS ${dirs[*]}"
for i in "$@"; do for n in bev cls pts; do "${A[@]}" pull -q "$R/frames/$i/$n.out.f32" "$F/$i/$n.out.f32"; done; done
