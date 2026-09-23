#!/bin/bash
# A/B the demo app's options on the phone: launches each configuration, lets it reach steady state,
# and reports median FPS / latency / stage times from the app's logcat lines (frames >= SKIP), plus
# startup: init ms (the engine's phases: env+EP, sessions+DSP, warm-up) and the first / tenth
# shown result, counted from process start.
#   ./bench.sh "<label>|<mode>|<pipe>|<opts>|<overlap true/false>" ...
# e.g. ./bench.sh "base|images|pipe_e_opt.txt||false" "lut|images|pipe_e_opt.txt|quant=lut|true"
set -euo pipefail
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
A=(adb -s "$DEVICE_SERIAL")
PKG=org.onnxsim.maskrcnndemo
RUN_S="${RUN_S:-45}"
SKIP="${SKIP:-40}"
printf '%-28s %6s %8s %8s %8s %8s %8s %8s %8s\n' config fps lat_ms pre bbone roi heads cpu stageB
for spec in "$@"; do
  IFS='|' read -r label mode pipe opts overlap <<<"$spec"
  "${A[@]}" shell am force-stop $PKG
  "${A[@]}" logcat -c
  "${A[@]}" shell am start -n $PKG/.MainActivity --es mode "$mode" --es pipe "$pipe" --es opts "'$opts'" \
    --ez overlap "$overlap" >/dev/null
  sleep "$RUN_S"
  "${A[@]}" logcat -d -s MaskRcnnDemo:V | python3 -c '
import sys, re, statistics as st
skip = int(sys.argv[2]); rows = []; start = {}; phases = ""
for l in sys.stdin:
    m = re.search(r"startup: result (\d+) shown (\d+) ms after process start \(init (\d+) ms\)", l)
    if m: start[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    m = re.search(r"init phases: env\+ep ([\d.]+), sessions\+dsp ([\d.]+), warmup ([\d.]+)", l)
    if m: phases = "ep %s sess %s warm %s" % m.groups()
    m = re.search(r"frame (\d+) fps ([\d.]+) lat ([\d.]+) \[([^\]]*)\]", l)
    if m and int(m.group(1)) >= skip:
        t = [float(x) for x in m.group(4).split(",")]
        rows.append([float(m.group(2)), float(m.group(3))] + t)
if not rows:
    print("%-28s no frames past %d" % (sys.argv[1], skip)); sys.exit()
med = lambda i: st.median(r[i] for r in rows)
# times: total pre backbone rpn roi heads cpu stageA stageB wait
print("%-28s %6.2f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f   (n=%d)" % (sys.argv[1], med(0), med(1), med(3), med(4),
      med(6), med(7), med(8), med(10), len(rows)))
if start:
    print("%-28s startup: init %s ms (%s), 1st result %s ms, 10th %s ms after process start" % ("", start.get(1, (0, 0))[1],
          phases, start.get(1, ("-",))[0], start.get(10, ("-",))[0]))
' "$label" "$SKIP"
done
"${A[@]}" shell am force-stop $PKG
