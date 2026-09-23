#!/bin/bash
# Build + run hmx_sim.c on hexagon-sim (v69 ISS models HMX): ./sim.sh <args for hmx_sim.elf...>
set -euo pipefail
T="${HEXAGON_TOOLCHAIN:-$HOME/.cache/hexagon-oa-19/Tools}"
OUT="${OUT:-$(cd "$(dirname "$0")" && pwd)/build}"; mkdir -p "$OUT"
"$T/bin/hexagon-clang" -mv69 -mhmx -mhvx -O2 "$(dirname "$0")/hmx_sim.c" -o "$OUT/hmx_sim.elf"
"$T/bin/hexagon-sim" -mv69 --mhmx 1 "$OUT/hmx_sim.elf" -- "$@" 2>&1 | grep -vE "^\s+T[0-9]:|Total:|rev_id"
