#!/bin/bash
# Build + run one of the hexagon-sim HMX programs here: ./sim/run.sh <prog.c> [args...]
# (login-free Hexagon_open_access 19 tools; libncurses.so.5 shim dir via NCSHIM)
set -euo pipefail
T="${HEXAGON_TOOLCHAIN:-$HOME/.cache/hexagon-oa-19/Tools}"
export LD_LIBRARY_PATH="${NCSHIM:-$HOME/.cache/ncshim}:${LD_LIBRARY_PATH:-}"
SRC="$1"; shift; OUT="${OUT:-${TMPDIR:-/tmp}/hmx_gemm_sim}"; mkdir -p "$OUT"
ELF="$OUT/$(basename "${SRC%.c}").elf"
"$T/bin/hexagon-clang" -mv69 -mhmx -mhvx -O2 "$SRC" -o "$ELF" -lm
"$T/bin/hexagon-sim" -mv69 --mhmx 1 "$ELF" -- "$@" 2>/dev/null | grep -vE "^\s+T[0-9]:|Total:|rev_id|^$"
