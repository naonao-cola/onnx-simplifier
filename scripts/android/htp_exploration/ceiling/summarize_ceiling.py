#!/usr/bin/env python3
"""Summarize run_ceiling.sh logs: steady-state median per model (runs after the first WARM),
per-layer time by differencing L=hi and L=lo, and per-layer TMAC/s.
usage: summarize_ceiling.py <modeldir> [perf_mode]"""
import json
import re
import statistics
import sys
from pathlib import Path

WARM = 3


def med(log):
    if not log.exists():
        return None, "missing"
    txt = log.read_text()
    if "PASS" not in txt:
        m = re.search(r"^FAIL (.*)$", txt, re.M)
        return None, (m.group(1)[:120] if m else "no PASS")
    ts = [float(x) for x in re.findall(r"^run \d+ ([\d.]+) ms", txt, re.M)]
    return statistics.median(ts[WARM:]), ""


def main():
    md = Path(sys.argv[1])
    perf = sys.argv[2] if len(sys.argv) > 2 else "burst"
    raw = md / "raw"
    tiny, _ = med(raw / f"tiny_L0.{perf}.log")
    print(f"tiny (Q->DQ only, fixed overhead): {tiny} ms\n")
    rows = []
    print("| shape | variant | GMAC/layer | t(L=lo) ms | t(L=hi) ms | per-layer ms | TMAC/s |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for e in json.loads((md / "manifest.json").read_text()):
        lo, elo = med(raw / f"{e['tag']}_L{e['lo']}.{perf}.log")
        hi, ehi = med(raw / f"{e['tag']}_L{e['hi']}.{perf}.log")
        g = e["macs_per_layer"] / 1e9
        shape = f"{e['kind']} {e['cin']}->{e['cout']} @{e['hw']}"
        if lo is None or hi is None:
            print(f"| {shape} | {e['variant']} | {g:.2f} | {lo} | {hi} | -- | FAIL: {elo or ehi} |")
            continue
        per = (hi - lo) / (e["hi"] - e["lo"])
        t = g / per if per > 0 else float("nan")
        rows.append({**e, "t_lo": lo, "t_hi": hi, "per_layer_ms": per, "tmacs": t})
        print(f"| {shape} | {e['variant']} | {g:.2f} | {lo:.2f} | {hi:.2f} | {per:.3f} | {t:.2f} |")
    (md / f"summary.{perf}.json").write_text(json.dumps({"tiny_ms": tiny, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
