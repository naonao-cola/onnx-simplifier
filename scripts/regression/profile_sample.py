#!/usr/bin/env python3
"""Run onnxsim's built-in fixed-point profiler (``ONNXSIM_PROFILE``) over a
sample of the model-regression set to find where time currently goes.

Each model runs in its own subprocess (isolation from a hang/crash, matching
the regression harness's own worker.py pattern) with ``simplify(path,
profile=<json>)``, which makes onnxsim's C++ core print a per-span summary
table to stdout (see ``onnxsim/profiler.cpp:PrintSummary``) and also write a
Chrome trace. This script captures that table plus wall-clock/peak-RSS for
each model, and a follow-up "skip this pass" isolation run for the models
where the ``Optimize`` span dominates, then prints an aggregate report.

Usage:
    python scripts/regression/profile_sample.py MODEL_NAME [MODEL_NAME ...]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORK_DIR = os.environ.get("PROFILE_SAMPLE_DIR", "/tmp/onnxsim_profile_sample")


CHILD_SRC = r'''
import json, os, re, resource, sys, time
import onnx
from onnxsim import simplify

onnx_path = sys.argv[1]
profile_path = sys.argv[2]
skip = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None

model = onnx.load(onnx_path, load_external_data=True)
orig_nodes = len(model.graph.node)
t0 = time.time()
skipped = list(skip) if skip else []
while True:
    try:
        model_simp, check = simplify(model, skipped_optimizers=skipped or None,
                                      profile=profile_path)
        break
    except RuntimeError as e:
        m = re.search(r"passes/([A-Za-z0-9_]+)\.h", str(e))
        if not m or m.group(1) in skipped:
            raise
        skipped.append(m.group(1))
dt = time.time() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print("__CHILDRESULT__" + json.dumps({
    "orig_nodes": orig_nodes,
    "simp_nodes": len(model_simp.graph.node),
    "valid": bool(check),
    "seconds": round(dt, 2),
    "peak_rss_mb": round(peak, 1),
    "skipped_optimizers": skipped,
}))
'''


def run_child(onnx_path, profile_path, skip=None):
    args = [sys.executable, "-c", CHILD_SRC, onnx_path, profile_path, ",".join(skip or [])]
    proc = subprocess.run(args, capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__CHILDRESULT__"):
            result = json.loads(line[len("__CHILDRESULT__"):])
    return result, proc.stdout, proc.stderr


def parse_summary_table(stdout):
    """Pull the 'onnxsim profiling summary' text table into {name: row}."""
    lines = stdout.splitlines()
    rows = {}
    in_table = False
    for line in lines:
        if "onnxsim profiling summary" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("function") or line.startswith("---"):
            continue
        if not line.strip():
            break
        m = re.match(r"^(\s*)(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if not m:
            continue
        indent, name, calls, wall, cpu, max_wall, peak = m.groups()
        depth = len(indent) // 2
        rows[name.strip()] = {
            "depth": depth,
            "calls": int(calls),
            "wall_ms": float(wall),
            "cpu_ms": float(cpu),
            "max_wall_ms": float(max_wall),
            "peak_mib": float(peak),
        }
    return rows


def fmt_pct(part, whole):
    return f"{100*part/whole:5.1f}%" if whole else "   n/a"


def main():
    from model_zoo import fetch_model

    names = sys.argv[1:]
    if not names:
        print("usage: profile_sample.py MODEL_NAME [MODEL_NAME ...]", file=sys.stderr)
        return 1

    os.makedirs(WORK_DIR, exist_ok=True)
    all_rows = []

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            onnx_path = fetch_model(name)
        except Exception as e:
            print(f"  fetch FAILED: {e}", flush=True)
            continue
        print(f"  fetched in {time.time()-t0:.1f}s -> {onnx_path}", flush=True)

        profile_path = os.path.join(WORK_DIR, f"{name.replace('/', '__')}.json")
        result, out, err = run_child(onnx_path, profile_path)
        if result is None:
            print(f"  onnxsim CRASHED\n--- stdout ---\n{out[-2000:]}\n--- stderr ---\n{err[-2000:]}", flush=True)
            continue

        table = parse_summary_table(out)
        on, sn = result["orig_nodes"], result["simp_nodes"]
        print(f"  {on} -> {sn} nodes ({100*(on-sn)/on:+.1f}%), "
              f"{result['seconds']}s wall, peak {result['peak_rss_mb']} MiB, "
              f"valid={result['valid']}, skipped={result['skipped_optimizers']}", flush=True)

        top = table.get("Simplify") or table.get("Pipeline")
        total_ms = top["wall_ms"] if top else None
        print(f"  {'span':<20}{'calls':>7}{'wall(ms)':>12}{'%total':>9}{'peak(MiB)':>12}", flush=True)
        for span in ("Pipeline", "OptAndShape", "FoldConstant", "Fingerprint",
                     "Optimize", "InferShapes", "OrtSession", "OrtSessionInit",
                     "OrtSessionRun"):
            r = table.get(span)
            if not r:
                continue
            pct = fmt_pct(r["wall_ms"], total_ms) if total_ms else "   n/a"
            print(f"  {span:<20}{r['calls']:>7}{r['wall_ms']:>12.1f}{pct:>9}{r['peak_mib']:>12.1f}", flush=True)

        all_rows.append({
            "model": name,
            "orig_nodes": on,
            "simp_nodes": sn,
            "seconds": result["seconds"],
            "peak_rss_mb": result["peak_rss_mb"],
            "skipped_optimizers": result["skipped_optimizers"],
            "spans": table,
        })

    summary_path = os.path.join(WORK_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
