#!/usr/bin/env python3
"""Run the Pulsar2 (Axera NPU) coverage heuristic over the model suite and report.

Drives ``worker.py`` once per model (isolated subprocess, hard timeout),
collects the JSON results, writes a CSV, prints a summary, and exits non-zero
if any model failed.

Unlike the QNN/OpenVINO/MIGraphX checks, this needs no vendor package or
device -- see ``pulsar2_backend.py`` -- so there is no ``--require-*`` flag or
``skipped`` status: it always runs.

Usage:
    run_pulsar2_compat.py --output pulsar2-compat.csv
    run_pulsar2_compat.py --models conv_bn_relu matmul_bias_tanh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import models  # noqa: E402

FAIL_STATUSES = {
    "pulsar2_regression",
    "pulsar2_data_corrupted",
    "simplify_check_failed",
    "simplify_error",
    "crash",
    "timeout",
    "error",
}


def run_one(model_name: str, timeout: int) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"), model_name],
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "status": "timeout",
            "error": f">{timeout}s",
            "seconds": timeout,
        }
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__") :])
    if result is None:
        result = {
            "model": model_name,
            "status": "crash",
            "error": (proc.stderr or proc.stdout or "no result line")[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subset of model names to run (default: the whole suite)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-model wall-clock cap in seconds; <=0 disables",
    )
    ap.add_argument("--output", default="pulsar2-compat.csv")
    args = ap.parse_args()

    selected = args.models or models.names()
    print(f"Pulsar2 (Axera NPU) coverage check | {len(selected)} models", flush=True)

    rows = []
    failures = []
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        r = run_one(name, args.timeout)
        rows.append(r)
        status = r.get("status")
        detail = ""
        if r.get("orig_nodes") is not None:
            detail = f"{r.get('orig_nodes')}->{r.get('simp_nodes')} nodes"
        if r.get("coverage_orig") or r.get("coverage_simp"):
            detail += f", pulsar2={r.get('coverage_orig')}->{r.get('coverage_simp')}"
        print(f"{status} ({detail}) {r.get('seconds')}s", flush=True)
        if status in FAIL_STATUSES:
            failures.append((name, status, str(r.get("error"))[:200]))

    fields = [
        "model",
        "status",
        "orig_nodes",
        "simp_nodes",
        "coverage_orig",
        "coverage_simp",
        "new_blocking_ops",
        "seconds",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for name, status, err in failures:
            print(f"  - {name}: {status} {err}", flush=True)
        return 1
    passed = sum(1 for r in rows if r.get("status") == "ok")
    partial = sum(1 for r in rows if r.get("coverage_simp") == "partial")
    print(f"\nall passed ({passed} ok, {partial} partial coverage)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
