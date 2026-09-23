#!/usr/bin/env python3
"""Run the tinygrad Hexagon / Metal compatibility check over the model suite and report.

Drives ``worker.py`` once per model (isolated subprocess, hard timeout),
collects the JSON results, writes a CSV, prints a summary, and exits non-zero
if any model failed. Mirrors ``scripts/apple/run_coreml_compat.py`` and
``scripts/qualcomm/run_qnn_compat.py``.

A model **fails** the check when simplification breaks device compatibility
or changes the on-device result (``hexagon_regression`` / ``metal_regression``),
when onnxsim raises (``simplify_error``), or when the worker crashes/times
out. A graph the device backend cannot handle even *before* simplification is
``unsupported`` and is **reported, not failed** -- that is a backend
limitation, not an onnxsim bug.

If the device is unavailable on the host (no qemu-hexagon/clang for the mock
DSP path, non-macOS for Metal, or no tinygrad at all), every model reports
``skipped`` and the run passes (nothing to test) unless ``--require-device``
is given.

Usage:
    run_hexagon_compat.py --output hexagon-compat.csv
    run_hexagon_compat.py --device METAL --require-device
    run_hexagon_compat.py --models conv_bn_relu matmul_bias_tanh
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
import tinygrad_hexagon_backend as backend  # noqa: E402


def run_one(model_name: str, device: str, timeout: int) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "worker.py"),
                model_name,
                "--device",
                device,
            ],
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "device": device,
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
            "device": device,
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
        "--device",
        default=backend.HEXAGON_DEVICE,
        help="tinygrad device to check (default: DSP)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="per-model wall-clock cap in seconds; <=0 disables",
    )
    ap.add_argument(
        "--require-device",
        action="store_true",
        help="fail if the device is unavailable instead of skipping",
    )
    ap.add_argument("--output", default="hexagon-compat.csv")
    args = ap.parse_args()

    tag = "hexagon" if args.device != "METAL" else "metal"
    fail_statuses = {f"{tag}_regression", "simplify_error", "crash", "timeout", "error"}

    selected = args.models or models.names()
    print(
        f"tinygrad {args.device} compatibility check | {len(selected)} models",
        flush=True,
    )

    rows = []
    failures = []
    skipped = 0
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        r = run_one(name, args.device, args.timeout)
        rows.append(r)
        status = r.get("status")
        if status == "skipped":
            skipped += 1
            print(f"skipped ({r.get('error')})", flush=True)
            continue
        detail = ""
        if r.get("orig_nodes") is not None:
            detail = f"{r.get('orig_nodes')}->{r.get('simp_nodes')} nodes"
        if r.get("diff_vs_orig") is not None:
            detail += f", d(orig)={r.get('diff_vs_orig'):.2g}"
        print(f"{status} ({detail}) {r.get('seconds')}s", flush=True)
        if status in fail_statuses:
            failures.append((name, status, str(r.get("error"))[:200]))

    fields = [
        "model",
        "device",
        "status",
        "orig_nodes",
        "simp_nodes",
        "diff_vs_orig",
        "diff_vs_reference",
        "seconds",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if skipped == len(selected):
        msg = f"tinygrad {args.device} unavailable on this host; all models skipped."
        if args.require_device:
            print(f"\n{msg} (--require-device set -> failing)", flush=True)
            return 1
        print(f"\n{msg} Nothing to test; passing.", flush=True)
        return 0

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for name, status, err in failures:
            print(f"  - {name}: {status} {err}", flush=True)
        return 1
    passed = sum(1 for r in rows if r.get("status") == "ok")
    unsupported = sum(1 for r in rows if r.get("status") == "unsupported")
    print(
        f"\nall passed ({passed} ok, {unsupported} unsupported, {skipped} skipped)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
