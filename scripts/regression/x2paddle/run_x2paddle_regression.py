#!/usr/bin/env python3
"""Run the onnxsim -> X2Paddle downstream regression over a shard of the set.

For each model (see ``models.json``) the worker converts the graph to
PaddlePaddle twice -- once from the original ONNX and once after
``onnxsim.simplify`` -- each conversion isolated in its own subprocess. onnxsim
is X2Paddle's built-in optimize step, so this catches the case that matters
most: an onnxsim change that X2Paddle could digest before but can't anymore.

A model **fails** the regression when:

* onnxsim itself crashes, times out, or produces a graph it can't validate
  (``onnxsim_fail``); or
* X2Paddle converts the original graph but **not** the onnxsim-simplified one
  (``regression``) -- onnxsim broke a working downstream conversion.

A model does **not** fail when X2Paddle can't convert the original either
(``baseline_unsupported``): that's an X2Paddle limitation on that model, not an
onnxsim regression. It's still recorded so the set's coverage is visible. The
``improved`` verdict (original fails, simplified converts) is likewise recorded,
not gated.

Usage:
    run_x2paddle_regression.py --shard 0 --num-shards 2 --output shard-0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(HERE, "models.json")
DL_DIR = os.path.join(HERE, "dl")

GATING_VERDICTS = ("onnxsim_fail", "regression")


def load_models():
    with open(MODELS_JSON) as f:
        return json.load(f)["models"]


def assign_shard(models, shard, num_shards):
    """Longest-processing-time balancing on baseline duration, so shards keep
    close wall-times even when per-model cost varies."""
    buckets = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for m in sorted(models, key=lambda m: -(m.get("baseline_seconds") or 0.0)):
        i = min(range(num_shards), key=lambda i: loads[i])
        buckets[i].append(m)
        loads[i] += m.get("baseline_seconds") or 0.0
    return buckets[shard]


def run_one(model_id, per_step_timeout):
    """Run one model through the worker in an isolated subprocess."""
    backstop = None if per_step_timeout <= 0 else per_step_timeout * 3 + 600
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "worker.py"),
            model_id,
            DL_DIR,
            str(per_step_timeout),
        ],
        capture_output=True,
        text=True,
        timeout=backstop,
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__") :])
    if result is None:
        result = {
            "model": model_id,
            "status": "crash",
            "verdict": "error",
            "error": (proc.stderr or proc.stdout or "no result line")[-400:],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="per-step wall-clock cap in seconds (onnxsim and each conversion "
        "get this budget individually); <=0 disables",
    )
    ap.add_argument("--output", default="x2paddle-regression.csv")
    args = ap.parse_args()

    models = load_models()
    selected = assign_shard(models, args.shard, args.num_shards)
    print(
        f"shard {args.shard}/{args.num_shards} | {len(selected)} models | "
        f"timeout {args.timeout}s",
        flush=True,
    )
    os.makedirs(DL_DIR, exist_ok=True)

    rows = []
    failures = []
    for i, m in enumerate(selected, 1):
        model_id = m["id"]
        print(f"[{i}/{len(selected)}] {model_id} ...", end=" ", flush=True)
        try:
            r = run_one(model_id, args.timeout)
        except subprocess.TimeoutExpired:
            r = {
                "model": model_id,
                "status": "timeout",
                "verdict": "onnxsim_fail",
                "error": f">{args.timeout}s",
            }
        finally:
            shutil.rmtree(
                os.path.join(DL_DIR, model_id.replace("/", "__")), ignore_errors=True
            )
        rows.append(r)

        v = r.get("verdict")
        on, sn = r.get("orig_nodes"), r.get("simp_nodes")
        red = (100 * (on - sn) / on) if on and sn is not None else 0
        bconv = r.get("baseline_conv_status")
        sconv = r.get("simp_conv_status")
        print(
            f"{v} | onnxsim {r.get('onnxsim_status')} {on}->{sn} ({red:+.0f}%) | "
            f"x2paddle base={bconv} simp={sconv}",
            flush=True,
        )
        if v not in ("pass",):
            detail = r.get("simp_error") or r.get("baseline_error") or r.get("error")
            if detail:
                print(f"      {str(detail)[:160]}", flush=True)
        if v in GATING_VERDICTS or r.get("status") == "error":
            failures.append(
                (model_id, v, str(r.get("simp_error") or r.get("error"))[:200])
            )

    fields = [
        "model",
        "verdict",
        "status",
        "orig_nodes",
        "simp_nodes",
        "reduction_pct",
        "onnxsim_status",
        "onnxsim_valid",
        "onnxsim_seconds",
        "skipped_optimizers",
        "onnxsim_peak_rss_mb",
        "baseline_conv_status",
        "baseline_ops",
        "baseline_conv_seconds",
        "baseline_peak_rss_mb",
        "baseline_error",
        "simp_conv_status",
        "simp_ops",
        "simp_conv_seconds",
        "simp_peak_rss_mb",
        "simp_error",
        "orig_bytes",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            on, sn = r.get("orig_nodes"), r.get("simp_nodes")
            r["reduction_pct"] = (
                round(100 * (on - sn) / on, 1) if on and sn is not None else ""
            )
            so = r.get("skipped_optimizers")
            r["skipped_optimizers"] = (
                ",".join(so) if isinstance(so, list) else (so or "")
            )
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for mid, v, err in failures:
            print(f"  - {mid}: {v} {err}", flush=True)
        return 1
    print("\nall models passed (no onnxsim-caused X2Paddle regressions)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
