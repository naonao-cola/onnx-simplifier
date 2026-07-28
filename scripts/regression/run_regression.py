#!/usr/bin/env python3
"""Run onnxsim (and onnxslim, for comparison) over a shard of the large-model set.

Each model is downloaded from the Hugging Face ``onnxmodelzoo`` org, run through
both simplifiers, correctness-checked, and then deleted before the next one, so
peak disk stays at roughly one model. Every model runs in its own subprocess and
each tool inside it runs in its own child with a hard timeout, so an OOM,
segfault, or hang on one model (or one tool) is contained and reported rather
than taking the whole job down.

onnxsim is the *authoritative* arm: the regression is about onnxsim's
*robustness*, and a model fails only when onnxsim crashes, times out, or produces
a graph onnxsim itself cannot validate. onnxslim is measured on the same graphs
purely for comparison -- its node counts, timing, peak memory and validity are
recorded (and summarized by ``summarize.py``) but never fail the job. Node counts
are also compared against the committed baseline for the summary, but drift there
does not fail the job (legitimate optimizer changes move those numbers).

Usage:
    run_regression.py --shard 0 --num-shards 6 --output shard-0.csv
    run_regression.py --slow-only --output slow.csv     # the known-slow models
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(HERE, "models.json")
DL_DIR = os.path.join(HERE, "dl")


def load_models():
    with open(MODELS_JSON) as f:
        return json.load(f)["models"]


def assign_shard(models, shard, num_shards):
    """Longest-processing-time balancing: sort by baseline duration descending
    and greedily place each model on the currently-lightest shard. Keeps shard
    wall-times close even though durations span 2s..600s."""
    buckets = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for m in sorted(models, key=lambda m: -m.get("baseline_seconds", 0.0)):
        i = min(range(num_shards), key=lambda i: loads[i])
        buckets[i].append(m)
        loads[i] += m.get("baseline_seconds", 0.0) or 0.0
    return buckets[shard]


def run_one(model_id, per_tool_timeout):
    """Run onnxsim + onnxslim on one model in an isolated worker subprocess.

    ``per_tool_timeout`` caps each tool individually inside the worker (each tool
    runs in its own child). The outer cap here is a generous backstop covering
    the download plus both tools, so a single-tool hang is reported by the worker
    rather than taking the whole model down."""
    t0 = time.time()
    backstop = None if per_tool_timeout <= 0 else per_tool_timeout * 2 + 600
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "worker.py"), model_id, DL_DIR,
         str(per_tool_timeout)],
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
            "error": (proc.stderr or proc.stdout or "no result line")[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--slow-only",
        action="store_true",
        help="run only the models marked known_slow (non-blocking job); "
        "otherwise known_slow models are excluded and the rest are sharded",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="per-tool wall-clock cap in seconds (each of onnxsim/onnxslim gets "
        "this budget); <=0 disables",
    )
    ap.add_argument("--output", default="regression.csv")
    args = ap.parse_args()

    models = load_models()
    if args.slow_only:
        selected = [m for m in models if m.get("known_slow")]
    else:
        pool = [m for m in models if not m.get("known_slow")]
        selected = assign_shard(pool, args.shard, args.num_shards)

    print(
        f"shard {args.shard}/{args.num_shards} | {len(selected)} models | "
        f"timeout {args.timeout}s",
        flush=True,
    )
    os.makedirs(DL_DIR, exist_ok=True)

    baseline = {m["id"]: m for m in models}
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
                "error": f">{args.timeout}s",
                "seconds": args.timeout,
            }
        finally:
            # Belt-and-braces cleanup in case the worker died before its own.
            shutil.rmtree(
                os.path.join(DL_DIR, model_id.replace("/", "__")),
                ignore_errors=True,
            )

        b = baseline.get(model_id, {})
        r["baseline_simp_nodes"] = b.get("baseline_simp_nodes")
        rows.append(r)

        status = r.get("status")
        on = r.get("orig_nodes")
        if status in ("ok", "unvalidated"):
            sn = r.get("simp_nodes")
            red = (100 * (on - sn) / on) if on and sn is not None else 0
            print(f"onnxsim {status}: {on}->{sn} ({red:+.0f}%) {r.get('seconds')}s", flush=True)
        else:
            print(f"onnxsim {status}: {str(r.get('error'))[:80]} {r.get('seconds')}s", flush=True)
        # onnxslim is comparison-only and never gates the run.
        sstatus, ssn = r.get("slim_status"), r.get("slim_simp_nodes")
        if sstatus in ("ok", "unvalidated") and ssn is not None:
            sred = (100 * (on - ssn) / on) if on else 0
            print(f"    onnxslim {sstatus}: {on}->{ssn} ({sred:+.0f}%) {r.get('slim_seconds')}s",
                  flush=True)
        elif sstatus:
            print(f"    onnxslim {sstatus}: {str(r.get('slim_error'))[:80]} {r.get('slim_seconds')}s",
                  flush=True)

        # A model fails the regression when it crashes, times out, or the
        # simplified graph does not pass onnxsim's own correctness check.
        if status not in ("ok",):
            failures.append((model_id, status, str(r.get("error"))[:200]))

    fields = [
        "model",
        "status",
        "orig_nodes",
        "simp_nodes",
        "baseline_simp_nodes",
        "reduction_pct",
        "seconds",
        "skipped_optimizers",
        "valid",
        "peak_rss_mb",
        "error",
        # onnxslim comparison arm
        "slim_status",
        "slim_simp_nodes",
        "slim_reduction_pct",
        "slim_seconds",
        "slim_valid",
        "slim_peak_rss_mb",
        "slim_error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            on = r.get("orig_nodes")
            sn = r.get("simp_nodes")
            r["reduction_pct"] = round(100 * (on - sn) / on, 1) if on and sn is not None else ""
            ssn = r.get("slim_simp_nodes")
            r["slim_reduction_pct"] = round(100 * (on - ssn) / on, 1) if on and ssn is not None else ""
            so = r.get("skipped_optimizers")
            r["skipped_optimizers"] = ",".join(so) if isinstance(so, list) else (so or "")
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for mid, st, err in failures:
            print(f"  - {mid}: {st} {err}", flush=True)
        return 1
    print("\nall models passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
