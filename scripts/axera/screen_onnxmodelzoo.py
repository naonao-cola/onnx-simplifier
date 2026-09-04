#!/usr/bin/env python3
"""Screen `onnxmodelzoo` models for AX650 compatibility -- no Docker, no device.

Fetches each named model via `scripts/regression/model_zoo.py`, then reports
`pulsar2_simulator.coverage()`/`partition()` and `pulsar2_backend.
ax650_build_risks()` for both the original ONNX and its onnxsim-simplified
twin. This is the fast, offline-capable screening step: it predicts whether
a real `pulsar2 build --target_hardware AX650` is likely to succeed and stay
fully on-NPU, using the confirmed real op list (`pulsar2_ops.
AX650_SUPPORTED_OPS`) -- see `pulsar2_ops.py`'s docstring for the two real
conversions (`resnet18d_Opset18`, `googlenet-6`) this was validated against.

It does not attempt a real `pulsar2 build` or touch real hardware -- use
`worker.py`/`run_pulsar2_compat.py` (the static onnxsim-regression check) or
a real Docker + device setup for that, following the pattern in this
directory's README.

Not every onnxmodelzoo model is a plain single-image classifier (NLP models
like `bert_Opset18` take token-id inputs; detection models like
`FasterRCNN-10` have multiple inputs/outputs) -- this script only looks at
op-type coverage, which is meaningful regardless of input shape, but a
"full" or "partial" result says nothing about whether the *deployment*
config (preprocessing, output postprocessing) would also need work.

Usage:
    python screen_onnxmodelzoo.py --models resnet18d_Opset18 googlenet-6
    python screen_onnxmodelzoo.py --all --output screen.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_REGRESSION_DIR = os.path.join(os.path.dirname(HERE), "regression")
if _REGRESSION_DIR not in sys.path:
    sys.path.insert(0, _REGRESSION_DIR)


def screen_one(name: str) -> dict:
    import onnx

    import model_zoo
    import pulsar2_backend as pulsar2
    import pulsar2_simulator as sim

    res = {
        "model": name,
        "status": "error",
        "nodes": None,
        "coverage": None,
        "npu_node_fraction": None,
        "cpu_op_types": None,
        "build_risks": None,
        "simplified_coverage": None,
        "simplified_build_risks": None,
        "error": None,
    }
    try:
        path = model_zoo.fetch_model(name)
        model = onnx.load(path)
        res["nodes"] = len(model.graph.node)

        p = sim.partition(model)
        res["coverage"] = sim.coverage(model)
        res["npu_node_fraction"] = round(p.npu_node_fraction, 3)
        res["cpu_op_types"] = dict(p.cpu_op_types)
        res["build_risks"] = pulsar2.ax650_build_risks(model)

        try:
            from onnxsim import simplify

            simp, _ = simplify(model)
            res["simplified_coverage"] = sim.coverage(simp)
            res["simplified_build_risks"] = pulsar2.ax650_build_risks(simp)
        except Exception as exc:
            res["simplified_coverage"] = f"simplify_error: {exc}"

        res["status"] = "ok"
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="model short names")
    ap.add_argument(
        "--all", action="store_true", help="screen every model in models.json"
    )
    ap.add_argument("--output", default="pulsar2-screen.csv")
    args = ap.parse_args()

    import model_zoo

    if args.all:
        selected = [m.split("/", 1)[1] for m in model_zoo.list_models()]
    elif args.models:
        selected = args.models
    else:
        ap.error("pass --models NAME [NAME ...] or --all")
        return 2

    rows = []
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        r = screen_one(name)
        rows.append(r)
        if r["status"] == "ok":
            print(
                f"{r['coverage']} ({r['npu_node_fraction']:.0%} NPU-eligible nodes)"
                + (f", risks={r['build_risks']}" if r["build_risks"] else "")
            )
        else:
            print(f"error: {r['error']}")

    fields = [
        "model",
        "status",
        "nodes",
        "coverage",
        "npu_node_fraction",
        "cpu_op_types",
        "build_risks",
        "simplified_coverage",
        "simplified_build_risks",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)")

    full = sum(1 for r in rows if r.get("coverage") == "full")
    partial = sum(1 for r in rows if r.get("coverage") == "partial")
    errors = sum(1 for r in rows if r["status"] != "ok")
    print(f"{full} full, {partial} partial, {errors} errored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
