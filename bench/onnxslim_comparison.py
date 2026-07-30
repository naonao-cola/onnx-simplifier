#!/usr/bin/env python3
"""Compare onnxsim against onnxslim on public models and print node-count deltas.

Usage:
    python bench/onnxslim_comparison.py swin_s_Opset18 FasterRCNN-10

Each argument is a model name from the Hugging Face ``onnxmodelzoo`` org
(``onnxmodelzoo/<name>`` containing ``<name>.onnx``) or a path to a local
``.onnx`` file. For every model the script runs both simplifiers, prints the
total node count before/after and the per-op-type differences, and (best effort)
reports whether each simplified model loads in onnxruntime.

Dependencies: ``onnxsim``, ``onnxslim``, ``onnx``, ``onnxruntime`` and
``huggingface_hub`` (only when fetching from the model zoo).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import onnx


def _resolve(name: str) -> str:
    if os.path.exists(name):
        return name
    if os.path.exists(f"{name}.onnx"):
        return f"{name}.onnx"
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=f"onnxmodelzoo/{name}", filename=f"{name}.onnx")


def _counts(model: onnx.ModelProto) -> Counter:
    return Counter(n.op_type for n in model.graph.node)


def _run_onnxsim(path: str):
    import onnxsim

    t = time.time()
    sim, ok = onnxsim.simplify(onnx.load(path))
    return sim, ok, time.time() - t


def _run_onnxslim(path: str):
    import onnxslim

    t = time.time()
    sim = onnxslim.slim(path)
    return sim, True, time.time() - t


def _loads(model: onnx.ModelProto) -> str:
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            ort.InferenceSession(f.name, so, providers=["CPUExecutionProvider"])
        return "loads"
    except Exception as e:  # noqa: BLE001
        return f"load-fail: {str(e)[:60]}"


def compare(name: str) -> None:
    path = _resolve(name)
    orig = onnx.load(path)
    oc = _counts(orig)
    print(f"\n===== {name} ({len(orig.graph.node)} nodes) =====")
    results = {}
    for tool, runner in (("onnxsim", _run_onnxsim), ("onnxslim", _run_onnxslim)):
        try:
            sim, ok, dt = runner(path)
        except Exception as e:  # noqa: BLE001
            print(f"[{tool}] FAILED: {type(e).__name__}: {str(e)[:80]}")
            continue
        results[tool] = _counts(sim)
        print(
            f"[{tool}] {len(orig.graph.node)} -> {len(sim.graph.node)} nodes "
            f"(ok={ok}, {dt:.0f}s, {_loads(sim)})"
        )
    if "onnxsim" in results and "onnxslim" in results:
        sim_c, slim_c = results["onnxsim"], results["onnxslim"]
        keys = set(oc) | set(sim_c) | set(slim_c)
        rows = [
            (k, oc.get(k, 0), sim_c.get(k, 0), slim_c.get(k, 0))
            for k in keys
            if sim_c.get(k, 0) != slim_c.get(k, 0)
        ]
        if rows:
            print(f"  {'op':18s} {'orig':>6s} {'onnxsim':>8s} {'onnxslim':>9s}")
            for k, o, a, b in sorted(rows, key=lambda r: -(abs(r[2] - r[3]))):
                print(f"  {k:18s} {o:6d} {a:8d} {b:9d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="model name(s) or path(s)")
    args = ap.parse_args()
    for name in args.models:
        compare(name)


if __name__ == "__main__":
    main()
