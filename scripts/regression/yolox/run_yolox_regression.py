#!/usr/bin/env python3
"""Regression test: run onnxsim over YOLOX (Megvii-BaseDetection) exports.

YOLOX is a long-standing onnxsim consumer -- its own ``tools/export_onnx.py``
calls ``onnxsim.simplify`` and asserts the correctness check passes. This
script reproduces that path as a stand-alone regression: for each variant it
exports the torch model to a *raw* ONNX graph, then runs ``onnxsim.simplify``
with its built-in correctness check (``check_n``) and records the node-count
reduction, validity, wall-clock and peak RSS.

Each graph is **also** run through `onnxslim <https://github.com/inisis/OnnxSlim>`_
on the same export, purely for comparison -- exactly as the onnxmodelzoo harness
does. onnxslim's numbers (node reduction, timing, best-effort equivalence check)
are recorded but **never** gate the run.

A variant *fails* when **onnxsim** crashes/aborts, hangs, or the simplified graph
does not pass onnxsim's own check -- the same failure definition as the
onnxmodelzoo harness in ``scripts/regression/``. Node-count drift and any
onnxslim outcome are reported but never fail the run.

Requirements (not onnxsim deps -- install separately):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install loguru thop tabulate tqdm psutil opencv-python-headless
    git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git
    pip install onnx onnxruntime onnxslim .   # onnxsim under test + comparison arm

Then, with the YOLOX checkout on PYTHONPATH:

    PYTHONPATH=/path/to/YOLOX python scripts/regression/yolox/run_yolox_regression.py \
        --download --workdir /tmp/yolox-reg --opset 11

Weights are the official 0.1.1rc0 release checkpoints; ``--download`` fetches
any that are missing from ``--weights-dir``.
"""
import argparse
import csv
import json
import os
import resource
import sys
import time
import traceback
import urllib.request
from collections import Counter

import torch
from torch import nn
import onnx
import onnxsim

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

# variant name -> release weight file
VARIANTS = [
    ("yolox-nano", "yolox_nano.pth"),
    ("yolox-tiny", "yolox_tiny.pth"),
    ("yolox-s", "yolox_s.pth"),
]
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"


def maybe_download(weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    for _, wfile in VARIANTS:
        dst = os.path.join(weights_dir, wfile)
        if os.path.exists(dst):
            continue
        url = f"{RELEASE}/{wfile}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dst)


def op_hist(model):
    return Counter(n.op_type for n in model.graph.node)


def run_onnxslim(raw_path, raw_nodes):
    """Comparison arm: optimize with onnxslim + a best-effort equivalence check.

    Mirrors scripts/regression/worker.py -- slim() mutates its input so each call
    gets a pristine reload, and model_check=True returns None when the slimmed
    graph fails onnxslim's own onnxruntime check. A check that *errors* (e.g. the
    model needs explicit input shapes) leaves validity unknown rather than
    counting as an onnxslim failure. Never affects the run's exit code.
    """
    import onnxslim
    out = {}
    try:
        t0 = time.perf_counter()
        model_simp = onnxslim.slim(onnx.load(raw_path))
        out["slim_seconds"] = round(time.perf_counter() - t0, 2)
        out["slim_simp_nodes"] = len(model_simp.graph.node)
        out["slim_reduction_pct"] = round(
            100.0 * (raw_nodes - out["slim_simp_nodes"]) / raw_nodes, 1
        )
        out["slim_status"] = "ok"
    except Exception as e:
        out["slim_status"] = "error"
        out["slim_error"] = f"{type(e).__name__}: {e}"
        return out
    if os.environ.get("SLIM_CHECK", "1") != "0":
        try:
            checked = onnxslim.slim(onnx.load(raw_path), model_check=True)
            out["slim_valid"] = checked is not None
        except Exception as e:
            out["slim_valid"] = None  # unknown, not a failure
            out["slim_check_error"] = f"{type(e).__name__}: {e}"
    return out


def export_raw(exp_name, ckpt_path, out_path, opset):
    exp = get_exp(None, exp_name)
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    # YOLOX ships its own SiLU so the graph is exportable on old opsets.
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    dummy = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["images"], output_names=["output"],
        opset_version=opset, dynamo=False,
    )
    return exp.test_size


def run_one(exp_name, ckpt_path, workdir, opset):
    raw_path = os.path.join(workdir, f"{exp_name}_raw.onnx")
    simp_path = os.path.join(workdir, f"{exp_name}_sim.onnx")
    rec = {"variant": exp_name, "opset": opset}
    try:
        test_size = export_raw(exp_name, ckpt_path, raw_path, opset)
        raw = onnx.load(raw_path)
        rec["input_hw"] = f"{test_size[0]}x{test_size[1]}"
        rec["raw_nodes"] = len(raw.graph.node)

        t0 = time.perf_counter()
        model_simp, check = onnxsim.simplify(raw)
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        rec["simp_nodes"] = len(model_simp.graph.node)
        rec["valid"] = bool(check)
        rec["reduction_pct"] = round(
            100.0 * (rec["raw_nodes"] - rec["simp_nodes"]) / rec["raw_nodes"], 1
        )
        onnx.save(model_simp, simp_path)
        onnx.checker.check_model(simp_path)  # structural validity of saved graph
        rec["status"] = "ok" if check else "check_failed"
        rec["top_ops_removed"] = json.dumps(
            (op_hist(raw) - op_hist(model_simp)).most_common(5)
        )
    except Exception as e:  # crash / C++ abort surfaces here
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["valid"] = False
        traceback.print_exc()
    rec["peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )

    # Comparison arm -- only if onnxsim got a raw graph to work from.
    if "raw_nodes" in rec:
        try:
            rec.update(run_onnxslim(raw_path, rec["raw_nodes"]))
        except Exception as e:  # onnxslim not installed, ...
            rec["slim_status"] = "error"
            rec["slim_error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", default="yolox-weights",
                    help="directory holding / receiving the .pth checkpoints")
    ap.add_argument("--download", action="store_true",
                    help="fetch any missing weights from the YOLOX release")
    ap.add_argument("--workdir", default="yolox-reg-work")
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--output", default="yolox-regression.csv")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    if args.download:
        maybe_download(args.weights_dir)

    rows = []
    for exp_name, wfile in VARIANTS:
        ckpt = os.path.join(args.weights_dir, wfile)
        print(f"\n=== {exp_name} (opset {args.opset}) ===", flush=True)
        rec = run_one(exp_name, ckpt, args.workdir, args.opset)
        print(json.dumps(rec, indent=2), flush=True)
        rows.append(rec)

    fields = ["variant", "opset", "input_hw", "raw_nodes",
              "simp_nodes", "reduction_pct", "valid", "seconds",
              "slim_simp_nodes", "slim_reduction_pct", "slim_valid",
              "slim_seconds", "slim_status",
              "peak_rss_mb", "status", "top_ops_removed", "error", "slim_error"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {args.output}", flush=True)

    failed = [r for r in rows if r["status"] != "ok" or not r.get("valid")]
    print(f"\nSUMMARY: {len(rows) - len(failed)}/{len(rows)} passed onnxsim")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
