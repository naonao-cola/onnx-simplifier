#!/usr/bin/env python3
"""Regression test: run onnxsim's output through tpu-mlir on real YOLOX exports.

``tests/test_tpu_mlir_integration.py`` covers the small, synthetic side of
tpu-mlir's own onnxsim dependency (see that module's docstring for the full
context: tpu-mlir's ``OnnxConverter.model_simplify()`` calls
``onnxsim.simplify()`` directly, with code comments citing real bugs it hit
running "yolox" and "ppyolo_tiny" against the old onnxsim version it pins).
This script is the real-model counterpart, mirroring
``scripts/regression/yolox/run_yolox_regression.py``'s own export step: it
reuses the official YOLOX 0.1.1rc0 checkpoints (already confirmed to pass
onnxsim cleanly, 6/6, in that harness) and additionally pushes each export all
the way through tpu-mlir's real Top-MLIR ingestion + ``tpuc-opt``
canonicalization + ``pymlir`` interpretation, checking that the result still
agrees with onnx's own reference evaluator.

(``ppyolo_tiny`` is not covered here: a real PaddleDetection ``ppyolo_tiny``
checkpoint was obtained and traced during this project's own investigation of
tpu-mlir's onnxsim dependency far enough to confirm its detection head is
NMS-based -- the same category of graph as onnxsim issue #60's
``PrepareForReduce`` crash -- but paddle2onnx has never supported the specific
op version (bare ``multiclass_nms``, pre-``multiclass_nms3``) that 2021-era
checkpoint uses, on any released paddle2onnx version. That is a permanent
paddle2onnx limitation, not something a regression harness here can route
around.)

A variant *fails* when onnxsim's own check fails, or when tpu-mlir's ingestion,
canonicalization, or interpretation of the *simplified* graph raises where it
didn't for the original -- the same "does simplification make this harder"
contract every other backend-integration test in this repo checks.

Requirements (not onnxsim's own deps -- install separately; see
``.github/workflows/backend-integration.yml``'s ``tpu_mlir`` job for the exact
mutually-compatible pin set this mirrors):

    # tpu_mlir is Python-3.10-only and only really runs on Ubuntu 22.04 (its
    # compiled tpuc-opt/pymlir extensions segfault against a newer glibc/
    # libstdc++ ABI) -- run this inside an ubuntu:22.04 container/chroot, not
    # directly on a newer host.
    pip install "tpu_mlir[onnx]"          # pins onnx==1.14.1, onnxruntime==1.16.3,
                                           # onnxsim==0.4.17, numpy==1.24.3, protobuf==3.20.3
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    pip install loguru thop tabulate tqdm psutil opencv-python-headless
    git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git
    pip install --force-reinstall --no-deps <onnxsim-under-test>.whl   # swap in current onnxsim

Then, with the YOLOX checkout on PYTHONPATH:

    PYTHONPATH=/path/to/YOLOX python scripts/regression/tpu_mlir/run_tpu_mlir_regression.py \
        --download --weights-dir yolox-weights --workdir tpu-mlir-reg-work \
        --opset 13 --output tpu-mlir-regression.csv
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
import urllib.request

import numpy as np
import onnx
import onnxsim
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

# variant name -> release weight file (same official checkpoints as
# scripts/regression/yolox/run_yolox_regression.py)
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


def export_raw(exp_name, ckpt_path, out_path, opset):
    """Identical to run_yolox_regression.py's own export_raw -- kept
    self-contained here since scripts/regression's harnesses aren't set up as
    an importable package (see sibling directories: each is standalone)."""
    exp = get_exp(None, exp_name)
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    dummy = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["images"], output_names=["output"],
        opset_version=opset, dynamo=False,
    )
    return exp.test_size


def run_through_tpu_mlir(model: onnx.ModelProto, name: str, workdir: str, feed: dict):
    """Ingest, canonicalize (tpuc-opt), and interpret (pymlir) `model`, inside
    `workdir` (tpu-mlir's own ModelTransformer writes several intermediate
    files relative to the current directory)."""
    from tools.model_runner import mlir_inference
    from tools.model_transform import OnnxTransformer

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        t = OnnxTransformer(
            name, model, input_shapes=[shape], output_names=[], do_onnx_sim=True
        )
        mlir_file = f"{name}.mlir"
        t.model_transform(mlir_file)
        out = mlir_inference(feed, mlir_file, dump_all=False)
    finally:
        os.chdir(cwd)
    return next(iter(out.values()))


def run_one(exp_name, ckpt_path, workdir, opset):
    raw_path = os.path.join(workdir, f"{exp_name}_raw.onnx")
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
        rec["onnxsim_valid"] = bool(check)
        rec["reduction_pct"] = round(
            100.0 * (rec["raw_nodes"] - rec["simp_nodes"]) / rec["raw_nodes"], 1
        )

        rng = np.random.RandomState(0)
        feed = {"images": rng.rand(1, 3, *test_size).astype(np.float32)}

        t0 = time.perf_counter()
        tpu_out = run_through_tpu_mlir(model_simp, f"{exp_name}_tpu_mlir", workdir, feed)
        rec["tpu_mlir_seconds"] = round(time.perf_counter() - t0, 2)

        ref_out = ReferenceEvaluator(raw).run(None, feed)[0]
        np.testing.assert_allclose(ref_out, tpu_out, rtol=1e-2, atol=1e-2)
        rec["tpu_mlir_status"] = "ok"
    except Exception as e:  # crash / abort / numeric mismatch surfaces here
        rec["tpu_mlir_status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", default="yolox-weights")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--workdir", default="tpu-mlir-reg-work")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--output", default="tpu-mlir-regression.csv")
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

    fields = ["variant", "opset", "input_hw", "raw_nodes", "simp_nodes",
              "reduction_pct", "onnxsim_valid", "seconds",
              "tpu_mlir_status", "tpu_mlir_seconds", "error"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {args.output}", flush=True)

    failed = [r for r in rows if not r.get("onnxsim_valid") or r["tpu_mlir_status"] != "ok"]
    print(f"\nSUMMARY: {len(rows) - len(failed)}/{len(rows)} passed onnxsim + tpu-mlir")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
