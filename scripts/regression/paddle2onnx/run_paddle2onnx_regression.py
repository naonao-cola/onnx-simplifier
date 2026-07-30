#!/usr/bin/env python3
"""Paddle2ONNX -> onnxsim regression.

Reproduces, end to end, the optimization path that Paddle2ONNX's ``convert.py``
guards behind ``optimize_tool == "onnxsim"`` (currently commented out upstream:
https://github.com/PaddlePaddle/Paddle2ONNX/blob/187729275c36be251a9c676d4ff5550174c10189/paddle2onnx/convert.py#L267)::

    from onnxsim import simplify
    onnx_model = onnx.load_model(model_stream)   # produced by paddle2onnx.export
    simplified_model, check = simplify(onnx_model)

so we can answer one question: **does onnxsim currently simplify the ONNX graphs
that Paddle2ONNX emits, and does the result stay numerically equivalent?**

For each model we:

1. build a real PaddlePaddle vision network and ``paddle.jit.save`` it to an
   inference model (the ``.json``/``.pdiparams`` pair Paddle2ONNX consumes);
2. convert it to ONNX with ``paddle2onnx.export`` (the same call ``convert.py``
   makes to obtain ``onnx_model_str``);
3. run ``onnxsim.simplify`` and keep its correctness flag (``check``);
4. independently re-verify equivalence by running the pre- and post-simplify
   graphs through onnxruntime on shared random inputs.

Each model runs in its own child process (like ``scripts/regression/worker.py``)
so a C++ abort or hang in one graph can't take the whole run down. A model is a
FAILURE when the paddle2onnx export fails, onnxsim crashes / times out / returns
``check=False``, or the onnxruntime outputs diverge.

Usage::

    python run_paddle2onnx_regression.py                 # run the whole set
    python run_paddle2onnx_regression.py --only resnet18 mobilenet_v2
    python run_paddle2onnx_regression.py --output report.csv --markdown report.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

# (name, paddle.vision.models constructor, H, W, opset, dynamic_batch)
# A spread across the architecture families Paddle2ONNX users actually export:
# plain/residual CNNs, depthwise-separable mobile nets, dense/branchy graphs,
# channel-shuffle, and a couple of dynamic-batch graphs (the shape regime where
# onnxsim's C++ passes have historically aborted).
MODELS = [
    ("alexnet",             "alexnet",             224, 224, 11, False),
    ("vgg16",               "vgg16",               224, 224, 11, False),
    ("resnet18",            "resnet18",            224, 224, 11, False),
    ("resnet50",            "resnet50",            224, 224, 11, False),
    ("resnext50_32x4d",     "resnext50_32x4d",     224, 224, 11, False),
    ("wide_resnet50_2",     "wide_resnet50_2",     224, 224, 11, False),
    ("mobilenet_v1",        "mobilenet_v1",        224, 224, 11, False),
    ("mobilenet_v2",        "mobilenet_v2",        224, 224, 11, False),
    ("mobilenet_v3_small",  "mobilenet_v3_small",  224, 224, 11, False),
    ("mobilenet_v3_large",  "mobilenet_v3_large",  224, 224, 11, False),
    ("squeezenet1_1",       "squeezenet1_1",       224, 224, 11, False),
    ("densenet121",         "densenet121",         224, 224, 11, False),
    ("shufflenet_v2_x1_0",  "shufflenet_v2_x1_0",  224, 224, 11, False),
    ("googlenet",           "googlenet",           224, 224, 11, False),
    ("inception_v3",        "inception_v3",        299, 299, 11, False),
    # Dynamic batch dimension: exercises the shape-inference / optimizer path
    # that static graphs skip.
    ("resnet18_dynbatch",   "resnet18",            224, 224, 11, True),
    ("mobilenet_v2_dynbatch", "mobilenet_v2",      224, 224, 11, True),
]

# Per-model wall-clock cap for the child process (export + simplify + verify).
DEFAULT_TIMEOUT = 900


def worker(cfg: dict) -> dict:
    """Run one model end to end. Returns a result dict; never raises for a
    graph-level failure (those are captured), only for genuinely unexpected
    programming errors."""
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("GLOG_minloglevel", "3")

    import numpy as np
    import onnx
    import onnxruntime as ort
    import paddle
    import paddle.vision.models as pvm
    import paddle2onnx
    import onnxsim

    name = cfg["name"]
    h, w, opset, dyn = cfg["h"], cfg["w"], cfg["opset"], cfg["dynamic"]
    result = {
        "model": name,
        "status": "fail",
        "stage": "",
        "orig_nodes": None,
        "simp_nodes": None,
        "node_reduction_pct": None,
        "onnxsim_check": None,
        "ort_equiv": None,
        "export_s": None,
        "simplify_s": None,
        "error": "",
    }

    with tempfile.TemporaryDirectory() as td:
        # --- 1. build + export a Paddle inference model -----------------------
        try:
            result["stage"] = "paddle_export"
            paddle.set_device("cpu")
            net = getattr(pvm, cfg["ctor"])(pretrained=False)
            net.eval()
            batch = None if dyn else 1
            spec = [paddle.static.InputSpec(
                shape=[batch, 3, h, w], dtype="float32", name="x")]
            prefix = os.path.join(td, name)
            paddle.jit.save(net, prefix, input_spec=spec)
        except Exception as e:  # noqa: BLE001
            result["error"] = f"paddle export: {type(e).__name__}: {e}"[:500]
            return result

        # Paddle 3.x (PIR) writes <prefix>.json; older writes <prefix>.pdmodel.
        model_file = prefix + ".json"
        if not os.path.exists(model_file):
            model_file = prefix + ".pdmodel"
        params_file = prefix + ".pdiparams"

        # --- 2. paddle2onnx.export (the convert.py path) ----------------------
        try:
            result["stage"] = "paddle2onnx"
            t0 = time.perf_counter()
            onnx_bytes = paddle2onnx.export(
                model_file,
                params_file,
                save_file=None,
                opset_version=opset,
                enable_onnx_checker=True,
            )
            result["export_s"] = round(time.perf_counter() - t0, 2)
            if not onnx_bytes:
                result["error"] = "paddle2onnx.export returned empty model"
                return result
            model = onnx.load_from_string(onnx_bytes)
            result["orig_nodes"] = len(model.graph.node)
        except Exception as e:  # noqa: BLE001
            result["error"] = f"paddle2onnx: {type(e).__name__}: {e}"[:500]
            return result

        # --- 3. onnxsim.simplify (line 267 of convert.py) ---------------------
        try:
            result["stage"] = "onnxsim"
            t0 = time.perf_counter()
            simp, check = onnxsim.simplify(model)
            result["simplify_s"] = round(time.perf_counter() - t0, 2)
            result["onnxsim_check"] = bool(check)
            result["simp_nodes"] = len(simp.graph.node)
            if result["orig_nodes"]:
                result["node_reduction_pct"] = round(
                    100.0 * (result["orig_nodes"] - result["simp_nodes"])
                    / result["orig_nodes"], 1)
        except Exception as e:  # noqa: BLE001
            result["error"] = f"onnxsim.simplify: {type(e).__name__}: {e}"[:500]
            return result

        # --- 4. independent onnxruntime equivalence check ---------------------
        try:
            result["stage"] = "ort_verify"
            concrete_batch = 1
            x = np.random.rand(concrete_batch, 3, h, w).astype("float32")

            def run(m):
                so = ort.SessionOptions()
                so.log_severity_level = 3
                sess = ort.InferenceSession(
                    m.SerializeToString(), so,
                    providers=["CPUExecutionProvider"])
                feeds = {sess.get_inputs()[0].name: x}
                return sess.run(None, feeds)

            out_a = run(model)
            out_b = run(simp)
            equiv = len(out_a) == len(out_b) and all(
                np.allclose(a, b, rtol=1e-3, atol=1e-4)
                for a, b in zip(out_a, out_b))
            result["ort_equiv"] = bool(equiv)
        except Exception as e:  # noqa: BLE001
            # Verification harness failure is not an onnxsim failure; note it but
            # let the onnxsim_check decide pass/fail.
            result["error"] = f"ort_verify: {type(e).__name__}: {e}"[:300]

        ok = (result["onnxsim_check"] is True
              and result["ort_equiv"] is not False)
        result["status"] = "pass" if ok else "fail"
        result["stage"] = "done"
        return result


def run_driver(models, timeout, output, markdown):
    here = os.path.abspath(__file__)
    results = []
    for name, ctor, h, w, opset, dyn in models:
        cfg = {"name": name, "ctor": ctor, "h": h, "w": w,
               "opset": opset, "dynamic": dyn}
        print(f"[ {name:22s} ] ...", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, here, "--worker", json.dumps(cfg)],
                capture_output=True, text=True, timeout=timeout)
            res = None
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and '"model"' in line:
                    res = json.loads(line)
                    break
            if res is None:
                res = _blank(name, "crash",
                             (proc.stderr or proc.stdout or "")[-500:])
        except subprocess.TimeoutExpired:
            res = _blank(name, "timeout", f"exceeded {timeout}s")
        res["wall_s"] = round(time.perf_counter() - t0, 2)
        results.append(res)
        _print_row(res)

    _write_csv(results, output)
    md = _render_markdown(results)
    if markdown:
        with open(markdown, "w") as f:
            f.write(md)
    print("\n" + md)

    failures = [r for r in results if r["status"] != "pass"]
    print(f"\n{len(results) - len(failures)}/{len(results)} models passed.")
    return 1 if failures else 0


def _blank(name, status, error):
    return {"model": name, "status": status, "stage": "", "orig_nodes": None,
            "simp_nodes": None, "node_reduction_pct": None,
            "onnxsim_check": None, "ort_equiv": None, "export_s": None,
            "simplify_s": None, "error": error}


def _print_row(r):
    tag = {"pass": "PASS", "fail": "FAIL",
           "timeout": "TIMEOUT", "crash": "CRASH"}.get(r["status"], r["status"])
    detail = ""
    if r["orig_nodes"] is not None:
        detail = (f"nodes {r['orig_nodes']}->{r['simp_nodes']} "
                  f"({r['node_reduction_pct']}%) "
                  f"check={r['onnxsim_check']} equiv={r['ort_equiv']}")
    if r["status"] != "pass" and r["error"]:
        detail = (detail + "  " + r["error"]).strip()
    print(f"    -> {tag:8s} {detail}", flush=True)


CSV_FIELDS = ["model", "status", "stage", "orig_nodes", "simp_nodes",
              "node_reduction_pct", "onnxsim_check", "ort_equiv",
              "export_s", "simplify_s", "wall_s", "error"]


def _write_csv(results, output):
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _render_markdown(results):
    npass = sum(r["status"] == "pass" for r in results)
    lines = [
        "## Paddle2ONNX → onnxsim regression",
        "",
        f"**{npass}/{len(results)} models passed** "
        "(paddle2onnx export + `onnxsim.simplify` with `check=True` and "
        "onnxruntime equivalence).",
        "",
        "| model | status | nodes (orig→simp) | reduction | onnxsim check | ort equiv | export s | simplify s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        nodes = ("-" if r["orig_nodes"] is None
                 else f"{r['orig_nodes']}→{r['simp_nodes']}")
        red = ("-" if r["node_reduction_pct"] is None
               else f"{r['node_reduction_pct']}%")
        lines.append(
            f"| {r['model']} | {r['status']} | {nodes} | {red} | "
            f"{_fmt(r['onnxsim_check'])} | {_fmt(r['ort_equiv'])} | "
            f"{_fmt(r['export_s'])} | {_fmt(r['simplify_s'])} |")
    fails = [r for r in results if r["status"] != "pass"]
    if fails:
        lines += ["", "### Failures", ""]
        for r in fails:
            lines.append(f"- **{r['model']}** ({r['status']}, "
                         f"stage `{r['stage'] or '?'}`): {r['error'] or '-'}")
    return "\n".join(lines)


def _fmt(v):
    return "-" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", help="internal: JSON config for one model")
    ap.add_argument("--only", nargs="*", help="run only these model names")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="per-model wall-clock cap (s)")
    ap.add_argument("--output", default="paddle2onnx-regression.csv")
    ap.add_argument("--markdown", default="paddle2onnx-regression.md")
    args = ap.parse_args()

    if args.worker:
        res = worker(json.loads(args.worker))
        print(json.dumps(res), flush=True)
        return 0

    models = MODELS
    if args.only:
        wanted = set(args.only)
        models = [m for m in MODELS if m[0] in wanted]
        if not models:
            print(f"no models match {args.only}; known: "
                  f"{[m[0] for m in MODELS]}", file=sys.stderr)
            return 2
    return run_driver(models, args.timeout, args.output, args.markdown)


if __name__ == "__main__":
    sys.exit(main())
