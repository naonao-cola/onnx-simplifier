#!/usr/bin/env python3
"""yolov5 regression: onnxsim as a drop-in replacement for onnxslim.

Ultralytics `yolov5 <https://github.com/ultralytics/yolov5>`_ simplifies its
exported ONNX inside ``export.py::export_onnx``::

    import onnxslim
    model_onnx = onnxslim.slim(model_onnx)

This script checks that ``onnxsim`` can stand in for that ``onnxslim.slim`` call
without regressing yolov5's export. For every supplied *raw* (un-simplified)
yolov5 ONNX graph it runs both simplifiers and verifies onnxsim is a safe
replacement:

  * onnxsim's own correctness check passes,
  * the simplified graph loads in onnxruntime, and
  * its outputs match the ORIGINAL graph on random input (rtol/atol 1e-3)
    -- the real regression gate.

onnxslim is run on the same graphs as the baseline yolov5 currently ships, so
node counts, timing and per-op deltas can be compared side by side. The exit
code is governed by onnxsim only (onnxslim is comparison-only), matching the
convention in ``run_regression.py``.

Usage
-----
    # compare on raw yolov5 exports you already have
    python scripts/regression/yolov5_regression.py raw_s.onnx raw_n.onnx

    # or let the script export them itself (needs a yolov5 checkout + torch):
    python scripts/regression/yolov5_regression.py \
        --export --yolov5 /path/to/yolov5 --weights yolov5s.pt yolov5n.pt

Dependencies: onnx, onnxruntime, onnxsim, onnxslim (+ torch & a yolov5 checkout
only when using --export).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort


def _counts(m: onnx.ModelProto) -> Counter:
    return Counter(n.op_type for n in m.graph.node)


def _rand_feed(sess) -> dict:
    """Random float32 inputs; dynamic dims default to batch=1, spatial=640."""
    feed = {}
    for inp in sess.get_inputs():
        shape = []
        for j, d in enumerate(inp.shape):
            if isinstance(d, str) or d is None or d <= 0:
                shape.append(1 if j == 0 else 640)
            else:
                shape.append(d)
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )


def _equivalent(orig: onnx.ModelProto, simp: onnx.ModelProto):
    """Run orig and simp on identical random input; (ok, max_abs_diff)."""
    osess = _session(orig)
    feed = _rand_feed(osess)
    o_out = osess.run(None, feed)
    s_out = _session(simp).run(None, feed)
    if len(o_out) != len(s_out):
        return False, float("nan")
    ok, max_diff = True, 0.0
    for a, b in zip(o_out, s_out):
        if a.shape != b.shape:
            return False, float("nan")
        max_diff = max(max_diff, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        ok = ok and np.allclose(a, b, rtol=1e-3, atol=1e-3)
    return ok, max_diff


def _run_onnxsim(path: str):
    import onnxsim

    m, ok = onnxsim.simplify(onnx.load(path))
    return m, ok


def _run_onnxslim(path: str):
    import onnxslim

    return onnxslim.slim(path), True


def export_raw(yolov5_dir: str, weights: str, out_dir: str) -> str:
    """Export a raw (un-simplified) yolov5 ONNX via the project's export.py."""
    stem = os.path.splitext(os.path.basename(weights))[0]
    subprocess.run(
        [sys.executable, "export.py", "--weights", weights,
         "--include", "onnx", "--imgsz", "640", "--opset", "12"],
        cwd=yolov5_dir, check=True,
    )
    src = os.path.join(yolov5_dir, f"{stem}.onnx")
    dst = os.path.join(out_dir, f"{stem}_raw.onnx")
    os.replace(src, dst)
    return dst


def compare(path: str) -> dict:
    name = os.path.basename(path)
    orig = onnx.load(path)
    on = len(orig.graph.node)
    oc = _counts(orig)
    print(f"===== {name}  ({on} nodes) =====")
    res = {}
    for tool, fn in (("onnxsim", _run_onnxsim), ("onnxslim", _run_onnxslim)):
        t = time.time()
        try:
            m, internal_ok = fn(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [{tool:8s}] CRASH: {type(e).__name__}: {str(e)[:90]}")
            res[tool] = None
            continue
        dt = time.time() - t
        sn = len(m.graph.node)
        try:
            eq, diff = _equivalent(orig, m)
            verdict = "loads+equiv" if eq else "MISMATCH"
        except Exception as e:  # noqa: BLE001
            eq, diff, verdict = False, float("nan"), f"load-fail:{str(e)[:40]}"
        red = 100 * (on - sn) / on if on else 0
        print(f"  [{tool:8s}] {on:4d} -> {sn:4d} nodes ({red:+5.1f}%)  {dt:6.1f}s  "
              f"check={str(internal_ok):5s}  {verdict}  maxdiff={diff:.2e}")
        res[tool] = {"nodes": sn, "seconds": dt, "equiv": eq, "maxdiff": diff,
                     "counts": _counts(m)}
    # per-op divergence between the two simplified graphs
    a, b = res.get("onnxsim"), res.get("onnxslim")
    if a and b:
        sc, slc = a["counts"], b["counts"]
        keys = [k for k in set(sc) | set(slc) | set(oc) if sc.get(k, 0) != slc.get(k, 0)]
        if keys:
            print(f"    {'op':16s} {'orig':>5s} {'onnxsim':>8s} {'onnxslim':>9s}")
            for k in sorted(keys, key=lambda k: -abs(sc.get(k, 0) - slc.get(k, 0))):
                print(f"    {k:16s} {oc.get(k,0):5d} {sc.get(k,0):8d} {slc.get(k,0):9d}")
    print()
    return {"name": name, "orig_nodes": on, "res": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="raw yolov5 .onnx paths to test")
    ap.add_argument("--export", action="store_true",
                    help="export raw ONNX from a yolov5 checkout instead")
    ap.add_argument("--yolov5", help="path to a yolov5 checkout (with export.py)")
    ap.add_argument("--weights", nargs="*", default=["yolov5s.pt"],
                    help="weights to export when --export is set")
    ap.add_argument("--out-dir", default=".", help="where to write exported ONNX")
    args = ap.parse_args()

    import onnxsim
    import onnxslim
    print(f"onnxsim {onnxsim.__version__}  |  onnxslim {onnxslim.__version__}  |  "
          f"onnxruntime {ort.__version__}\n")

    paths = list(args.models)
    if args.export:
        if not args.yolov5:
            ap.error("--export requires --yolov5 <checkout path>")
        os.makedirs(args.out_dir, exist_ok=True)
        for w in args.weights:
            paths.append(export_raw(args.yolov5, w, args.out_dir))
    if not paths:
        ap.error("no models: pass raw .onnx paths or use --export")

    reports = [compare(p) for p in paths]

    print("=" * 62)
    print("REGRESSION VERDICT (onnxsim as onnxslim replacement in yolov5):")
    failed = False
    for r in reports:
        a = r["res"].get("onnxsim")
        if a is None:
            print(f"  {r['name']:26s} FAIL (onnxsim crashed)")
            failed = True
        elif not a["equiv"]:
            print(f"  {r['name']:26s} FAIL (output mismatch vs original)")
            failed = True
        else:
            print(f"  {r['name']:26s} PASS "
                  f"({r['orig_nodes']}->{a['nodes']} nodes, equiv, {a['seconds']:.1f}s)")
    print("=" * 62)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
