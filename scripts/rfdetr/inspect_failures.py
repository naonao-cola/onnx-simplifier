#!/usr/bin/env python3
"""Inspect the onnxsim ``check_n`` failures of RF-DETR's XLarge/2XLarge
segmentation variants using onnxruntime.

These two variants simplify to a structurally correct graph but report
``check_ok=False``. This script uses onnxruntime to answer three questions:

1. **Input regime** -- does the divergence depend on the (random) input, or
   does it also appear with RF-DETR-normalized input and with zeros?
2. **Prediction impact** -- does the raw-tensor drift actually change
   detections (class argmax, top-k queries, boxes, binarized masks)?
3. **Localization** -- where in the graph does the divergence originate, and
   which onnxsim transform introduces it?

It requires the ``.onnx`` and ``.sim.onnx`` files produced by
``simplify_rfdetr.py`` (the harness saves the simplified graph even when the
check fails). Point ``--stem`` at their shared basename.

Usage::

    python inspect_failures.py --dir rfdetr_onnx --stem rfdetr-seg-xlarge
"""

from __future__ import annotations

import argparse
import collections
import warnings

import numpy as np
import onnx
import onnxruntime as ort

warnings.filterwarnings("ignore")

RTOL, ATOL = 1e-4, 1e-5  # onnxsim/model_checking.py np.allclose defaults
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so)


def _make_input(kind: str, dims, seed: int) -> np.ndarray:
    r = np.random.RandomState(seed)
    if kind == "uniform01":
        return r.rand(*dims).astype(np.float32)
    if kind == "imagenet":  # what RF-DETR actually feeds the simplifier
        return ((r.rand(*dims).astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    if kind == "zeros":
        return np.zeros(dims, np.float32)
    raise ValueError(kind)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def input_regimes(orig, sim, name, dims, outnames):
    print("\n== 1. divergence across input regimes (allclose rtol=1e-4, atol=1e-5) ==")
    s0, s1 = _session(orig), _session(sim)
    for kind in ("uniform01", "imagenet", "zeros"):
        for seed in range(2):
            x = _make_input(kind, dims, seed)
            o0, o1 = s0.run(None, {name: x}), s1.run(None, {name: x})
            parts = []
            for nm, a, b in zip(outnames, o0, o1):
                a, b = a.astype(np.float64), b.astype(np.float64)
                ok = np.allclose(b, a, rtol=RTOL, atol=ATOL)
                parts.append(f"{nm}:max={np.abs(a - b).max():.2e} ok={ok}")
            print(f"  {kind:9s} seed{seed}: " + " | ".join(parts))


def prediction_impact(orig, sim, name, dims, outnames):
    print("\n== 2. prediction-level impact (does the drift change detections?) ==")
    s0, s1 = _session(orig), _session(sim)
    for seed in range(3):
        x = _make_input("imagenet", dims, seed)
        o0 = dict(zip(outnames, s0.run(None, {name: x})))
        o1 = dict(zip(outnames, s1.run(None, {name: x})))
        l0, l1 = o0["labels"][0], o1["labels"][0]
        cls_agree = np.mean(l0.argmax(1) == l1.argmax(1)) * 100
        sc0, sc1 = _sigmoid(l0).max(1), _sigmoid(l1).max(1)
        top0, top1 = set(np.argsort(-sc0)[:10]), set(np.argsort(-sc1)[:10])
        box_diff = np.abs(o0["dets"][0] - o1["dets"][0]).max()
        line = (
            f"  seed{seed}: argmax-class={cls_agree:.2f}% | top10 match={len(top0 & top1)}/10 | "
            f"score maxdiff={np.abs(sc0 - sc1).max():.2e} | box maxdiff={box_diff:.2e}"
        )
        if "masks" in o0:
            flip = (( _sigmoid(o0["masks"][0]) > 0.5) != (_sigmoid(o1["masks"][0]) > 0.5)).mean() * 100
            line += f" | mask flips@0.5={flip:.4f}%"
        print(line)


def localize(orig, sim, name, dims):
    print("\n== 3. where does the divergence originate? ==")
    from onnx import shape_inference

    g0, g1 = onnx.load(orig), onnx.load(sim)

    def prod_map(g):
        return {o: n.op_type for n in g.graph.node for o in n.output if o}

    pm0, pm1 = prod_map(g0), prod_map(g1)
    common = set(pm0) & set(pm1)

    si = shape_inference.infer_shapes(g0)
    shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(si.graph.value_info) + list(si.graph.output)
        if all(d.dim_value > 0 for d in vi.type.tensor_type.shape.dim)
    }
    order, seen = [], set()
    for n in g0.graph.node:
        for o in n.output:
            ne = int(np.prod(shapes[o])) if o in shapes else 0
            if o in common and o not in seen and 0 < ne <= 4_000_000:
                seen.add(o)
                order.append(o)

    x = _make_input("imagenet", dims, 0)

    def run_all(path, names):
        m = onnx.load(path)
        existing = {o.name for o in m.graph.output}
        for nm in names:
            if nm not in existing:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
        return dict(zip(names, _session_from_model(m).run(names, {name: x})))

    def _session_from_model(m):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(m.SerializeToString(), so)

    d0, d1 = run_all(orig, order), run_all(sim, order)
    div = []
    for nm in order:
        a, b = d0[nm].astype(np.float64), d1[nm].astype(np.float64)
        if a.shape != b.shape:
            continue
        if not np.allclose(b, a, rtol=RTOL, atol=ATOL):
            div.append((nm, a.shape, np.abs(a - b).max()))
    print(f"  {len(g0.graph.node)} orig nodes -> {len(g1.graph.node)} sim nodes; "
          f"{len(order)} common tensors compared, {len(div)} exceed tolerance")
    print("  first 8 diverging tensors (topological order):")
    for nm, shp, md in div[:8]:
        print(f"    maxdiff={md:.2e} shape={str(shp):18s} [{pm0.get(nm)}] {nm[:60]}")

    # which ops did onnxsim fold away, and where?
    def outs(g, op):
        return {o for n in g.graph.node if n.op_type == op for o in n.output}

    for op in ("BatchNormalization", "Mul", "Add"):
        n0 = sum(1 for n in g0.graph.node if n.op_type == op)
        n1 = sum(1 for n in g1.graph.node if n.op_type == op)
        removed = outs(g0, op) - outs(g1, op)
        loc = collections.Counter("/".join(t.split("/")[:5]) for t in removed)
        top = ", ".join(f"{k}(x{v})" for k, v in loc.most_common(4))
        print(f"  op {op:20s} {n0:4d} -> {n1:4d}   folded-away locations: {top or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="rfdetr_onnx")
    ap.add_argument("--stem", default="rfdetr-seg-xlarge")
    args = ap.parse_args()
    orig = f"{args.dir}/{args.stem}.onnx"
    sim = f"{args.dir}/{args.stem}.sim.onnx"

    g = onnx.load(orig)
    inp = g.graph.input[0]
    name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    outnames = [o.name for o in g.graph.output]
    print(f"### {args.stem}: input {name}{dims}, outputs {outnames}")

    input_regimes(orig, sim, name, dims, outnames)
    prediction_impact(orig, sim, name, dims, outnames)
    localize(orig, sim, name, dims)


if __name__ == "__main__":
    main()
