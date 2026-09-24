#!/usr/bin/env python3
"""SmoothQuant (onnxsim.apply_smoothquant) for RF-DETR's DINOv2 backbone, with no extra ops at run time.

apply_smoothquant migrates activation outliers into the weights of every plain 2-D Linear: it scales
the weight's reduction rows by s and inserts a `Mul(x, 1/s)` before the layer. Afterwards:
- a Mul whose input is a LayerNorm output (directly or through the qkv Reshape) is folded into that
  LayerNorm's gamma and beta (exact: LN(x)*g/s + b/s), so the backbone's qkv and fc1 -- and the
  decoder's LayerNorm-fed Linears -- get SmoothQuant for free;
- `keep_fc2`: the backbone's fc2 (input: the GELU output, where DINOv2's largest outliers live) keeps
  its Mul, a real extra elementwise op;
- every other Mul is undone (the weight's rows are scaled back, the Mul removed).
The result is float-equivalent to the input model (checked by `check`).

usage: smooth.py <variant> <alpha> [--keep-fc2]   -> <work>/<variant>_sq<alpha>[f].f255.onnx
Run with this checkout's onnxsim (and its built extension) on PYTHONPATH.
"""

from __future__ import annotations

import argparse

import common as C
import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnxsim.smoothquant import apply_smoothquant


def smooth(m: onnx.ModelProto, cal, alpha: float, keep_fc2: bool):
    import quantize

    bb, _, _ = quantize.regions(m)
    s = apply_smoothquant(m, calibration_data=cal, alpha=alpha)
    g = s.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons: dict = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    arr = lambda x: numpy_helper.to_array(inits[x]).astype(np.float64)  # noqa: E731

    def put(name, a):
        t = numpy_helper.from_array(a.astype(np.float32), name)
        inits[name] = t
        g.initializer.append(t)

    stats = {"ln_fold": 0, "kept": 0, "undone": 0}
    for mul in [n for n in g.node if n.name.endswith("_smoothquant_mul")]:
        x, inv = mul.input
        v = arr(inv).reshape(-1)  # 1/s per input channel
        gemm = cons[mul.output[0]][0]
        p = prod.get(x)
        ln = p
        if p is not None and p.op_type == "Reshape" and len(cons[p.output[0]]) == 1:
            ln = prod.get(p.input[0])
        ok_ln = (
            ln is not None
            and ln.op_type == "LayerNormalization"
            and len(cons[ln.output[0]]) == 1
            and len(cons[x]) == 1
        )
        if ok_ln:
            gname, bname = ln.input[1], ln.input[2]
            put(gname + "_sq", arr(gname) * v)
            put(bname + "_sq", arr(bname) * v)
            ln.input[1], ln.input[2] = gname + "_sq", bname + "_sq"
            stats["ln_fold"] += 1
        elif keep_fc2 and p is not None and p.op_type == "Gelu" and p.name in bb:
            stats["kept"] += 1
            continue
        else:  # undo: scale the weight's reduction dimension back by 1/s
            at = {a.name: helper.get_attribute_value(a) for a in gemm.attribute}
            w = arr(gemm.input[1])
            w = w * (v[None, :] if at.get("transB", 0) else v[:, None])
            put(gemm.input[1] + "_unsq", w)
            gemm.input[1] = gemm.input[1] + "_unsq"
            stats["undone"] += 1
        for c in cons[mul.output[0]]:
            c.input[:] = [x if y == mul.output[0] else y for y in c.input]
        g.node.remove(mul)
    used = {x for n in g.node for x in n.input}
    for i in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(i)
    return s, stats


def calib(res, lo, hi):
    return [
        {"image": C.load_rgb_u8(p, res)[None].astype("float32")}
        for p in C.image_paths("calibration")[lo:hi]
    ]


def check(m0, m1, data):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(
        m0.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    b = ort.InferenceSession(
        m1.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    mx = 0.0
    for d in data:
        oa, ob = a.run(["boxes"], d)[0], b.run(["boxes"], d)[0]
        mx = max(mx, float(np.abs(oa - ob).max()))
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("alpha", type=float)
    ap.add_argument("--keep-fc2", action="store_true")
    a = ap.parse_args()
    m = onnx.load(str(C.WORK / f"{a.variant}.f255.onnx"))
    res = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    s, stats = smooth(m, calib(res, 0, 16), a.alpha, a.keep_fc2)
    tag = f"{a.variant}_sq{a.alpha:g}" + ("f" if a.keep_fc2 else "")
    print(tag, stats, "float boxes max abs vs input:", check(m, s, calib(res, 56, 58)))
    onnx.save(s, str(C.WORK / f"{tag}.f255.onnx"))


if __name__ == "__main__":
    main()
