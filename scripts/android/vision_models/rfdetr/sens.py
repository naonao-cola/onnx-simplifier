#!/usr/bin/env python3
"""Activation-quantization sensitivity of RF-DETR per DINOv2 block, and a mixed-precision search
against a detection-match budget, with onnxsim's activation_sensitivity (PR #1935).

  sens.py analyze <variant>                 per-group sensitivity (mode "only", uint8)
  sens.py search <variant> <budget>         greedy uint8 -> uint16 -> float promotion until the
                                            held-out matched fraction >= budget; writes
                                            <work>/<variant>.search<budget>.onnx (uint8 input)
Groups: the 12 DINOv2 layers (onnxsim renames their Linear layers, so each compute node joins the
layer of the nearest named node before it in topological order), the backbone rest (embed,
projector), and "decoder" (query selection + decoder + heads).
Metric: detections of the float model matched by the quantized one (score >= 0.3, IoU >= 0.5, same
class), summed over held-out COCO images (the calibration list's ids 32-55: disjoint from the
calibration images 0-15 and from the 20 eval images).
"""

from __future__ import annotations

import re
import sys

import common as C
import numpy as np
import onnx

from onnxsim import activation_sensitivity as AS
from onnxsim import full_qdq as F


def groups(m: onnx.ModelProto):
    import quantize

    bb, _, _ = quantize.regions(m)
    keyset = {k for ks in AS.group_nodes(m, "node").values() for k in ks}
    out: dict = {}
    cur = None  # the DINOv2 layer of the last named backbone node seen
    for n in m.graph.node:
        mt = re.search(r"/layer\.(\d+)/", n.name)
        if mt:
            cur = f"layer.{int(mt.group(1)):02d}"
        elif "/projector" in n.name or "/embeddings/" in n.name:
            cur = None
        k = n.name or n.output[0]
        if k not in keyset:
            continue
        g = "decoder" if n.name not in bb else (cur or "backbone rest")
        out.setdefault(g, []).append(k)
    return out


def metric_for(res):
    def metric(fo, qo):
        tot = 0
        for f, q in zip(fo, qo):
            tot += C.match((f["logits"], f["boxes"]), (q["logits"], q["boxes"]), res)[
                "matched"
            ]
        return float(tot)

    return metric


def data(res, lo, hi):
    return [
        {"image": C.load_rgb_u8(p, res)[None].astype("float32")}
        for p in C.image_paths("calibration")[lo:hi]
    ]


def main():
    cmd, variant = sys.argv[1], sys.argv[2]
    m = onnx.load(str(C.WORK / f"{variant}.f255.onnx"))
    res = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    cal, ev = data(res, 0, 16), data(res, 32, 56)
    grp = groups(m)
    print({g: len(v) for g, v in grp.items()})
    met = metric_for(res)
    if cmd == "analyze":
        rep = AS.analyze_activation_sensitivity(
            m,
            cal,
            ev,
            groups=grp,
            mode="only",
            activation_dtypes=("uint8",),
            metric=met,
            verbose=False,
        )
        print(f"float {rep.float_score:.0f}, all uint8 {rep.all_quantized_score}")
        for gs in rep.groups:
            print(
                f"  {gs.group:14s} only-this-uint8: {gs.score:.0f} (sensitivity {gs.sensitivity:.0f})"
            )
        return
    budget = float(sys.argv[3])
    res_ = AS.search_activation_precision_for_budget(
        m,
        cal,
        ev,
        budget=budget,
        ladder=("uint8", "uint16", "float"),
        groups=grp,
        metric=met,
        costs="macs",
        verbose=True,
    )
    print(
        f"meets {res_.meets_budget} score {res_.score:.0f} / float {res_.float_score:.0f}; levels {res_.levels}"
    )
    tdt = dict(res_.tensor_dtypes)
    tdt["image"] = "uint8"
    q = F.quantize_full_qdq(
        m,
        cal,
        exclude_nodes=res_.exclude_nodes,
        tensor_dtypes=tdt,
        ranges={"image": (0.0, 255.0)},
    )
    q, info = F.quantized_io(q, inputs=["image"], outputs=[])
    if not info:
        import quantize

        quantize.float_image_to_u8(q)
    out = C.WORK / f"{variant}.search{budget:g}.onnx"
    onnx.save(q, str(out))
    print(out)


if __name__ == "__main__":
    np.seterr(over="ignore")
    main()
