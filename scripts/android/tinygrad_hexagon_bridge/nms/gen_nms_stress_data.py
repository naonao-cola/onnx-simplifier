#!/usr/bin/env python3
"""Adversarial NonMaxSuppression test sets, in gen_nms_test_data.py's file format (so the same
host check, qemu harness and phone client run them), with ONNX Runtime CPU as the reference.
Real data from one image rarely lands exactly on the IoU threshold; these do on purpose: box
coordinates on a coarse grid (many IoUs exactly equal to 0.5 / 0.7 in real arithmetic, and a few
ulp either side in fp32), heavy score ties (ORT breaks them by lower index), zero-area boxes and
boxes with swapped corners (ORT's MaxMin handles those), across n = 1..2000.
`level_*` files get iou=0.7 calls, `class_*` files iou=0.5 calls (just reusing the two group names).

    python gen_nms_stress_data.py OUT_DIR [--calls 40] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--calls", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)


def model(iou):
    node = helper.make_node("NonMaxSuppression", ["b", "s", "m", "i"], ["o"], center_point_box=0)
    g = helper.make_graph([node], "nms", [helper.make_tensor_value_info("b", TensorProto.FLOAT, None),
                                          helper.make_tensor_value_info("s", TensorProto.FLOAT, None)],
                          [helper.make_tensor_value_info("o", TensorProto.INT64, None)],
                          [numpy_helper.from_array(np.array([2000], np.int64), "m"),
                           numpy_helper.from_array(np.array([iou], np.float32), "i")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)]); m.ir_version = 7
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


def boxes(n):
    grid = rng.choice([1.0, 0.5, 0.25, 10.0])
    span = rng.choice([120.0, 400.0, 1000.0])  # small spans -> dense overlap, many suppressions
    off = rng.choice([0.0, 500.0])  # also exercise large coordinates (bigger fp32 error bound)
    y1 = off + rng.integers(0, span / grid, n) * grid; x1 = off + rng.integers(0, span / grid, n) * grid
    h = rng.integers(0, 60, n) * grid; w = rng.integers(0, 60, n) * grid  # includes zero-size
    b = np.stack([y1, x1, y1 + h, x1 + w], 1).astype(np.float32)
    # near-duplicates of earlier boxes, shifted by grid steps -> IoUs at/near simple fractions
    k = n // 2
    src = rng.integers(0, n, k); dst = rng.integers(0, n, k)
    b[dst] = b[src] + rng.integers(-3, 4, (k, 4)) * grid
    flip = rng.random(n) < 0.1
    b[flip] = b[flip][:, [2, 3, 0, 1]]  # swapped corners
    return b


for g, iou in (("level", 0.7), ("class", 0.5)):
    s_ = model(iou)
    calls = []
    for c in range(a.calls):
        n = int(rng.choice([1, 2, 7, 33, 64, 65, 200, 1000, 2000]))
        b = boxes(n)
        sc = (rng.integers(0, rng.choice([2, 5, 50, 1000]), n) / 7.0).astype(np.float32)  # ties
        sel = s_.run(None, {"b": b[None], "s": sc[None, None]})[0][:, 2].astype(np.int32)
        calls.append((b, sc, sel))
    (a.out / f"{g}_calls.txt").write_text("".join(f"{len(s)} {float(np.float32(iou))!r} 2000 {len(r)}\n" for b, s, r in calls))
    np.concatenate([b.reshape(-1) for b, _, _ in calls]).tofile(a.out / f"{g}_boxes.bin")
    np.concatenate([s for _, s, _ in calls]).tofile(a.out / f"{g}_scores.bin")
    np.concatenate([r for _, _, r in calls]).tofile(a.out / f"{g}_ref.bin")
    np.full(len(calls), iou, np.float32).tofile(a.out / f"{g}_thr.bin")
    print(f"{g} (iou={iou}): {len(calls)} calls, boxes={sum(len(s) for _, s, _ in calls)}, selected={sum(len(r) for *_, r in calls)}")
