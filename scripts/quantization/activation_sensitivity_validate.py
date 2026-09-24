"""Host-only check that onnxsim.activation_sensitivity reproduces the
mixed-precision findings the per-model phone work reached by hand
(BEVFormer-tiny encoder, EdgeSAM encoder, RF-DETR-Nano).

    python activation_sensitivity_validate.py bevformer|edgesam|rfdetr [--out DIR]

Inputs are the models and data those agents left under ~/.cache (see each
loader); a missing model is reported and skipped. Prints and writes JSON.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import onnx

from onnxsim.activation_sensitivity import (
    analyze_activation_sensitivity,
    group_nodes,
    search_activation_precision_for_budget,
)

C = os.path.expanduser("~/.cache")


def _images(n, start=0):
    from PIL import Image

    files = sorted(glob.glob(f"{C}/onnxsim-deploy/_images/*.jpg"))[start : start + n]
    return [Image.open(f).convert("RGB") for f in files]


def bevformer():
    p = f"{C}/onnxsim-bevformer/work/enc1.sim.onnx"
    m = onnx.load(p)
    names = [i.name for i in m.graph.input]
    frames = sorted(
        glob.glob(f"{C}/onnxsim-bevformer/work/calib/*.npz"),
        key=lambda f: int(os.path.basename(f)[:-4]),
    )
    data = [{k: np.load(f)[k].astype(np.float32) for k in names} for f in frames]
    return m, data[:8], data[8:12], dict(groups="op_type")


def edgesam():
    m = onnx.load(f"{C}/onnxsim-sam/edgesam/enc.sim.onnx")
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)

    def prep(img):
        s = 1024 / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)))
        a = (np.asarray(img, np.float32) - mean) / std
        out = np.zeros((1024, 1024, 3), np.float32)
        out[: a.shape[0], : a.shape[1]] = a
        return {"pixels": out.transpose(2, 0, 1)[None]}

    return (
        m,
        [prep(i) for i in _images(6)],
        [prep(i) for i in _images(3, 6)],
        dict(groups="block"),
    )


def rfdetr():
    m = onnx.load(f"{C}/onnxsim-rfdetr/work/nano.u8.onnx")

    def prep(img):
        return {"image": np.asarray(img.resize((384, 384)), np.uint8)[None]}

    return (
        m,
        [prep(i) for i in _images(8)],
        [prep(i) for i in _images(4, 8)],
        dict(groups="block", block_regex=r"^(.*?/(?:layer|layers)\.\d+)/"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["bevformer", "edgesam", "rfdetr"])
    ap.add_argument("--out", default=f"{C}/actsens_val")
    ap.add_argument("--budget", type=float, default=20.0)
    a = ap.parse_args()
    try:
        m, cal, ev, gkw = globals()[a.model]()
    except FileNotFoundError as e:
        print(f"skip {a.model}: {e}")
        return 0
    grp = group_nodes(m, **gkw)
    print(f"{a.model}: {len(grp)} groups, {sum(map(len, grp.values()))} compute nodes")
    res = {"model": a.model, "groups": len(grp)}
    for mode in ("only", "all_but"):
        t = time.time()
        rep = analyze_activation_sensitivity(m, cal, ev, mode=mode, method="mse", **gkw)
        sens = np.array([r.sensitivity for r in rep.groups])
        res[mode] = {
            "float": rep.float_score,
            "all_uint8": rep.all_quantized_score["uint8"],
            "top": [
                (r.group, round(r.score, 2), round(r.sensitivity, 2))
                for r in rep.top(8)
            ],
            "sens_mean": float(sens.mean()),
            "sens_cv": float(sens.std() / max(abs(sens.mean()), 1e-9)),
            "seconds": round(time.time() - t, 1),
        }
        print(json.dumps(res[mode], indent=1))
    t = time.time()
    s = search_activation_precision_for_budget(
        m, cal, ev, budget=a.budget, method="mse", costs="macs", **gkw
    )
    res["search"] = {
        "budget": a.budget,
        "meets": s.meets_budget,
        "score": s.score,
        "promoted_groups": sum(lv != "uint8" for lv in s.levels.values()),
        "promoted_nodes": s.promoted_nodes,
        "levels": {k: v for k, v in s.levels.items() if v != "uint8"},
        "steps": len(s.trace) - 1,
        "seconds": round(time.time() - t, 1),
    }
    print(json.dumps(res["search"], indent=1))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/{a.model}.json", "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
