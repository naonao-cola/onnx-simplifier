#!/usr/bin/env python3
"""RF-DETR -> ONNX for the HTP.

  export.py refs <variant>       fp32 reference outputs of the unpatched library model (20 eval images)
  export.py validate <variant>   rank<=4 MSDA patch vs the library model: max abs diff + detections
  export.py full <variant>       -> <work>/<variant>.sim.onnx (f32 normalized NCHW input)
                                    <work>/<variant>.u8.onnx  (uint8 NHWC input, normalization folded)
                                 then ORT checks both against the reference

Run in the rfdetr venv (torch + rfdetr); the ONNX passes (graph.py) run under REPO_PY with ONNXSIM_REPO (a
checkout with the built onnxsim extension) on PYTHONPATH. <variant> is nano | small | medium.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import common as C
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
# a checkout with the built onnxsim extension, and its python (a worktree usually has neither)
ONNXSIM_REPO = Path(os.environ.get("ONNXSIM_REPO", str(REPO)))
REPO_PY = os.environ.get(
    "REPO_PY", str(ONNXSIM_REPO / ".direnv/python-3.12/bin/python")
)


def refs(variant, paths=None):
    import model as M

    paths = paths or C.image_paths("eval")
    d = C.WORK / f"ref_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (d / f"{p.stem}.npz").exists()]
    if todo:
        core, res = M.load(variant, patched=False)
        w = M.Wrapped(core)
        with torch.no_grad():
            for p in todo:
                lg, bx = w(torch.from_numpy(C.to_pixels(C.load_rgb_u8(p, res))))
                np.savez(
                    d / f"{p.stem}.npz", logits=lg.numpy(), boxes=bx.numpy(), res=res
                )
    return [np.load(d / f"{p.stem}.npz") for p in paths]


def tally(pairs, res):
    tot = {"ref": 0, "det": 0, "matched": 0}
    for r, g in pairs:
        m = C.match(r, g, res)
        for k in tot:
            tot[k] += m[k]
    return tot


def validate(variant):
    import model as M

    paths = C.image_paths("eval")
    ref = refs(variant, paths)
    core, res = M.load(variant, patched=True)
    w = M.Wrapped(core)
    mx, pairs = 0.0, []
    with torch.no_grad():
        for p, r in zip(paths, ref):
            lg, bx = w(torch.from_numpy(C.to_pixels(C.load_rgb_u8(p, res))))
            mx = max(
                mx,
                float(np.abs(lg.numpy() - r["logits"]).max()),
                float(np.abs(bx.numpy() - r["boxes"]).max()),
            )
            pairs.append(((r["logits"], r["boxes"]), (lg.numpy(), bx.numpy())))
    t = tally(pairs, res)
    print(
        f"{variant} @{res}: patched vs library max abs {mx:.2e}, matched {t['matched']}/{t['ref']} (det {t['det']})"
    )


def full(variant):
    import model as M
    import onnxruntime as ort

    core, res = M.load(variant, patched=True)
    w = M.Wrapped(core)
    raw = C.WORK / f"{variant}.raw.onnx"
    x = torch.from_numpy(C.to_pixels(np.zeros((res, res, 3), np.uint8)))
    torch.onnx.export(
        w,
        (x,),
        str(raw),
        opset_version=20,
        input_names=["pixels"],
        output_names=["logits", "boxes"],
        dynamo=False,
    )
    sim, u8 = C.WORK / f"{variant}.sim.onnx", C.WORK / f"{variant}.u8.onnx"
    subprocess.run(
        [REPO_PY, str(HERE / "graph.py"), str(raw), str(sim), str(u8)],
        check=True,
        env=dict(os.environ, PYTHONPATH=str(ONNXSIM_REPO)),
    )
    paths = C.image_paths("eval")[:5]
    ref = refs(variant, paths)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for f, mk in [(sim, lambda rgb: C.to_pixels(rgb)), (u8, lambda rgb: rgb[None])]:
        s = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        mx, pairs = 0.0, []
        for p, r in zip(paths, ref):
            lg, bx = s.run(None, {s.get_inputs()[0].name: mk(C.load_rgb_u8(p, res))})
            mx = max(
                mx,
                float(np.abs(lg - r["logits"]).max()),
                float(np.abs(bx - r["boxes"]).max()),
            )
            pairs.append(((r["logits"], r["boxes"]), (lg, bx)))
        t = tally(pairs, res)
        print(
            f"{f.name}: ORT vs library max abs {mx:.2e}, matched {t['matched']}/{t['ref']} over {len(paths)} images"
        )


if __name__ == "__main__":
    cmd, variant = sys.argv[1], sys.argv[2]
    torch.set_grad_enabled(False)
    {"refs": refs, "validate": validate, "full": full}[cmd](variant)
