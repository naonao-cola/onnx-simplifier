#!/usr/bin/env python3
"""Validate the rank-<=4 MSDA rewrite against HF's own RT-DETR, then export ONNX pieces.

usage: export.py validate --work <dir>          patched vs unpatched HF on the eval images
       export.py <piece> --work <dir>           export + ORT check + onnxsim

pieces:
  full      pixels (1,3,640,640) f32 -> logits (1,300,80), boxes (1,300,4)
  full_u8   the same with a uint8 NHWC image input (RGB 0..255; the /255 + NCHW is in the graph,
            where the uint8_input rewrite folds it into the first Conv's quantization)

TorchScript exporter (dynamo=False), opset 17, fixed shapes, no_grad. Test inputs: the first eval
image; phone inputs go to <work>/<piece>.in/ (manifest.txt + raw .bin), the fp32 torch reference
outputs to <work>/<piece>.in/ref_o{0,1}.bin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as C
import model as M
import numpy as np
import torch


class FullU8(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):  # (1, 640, 640, 3) uint8
        x = image.permute(0, 3, 1, 2).float() * (1.0 / 255)
        o = self.m(pixel_values=x)
        return o.logits, o.pred_boxes


def validate(work: Path):
    torch.set_grad_enabled(False)
    ref, pat = M.load(patched=False), M.load(patched=True)
    worst, tot = 0.0, {"ref": 0, "det": 0, "matched": 0}
    for p in C.image_paths("eval"):
        x = torch.from_numpy(C.to_pixels(C.load_rgb_u8(p)))
        a, b = ref(pixel_values=x), pat(pixel_values=x)
        d = max(
            float((a.logits - b.logits).abs().max()),
            float((a.pred_boxes - b.pred_boxes).abs().max()),
        )
        worst = max(worst, d)
        r = C.match(
            (a.logits.numpy(), a.pred_boxes.numpy()),
            (b.logits.numpy(), b.pred_boxes.numpy()),
        )
        for k in tot:
            tot[k] += r[k]
    print(
        f"patched vs HF over {len(C.coco_ids('eval'))} images: max abs {worst:.2e}, "
        f"matched {tot['matched']}/{tot['ref']} (det {tot['det']})"
    )


def write_inputs(d: Path, feeds: dict, outs):
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    dt = {
        np.dtype(np.float32): "f32",
        np.dtype(np.uint8): "u8",
        np.dtype(np.int64): "i64",
    }
    for name, v in feeds.items():
        v.tofile(d / f"{name}.bin")
        lines.append(
            f"{name} {dt[v.dtype]} {d / (name + '.bin')} {','.join(map(str, v.shape))}"
        )
    (d / "manifest.txt").write_text("\n".join(lines) + "\n")
    for i, o in enumerate(outs):
        np.asarray(o, np.float32).tofile(d / f"ref_o{i}.bin")


def export(name: str, work: Path):
    import onnx
    import onnxruntime as ort

    import onnxsim

    torch.set_grad_enabled(False)
    m = M.load(patched=True)
    rgb = C.load_rgb_u8(C.image_paths("eval")[0])
    if name == "full":
        mod, in_name, x = M.Full(m).eval(), "pixel_values", C.to_pixels(rgb)
    elif name == "full_u8":
        mod, in_name, x = FullU8(m).eval(), "image", rgb[None].copy()
    else:
        raise SystemExit(f"unknown piece {name}")
    xt = torch.from_numpy(x)
    ref = [t.numpy() for t in mod(xt)]
    raw = work / f"{name}.onnx"
    torch.onnx.export(
        mod,
        (xt,),
        str(raw),
        input_names=[in_name],
        output_names=["logits", "boxes"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    sim, ok = onnxsim.simplify(onnx.load(str(raw)))
    assert ok
    out = work / f"{name}.sim.onnx"
    onnx.save(sim, str(out))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    got = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"]).run(
        None, {in_name: x}
    )
    md = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    r = C.match(ref, got)
    ranks = max(
        (
            len(v.type.tensor_type.shape.dim)
            for v in onnx.shape_inference.infer_shapes(sim).graph.value_info
        ),
        default=0,
    )
    ops = {}
    for n in sim.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(
        f"{out}: {len(sim.graph.node)} nodes, max rank {ranks}, ORT vs torch max abs {md:.2e}, "
        f"matched {r['matched']}/{r['ref']}"
    )
    print("  ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    write_inputs(work / f"{name}.in", {in_name: x}, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.what == "validate":
        validate(work)
    else:
        export(a.what, work)


if __name__ == "__main__":
    sys.exit(main())
