#!/usr/bin/env python3
"""Detections of a (quantized) RT-DETR ONNX model on ORT CPU vs the fp32 torch model, eval images.

usage: host_eval.py <model.onnx>... [--set eval|calibration]

ORT runs with graph optimizations disabled, so a QDQ model computes exactly its DQ -> float op -> Q
graph (no fused integer kernels, which saturate on CPUs without VNNI): the quantization's own error,
a fast proxy for the HTP when bisecting precision policies. Input dtype comes from the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as C
import onnxruntime as ort
from phone_eval import refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--set", default="eval")
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    a = ap.parse_args()
    paths = C.image_paths(a.set)
    ref = refs(Path(a.work), paths)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for mp in a.models:
        s = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0]
        tot = {"ref": 0, "det": 0, "matched": 0}
        for p, r in zip(paths, ref):
            rgb = C.load_rgb_u8(p)
            x = rgb[None].copy() if inp.type == "tensor(uint8)" else C.to_pixels(rgb)
            lg, bx = s.run(None, {inp.name: x})[:2]
            m = C.match((r["logits"], r["boxes"]), (lg, bx))
            for k in tot:
                tot[k] += m[k]
        print(
            f"{Path(mp).name}: matched {tot['matched']}/{tot['ref']} (det {tot['det']})",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
