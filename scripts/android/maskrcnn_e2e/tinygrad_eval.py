#!/usr/bin/env python3
"""Mask R-CNN in tinygrad: full graph from the ONNX file, compared with ONNX Runtime.

tinygrad's ONNX frontend loads the whole model (about 5 s; TVM's Relay needs ~10 minutes) and
implements 41 of its 43 operator types; `tinygrad_ops.py` adds NonMaxSuppression and RoiAlign.
`--rest-only` feeds ONNX Runtime's backbone features into the remainder graph (an exact test of
the dynamic-shape operators); otherwise the whole model runs in tinygrad.

Select the device with `--device` (NV, CUDA, ... ; tinygrad reads DEV before importing). The CPU
device needs `clang` on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("maskrcnn_work"))
    parser.add_argument("--model", type=Path, required=True, help="original MaskRCNN-12-qdq.onnx")
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="NV")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("tinygrad_results.json"))
    args = parser.parse_args()
    os.environ["DEV"] = args.device

    import numpy as np
    import onnxruntime as ort
    from eval_common import canvas, compare
    from tinygrad import Tensor
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad_ops import register

    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    full = ort.InferenceSession(str(args.model), options, providers=providers)
    names = [n for n in (args.workdir / "backbone_outputs.txt").read_text().split("\n") if n]
    outputs = [o.name for o in full.get_outputs()]
    if args.rest_only:
        backbone = ort.InferenceSession(str(args.workdir / "backbone.onnx"), options, providers=providers)
        rest_ort = ort.InferenceSession(str(args.workdir / "rest.onnx"), options, providers=providers)
    graph = args.workdir / ("rest.onnx" if args.rest_only else "maskrcnn_sim.onnx")

    rows = []
    for path in args.images:
        image = canvas(path, args.height, args.width)
        reference = full.run(None, {"image": image})
        # A fresh runner per image: OnnxRunner caches Python constants across calls, which goes
        # stale when the graph has data-dependent shapes (the second image fails otherwise).
        runner = register(OnnxRunner(graph))
        start = time.time()
        if args.rest_only:
            features = dict(zip(names, backbone.run(None, {"image": image})))
            expected = rest_ort.run(None, features)
            start = time.time()
            result = runner({n: Tensor(v) for n, v in features.items()})
        else:
            expected = reference
            result = runner({"image": Tensor(image)})
        detections = [result[n].numpy() for n in outputs]
        seconds = time.time() - start
        row = {"image": path.name, **compare(reference, detections), "seconds": round(seconds, 2)}
        if args.rest_only:  # exactness against ONNX Runtime's own remainder
            row["max_abs_vs_ort_rest"] = [
                float(np.abs(d.astype("float64") - e.astype("float64")).max()) if d.shape == e.shape else "shape"
                for d, e in zip(detections, expected)
            ]
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
