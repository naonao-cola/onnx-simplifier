#!/usr/bin/env python3
"""Which op types does int8 hurt in an encoder/decoder piece? ORT CPU (QDQ = fake quant in fp32).

  sensitivity.py <piece> --work <work> [--dtype uint8]
For every op type in the piece: cosine vs the fp32 reference (export.py's <piece>.in/ref_*) of
  * "all but T": every op quantized except type T (kept float, onnxsim.full_qdq exclusion), and
  * "only T":    only type T quantized.
Calibration ranges are computed once for every tensor (quantize.py's calibration set) and reused.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from quantize import ENC_NAMES, load_onnxsim, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--work", required=True)
    ap.add_argument("--dtype", default="uint8")
    a = ap.parse_args()
    import onnx
    import onnxruntime as ort

    F = load_onnxsim()
    from onnxsim.calibration import calibrate

    work = Path(a.work)
    m = onnx.load(str(work / f"{a.piece}.sim.onnx"))
    names = ENC_NAMES if a.piece.startswith("enc") else ["bev_embed"]
    every = [o for n in m.graph.node for o in n.output] + [i.name for i in m.graph.input]
    ranges = calibrate(m, list(samples(work, names)), extra_tensor_names=every)
    ind = work / f"{a.piece}.in"
    feed = {}
    for line in (ind / "manifest.txt").read_text().splitlines():
        k, _, f, dims = line.split()
        feed[k] = np.fromfile(f, np.float32).reshape([int(d) for d in dims.split(",")])
    outs = [o.name for o in m.graph.output]
    refs = [np.fromfile(ind / f"ref_{o}.bin", np.float32) for o in outs]

    def cos(q):
        so = ort.SessionOptions()  # no DQ->op->Q fusion into host-CPU int8 kernels: test the QDQ math
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        got = ort.InferenceSession(q.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(outs, feed)
        c = []
        for r, g in zip(refs, got):
            g = g.ravel().astype(np.float64)
            c.append(float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g))))
        return " ".join(f"{x:.5f}" for x in c)

    types = collections.Counter(n.op_type for n in m.graph.node if n.op_type not in ("Constant", "Cast"))
    print(f"{a.piece} {a.dtype}: all quantized {cos(F.quantize_full_qdq(m, activation_dtype=a.dtype, ranges=ranges))}")
    for t, c in types.most_common():
        but = F.quantize_full_qdq(m, activation_dtype=a.dtype, ranges=ranges, exclude_op_types=[t])
        only = F.quantize_full_qdq(m, activation_dtype=a.dtype, ranges=ranges, op_types=[t])
        print(f"  {t:20s} x{c:3d}  all but: {cos(but)}  only: {cos(only)}")


if __name__ == "__main__":
    main()
