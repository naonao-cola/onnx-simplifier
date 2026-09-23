"""Which of the head's op types / regions quantization hurts, on the host (ORT CPU, no QDQ fusion).

  python sensitivity.py --work <work> [--act uint16] [--variants all,-Gemm,only:MatMul,...]

Calibrates the float head once (quantize.py's calibration frames, all tensors), then quantizes it per
variant with those ranges (quantize_full_qdq(ranges=...)) and runs every scene-0103 frame
teacher-forced (the fp32 chain's inputs from validate.py's dumps). Reports per variant the worst
per-frame cosine of cls / reg / dec vs the float model and the score-weighted box error. Variants:
``all`` (everything quantized), ``-T`` (everything but op type T kept float), ``only:T`` (only T
quantized), ``-node:substr`` (nodes whose name contains substr kept float).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import quantize as Qz


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--src", default="head.sim")
    ap.add_argument("--act", default="uint16")
    ap.add_argument("--variants", default="all,-Gemm,-MatMul,-Softmax,-LayerNormalization,only:Gemm,only:MatMul")
    a = ap.parse_args()
    F = Qz.load_onnxsim()
    from onnxsim.calibration import calibrate

    work = Path(a.work)
    m = onnx.load(str(work / f"{a.src}.onnx"))
    names = [o for n in m.graph.node for o in n.output] + [i.name for i in m.graph.input]
    ranges = calibrate(m, list(Qz.samples(work, "head")), extra_tensor_names=names)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    evals = [dict(np.load(p)) for p in sorted((work / "frames" / "scene-0103").glob("*.npz"), key=lambda p: int(p.stem))]
    fsess = ort.InferenceSession(str(work / f"{a.src}.onnx"), so, providers=["CPUExecutionProvider"])
    ref = [fsess.run(None, {k: z[k] for k in Qz.HEAD_IN}) for z in evals]
    ops = sorted({n.op_type for n in m.graph.node})
    for v in a.variants.split(","):
        kw = {}
        if v.startswith("only:"):
            kw["exclude_op_types"] = [o for o in ops if o != v[5:]]
        elif v.startswith("-node:"):
            kw["exclude_nodes"] = [n.name for n in m.graph.node if v[6:] in n.name]
        elif v.startswith("-"):
            kw["exclude_op_types"] = [v[1:]]
        q = F.quantize_full_qdq(m, ranges=ranges, activation_dtype=a.act, **kw)
        sess = ort.InferenceSession(q.SerializeToString(), so, providers=["CPUExecutionProvider"])
        worst = [1.0, 1.0, 1.0]
        for z, r in zip(evals, ref):
            got = sess.run(None, {k: z[k] for k in Qz.HEAD_IN})
            worst = [min(w, cos(x, y)) for w, x, y in zip(worst, r, got)]
        print(f"{a.act} {v:28s} worst cos cls {worst[0]:.5f} reg {worst[1]:.5f} dec {worst[2]:.5f}", flush=True)


if __name__ == "__main__":
    main()
