"""Quantize StreamPETR's HTP pieces with onnxsim's whole-graph QDQ quantizer (onnxsim.full_qdq).

  python quantize.py {img,head} --work <work> [--src img.sim] [--act uint8|uint16] [--method minmax]
                     [--exclude-ops Softmax,LayerNormalization] [--tag x]

Calibration: validate.py's dumps of scene-0061/-0553/-0757/-1077 x 6 keyframes (disjoint from the
scene-0103 evaluation frames), each frame's actual inputs -- for the head that is the memory queue as
the fp32 chain carried it. Every activation gets a calibrated scale, int8 per-channel weights,
int32 biases (quantize_full_qdq); quantized_io then drops the float graph boundary: the image
comes in as uint8 NHWC, the rest as the quantized dtype, and <stem>.json records the qparams the
host (e2e_phone.py) uses to (de)quantize.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CALIB = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]
HEAD_IN = ["feat", "pe", "sa_gamma", "sa_beta", "mem_emb", "mem_pe3d", "mem_time", "mem_motion"]
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)


def load_onnxsim():
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo))
    import onnxsim.full_qdq as F

    return F


def samples(work: Path, piece: str):
    for scene in CALIB:
        for p in sorted((work / "frames" / scene).glob("*.npz"), key=lambda p: int(p.stem)):
            z = np.load(p)
            if piece.startswith("img"):
                x = z["img_u8"].astype(np.float32).transpose(0, 3, 1, 2)
                yield {"img": x if piece.endswith("raw") else (x - MEAN) / STD}
            else:
                yield {k: z[k] for k in HEAD_IN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=["img", "img_raw", "head"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--src", default=None, help="float model stem in <work> (default <piece>.sim)")
    ap.add_argument("--act", default="uint8", choices=["uint8", "uint16"])
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--exclude-ops", default="")
    ap.add_argument("--u16", default="", help="comma-separated op types whose outputs get uint16")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import onnx

    F = load_onnxsim()
    work = Path(a.work)
    src = a.src or f"{a.piece}.sim"
    m = onnx.load(str(work / f"{src}.onnx"))
    kw = {}
    if a.exclude_ops:
        kw["exclude_op_types"] = a.exclude_ops.split(",")
    if a.u16:
        ops = set(a.u16.split(","))
        kw["tensor_dtypes"] = {o: "uint16" for n in m.graph.node if n.op_type in ops for o in n.output}
    q = F.quantize_full_qdq(m, list(samples(work, a.piece)), activation_dtype=a.act, method=a.method, **kw)
    q, info = F.quantized_io(q, nhwc_inputs=["img"] if a.piece.startswith("img") else [])
    stem = f"{src}.{'q8' if a.act == 'uint8' else 'q16'}{'' if a.method == 'minmax' else '.' + a.method}{a.tag}"
    onnx.save(q, str(work / f"{stem}.onnx"))
    (work / f"{stem}.json").write_text(json.dumps(info, indent=1))
    ops = {}
    for n in q.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"{stem}: {len(q.graph.node)} nodes; Q {ops.get('QuantizeLinear', 0)} DQ {ops.get('DequantizeLinear', 0)}; "
          f"io {json.dumps(info)[:300]}")


if __name__ == "__main__":
    sys.exit(main())
