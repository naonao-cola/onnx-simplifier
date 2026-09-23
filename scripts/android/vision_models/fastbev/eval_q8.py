#!/usr/bin/env python3
"""Host check of the int8 pieces end to end (ORT CPU, graph optimizations off so the QDQ graph is
evaluated as written), on the frames validate.py saved: GT matches and cosines vs fp32 torch.

usage: eval_q8.py m0|pp --work <dir>
m0: q8 encoder on each time step's 6 uint8 images -> uint8 feature tables; the view transform is a
    plain byte gather (numpy) with the host LUTs ("no camera" = the zero point) -> q8 BEV net.
pp: q8 encoder -> q8 view+BEV (Gather on uint8 inside the graph).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import data as D
import decode as DEC
import numpy as np
import onnxruntime as ort
import torch


def sess(p):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 8
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def deq(x, q):
    return (x.astype(np.float32) - q["zero_point"]) * q["scale"]


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def m0_gather(tables, luts, zp):
    """tables: 4 x (6*64*176, 64) uint8; luts: 4 x (160000,) int32 into [table; zero row]
    -> (1, 200, 200, 1024) uint8, channel = z*256 + t*64 + c."""
    outs = []
    for tab, lut in zip(tables, luts):
        t = np.concatenate([tab, np.full((1, 64), zp, np.uint8)])
        outs.append(t[lut].reshape(40000, 4, 64))
    return np.concatenate(outs, 2).reshape(1, 200, 200, 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam", choices=["m0", "pp"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    io = json.loads((work / f"{a.fam}_io.json").read_text())
    frames = [torch.load(p, weights_only=False)
              for p in sorted((work / f"{a.fam}_frames").glob("*.pt"), key=lambda p: int(p.stem))]
    enc = sess(work / f"{a.fam}_enc.q8.onnx")
    tot = np.zeros(3, int)
    if a.fam == "m0":
        bev = sess(work / "m0_bev.q8.onnx")
        qf, qb = io["m0_enc.q8"]["feats"], io["m0_bev.q8"]
        for i, f in enumerate(frames):
            tables = [enc.run(None, {"img": f["img_u8"][t]})[0].reshape(-1, 64) for t in range(4)]
            fc = cos(deq(tables[0], qf), f["feats"][:6])
            vol = m0_gather(tables, f["luts"], qf["zero_point"])
            cls, reg, dirc = bev.run(None, {"vol": vol})
            cls, reg, dirc = deq(cls, qb["cls"])[0], deq(reg, qb["reg"])[0], deq(dirc, qb["dir"])[0]
            hc = [cos(x, y) for x, y in zip((cls, reg, dirc), f["head"])]
            b, s, names = DEC.m0_decode(cls, reg, dirc)
            m = D.match(b[:, :2], s, names, f["gt"])
            tot += m
            print(f"frame {i}: feats cos {fc:.4f}  head cos {' '.join(f'{c:.4f}' for c in hc)}  "
                  f"{m[0]} of {m[2]} GT ({m[1]} dets)")
    else:
        vb = sess(work / "pp_viewbev.q8.onnx")
        qe, qv = io["pp_enc.q8"], io["pp_viewbev.q8"]
        names_out = [o.name for o in vb.get_outputs()]
        for i, f in enumerate(frames):
            feats, depth = enc.run(None, {"img": f["img_u8"]})
            fc = cos(deq(feats, qe["feats"]), f["feats"])
            dc = cos(deq(depth, qe["depth"]), f["depth"])
            outs = vb.run(None, {"feats": feats.reshape(-1, 64), "depth": depth.reshape(-1),
                                 "idx": f["idx"], "didx": f["didx"]})
            head = [deq(o, qv[n])[0] for o, n in zip(outs, names_out)]
            hc = [cos(x, y) for x, y in zip(head, f["head"])]
            b, s, names = DEC.pp_decode(head)
            m = D.match(b[:, :2], s, names, f["gt"])
            tot += m
            print(f"frame {i}: feats cos {fc:.4f} depth cos {dc:.4f}  head cos {' '.join(f'{c:.4f}' for c in hc)}  "
                  f"{m[0]} of {m[2]} GT ({m[1]} dets)")
    print(f"{a.fam} int8 (ORT CPU): {tot[0]} matched of {tot[2]} GT ({tot[1]} dets >= 0.3) over {len(frames)} frames")


if __name__ == "__main__":
    main()
