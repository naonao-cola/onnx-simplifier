#!/usr/bin/env python3
"""Export one Fast-BEV / Fast-BEV++ piece to ONNX, check it with ORT CPU, onnxsim it, write phone inputs.

usage: export.py <piece> --ckpt <pth> --work <dir> [--uint8]
  m0_enc      EncoderM0          img (6, 3, 256, 704) [or uint8 (6, 256, 704, 3)] -> feats (6, 64, 176, 64)
  m0_view     ViewM0             f0..f3 (67584, 64), i0..i3 int32 (160000,) -> vol (1, 200, 200, 1024)
  m0_bev      BevM0              vol -> cls (1, 100, 100, 80), reg (.., 72), dir (.., 16)
  m0_viewbev  ViewM0 + BevM0     f0..f3, i0..i3 -> cls, reg, dir  (the volume never leaves the HTP)
  pp_enc      EncoderPP          img -> feats (6, 16, 44, 64), depth (6, 16, 44, 59)
  pp_view     ViewPP             feats (4224, 64), depth (249216,), idx, didx int32 (114688,) -> bev
  pp_bev      BevPP              bev (1, 128, 128, 64) -> heatmap, reg, height, dim, rot, vel (1, 128, 128, k)
  pp_viewbev  ViewPP + BevPP
Test inputs: the frames validate.py saved in <work>/{m0,pp}_frames/<i>.pt (frame 1 for the phone).
Exported with the TorchScript exporter (opset 17), fixed shapes, checked with ORT CPU (graph
optimizations off) against torch, simplified with onnxsim and checked again. Outputs
<work>/<piece>[.u8].onnx, .sim.onnx and <work>/<piece>.in/ (manifest + raw inputs + ref outputs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import model as M
import numpy as np
import torch

HEADS = [n for n, _ in M.HEAD_PP]


class Chain(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, *x):
        return self.b(self.a(*x))


def piece(name, ckpt, uint8):
    fam = name.split("_")[0]
    enc, bev = (M.load_m0 if fam == "m0" else M.load_pp)(ckpt, uint8_input=uint8)
    view = M.ViewM0() if fam == "m0" else M.ViewPP()

    def img(f):
        return (torch.from_numpy(f["img_u8"].reshape(-1, 256, 704, 3)[:6].astype(np.float32)),) if uint8 else (
            torch.from_numpy(f["img"][:6].copy()),)

    if fam == "m0":
        def view_in(f):
            feats = torch.from_numpy(f["feats"])
            return (*[feats[t * 6:(t + 1) * 6].reshape(-1, 64) for t in range(4)],
                    *[torch.from_numpy(x) for x in f["luts"]])
        vnames = ["f0", "f1", "f2", "f3", "i0", "i1", "i2", "i3"]
        table = {
            "m0_enc": (enc, ["img"], ["feats"], img),
            "m0_view": (view, vnames, ["vol"], view_in),
            "m0_bev": (bev, ["vol"], ["cls", "reg", "dir"], lambda f: (torch.from_numpy(f["vol"]),)),
            "m0_viewbev": (Chain(view, bev).eval(), vnames, ["cls", "reg", "dir"], view_in),
        }
    else:
        def view_in(f):
            return (torch.from_numpy(f["feats"]).reshape(-1, 64), torch.from_numpy(f["depth"]).reshape(-1),
                    torch.from_numpy(f["idx"]), torch.from_numpy(f["didx"]))
        vnames = ["feats", "depth", "idx", "didx"]
        table = {
            "pp_enc": (enc, ["img"], ["feats", "depth"], img),
            "pp_view": (view, vnames, ["bev"], view_in),
            "pp_bev": (bev, ["bev"], HEADS, lambda f: (torch.from_numpy(f["bev"]),)),
            "pp_viewbev": (Chain(view, bev).eval(), vnames, HEADS, view_in),
        }
    if name not in table:
        raise SystemExit(f"unknown piece {name}")
    return table[name]


def compare(sess, mod, inputs_of, frames, in_names, label):
    worst = []
    for f in frames:
        x = inputs_of(f)
        ref = mod(*x)
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = sess.run(None, {k: v.numpy() for k, v in zip(in_names, x)})
        for r, g in zip(ref, got):
            r = r.numpy().ravel().astype(np.float64)
            g = g.ravel().astype(np.float64)
            cos = float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30))
            worst.append((float(np.abs(r - g).max()), cos))
    md, mc = max(w[0] for w in worst), min(w[1] for w in worst)
    print(f"  {label}: ORT CPU vs torch over {len(frames)} frames: max abs {md:.2e}, min cos {mc:.7f}")
    return md, mc


DT = {torch.float32: "f32", torch.int32: "i32", torch.uint8: "u8", torch.int64: "i64"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--uint8", action="store_true", help="NHWC RGB pixel input (encoders; uint8 once quantized)")
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import onnx
    import onnxruntime as ort

    import onnxsim

    work = Path(a.work)
    fam = a.piece.split("_")[0]
    frames = [torch.load(p, weights_only=False)
              for p in sorted((work / f"{fam}_frames").glob("*.pt"), key=lambda p: int(p.stem))][: a.frames]
    mod, in_names, out_names, inputs_of = piece(a.piece, a.ckpt, a.uint8)
    tag = a.piece + (".u8" if a.uint8 else "")
    x = inputs_of(frames[min(1, len(frames) - 1)])
    raw = work / f"{tag}.onnx"
    t = time.time()
    torch.onnx.export(mod, x, str(raw), input_names=in_names, output_names=out_names, opset_version=17,
                      dynamo=False, do_constant_folding=True)
    print(f"{tag}: exported in {time.time() - t:.1f}s, {raw.stat().st_size / 1e6:.1f} MB")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    compare(ort.InferenceSession(str(raw), so, providers=["CPUExecutionProvider"]), mod, inputs_of, frames,
            in_names, "raw")
    sim, _ = onnxsim.simplify(str(raw), check_n=0)
    simp = work / f"{tag}.sim.onnx"
    onnx.save(sim, str(simp))
    ops = {}
    for n in sim.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("  onnxsim: " + " ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
    compare(ort.InferenceSession(str(simp), so, providers=["CPUExecutionProvider"]), mod, inputs_of, frames,
            in_names, "sim")
    ind = work / f"{tag}.in"
    ind.mkdir(exist_ok=True)
    with open(ind / "manifest.txt", "w") as man:
        for k, v in zip(in_names, x):
            v.numpy().tofile(ind / f"{k}.bin")
            man.write(f"{k} {DT[v.dtype]} {(ind / f'{k}.bin').resolve()} {','.join(map(str, v.shape))}\n")
    ref = mod(*x)
    for k, v in zip(out_names, ref if isinstance(ref, tuple) else (ref,)):
        v.numpy().astype(np.float32).tofile(ind / f"ref_{k}.bin")


if __name__ == "__main__":
    sys.exit(main())
