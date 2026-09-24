#!/usr/bin/env python3
"""Export one BEVFormer-tiny piece to ONNX, check it against the torch model, then onnxsim it.

usage: export.py <piece> --ckpt <pth> --work <dir>
  pieces (smallest first): backbone1 backbone6 enc1 enc3 decoder

Each piece is exported with torch.onnx.export (TorchScript exporter, dynamo=False, opset 17) under
no_grad, all shapes fixed. Test inputs are the real nuScenes frames validate.py saved in
<work>/frames/*.pt (the backbone pieces use the saved camera images, the encoder the saved encoder
inputs, the decoder the saved BEV). The exported graph is checked with ORT CPU against the torch
piece (max abs diff, cosine), simplified with onnxsim (fixed shapes; no onnx.checker), and checked
again. Outputs: <work>/<piece>.onnx (raw) and <work>/<piece>.sim.onnx, plus <work>/<piece>.in/*.bin
(raw fp32 inputs of frame 1 for the phone) and a line in <work>/export_log.txt.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

import model as M

ENC_NAMES = ["feats", "prev_bev", "has_prev", "shift", "can_bus", "ref_cam", "bev_mask"]


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def piece(name, ckpt):
    bb, enc, dec = M.load_official(ckpt)
    if name.startswith("backbone"):
        n = int(name[len("backbone"):])
        return bb, ["img"], ["feats"], lambda f: (f["img"][:n],)
    if name.startswith("enc"):
        n = int(name[3:])
        if n != len(enc.layers):
            enc.layers = enc.layers[:n]
        return enc, ENC_NAMES, ["bev_embed"], lambda f: tuple(f["enc_in"])
    if name == "decoder":
        return dec, ["bev_embed"], ["cls_scores", "bbox_preds"], lambda f: (f["bev"],)
    raise SystemExit(f"unknown piece {name}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import onnx
    import onnxruntime as ort

    work = Path(a.work)
    frames = [torch.load(p, weights_only=False) for p in sorted((work / "frames").glob("*.pt"))]
    mod, in_names, out_names, inputs_of = piece(a.piece, a.ckpt)
    x = inputs_of(frames[min(1, len(frames) - 1)])
    raw = work / f"{a.piece}.onnx"
    t = time.time()
    torch.onnx.export(mod, x, str(raw), input_names=in_names, output_names=out_names, opset_version=17,
                      dynamo=False, do_constant_folding=True)
    m = onnx.load(str(raw))
    size = raw.stat().st_size
    print(f"{a.piece}: exported in {time.time() - t:.1f}s, {len(m.graph.node)} nodes, {size / 1e6:.1f} MB, "
          f"peak rss {peak_mb():.0f} MB")
    del m
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    compare(ort.InferenceSession(str(raw), so, providers=["CPUExecutionProvider"]), mod, inputs_of, frames,
            in_names, "raw")
    import onnxsim

    t = time.time()
    sim, ok = onnxsim.simplify(str(raw), check_n=0)
    simp = work / f"{a.piece}.sim.onnx"
    onnx.save(sim, str(simp))
    ops = {}
    for n in sim.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  onnxsim: {time.time() - t:.1f}s, {len(sim.graph.node)} nodes, peak rss {peak_mb():.0f} MB; ops "
          + " ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
    md, mc = compare(ort.InferenceSession(str(simp), so, providers=["CPUExecutionProvider"]), mod, inputs_of,
                     frames, in_names, "sim")
    ind = work / f"{a.piece}.in"
    ind.mkdir(exist_ok=True)
    with open(ind / "manifest.txt", "w") as man:  # partition_report.sh's manifest format
        for k, v in zip(in_names, x):
            v.numpy().astype(np.float32).tofile(ind / f"{k}.bin")
            man.write(f"{k} f32 {(ind / f'{k}.bin').resolve()} {','.join(map(str, v.shape))}\n")
    for k, v in zip(out_names, (mod(*x) if len(out_names) > 1 else (mod(*x),))):
        v.numpy().astype(np.float32).tofile(ind / f"ref_{k}.bin")
    with open(work / "export_log.txt", "a") as fh:
        fh.write(f"{a.piece} nodes {len(sim.graph.node)} maxabs {md:.2e} cos {mc:.7f} peak_mb {peak_mb():.0f}\n")
    print(f"peak rss {peak_mb():.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
