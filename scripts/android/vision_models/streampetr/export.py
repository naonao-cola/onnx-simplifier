"""Export StreamPETR's two HTP pieces to ONNX, check them against torch, onnxsim them, write phone inputs.

  python export.py {img,head} --ckpt <pth> --work <work> [--frame scene-0103/3]

  img   (6, 3, 256, 704) normalized f32 -> feat (6, 16, 44, 256), channels-last so the head reads it
        as (4224, 256) tokens with a free reshape
  head  HeadCore: feat (4224, 256), pe / sa_gamma / sa_beta (4224, 256), mem_emb (512, 256),
        mem_pe3d (512, 384), mem_time (512, 256), mem_motion (512, 180) -> cls, reg (428, 10), dec (428, 256)

TorchScript exporter (dynamo=False), opset 17, checked on ORT CPU (graph optimizations off) against
the torch module on the validate.py dumps, then onnxsim (fixed shapes). Writes <work>/<piece>.onnx,
<work>/<piece>.sim.onnx and <work>/<piece>.in/ (partition_report.sh manifest + f32 inputs + ref_*
torch outputs) for one frame.
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import data as D
import model as M
import numpy as np
import onnx
import onnxruntime as ort
import torch

HEAD_IN = ["feat", "pe", "sa_gamma", "sa_beta", "mem_emb", "mem_pe3d", "mem_time", "mem_motion"]


class ImgNHWC(torch.nn.Module):
    def __init__(self, img_net):
        super().__init__()
        self.net = img_net

    def forward(self, img):
        return self.net(img).permute(0, 2, 3, 1)


def pieces(name, ckpt):
    img_net, head = M.load_official(ckpt)
    if name == "img":
        return ImgNHWC(img_net).eval(), ["img"], ["feat"], lambda z: {"img": D.normalize(z["img_u8"])}
    core = M.HeadCore(head).eval()
    for layer in core.h.decoder_layers:  # headT: cross-attention transposed; headTT: self-attention too
        layer.attentions[1].attn.transposed = name in ("headT", "headTT")
        layer.attentions[0].attn.transposed = name == "headTT"
    return core, HEAD_IN, ["cls", "reg", "dec"], lambda z: {k: torch.from_numpy(z[k]) for k in HEAD_IN}


def compare(sess, mod, inputs, label):
    with torch.no_grad():
        ref = mod(*inputs.values())
    ref = ref if isinstance(ref, tuple) else (ref,)
    got = sess.run(None, {k: v.numpy() for k, v in inputs.items()})
    for r, g, o in zip(ref, got, sess.get_outputs()):
        r, g = r.numpy().astype(np.float64).ravel(), g.astype(np.float64).ravel()
        cos = r @ g / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30)
        print(f"  {label} {o.name}: max abs {np.abs(r - g).max():.3g} cos {cos:.7f}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=["img", "head", "headT", "headTT"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--frame", default="scene-0103/3")
    a = ap.parse_args()
    w = Path(a.work)
    mod, in_names, out_names, inputs_of = pieces(a.piece, a.ckpt)
    z = np.load(w / "frames" / f"{a.frame}.npz")
    inputs = inputs_of(z)
    raw, simp = w / f"{a.piece}.onnx", w / f"{a.piece}.sim.onnx"
    t = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, tuple(inputs.values()), str(raw), input_names=in_names, output_names=out_names,
                          opset_version=17, dynamo=False, do_constant_folding=True)
    print(f"{a.piece}: exported in {time.time() - t:.1f}s")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    compare(ort.InferenceSession(str(raw), so, providers=["CPUExecutionProvider"]), mod, inputs, "raw")
    import onnxsim

    sim, ok = onnxsim.simplify(str(raw), check_n=0)
    assert ok
    onnx.save(sim, str(simp))
    ops = {}
    for n in sim.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"  onnxsim: {len(sim.graph.node)} nodes {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    ref = compare(ort.InferenceSession(str(simp), so, providers=["CPUExecutionProvider"]), mod, inputs, "sim")
    ind = w / f"{a.piece}.in"
    ind.mkdir(exist_ok=True)
    with open(ind / "manifest.txt", "w") as man:  # partition_report.sh's manifest format
        for k, v in inputs.items():
            v.numpy().astype(np.float32).tofile(ind / f"{k}.bin")
            man.write(f"{k} f32 {ind / f'{k}.bin'} {','.join(map(str, v.shape))}\n")
    for k, v in zip(out_names, ref):
        v.numpy().astype(np.float32).tofile(ind / f"ref_{k}.bin")
    print(f"  peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
