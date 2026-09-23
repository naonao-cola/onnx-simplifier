"""Real Sparse4D DFA calls as case files for dfa_host_check.c / dfa_sim.c.

  python dfa_case.py --ckpt <pth> --work <work> --out <case dir> [--frame 0] [--layer 0]

Runs the fp32 model on a saved frame (validate.py --work) up to one decoder layer's DFA and writes
the kernel's inputs in its layouts (dfa_core.h): the 4 FPN levels quantized to uint8 per level
(the HTP emits one scale per tensor), channels-last (6, H, W, 256); the projected points padded
13 -> 16, (6, Q, 16, 2); the weights (6, 4, Q, 8, 16) with 0 on the pads. Two references:
  ref_u8.f32: model.dfa_upstream on the *dequantized* value maps (what the kernel must match up to
              its Q15 weight rounding),
  ref_f32.f32: the same on the fp32 maps (the total error, uint8 values included).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import MEAN, STD  # noqa: E402
from model import (  # noqa: E402
    CAMS,
    GROUPS,
    LEVELS,
    OPS,
    PTS,
    Runner,
    Sparse4D,
    dfa_upstream,
)


def quantize(t):
    lo, hi = min(float(t.min()), 0.0), max(float(t.max()), 0.0)
    scale = (hi - lo) / 255 or 1.0
    zp = int(round(-lo / scale))
    q = torch.clamp(torch.round(t / scale) + zp, 0, 255).to(torch.uint8)
    return q, scale, zp


def to_kernel(pts, w):
    """pts (6, N, 13, 2), w (N, 6, 4, 13, 8) -> (6, N, 16, 2), (6, 4, N, 8, 16)."""
    n = pts.shape[1]
    p16 = torch.zeros(CAMS, n, 16, 2)
    p16[:, :, :PTS] = pts
    w16 = torch.zeros(CAMS, LEVELS, n, GROUPS, 16)
    w16[..., :PTS] = w.permute(1, 2, 0, 4, 3)
    return p16, w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0, help="DFA layer 0..5")
    ap.add_argument("--q", type=int, default=900, help="first q anchors only (for hexagon-sim)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    m = Sparse4D().load_official(a.ckpt).eval()
    grab = {}
    deform = [i for i, op in enumerate(OPS) if op == "deformable"]

    def dfa(fmaps, pts, w, layer):
        if layer == deform[a.layer] and "pts" not in grab:
            grab.update(fmaps=fmaps, pts=pts, w=w)
        return dfa_upstream(fmaps, pts, w)

    run = Runner(m, dfa)
    frames = sorted((Path(a.work) / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
    for p in frames[: a.frame + 1]:
        fr = pickle.load(open(p, "rb"))
        img = torch.from_numpy((fr["rgb"].astype(np.float32) - MEAN) / STD).permute(0, 3, 1, 2).contiguous()
        grab.clear()
        run.frame(img, fr["metas"])
    fmaps, pts, w = grab["fmaps"], grab["pts"][:, : a.q], grab["w"][: a.q]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deq, meta = [], []
    for lv, fm in enumerate(fmaps):
        nhwc = fm.permute(0, 2, 3, 1).contiguous()
        q, s, z = quantize(nhwc)
        q.numpy().tofile(out / f"val{lv}.u8")
        deq.append(((q.float() - z) * s).permute(0, 3, 1, 2))
        meta.append(f"{fm.shape[2]} {fm.shape[3]} {s!r} {z}")
    p16, w16 = to_kernel(pts, w)
    p16.numpy().astype(np.float32).tofile(out / "pts.f32")
    w16.numpy().astype(np.float32).tofile(out / "w.f32")
    dfa_upstream(deq, pts, w).numpy().astype(np.float32).tofile(out / "ref_u8.f32")
    dfa_upstream(list(fmaps), pts, w).numpy().astype(np.float32).tofile(out / "ref_f32.f32")
    (out / "meta.txt").write_text(f"{pts.shape[1]}\n" + "\n".join(meta) + "\n")
    print(f"{out}: Q {pts.shape[1]}, levels {[tuple(f.shape[2:]) for f in fmaps]}")


if __name__ == "__main__":
    main()
