#!/usr/bin/env python3
"""Check the plain-torch MapTR-tiny against upstream semantics, on real nuScenes-mini frames.

  1. msda_rank5 == mmcv's multi_scale_deformable_attn_pytorch (verbatim copy) at MapTR's shapes
  2. the official checkpoint loads with no missing/unexpected keys (beyond model.unused)
  3. per frame: encoder/decoder with msda_rank5 vs the upstream-literal path (6-D MSDA, SCA
     nonzero rebatch) -> max abs diff; ref_cam clamped vs unclamped (exact); decoded polylines
  4. saves each frame's piece inputs/outputs to <work>/frames/<i>.pt for export.py

No map GT: nuScenes-mini's map expansion needs a nuScenes account, so accuracy is vs this fp32
model (the upstream-literal path) only.

usage: validate.py --ckpt maptr_tiny_r50_24e_bevformer.pth --data <nuscenes-mini>
       [--scene scene-0103] [--frames 6] [--work work]
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bevformer_tiny"))  # nuScenes-mini loader (same pipeline)
sys.path.insert(0, str(HERE))  # this dir's model.py first
import model as M  # noqa: E402
from nuscenes import NuScenesMini  # noqa: E402


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def check_msda():
    g = torch.Generator().manual_seed(0)
    for (b, q, m, p, h, w) in ((2, 3000, 8, 4, M.BEV_H, M.BEV_W), (1, 1000, 8, 4, M.BEV_H, M.BEV_W),
                               (1, 3000, 8, 8, M.FH, M.FW)):
        v = torch.randn(b, h * w, m, 32, generator=g)
        loc = torch.rand(b, q, m, p, 2, generator=g) * 1.2 - 0.1
        a = torch.softmax(torch.randn(b, q, m, p, generator=g), -1)
        r = M.msda_mmcv_reference(v, [(h, w)], loc.reshape(b, q, m, 1, p, 2), a.reshape(b, q, m, 1, p))
        o = M.msda_rank5(v, (h, w), loc, a)
        print(f"msda_rank5 vs mmcv {tuple(loc.shape)}: max abs {(o - r).abs().max():.2e}")


def fmt(pts, scores, labels, thr):
    k = scores >= thr
    counts = {c: int((labels[k] == i).sum()) for i, c in enumerate(M.CLASSES)}
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    check_msda()
    bb, enc, dec = M.load_official(a.ckpt)
    print("checkpoint loaded: every key mapped")
    ns = NuScenesMini(a.data)
    out = Path(a.work) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        t0 = time.time()
        feats = bb(f["img"])
        can_bus = M.test_can_bus(f["can_bus_abs"])
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        bev = enc(feats, can_bus, ref_cam, bev_mask)
        bev_ref = enc(feats, can_bus, ref_cam, bev_mask, reference=True)
        rc_raw, bm_raw = M.reference_points_cam(f["lidar2img"], clamp=None)
        bev_raw = enc(feats, can_bus, rc_raw, bm_raw, reference=True)
        cls, pts = dec(bev)
        cls_r, pts_r = dec(bev_ref, reference=True)
        p, s, lab = M.decode(cls, pts)
        print(f"frame {i} {tok[:8]}: enc rank5 vs upstream max abs {(bev - bev_ref).abs().max():.2e}, "
              f"clamped vs unclamped {(bev_ref - bev_raw).abs().max():.2e}; dec cls {(cls - cls_r).abs().max():.2e} "
              f"pts {(pts - pts_r).abs().max():.2e}; visible (cam,query) pairs {int((bev_mask.sum(-1) > 0).sum())}/"
              f"{M.NUM_CAMS * M.NQ}; >=0.4: {fmt(p, s, lab, 0.4)}  ({time.time() - t0:.1f} s, peak {peak_gb():.1f} GB)")
        torch.save({"img": f["img"], "feats": feats, "can_bus": can_bus, "ref_cam": ref_cam, "bev_mask": bev_mask,
                    "bev": bev, "cls": cls, "pts": pts, "token": tok}, out / f"{i}.pt")


if __name__ == "__main__":
    main()
