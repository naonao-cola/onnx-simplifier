#!/usr/bin/env python3
"""Check the plain-torch BEVFormer-tiny against upstream semantics, on real nuScenes-mini frames.

  1. msda_rank5 == mmcv's multi_scale_deformable_attn_pytorch (verbatim copy) on random inputs
  2. the official checkpoint loads with no missing/unexpected keys (beyond model.UNUSED)
  3. per frame of a scene, run temporally (prev_bev carried, rotated, shifted as upstream):
     encoder/decoder with msda_rank5 vs the upstream-literal path (6D MSDA, SCA nonzero rebatch)
     -> max abs diff; decoded boxes vs GT (2 m center match at score >= 0.3)
  4. saves each frame's piece inputs/outputs to <work>/frames/<i>.pt for export.py to check against

usage: validate.py --ckpt bevformer_tiny_epoch_24.pth --data <nuscenes-mini> [--scene scene-0103]
       [--frames 3] [--work work]
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import torch

import model as M
from nuscenes import NuScenesMini, match, temporal_can_bus


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def check_msda():
    g = torch.Generator().manual_seed(0)
    for (b, q, m, p, h, w) in ((2, 2500, 8, 4, 50, 50), (6, 700, 8, 8, 15, 25)):
        v = torch.randn(b, h * w, m, 32, generator=g)
        loc = torch.rand(b, q, m, p, 2, generator=g) * 1.2 - 0.1  # some samples fall outside
        a = torch.softmax(torch.randn(b, q, m, p, generator=g), -1)
        r = M.msda_mmcv_reference(v, [(h, w)], loc.reshape(b, q, m, 1, p, 2), a.reshape(b, q, m, 1, p))
        o = M.msda_rank5(v, (h, w), loc, a)
        print(f"msda_rank5 vs mmcv reference {tuple(loc.shape)}: max abs diff {(o - r).abs().max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    check_msda()
    t = time.time()
    backbone, encoder, decoder = M.load_official(a.ckpt)
    print(f"checkpoint loaded: all keys mapped, unused: {len(M.UNUSED)} ({time.time() - t:.1f}s, peak {peak_mb():.0f} MB)")
    ns = NuScenesMini(a.data)
    out = Path(a.work) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    prev_bev, prev_abs = None, None
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        t = time.time()
        feats = backbone(f["img"])
        tb = time.time() - t
        if prev_bev is None:
            has_prev, shift, prev_in = torch.zeros(1), torch.zeros(2), torch.zeros(M.NQ, M.EMBED)
        else:
            has_prev, shift = torch.ones(1), M.can_bus_shift(can_bus)
            prev_in = M.rotate_prev_bev(prev_bev, can_bus)
        enc_in = (feats, prev_in, has_prev, shift, can_bus, ref_cam, bev_mask)
        t = time.time()
        bev = encoder(*enc_in)
        te = time.time() - t
        bev_ref = encoder(*enc_in, reference=True)
        rc_raw, _ = M.reference_points_cam(f["lidar2img"], clamp=None)
        bev_raw = encoder(feats, prev_in, has_prev, shift, can_bus, rc_raw, bev_mask, reference=True)
        t = time.time()
        cls, bbox = decoder(bev)
        td = time.time() - t
        cls_r, bbox_r = decoder(bev_ref, reference=True)
        boxes, scores, labels = M.decode(cls, bbox)
        tp, npred, ngt = match(boxes, scores, labels, f["gt"])
        vis = (bev_mask.sum(-1) > 0).float().sum(0)
        print(f"frame {i} {tok[:8]} has_prev {int(has_prev)} shift {shift.tolist()} | visible cams/query "
              f"mean {vis.mean():.2f} none {int((vis == 0).sum())}")
        print(f"  ref_cam |max| unclamped {rc_raw.abs().max():.3g}; encoder clamped vs unclamped (upstream-literal) "
              f"max abs {(bev_ref - bev_raw).abs().max():.2e}")
        print(f"  encoder rank5 vs upstream-literal: max abs {(bev - bev_ref).abs().max():.2e}  "
              f"decoder cls {(cls - cls_r).abs().max():.2e} bbox {(bbox - bbox_r).abs().max():.2e}")
        print(f"  cpu fp32 ms: backbone(6 cams) {tb * 1e3:.0f}  encoder {te * 1e3:.0f}  decoder {td * 1e3:.0f}")
        print(f"  detections >=0.3: {npred}, GT {ngt}, matched (same class, <2 m) {tp}; top5 "
              + ", ".join(f"{M_cls(labels[j])}@{scores[j]:.2f}" for j in range(5)))
        torch.save({"img": f["img"], "enc_in": enc_in, "bev": bev, "cls": cls, "bbox": bbox, "gt": f["gt"]},
                   out / f"{i}.pt")
        prev_bev, prev_abs = bev, f["can_bus_abs"]
    print(f"peak rss {peak_mb():.0f} MB")


def M_cls(i):
    from nuscenes import CLASSES

    return CLASSES[int(i)]


if __name__ == "__main__":
    main()
