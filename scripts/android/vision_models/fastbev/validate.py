#!/usr/bin/env python3
"""Check the rebuilt Fast-BEV M0 / Fast-BEV++ against the upstream-literal view transform, and
their detections against GT, on real nuScenes-mini frames; save the frames for export/phone runs.

usage: validate.py m0|pp --ckpt <pth> --data <nuscenes-mini> --work <dir> [--scene scene-0103]
                   [--frames 6] [--no-adj-swap]

m0: per frame, the 4 time steps' encoder features (current frame + the previous 3 keyframes,
    data.py) -> geometry.ref_backproject_inplace (verbatim upstream loop) vs model.ViewM0 on the
    host LUT (must be identical), then BevM0 + decode.m0_decode -> GT match.
pp: EncoderPP -> geometry.ref_fastray (verbatim upstream loop, incl. the cam-0 pixel zeroing) vs
    model.ViewPP on the host LUT, then BevPP + decode.pp_decode -> GT match.
Match criteria as ../bevformer_tiny (score >= 0.3, same class, center within 2 m).
Saves <work>/<model>_frames/<i>.pt with everything export.py / e2e_phone.py need.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import data as D
import decode as DEC
import geometry as G
import model as M
import numpy as np
import torch


def run_m0(a, ns, toks, out):
    enc, bev = M.load_m0(a.ckpt)
    view = M.ViewM0()
    points = G.m0_points()
    tot = np.zeros(3, int)
    for i, tok in enumerate(toks):
        f = ns.m0_frame(tok, adj_cam_swap=not a.no_adj_swap)
        img = torch.from_numpy(f["img"])  # (24, 3, 256, 704)
        t0 = time.time()
        feats = enc(img)  # (24, 64, 176, 64) NHWC
        t_enc = time.time() - t0
        # upstream-literal: per time step, backproject_inplace on NCHW features
        vols = []
        luts = []
        for t in range(4):
            ft = feats[t * 6:(t + 1) * 6].permute(0, 3, 1, 2)
            vols.append(G.ref_backproject_inplace(ft, points, G.m0_projection(f["lidar2img"][t])))
            luts.append(G.m0_lut(f["lidar2img"][t], points))
        ref = torch.cat(vols, 0)  # (256, X, Y, Z), channel = t*64 + c
        C, X, Y, Z = ref.shape
        ref_nhwc = ref.permute(1, 2, 3, 0).reshape(1, X, Y, Z * C)  # M2BevNeck: (x, y, z*C + c)
        got = view(*[feats[t * 6:(t + 1) * 6].reshape(-1, 64) for t in range(4)], *luts)
        diff = float((got - ref_nhwc).abs().max())
        cls, reg, dirc = bev(got)
        b, s, names = DEC.m0_decode(cls[0].numpy(), reg[0].numpy(), dirc[0].numpy())
        m = D.match(b[:, :2], s, names, f["gt"])
        tot += m
        print(f"frame {i} {tok[:8]}: view max|diff| vs upstream loop {diff:.1e}; enc {t_enc:.2f}s; "
              f"{m[0]} of {m[2]} GT matched ({m[1]} dets >= 0.3)")
        torch.save({"img_u8": f["img_u8"], "img": f["img"], "lidar2img": f["lidar2img"], "gt": f["gt"],
                    "luts": [x.numpy() for x in luts], "feats": feats.numpy(), "vol": got.numpy(),
                    "head": [x[0].numpy() for x in (cls, reg, dirc)], "token": tok}, out / f"{i}.pt")
    print(f"m0 total: {tot[0]} matched of {tot[2]} GT ({tot[1]} dets >= 0.3) over {len(toks)} frames")


def run_pp(a, ns, toks, out):
    enc, bev = M.load_pp(a.ckpt)
    vc = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]["img_view_transformer.voxel_coords"]
    view = M.ViewPP()
    tot = np.zeros(3, int)
    for i, tok in enumerate(toks):
        f = ns.pp_frame(tok)
        img = torch.from_numpy(f["img"])
        t0 = time.time()
        raw = enc.raw(img)
        t_enc = time.time() - t0
        x = raw.permute(0, 2, 3, 1)
        feats, depth = x[..., M.PP_D:], torch.sigmoid(x[..., :M.PP_D])  # == enc(img)
        ref = G.ref_fastray(None, raw, vc, f["sensor2keyego"], f["intrin"], f["post"])  # (1, 64, Y, X)
        idx, didx = G.pp_lut(vc, f["sensor2keyego"], f["intrin"], f["post"])
        got = view(feats.reshape(-1, 64), depth.reshape(-1), idx, didx)  # (1, Y, X, 64)
        diff = float((got - ref.permute(0, 2, 3, 1)).abs().max())
        head = bev(got)[0].numpy()
        b, s, names = DEC.pp_decode(head)
        m = D.match(b[:, :2], s, names, f["gt"])
        tot += m
        print(f"frame {i} {tok[:8]}: view max|diff| vs upstream loop {diff:.1e}; enc {t_enc:.2f}s; "
              f"{m[0]} of {m[2]} GT matched ({m[1]} dets >= 0.3)")
        torch.save({"img_u8": f["img_u8"], "img": f["img"], "gt": f["gt"], "idx": idx.numpy(),
                    "didx": didx.numpy(), "feats": feats.numpy(), "depth": depth.numpy(), "bev": got.numpy(),
                    "head": head, "sensor2keyego": f["sensor2keyego"], "intrin": f["intrin"], "post": f["post"],
                    "token": tok}, out / f"{i}.pt")
    print(f"pp total: {tot[0]} matched of {tot[2]} GT ({tot[1]} dets >= 0.3) over {len(toks)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["m0", "pp"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--no-adj-swap", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    ns = D.NuScenesMini(a.data)
    toks = ns.scene_samples(a.scene)[: a.frames]
    out = Path(a.work) / f"{a.model}_frames"
    out.mkdir(parents=True, exist_ok=True)
    (run_m0 if a.model == "m0" else run_pp)(a, ns, toks, out)


if __name__ == "__main__":
    main()
