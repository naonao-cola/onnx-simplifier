#!/usr/bin/env python3
"""Whole BEVFormer-tiny on the phone's HTP, frame after frame, vs fp32 torch and nuScenes GT.

Per keyframe of a scene: host computes the geometry (ref_cam, bev_mask, can_bus, shift) and
rotates the previous BEV; backbone6 -> enc3 -> decoder each run strict all-HTP through
qnn_run_multi (pieces from export.py, already on the phone via run_phone.sh), feeding the phone's
own outputs forward (the HTP prev_bev is carried, so fp16 error accumulates over time like it
would in the app). fp32 torch runs the same frames alongside. Reports per frame: cosine of
feats/bev/cls/bbox vs fp32, and GT matches for both.

usage: e2e_phone.py --ckpt <pth> --data <nuscenes-mini> --work <work> [--scene scene-0103] [--frames 6]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

import model as M
from nuscenes import NuScenesMini, match, temporal_can_bus

DEV = os.environ.get("DEVICE_SERIAL", "239dbd8f")
R = os.environ.get("R", "/data/local/tmp/bevformer_tiny")
ADB = ["adb", "-s", DEV]


def phone(work: Path, piece: str, inputs: dict) -> tuple[list[np.ndarray], float]:
    tmp = work / "e2e_tmp"
    tmp.mkdir(exist_ok=True)
    lines = []
    for k, v in inputs.items():
        v = np.ascontiguousarray(v, np.float32)
        v.tofile(tmp / f"{k}.bin")
        subprocess.run([*ADB, "push", "-q", str(tmp / f"{k}.bin"), f"{R}/{k}.bin"], check=True)
        lines.append(f"{k} f32 {k}.bin {','.join(map(str, v.shape))}")
    (tmp / "m.txt").write_text("\n".join(lines) + "\n")
    subprocess.run([*ADB, "push", "-q", str(tmp / "m.txt"), f"{R}/e2e_{piece}.txt"], check=True)
    out = subprocess.run([*ADB, "shell", f"cd {R} && LD_LIBRARY_PATH={R} QNN_PERF=burst "
                          f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
                          f"./qnn_run_multi {piece}.sim.onnx e2e_{piece}.txt htp 3 e2e_{piece} 2>&1"],
                         capture_output=True, text=True).stdout
    if "PASS" not in out:
        raise RuntimeError(f"{piece}: {out[-400:]}")
    res = []
    for i, _, dt, shape in re.findall(r"^out (\d+) (\S+) (\S+) (\S+)", out, re.M):
        subprocess.run([*ADB, "pull", "-q", f"{R}/e2e_{piece}_o{i}.bin", str(tmp / f"o{i}.bin")], check=True)
        dims = [int(d) for d in re.findall(r"\d+", shape)]
        res.append(np.fromfile(tmp / f"o{i}.bin", np.float32).reshape(dims))
    return res, float(re.search(r"median_ms ([0-9.]+)", out).group(1))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    bb, enc, dec = M.load_official(a.ckpt)
    ns = NuScenesMini(a.data)
    prev = {"cpu": None, "htp": None}
    prev_abs = None
    tot = {"cpu": [0, 0, 0], "htp": [0, 0, 0]}
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        first = prev["cpu"] is None
        has_prev = torch.zeros(1) if first else torch.ones(1)
        shift = torch.zeros(2) if first else M.can_bus_shift(can_bus)
        # fp32 torch
        feats = bb(f["img"])
        p_in = torch.zeros(M.NQ, M.EMBED) if first else M.rotate_prev_bev(prev["cpu"], can_bus)
        bev = enc(feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)
        cls, bbox = dec(bev)
        # HTP, chained on its own outputs
        (feats_h,), t_b = phone(work, "backbone6", {"img": f["img"].numpy()})
        p_h = np.zeros((M.NQ, M.EMBED), np.float32) if first else \
            M.rotate_prev_bev(torch.from_numpy(prev["htp"]), can_bus).numpy()
        (bev_h,), t_e = phone(work, "enc3", {"feats": feats_h, "prev_bev": p_h, "has_prev": has_prev.numpy(),
                                             "shift": shift.numpy(), "can_bus": can_bus.numpy(),
                                             "ref_cam": ref_cam.numpy(), "bev_mask": bev_mask.numpy()})
        (cls_h, bbox_h), t_d = phone(work, "decoder", {"bev_embed": bev_h})
        r = {}
        for k, (c, b) in {"cpu": (cls, bbox), "htp": (torch.from_numpy(cls_h), torch.from_numpy(bbox_h))}.items():
            boxes, scores, labels = M.decode(c, b)
            r[k] = match(boxes, scores, labels, f["gt"])
            for j in range(3):
                tot[k][j] += r[k][j]
        print(f"frame {i}: HTP ms backbone6 {t_b:.0f} enc3 {t_e:.0f} decoder {t_d:.0f} | cos vs fp32: feats "
              f"{cos(feats, feats_h):.5f} bev {cos(bev, bev_h):.5f} cls {cos(cls, cls_h):.5f} bbox {cos(bbox, bbox_h):.5f}"
              f" | matched/pred/GT fp32 {r['cpu']} htp {r['htp']}")
        prev["cpu"], prev["htp"], prev_abs = bev, bev_h, f["can_bus_abs"]
    for k, (tp, npred, ngt) in tot.items():
        print(f"{k}: {tp} matched of {ngt} GT ({npred} detections >= 0.3) over {a.frames} frames")


if __name__ == "__main__":
    main()
