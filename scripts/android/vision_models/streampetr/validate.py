"""fp32 check of the rebuild on real nuScenes-mini frames, and the frame dumps export.py / the phone use.

  python validate.py --ckpt <pth> --data <nuscenes-mini> --work <work> [--scene scene-0103] [--frames 6]

Per frame (a scene's first frame resets the memory queue, like Petr3D.simple_test_pts):
  * image branch (fp32 torch) -> UpstreamHead (the literal transcription) and HostState + HeadCore
    (the deployment split) on the same features: max |diff| of the final cls logits / boxes, and of
    the propagated memory, must be float noise;
  * decoded detections of both vs GT (score >= 0.3, same class, BEV center within 2 m);
  * dumps <work>/frames/<scene>/<i>.npz: uint8 images, rig + pose tensors, the image features,
    HeadCore's inputs and outputs (fp32) -- the calibration / phone reference data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import data as D
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", nargs="+", default=["scene-0103"])
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--no-upstream", action="store_true", help="dump only (calibration scenes)")
    ap.add_argument("--extra-thr", type=float, nargs="*", default=[], help="also report these score thresholds")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    img_net, head = M.load_official(a.ckpt)
    core = M.HeadCore(head).eval()
    ns = D.NuScenesStream(a.data)
    tot = {"up": [0, 0, 0], "split": [0, 0, 0], **{f"split@{t}": [0, 0, 0] for t in a.extra_thr}}
    for scene in a.scene:
        out = Path(a.work) / "frames" / scene
        out.mkdir(parents=True, exist_ok=True)
        up, host = M.UpstreamHead(head), M.HostState(head)
        prev_scene = None
        for i, tok in enumerate(ns.scene_samples(scene)[: a.frames]):
            f = ns.stream_frame(tok)
            t = M.to_torch(f)
            prev = f["scene_token"] == prev_scene
            prev_scene = f["scene_token"]
            feats = img_net(D.normalize(f["img_u8"]))  # (6, 256, 16, 44)
            feat = feats.permute(0, 2, 3, 1).reshape(-1, M.EMBED)  # (n, h, w) tokens = NHWC
            ins = {"feat": feat, **host.rig_inputs(t), **host.pre(t, prev)}
            cls, reg, dec = core(*[ins[k] for k in ("feat", "pe", "sa_gamma", "sa_beta", "mem_emb", "mem_pe3d",
                                                     "mem_time", "mem_motion")])
            s_cls, s_box = host.post(t, cls, reg, dec)
            line = f"{scene} {i}: prev {int(prev)}"
            res = {"split": (s_cls, s_box)}
            if not a.no_upstream:
                u_cls, u_box = up(feats, t, prev)
                res["up"] = (u_cls, u_box)
                d_cls = (u_cls - s_cls).abs().max().item()
                d_box = (u_box - s_box).abs().max().item()
                d_mem = (up.memory_embedding[0, :M.MEMORY_LEN] - host.emb[:M.MEMORY_LEN]).abs().max().item()
                d_ref = (up.memory_reference_point[0, :M.MEMORY_LEN] - host.ref[:M.MEMORY_LEN]).abs().max().item()
                line += f"  |d| cls {d_cls:.2e} box {d_box:.2e} mem {d_mem:.2e} ref {d_ref:.2e}"
            for k, (c, b) in res.items():
                boxes, scores, labels = M.decode(c, b)
                tp, npred, ngt = D.match(boxes, scores, labels, f["gt"])
                tot[k][0] += tp
                tot[k][1] += npred
                tot[k][2] += ngt
                line += f"  {k} {tp}/{ngt} (pred {npred})"
            for thr in a.extra_thr:
                boxes, scores, labels = M.decode(s_cls, s_box)
                r = D.match(boxes, scores, labels, f["gt"], thr=thr)
                tot[f"split@{thr}"] = [x + y for x, y in zip(tot[f"split@{thr}"], r)]
            print(line, flush=True)
            np.savez(out / f"{i}.npz", img_u8=f["img_u8"], lidar2img=f["lidar2img"], intrinsics=f["intrinsics"],
                     ego_pose=f["ego_pose"], ego_pose_inv=f["ego_pose_inv"], timestamp=f["timestamp"],
                     prev=prev, feats=feats.numpy(), **{k: v.numpy() for k, v in ins.items()},
                     cls=cls.numpy(), reg=reg.numpy(), dec=dec.numpy(),
                     gt_names=np.array([g[0] for g in f["gt"]]), gt_xyz=np.array([g[1] for g in f["gt"]]).reshape(-1, 3))
    for k, (tp, npred, ngt) in tot.items():
        if ngt:
            print(f"total {k}: {tp}/{ngt} GT matched, {npred} predictions >= 0.3")


if __name__ == "__main__":
    main()
