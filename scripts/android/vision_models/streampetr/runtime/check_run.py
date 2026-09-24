"""Checks petr_run's C++ host against model.py and scores its detections:

  python runtime/check_run.py --ckpt <pth> --work <work> --prep <prep dir> --run <petr_run out dir> [--scene ...]

Replays HostState (model.py) frame by frame on the *phone's* head outputs (DUMP=1's f<i>_{cls,reg,dec}),
so any difference is the C++ port, not the HTP: compares every f<i>_mem_*.bin and f<i>_{pe_in,cone}.bin
with what Python computes from the same state -- the memory rows as a set (C++ ranks the top-128 by the
raw logit, model.py by sigmoid(logit), whose fp32 rounding ties close scores and lets topk order them
differently; the head is permutation-equivariant in its memory rows and the queue drops whole 128-row
chunks, so only the order differs) -- then matches f<i>_{cls,box}.bin against GT like
e2e_phone.py (score >= 0.3 / 0.2, same class, 2 m)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
import model as M  # noqa: E402

SHAPES = {"mem_emb": (512, 256), "mem_pe3d": (512, 384), "mem_time": (512, 256), "mem_motion": (512, 180)}


def as_set(x):
    """rows sorted lexicographically (order-free comparison)"""
    return x[np.lexsort(x.T[::-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", default="scene-0103")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    _, head = M.load_official(a.ckpt)
    host = M.HostState(head)
    run, prep = Path(a.run), Path(a.prep)
    ld = lambda p, s: np.fromfile(p, np.float32).reshape(s)  # noqa: E731
    frames = sorted((Path(a.work) / "frames" / a.scene).glob("*.npz"), key=lambda p: int(p.stem))
    tot = {0.3: 0, 0.2: 0}
    worst = {}
    for i, fp in enumerate(frames):
        z = np.load(fp)
        t = M.to_torch({k: z[k] for k in ("lidar2img", "intrinsics", "ego_pose", "ego_pose_inv")} | {"timestamp": float(z["timestamp"])})
        mem = host.pre(t, bool(z["prev"]))
        got = {k: ld(run / f"f{i}_{k}.bin", sh) for k, sh in SHAPES.items()}
        # the permutation that takes model.py's queue order to C++'s, matched on every memory feature
        feats = lambda d: np.round(np.concatenate([np.asarray(d[k]) for k in SHAPES], 1), 4)  # noqa: E731
        ig, ir = np.lexsort(feats(got).T[::-1]), np.lexsort(feats({k: v.numpy() for k, v in mem.items()}).T[::-1])
        perm = np.empty(len(ig), np.int64)
        perm[ig] = ir  # C++ row r <-> model.py row perm[r]
        for k in SHAPES:
            worst[k] = max(worst.get(k, 0), float(np.abs(got[k] - mem[k].numpy()[perm]).max()))
        # continue model.py in C++'s order, so the propagated queries' boxes line up row for row
        for name in ("emb", "ref", "ts", "pose", "velo"):
            setattr(host, name, getattr(host, name)[torch.from_numpy(perm)])
        host.refn_prop = ((host.ref - host.pc[:3]) / (host.pc[3:6] - host.pc[0:3]))[: M.NUM_PROP]
        for k, s in (("pe_in", (4224, 192)), ("cone", (4224, 8))):
            e = np.abs(ld(run / f"f{i}_{k}.bin", s) - ld(prep / f"f{i}" / f"ref_{k}.bin", s)).max()
            worst[k] = max(worst.get(k, 0), float(e))
        cls, reg, dec = (torch.from_numpy(ld(run / f"f{i}_{k}.bin", s)) for k, s in
                         (("cls", (428, 10)), ("reg", (428, 10)), ("dec", (428, 256))))
        _, box = host.post(t, cls, reg, dec)
        e = np.abs(ld(run / f"f{i}_box.bin", (428, 10)) - box.numpy()).max()  # per query: same order
        worst["box"] = max(worst.get("box", 0), float(e))
        gt = list(zip(z["gt_names"].tolist(), z["gt_xyz"]))
        line = f"frame {i}:"
        for thr in tot:
            m = D.match(*M.decode(torch.from_numpy(ld(run / f"f{i}_cls.bin", (428, 10))),
                                  torch.from_numpy(ld(run / f"f{i}_box.bin", (428, 10)))), gt, thr=thr)[0]
            tot[thr] += m
            line += f" @{thr} {m}"
        print(line)
    print("C++ host vs model.py (max abs over frames): " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    for thr, m in tot.items():
        print(f"total @{thr}: {m}/190")


if __name__ == "__main__":
    main()
