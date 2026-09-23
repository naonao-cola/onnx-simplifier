"""Sparse4D v3 fp32 on nuScenes-mini, frames chained with the temporal instance bank.

  python validate.py --ckpt sparse4dv3_r50.pth --data <nuscenes-mini> [--work <dir>] [--frames 6]

Runs scene-0103's first N keyframes with DFA as upstream wrote it (rank 6), and at every one of
the 36 DFA calls also runs the rank-4 rewrite the HTP graphs use on the same inputs (a whole-run
comparison is meaningless past frame 0: the top-k instance cache reorders on 1e-5 differences).
Reports the max abs difference and the GT match (score >= 0.3, same class, BEV center within 2 m; same as bevformer_tiny). With --work it
saves each frame's inputs, the backbone features and the fp32 reference outputs for export.py and
the phone runs.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from data import NuScenesMini, match
from model import Runner, Sparse4D, dfa_rank4, dfa_upstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    m = Sparse4D().load_official(a.ckpt).eval()
    ns = NuScenesMini(a.data)
    toks = ns.scene_samples(a.scene)[: a.frames]
    diff = [0.0]

    def checked(fmaps, pts, w, layer):
        ref = dfa_upstream(fmaps, pts, w)
        diff[0] = max(diff[0], float((dfa_rank4(fmaps, pts, w) - ref).abs().max()))
        return ref

    run = Runner(m, checked)
    work = Path(a.work) if a.work else None
    if work:
        (work / "frames").mkdir(parents=True, exist_ok=True)
    thrs = (0.3, 0.2, 0.1)
    tot = {t: [0, 0, 0] for t in thrs}
    for k, tok in enumerate(toks):
        f = ns.frame(tok)
        (boxes, scores, labels), raw = run.frame(f["img"], f["metas"])
        res = {t: match(boxes, scores, labels, f["gt"], thr=t) for t in thrs}
        for t, r in res.items():
            tot[t] = [x + y for x, y in zip(tot[t], r)]
        tp, npred, ngt = res[0.3]
        print(f"frame {k} {tok[:8]}: GT {tp}/{ngt} (pred {npred} >= 0.3); >= 0.2: {res[0.2][0]}, >= 0.1: {res[0.1][0]}")
        if work:
            with open(work / "frames" / f"{k}.pkl", "wb") as fh:
                pickle.dump({"token": tok, "rgb": f["rgb"], "metas": f["metas"], "gt": f["gt"],
                             "ref": [t.numpy() for t in raw]}, fh)
    print(f"DFA rank4 vs upstream, all {6 * len(toks)} calls: max abs {diff[0]:.2e}")
    for t, (tp, npred, ngt) in tot.items():
        print(f"score >= {t}: GT matched {tp}/{ngt}, predictions {npred}")


if __name__ == "__main__":
    main()
