#!/usr/bin/env python3
"""compare.py <frames root> <frame index>...: the phone chain's outputs (map_run's *.out.f32) vs the fp32
torch model (split.py host's ref_*.f32): bev / cls / pts cosines and polyline matches.

A phone polyline matches an fp32 one of the same class when their symmetric Chamfer distance (mean
nearest-point distance both ways, metres) is < 1.0 m (MapTR's middle AP threshold), greedy by score;
only predictions with score >= 0.4 on either side count. No map GT: nuScenes-mini's map expansion
needs an account."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M  # noqa: E402
import torch  # noqa: E402

THR, CD = 0.4, 1.0


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def polylines(cls, pts):
    p, s, lab = M.decode(torch.from_numpy(cls), torch.from_numpy(pts))
    k = s >= THR
    return p[k].numpy(), s[k].numpy(), lab[k].numpy()


def chamfer(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def match(ref, got):
    (rp, _, rl), (gp, gs, gl) = ref, got
    used, n = set(), 0
    for i in np.argsort(-gs):
        best, bj = CD, -1
        for j in range(len(rp)):
            if j in used or rl[j] != gl[i]:
                continue
            c = chamfer(gp[i], rp[j])
            if c < best:
                best, bj = c, j
        if bj >= 0:
            used.add(bj)
            n += 1
    return n


def main():
    root = Path(sys.argv[1])
    tm = tr = tg = 0
    for i in sys.argv[2:]:
        d = root / i
        ld = lambda n, shape: np.fromfile(d / n, np.float32).reshape(shape)  # noqa: E731
        bev, rbev = ld("bev.out.f32", (M.NQ, M.EMBED)), ld("ref_bev.f32", (M.NQ, M.EMBED))
        cls, rcls = ld("cls.out.f32", (M.NUM_VEC, M.NUM_CLASSES)), ld("ref_cls.f32", (M.NUM_VEC, M.NUM_CLASSES))
        pts, rpts = ld("pts.out.f32", (M.NUM_VEC, M.NUM_PTS, 2)), ld("ref_pts.f32", (M.NUM_VEC, M.NUM_PTS, 2))
        r, g = polylines(rcls, rpts), polylines(cls, pts)
        m = match(r, g)
        tm, tr, tg = tm + m, tr + len(r[0]), tg + len(g[0])
        perr = np.abs(pts - rpts).max() * 60
        print(f"frame {i}: bev cos {cos(bev, rbev):.6f}  cls cos {cos(cls, rcls):.6f}  pts max err {perr:.2f} m;  "
              f"polylines fp32 {len(r[0])} phone {len(g[0])} matched {m}")
    print(f"total: matched {tm} of {tr} fp32 polylines (phone has {tg}), score >= {THR}, Chamfer < {CD} m")


if __name__ == "__main__":
    main()
