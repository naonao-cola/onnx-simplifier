"""Query-reduction strategies, scored exactly against the dense fp32 references.

  python queries.py --work <dir> [--demo quest2] [--thr 0.3] [--us-per-query 64]

Every query's prediction depends only on its own xyz (the seen K/V are fixed per image), and
the 0.2 / 0.1 / 0.05 grids nest (every coarse point is a fine point). So a strategy that
queries a subset of the dense grid gets exactly the dense predictions at those points: its
only loss is occupied points it never queries. Scored per strategy:
  queries  -- how many decoder evaluations it needs (x phone us/query -> projected time)
  recall   -- fraction of the dense grid's occupied points (p > thr) it recovers
  chamfer  -- symmetric mean nearest-neighbor distance, recovered vs dense occupied set
"""

import argparse
import os

import numpy as np


def load(work, demo, g):
    r = np.load(os.path.join(work, f"ref_{demo}_{g}.npz"))
    n = int(round(r["xyz"].shape[0] ** (1 / 3)))
    p = 1 / (1 + np.exp(-r["occ"].astype(np.float64)))
    return p.reshape(n, n, n), n


def dilate(mask, r):
    out = mask.copy()
    for ax in range(3):
        acc = out.copy()
        for s in range(1, r + 1):
            acc |= np.roll(out, s, ax) | np.roll(out, -s, ax)
        out = acc
    return out


def embed(coarse_q):
    """queried coarse points as a mask on the next finer grid."""
    n = coarse_q.shape[0] * 2
    m = np.zeros((n, n, n), bool)
    m[::2, ::2, ::2] = coarse_q
    return m


def refine(active_coarse, n_fine):
    """coarse active cells (grid index c) -> fine indices 2c-1 .. 2c+1 (the fine points
    between a coarse point and its neighbors)."""
    fine = np.zeros((n_fine,) * 3, bool)
    idx = np.argwhere(active_coarse)
    for d in (
        np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"))
        .reshape(3, -1)
        .T
    ):
        f = 2 * idx + d
        ok = ((f >= 0) & (f < n_fine)).all(1)
        fine[tuple(f[ok].T)] = True
    return fine


def chamfer(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    import torch

    ta, tb = torch.from_numpy(a).float(), torch.from_numpy(b).float()

    def nn(x, y):
        return torch.cat(
            [torch.cdist(x[i : i + 4096], y).min(1)[0] for i in range(0, len(x), 4096)]
        )

    return 0.5 * float(nn(ta, tb).mean() + nn(tb, ta).mean())


def coords(mask, n, world=3.0):
    return (np.argwhere(mask) - n / 2.0) / ((n / 2.0) / world)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--demo", default="quest2")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--us-per-query", type=float, default=64.0)
    ap.add_argument("--enc-ms", type=float, default=202.0)
    a = ap.parse_args()
    p05, n05 = load(a.work, a.demo, 0.05)
    p10, n10 = load(a.work, a.demo, 0.1)
    # same points, different chunking: logits agree to ~3e-4, occupancy decisions exactly
    assert ((p05[::2, ::2, ::2] > a.thr) == (p10 > a.thr)).all(), "grids do not nest"
    p20, n20 = p10[::2, ::2, ::2], n10 // 2

    rows = []

    def add(name, target_p, target_n, queried, count):
        occ = target_p > a.thr
        got = occ & queried
        rows.append(
            (
                name,
                count,
                got.sum() / max(occ.sum(), 1),
                chamfer(coords(got, target_n), coords(occ, target_n)),
                target_n,
            )
        )

    for tgt, (p, n) in {"0.1": (p10, n10), "0.05": (p05, n05)}.items():
        dense = np.ones_like(p, bool)
        add(f"dense {tgt}", p, n, dense, dense.size)
    # coarse-to-fine: evaluate the coarse grid, refine around cells whose p exceeds `lo`
    p40 = p20[::2, ::2, ::2]
    for start in ("0.2", "0.4"):
        for lo in (0.05, 0.1, 0.2, 0.3):
            dil = 0
            if start == "0.4":
                q40 = np.ones_like(p40, bool)
                q20 = embed(q40) | refine(dilate(p40 > lo, dil), n20)
            else:
                q20 = np.ones_like(p20, bool)
            act20 = q20 & (np.where(q20, p20, 0) > lo)
            q10 = embed(q20) | refine(dilate(act20, dil), n10)
            add(f"{start}->0.1 lo={lo}", p10, n10, q10, int(q10.sum()))
            act10 = q10 & dilate(np.where(q10, p10, 0) > lo, dil)
            q05 = embed(q10) | refine(act10, n05)
            add(f"{start}->0.05 lo={lo}", p05, n05, q05, int(q05.sum()))

    print(
        f"{'strategy':32s} {'queries':>10s} {'recall':>7s} {'chamfer':>8s} {'phone s (proj)':>14s}"
    )
    for name, q, rec, ch, n in rows:
        t = a.enc_ms / 1e3 + q * a.us_per_query / 1e6
        print(f"{name:32s} {q:10d} {rec:7.4f} {ch:8.4f} {t:14.2f}")


if __name__ == "__main__":
    main()
