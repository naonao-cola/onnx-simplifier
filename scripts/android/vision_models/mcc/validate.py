"""Check the deployment split against upstream MCC and write the fp32 reference.

  python validate.py --ckpt co3dv2_all_categories.pth --work <dir> [--gran 0.1 0.05]

1. upstream ``MCC.forward`` (full (197+Q)^2 masked decoder) vs ``Encoder`` + ``QueryDecoder``
   on the first queries of the demo grid: max |occ logit| and |rgb| difference.
2. The split over the whole grid at each granularity -> <work>/ref_<g>.npz
   (occ logits, rgb) -- the reference every phone/quantized run is scored against.
"""

import argparse
import os
import time

import model as M
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--demo", default="quest2")
    ap.add_argument("--gran", type=float, nargs="+", default=[0.1])
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    torch.set_grad_enabled(False)
    repo = os.environ.get("MCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/MCC"))
    m = M.load_mcc(a.ckpt, repo)
    img, xyz = M.load_demo(repo, a.demo)
    win, val = M.xyz_windows(xyz)
    np.savez(
        os.path.join(a.work, f"inputs_{a.demo}.npz"),
        img=img.numpy(),
        xyz_win=win.numpy(),
        valid=val.numpy(),
    )

    enc, dec = M.Encoder(m).eval(), M.QueryDecoder(m).eval()
    k, v = enc(img, win, val)

    # 1. vs upstream forward (valid_seen_xyz / -100 / shrink handled inside upstream)
    q = M.grid(0.1)[:, :2000]
    seen = xyz.clone()[None]
    valid = torch.isfinite(seen.sum(-1))
    seen[~valid] = -100.0
    m.args.regress_color = False
    _, pred = m(
        img * torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        + torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        seen,
        q,
        torch.zeros_like(q),
        torch.zeros(q.shape[:2]),
        valid,
    )
    up_occ = pred[..., 0]
    up_rgb = (
        torch.softmax(pred[..., 1:].reshape(1, -1, 3, 256) / M.TEMPERATURE, -1)
        * torch.linspace(0, 1, 256)
    ).sum(-1)
    occ, rgb = dec(q, k, v)
    print(
        f"split vs upstream on 2000 queries: occ max abs {float((occ - up_occ).abs().max()):.3g}, "
        f"rgb max abs {float((rgb - up_rgb).abs().max()):.3g}"
    )

    # 2. full-grid reference
    for g in a.gran:
        pts = M.grid(g)
        t = time.time()
        occs, rgbs = [], []
        for s in range(0, pts.shape[1], a.chunk):
            o, r = dec(pts[:, s : s + a.chunk], k, v)
            occs.append(o[0])
            rgbs.append(r[0])
        occ, rgb = torch.cat(occs).numpy(), torch.cat(rgbs).numpy()
        p = torch.sigmoid(torch.from_numpy(occ))
        print(
            f"g={g}: {pts.shape[1]} queries in {time.time() - t:.1f} s host fp32; "
            + ", ".join(f">{th}: {int((p > th).sum())}" for th in (0.1, 0.3, 0.5))
        )
        np.savez(
            os.path.join(a.work, f"ref_{a.demo}_{g}.npz"),
            xyz=pts[0].numpy(),
            occ=occ,
            rgb=rgb,
        )


if __name__ == "__main__":
    main()
