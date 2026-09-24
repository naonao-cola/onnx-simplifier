"""Massive-activation channels of the ViT residual stream per checkpoint: python outliers.py <model>..."""

import sys

import numpy as np
import torch
from common import load_batches, norm_params, split, torch_model

for name in sys.argv[1:]:
    m = torch_model(name)
    mean, std = norm_params(name)
    cal, _ = split("A")
    xs = [torch.from_numpy(b["input"]) for b in load_batches(cal[:32], mean, std)]
    amax = {}

    def hook(i):
        def f(_, __, out):
            v = (
                out.detach().abs().amax(dim=(0, 1)).numpy()
            )  # per channel over batch and tokens
            amax[i] = np.maximum(amax.get(i, v), v)

        return f

    hs = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(m.blocks)]
    with torch.no_grad():
        for x in xs:
            m(x)
    for h in hs:
        h.remove()
    rows = []
    for i in sorted(amax):
        v = amax[i]
        med = float(np.median(v))
        top = np.argsort(-v)[:3]
        rows.append(
            f"L{i}: "
            + " ".join(f"ch{c}={v[c] / med:.0f}x" for c in top if v[c] > 20 * med)
        )
    print(
        f"{name}: residual-stream channels with max|x| > 20x the median channel (per block output)"
    )
    print("  " + " | ".join(r for r in rows if ":" in r and "ch" in r) or "  none")
