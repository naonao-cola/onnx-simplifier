"""Teacher-forced head check on the phone: each scene-0103 frame's fp32-chain inputs -> <head> on the
HTP -> cos of cls / reg / dec vs the fp32 outputs (validate.py's dumps). Separates the HTP's own
error from what the memory chain compounds.   python tf_phone.py --work <work> --head head.sim.q16 ..."""
import argparse
from pathlib import Path

import e2e_phone as E
import numpy as np
import quantize as Qz

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--head", nargs="+", required=True)
a = ap.parse_args()
work = Path(a.work)
E.setup(work, a.head)
frames = sorted((work / "frames" / "scene-0103").glob("*.npz"), key=lambda p: int(p.stem))
for h in a.head:
    worst = [1.0, 1.0, 1.0]
    for fp in frames:
        z = np.load(fp)
        outs, _ = E.phone(work, h, {k: z[k] for k in Qz.HEAD_IN})
        worst = [min(w, E.cos(o, z[k])) for w, o, k in zip(worst, outs, ("cls", "reg", "dec"))]
    print(f"{h}: worst cos cls {worst[0]:.5f} reg {worst[1]:.5f} dec {worst[2]:.5f}", flush=True)
