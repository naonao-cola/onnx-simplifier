#!/usr/bin/env python3
"""Inputs for the fbgather DSP kernel checks (dsp/build_dsp.sh): frame 1's 4 time steps through the
host int8 encoder -> t0..t3.bin ((rows + 1) * 64 uint8, last row = zero point), the LUTs
l0..l3.bin (int32) and the expected volume vol_ref.bin (eval_q8.m0_gather).
usage: dsp_inputs.py --work <dir> --out <dir>"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from eval_q8 import m0_gather, sess

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
zp = json.loads((work / "m0_io.json").read_text())["m0_enc.q8"]["feats"]["zero_point"]
f = torch.load(work / "m0_frames" / "1.pt", weights_only=False)
enc = sess(work / "m0_enc.q8.onnx")
tables = [enc.run(None, {"img": f["img_u8"][t]})[0].reshape(-1, 64) for t in range(4)]
for t in range(4):
    np.concatenate([tables[t], np.full((1, 64), zp, np.uint8)]).tofile(out / f"t{t}.bin")
    f["luts"][t].astype(np.int32).tofile(out / f"l{t}.bin")
m0_gather(tables, f["luts"], zp).tofile(out / "vol_ref.bin")
print(f"rows {tables[0].shape[0]} zp {zp} -> {out}")
