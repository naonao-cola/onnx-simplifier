#!/usr/bin/env python3
"""Phone inputs for the int8 pieces: <work>/<stem>.in/{manifest.txt, *.bin, ref_*.bin} from frame 1,
with the host ORT CPU int8 outputs as the reference (compare_q8.py checks the phone's bytes).
usage: q8_inputs.py m0|pp --work <dir>"""
import argparse
from pathlib import Path

import numpy as np
import torch
from eval_q8 import m0_gather, sess

DT = {np.dtype("uint8"): "u8", np.dtype("uint16"): "u16", np.dtype("int32"): "i32", np.dtype("float32"): "f32"}


def write(work, stem, feeds, s):
    d = work / f"{stem}.in"
    d.mkdir(exist_ok=True)
    with open(d / "manifest.txt", "w") as man:
        for k, v in feeds.items():
            v = np.ascontiguousarray(v)
            v.tofile(d / f"{k}.bin")
            man.write(f"{k} {DT[v.dtype]} {(d / f'{k}.bin').resolve()} {','.join(map(str, v.shape))}\n")
    for o, v in zip(s.get_outputs(), s.run(None, feeds)):
        v.tofile(d / f"ref_{o.name}.bin")
    return s.run(None, feeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam", choices=["m0", "pp"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    f = torch.load(work / f"{a.fam}_frames" / "1.pt", weights_only=False)
    enc = sess(work / f"{a.fam}_enc.q8.onnx")
    if a.fam == "m0":
        write(work, "m0_enc.q8", {"img": f["img_u8"][0]}, enc)
        tables = [enc.run(None, {"img": f["img_u8"][t]})[0].reshape(-1, 64) for t in range(4)]
        import json
        zp = json.loads((work / "m0_io.json").read_text())["m0_enc.q8"]["feats"]["zero_point"]
        write(work, "m0_bev.q8", {"vol": m0_gather(tables, f["luts"], zp)}, sess(work / "m0_bev.q8.onnx"))
    else:
        feats, depth = write(work, "pp_enc.q8", {"img": f["img_u8"]}, enc)
        write(work, "pp_viewbev.q8", {"feats": feats.reshape(-1, 64), "depth": depth.reshape(-1), "idx": f["idx"],
                                      "didx": f["didx"]}, sess(work / "pp_viewbev.q8.onnx"))


if __name__ == "__main__":
    main()
