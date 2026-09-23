#!/usr/bin/env python3
"""Put Fast-BEV M0's view transform (4 uint8 Gathers) in front of the int8 BEV net: one HTP graph.

usage: compose_m0.py --work <dir> [--chunks K]
  in:  f0..f3 uint8 (67585, 64)  -- each time step's encoder output (6*64*176 rows) + one row of the
                                    feature zero point ("no camera"; the host keeps it at the end of
                                    each ring-buffer slot, so nothing is copied)
       i0..i3 int32 (160000,)    -- the host LUTs (geometry.m0_lut)
  out: cls, reg, dir of m0_bev.q8.onnx
The gathers are DequantizeLinear -> Gather -> QuantizeLinear with the encoder's output qparams (the
QDQ form QNN turns into a uint8 Gather); K > 1 splits each gather's indices into K chunks.
-> <work>/m0_viewbev.q8[.cK].onnx, and its phone inputs (frame 1) in <work>/m0_viewbev.q8[.cK].in
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import compose, parser

N_ROWS, N_VOX = 6 * 64 * 176 + 1, 200 * 200 * 4


def gather_graph(scale, zp, chunks):
    ins = ", ".join([f"uint8[{N_ROWS},64] f{t}" for t in range(4)] + [f"int32[{N_VOX}] i{t}" for t in range(4)])
    body, n = [], N_VOX // chunks
    for t in range(4):
        body.append(f"d{t} = DequantizeLinear(f{t}, s, z)")
        parts = []
        for k in range(chunks):
            if chunks == 1:
                idx = f"i{t}"
            else:
                body.append(f"i{t}_{k} = Slice(i{t}, st{k}, en{k}, ax0)")
                idx = f"i{t}_{k}"
            body.append(f"g{t}_{k} = Gather<axis=0>(d{t}, {idx})")
            body.append(f"q{t}_{k} = QuantizeLinear(g{t}_{k}, s, z)")
            body.append(f"u{t}_{k} = DequantizeLinear(q{t}_{k}, s, z)")
            parts.append(f"u{t}_{k}")
        body.append(f"c{t} = Concat<axis=0>({', '.join(parts)})" if chunks > 1 else f"c{t} = Identity({parts[0]})")
        body.append(f"r{t} = Reshape(c{t}, shp3)")
    body.append("cat = Concat<axis=2>(r0, r1, r2, r3)")
    body.append("v = Reshape(cat, shp4)")
    body.append("vol = QuantizeLinear(v, s, z)")
    consts = [f"float s = {{{scale!r}}}", f"uint8 z = {{{zp}}}", "int64[3] shp3 = {40000, 4, 64}",
              "int64[4] shp4 = {1, 200, 200, 1024}", "int64[1] ax0 = {0}"]
    for k in range(chunks):
        consts += [f"int64[1] st{k} = {{{k * n}}}", f"int64[1] en{k} = {{{(k + 1) * n}}}"]
    text = f"""<ir_version: 8, opset_import: ["": 17]>
view ({ins}) => (uint8[1,200,200,1024] vol) <{", ".join(consts)}> {{ {chr(10).join(body)} }}"""
    return parser.parse_model(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--chunks", type=int, default=1)
    a = ap.parse_args()
    work = Path(a.work)
    io = json.loads((work / "m0_io.json").read_text())
    q = io["m0_enc.q8"]["feats"]
    assert io["m0_bev.q8"]["vol"]["scale"] == q["scale"] and io["m0_bev.q8"]["vol"]["zero_point"] == q["zero_point"]
    view = gather_graph(np.float32(q["scale"]).item(), q["zero_point"], a.chunks)
    bev = onnx.load(str(work / "m0_bev.q8.onnx"))
    view.ir_version = bev.ir_version
    del view.opset_import[:]
    view.opset_import.extend(bev.opset_import)
    m = compose.merge_models(view, bev, io_map=[("vol", "vol")])
    onnx.checker.check_model(m)
    stem = "m0_viewbev.q8" + (f".c{a.chunks}" if a.chunks > 1 else "")
    onnx.save(m, str(work / f"{stem}.onnx"))
    io[stem] = {k: v for k, v in io["m0_bev.q8"].items() if k != "vol"}
    (work / "m0_io.json").write_text(json.dumps(io, indent=1))
    # phone inputs: frame 1, the host int8 encoder's tables (eval_q8.py) + LUTs; ref = host ORT int8
    from eval_q8 import sess
    from q8_inputs import write

    f = torch.load(work / "m0_frames" / "1.pt", weights_only=False)
    enc = sess(work / "m0_enc.q8.onnx")
    feeds = {}
    for t in range(4):
        tab = enc.run(None, {"img": f["img_u8"][t]})[0].reshape(-1, 64)
        feeds[f"f{t}"] = np.concatenate([tab, np.full((1, 64), q["zero_point"], np.uint8)])
    for t in range(4):
        feeds[f"i{t}"] = f["luts"][t]
    write(work, stem, feeds, sess(work / f"{stem}.onnx"))
    print(f"{stem}: {len(m.graph.node)} nodes")


if __name__ == "__main__":
    main()
