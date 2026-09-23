#!/usr/bin/env python3
"""int8 whole-graph QDQ for the Fast-BEV / Fast-BEV++ HTP pieces with onnxsim's quantizer.

usage:
  quantize.py calib m0|pp --ckpt <pth> --data <nuscenes-mini> --work <dir> [--frames 3]
      fp32 torch over the calibration scenes (disjoint from the eval scene-0103; the same four
      scenes ../bevformer_tiny calibrates on, scene-1077 is at night), saving each piece's real
      inputs to <work>/<fam>_calib/<i>.npz
  quantize.py enc m0|pp --work <dir>     <fam>_enc.u8.sim.onnx -> <fam>_enc.q8.onnx (+ .json io qparams)
  quantize.py bev m0 --work <dir>        m0_bev.sim.onnx       -> m0_bev.q8.onnx
  quantize.py viewbev pp --work <dir>    pp_viewbev.sim.onnx   -> pp_viewbev.q8.onnx

onnxsim.full_qdq.quantize_full_qdq: uint8 activations, per-channel int8 weights, int32 biases,
Relu folded, data-movement ops sharing their input's qparams; onnxsim.full_qdq.quantized_io then
makes the graph I/O integer. Pinned ranges make the pieces hand bytes to each other unchanged:
  * encoder input: the NHWC pixels (export.py --uint8), range [0, 255] -> scale 1, zero point 0: the
    camera's uint8 RGB goes in as is (normalization is in the graph);
  * the gathered BEV volume (M0) / feature table (PP) uses the encoder output's range, so the
    view transform copies uint8 bytes (a "no camera" voxel is the zero point).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CALIB_SCENES = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]


def calib(a):
    import data as D
    import geometry as G
    import model as M

    torch.set_grad_enabled(False)
    ns = D.NuScenesMini(a.data)
    out = Path(a.work) / f"{a.fam}_calib"
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    if a.fam == "m0":
        enc, _ = M.load_m0(a.ckpt)
        view = M.ViewM0()
        points = G.m0_points()
    else:
        enc, _ = M.load_pp(a.ckpt)
        view = M.ViewPP()
        vc = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]["img_view_transformer.voxel_coords"]
    for scene in CALIB_SCENES:
        for tok in ns.scene_samples(scene)[: a.frames]:
            if a.fam == "m0":
                f = ns.m0_frame(tok, keyframe_fallback=True)
                feats = enc(torch.from_numpy(f["img"]))
                luts = [G.m0_lut(f["lidar2img"][t], points) for t in range(4)]
                vol = view(*[feats[t * 6:(t + 1) * 6].reshape(-1, 64) for t in range(4)], *luts)
                np.savez(out / f"{k}.npz", img=f["img_u8"][0].astype(np.float32), vol=vol.numpy().astype(np.float16))
            else:
                f = ns.pp_frame(tok)
                feats, depth = enc(torch.from_numpy(f["img"]))
                idx, didx = G.pp_lut(vc, f["sensor2keyego"], f["intrin"], f["post"])
                np.savez(out / f"{k}.npz", img=f["img_u8"].astype(np.float32), feats=feats.reshape(-1, 64).numpy(),
                         depth=depth.reshape(-1).numpy(), idx=idx.numpy(), didx=didx.numpy())
            print(f"calib {k}: {scene} {tok[:8]}")
            k += 1


def samples(work, fam, names, cast=None):
    for p in sorted((Path(work) / f"{fam}_calib").glob("*.npz"), key=lambda p: int(p.stem)):
        z = np.load(p)
        yield {n: (z[s].astype(np.float32) if z[s].dtype == np.float16 else z[s]) for n, s in names.items()}


def run(a):
    import onnx

    import onnxsim.full_qdq as F

    work = Path(a.work)
    io_path = work / f"{a.fam}_io.json"
    io = json.loads(io_path.read_text()) if io_path.exists() else {}
    if a.what == "enc":
        m = onnx.load(str(work / f"{a.fam}_enc.u8.sim.onnx"))
        data = list(samples(work, a.fam, {"img": "img"}))
        q = F.quantize_full_qdq(m, data, method=a.method, ranges={"img": (0.0, 255.0)})
        q, info = F.quantized_io(q)
        stem = f"{a.fam}_enc.q8"
    elif a.what == "bev":
        assert a.fam == "m0"
        m = onnx.load(str(work / "m0_bev.sim.onnx"))
        f = io["m0_enc.q8"]["feats"]
        lo, hi = (0 - f["zero_point"]) * f["scale"], (255 - f["zero_point"]) * f["scale"]
        data = list(samples(work, "m0", {"vol": "vol"}))
        # box deltas share one conv output: uint16 keeps their resolution (small tensor)
        q = F.quantize_full_qdq(m, data, method=a.method, ranges={"vol": (lo, hi)}, tensor_dtypes={"reg": "uint16"})
        q, info = F.quantized_io(q, nhwc_inputs=())
        stem = "m0_bev.q8"
    else:
        assert a.fam == "pp"
        m = onnx.load(str(work / "pp_viewbev.sim.onnx"))
        e = io["pp_enc.q8"]
        ranges = {n: ((0 - e[n]["zero_point"]) * e[n]["scale"], (255 - e[n]["zero_point"]) * e[n]["scale"])
                  for n in ("feats", "depth")}
        data = list(samples(work, "pp", {k: k for k in ("feats", "depth", "idx", "didx")}))
        # regression maps (sub-cell offsets, sin/cos, log dims) in uint16; the heatmap in uint8
        u16 = {n: "uint16" for n in ("reg", "height", "dim", "rot", "vel")}
        q = F.quantize_full_qdq(m, data, method=a.method, ranges=ranges, tensor_dtypes=u16)
        q, info = F.quantized_io(q)
        stem = "pp_viewbev.q8"
    onnx.save(q, str(work / f"{stem}.onnx"))
    io[stem] = info
    io_path.write_text(json.dumps(io, indent=1))
    ops = {}
    for n in q.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"{stem}: {len(q.graph.node)} nodes, Q {ops.get('QuantizeLinear', 0)} DQ {ops.get('DequantizeLinear', 0)}; io {info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calib", "enc", "bev", "viewbev"])
    ap.add_argument("fam", choices=["m0", "pp"])
    ap.add_argument("--ckpt")
    ap.add_argument("--data")
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--method", default="minmax")
    a = ap.parse_args()
    calib(a) if a.what == "calib" else run(a)


if __name__ == "__main__":
    main()
