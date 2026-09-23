#!/usr/bin/env python3
"""int8 (QDQ) BEVFormer-tiny pieces for the HTP, with onnxsim's whole-graph QDQ quantizer.

  quantize.py calib    --ckpt <pth> --data <nuscenes-mini> --work <work>
      fp32 torch over the calibration scenes (disjoint from e2e_phone.py's scene-0103), chaining
      prev_bev like the real model; saves every piece's inputs to <work>/calib/<i>.npz.
  quantize.py backbone --work <work> [--method minmax]
      calibrate backbone1 (one camera per batch, 6 x frames batches) and apply the ranges to
      backbone6 and backbone1 (same tensor names): <work>/backbone{6,1}.q8.onnx, uint8 NHWC image
      in, uint8 feats out; qparams in <work>/backbone{6,1}.q8.json, ranges in backbone.ranges.json.
  quantize.py <enc3|decoder> --work <work> --policy <policy> [--method minmax]
      quantize with a mixed-precision policy (POLICIES below): <work>/<piece>.<policy>.onnx.

Every step is plain onnxsim.full_qdq.quantize_full_qdq on the float export.py graph; the
rewrites the Mask R-CNN backbone needed by hand (int8 residual Adds, Relu folded into Q, uint8
graph I/O, NHWC image input) are what that function and quantized_io produce directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CALIB_SCENES = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]  # 1077: night
ENC_NAMES = ["feats", "prev_bev", "has_prev", "shift", "can_bus", "ref_cam", "bev_mask"]

# Mixed-precision policies for the attention pieces: (activation dtype, quantize_full_qdq kwargs).
# LayerNorm/Softmax and the deformable sampling (GridSample + its coordinates) are the usual
# quantization-sensitive parts; the policies peel them off one at a time.
FLOAT_SENSITIVE = ["LayerNormalization", "Softmax", "GridSample"]
POLICIES = {
    # only the Linear layers (Gemm/MatMul) in int8, everything else fp16
    "lin8": ("uint8", {"op_types": ["Gemm", "MatMul"]}),
    # int8 everywhere except LayerNorm, Softmax, GridSample
    "all8": ("uint8", {"exclude_op_types": FLOAT_SENSITIVE}),
    # W8A16 Linear layers
    "lin16": ("uint16", {"op_types": ["Gemm", "MatMul"]}),
    # W8A16 everywhere except LayerNorm, Softmax, GridSample
    "all16": ("uint16", {"exclude_op_types": FLOAT_SENSITIVE}),
    # the same two, with GridSample quantized too
    "all8gs": ("uint8", {"exclude_op_types": ["LayerNormalization", "Softmax"]}),
    "all16gs": ("uint16", {"exclude_op_types": ["LayerNormalization", "Softmax"]}),
    # uint8 everywhere (GridSample too) except the sampling coordinates: uint16
    # (onnxsim.full_qdq.sampling_coordinate_tensors)
    "mix8": ("uint8", {"coords16": True}),
    # the same, LayerNorm and Softmax in fp16
    "mix8f": ("uint8", {"coords16": True, "exclude_op_types": ["LayerNormalization", "Softmax"]}),
}


def load_onnxsim():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[4]))  # repo root: this checkout's onnxsim
    from onnxsim import full_qdq

    return full_qdq


def calib(a):
    import torch

    import model as M
    from nuscenes import NuScenesMini, temporal_can_bus

    torch.set_grad_enabled(False)
    bb, enc, dec = M.load_official(a.ckpt)
    ns = NuScenesMini(a.data)
    out = Path(a.work) / "calib"
    out.mkdir(exist_ok=True)
    k = 0
    for scene in CALIB_SCENES:
        prev, prev_abs = None, None
        for tok in ns.scene_samples(scene)[: a.frames]:
            f = ns.frame(tok)
            can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
            ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
            first = prev is None
            has_prev = torch.zeros(1) if first else torch.ones(1)
            shift = torch.zeros(2) if first else M.can_bus_shift(can_bus)
            feats = bb(f["img"])
            p_in = torch.zeros(M.NQ, M.EMBED) if first else M.rotate_prev_bev(prev, can_bus)
            bev = enc(feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)
            x = dict(zip(ENC_NAMES, (feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)))
            np.savez(out / f"{k}.npz", img=f["img"].numpy(), bev_embed=bev.numpy(),
                     **{n: v.numpy().astype(np.float32) for n, v in x.items()})
            print(f"calib {k}: {scene} {tok[:8]} has_prev {int(has_prev)}")
            prev, prev_abs, k = bev, f["can_bus_abs"], k + 1


def samples(work: Path, names):
    for p in sorted((work / "calib").glob("*.npz"), key=lambda p: int(p.stem)):
        z = np.load(p)
        yield {n: z[n] for n in names}


def backbone(a, F):
    import onnx
    from onnxsim.calibration import calibrate

    work = Path(a.work)
    one = onnx.load(str(work / "backbone1.sim.onnx"))
    data = [{"img": s["img"][c : c + 1]} for s in samples(work, ["img"]) for c in range(6)]
    names = [o for n in one.graph.node for o in n.output] + ["img"]
    ranges = calibrate(one, data, method=a.method, extra_tensor_names=names)
    del one, data
    sfx = "" if a.method == "minmax" else "." + a.method
    (work / f"backbone.ranges{sfx}.json").write_text(json.dumps({k: list(v) for k, v in ranges.items()}))
    # the same ranges at batch 6 (one execute for all cameras) and batch 1 (one per camera, so a
    # pipelined runner can interleave the backbone with other HTP work at a finer grain)
    for b in (6, 1):
        q = F.quantize_full_qdq(onnx.load(str(work / f"backbone{b}.sim.onnx")), ranges=ranges)
        q, info = F.quantized_io(q, nhwc_inputs=["img"])
        stem = f"backbone{b}.q8{sfx}"
        onnx.save(q, str(work / f"{stem}.onnx"))
        (work / f"{stem}.json").write_text(json.dumps(info, indent=1))
        print(f"{stem}: {len(q.graph.node)} nodes, io {info}")


def attention_piece(a, F):
    import onnx

    work = Path(a.work)
    names = ENC_NAMES if a.piece.startswith("enc") else ["bev_embed"]
    m = onnx.load(str(work / f"{a.piece}.sim.onnx"))
    dtype, kw = POLICIES[a.policy]
    kw = dict(kw)
    if kw.pop("coords16", False):
        kw["tensor_dtypes"] = {t: "uint16" for t in F.sampling_coordinate_tensors(m)}
    if a.exclude:
        kw["exclude_nodes"] = a.exclude.split(",")
    q = F.quantize_full_qdq(m, list(samples(work, names)), activation_dtype=dtype, method=a.method, **kw)
    stem = f"{a.piece}.{a.policy}{'' if a.method == 'minmax' else '.' + a.method}{a.tag}"
    onnx.save(q, str(work / f"{stem}.onnx"))
    ops = {}
    for n in q.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"{stem}: {len(q.graph.node)} nodes; Q {ops.get('QuantizeLinear', 0)} DQ {ops.get('DequantizeLinear', 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=["calib", "backbone", "enc1", "enc3", "decoder"])
    ap.add_argument("--ckpt")
    ap.add_argument("--data")
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--policy", default="lin8", choices=sorted(POLICIES))
    ap.add_argument("--exclude", default="", help="comma-separated node/output names kept in float")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    a = ap.parse_args()
    if a.piece == "calib":
        return calib(a)
    F = load_onnxsim()
    return backbone(a, F) if a.piece == "backbone" else attention_piece(a, F)


if __name__ == "__main__":
    sys.exit(main())
