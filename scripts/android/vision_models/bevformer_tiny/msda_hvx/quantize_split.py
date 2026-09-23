#!/usr/bin/env python3
"""int8 (QDQ) versions of the split encoder's HTP pieces (split.py export's pre / mid<L> / post<L>).

With the sampling on the HVX, the pieces are Linears + LayerNorm/Softmax + elementwise: no GridSample
grid, which is what cost the earlier int8 encoder its accuracy (../README.md "Encoder: exact rewrite
first, then mixed precision"). Each piece's graph I/O stays fp32 (frame_run / enc_run unchanged); the
QDQ units are inside.

  quantize_split.py quant --ckpt <pth> --work <work> --policy <policy> [--method minmax]
      run the split encoder in torch over quantize.py calib's frames (other scenes than scene-0103),
      record every piece's inputs, and quantize each piece with onnxsim.full_qdq
      -> <work>/msda_split_<policy>[.<method>]/<piece>.sim.onnx (+ feats_q.txt)
  quantize_split.py eval --ckpt <pth> --work <work> --pieces msda_split_<policy>
      the split encoder with those pieces on ORT CPU (graph optimizations at the basic level: no
      fused int8 kernels) + msda_fused, vs the fp32 Encoder: bev cos per frame, on the saved
      scene-0103 frames and on the calibration frames
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import split as S  # noqa: E402

PIECES = ["pre", "mid0", "post0", "mid1", "post1", "mid2", "post2"]
OFFSETS = [
    "tsa_off",
    "sca_off",
]  # sampling offsets in pixels: the kernel's sampling coordinates

# (activation dtype, quantize_full_qdq kwargs, offsets16): LayerNorm / Softmax stay fp16 in all of them
POLICIES = {
    "all8": ("uint8", {"exclude_op_types": ["LayerNormalization", "Softmax"]}, False),
    # the same, the sampling-offset outputs as uint16 (the grid-precision lesson)
    "all8o16": ("uint8", {"exclude_op_types": ["LayerNormalization", "Softmax"]}, True),
    "lin8": ("uint8", {"op_types": ["Gemm", "MatMul"]}, False),
    "all16": ("uint16", {"exclude_op_types": ["LayerNormalization", "Softmax"]}, False),
}


def piece_feeds(enc, enc_in):
    """{piece: {input name: array}} of one frame, as the phone feeds them (fp32)."""
    feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask = enc_in
    tsa_ref, ref_cam_h, vis = S.host_inputs(enc_in)
    n = len(enc.layers)
    f32 = lambda t: np.ascontiguousarray(t.detach().numpy().astype(np.float32))  # noqa: E731
    feeds = {
        "pre": {
            "feats": f32(feats),
            "prev_bev": f32(prev_bev),
            "has_prev": f32(has_prev),
            "can_bus": f32(can_bus),
        }
    }
    sca_v, q0, v, off, w = S.Pre(enc)(feats, prev_bev, has_prev, can_bus)
    q = q0
    for i in range(n):
        tp, sp = enc.layers[i].tsa.p, enc.layers[i].sca.p
        tsa_out = S.msda_fused(
            v,
            (M.BEV_H, M.BEV_W),
            tsa_ref,
            off.reshape(M.NQ, M.HEADS, 2, tp, 2),
            w.reshape(M.NQ, M.HEADS, 2, tp),
        )
        feeds[f"mid{i}"] = {"tsa_out": f32(tsa_out), "q": f32(q)}
        q1, soff, sw = S.Mid(enc.layers[i])(tsa_out, q)
        sca_out = S.msda_fused(
            sca_v[i],
            (M.FH, M.FW),
            ref_cam_h,
            soff.reshape(M.NQ, M.HEADS, 1, sp, 2),
            sw.reshape(M.NQ, M.HEADS, 1, sp),
            vis,
        )
        if i == n - 1:
            feeds[f"post{i}"] = {"sca_out": f32(sca_out), "q1": f32(q1)}
        else:
            feeds[f"post{i}"] = {
                "sca_out": f32(sca_out),
                "q1": f32(q1),
                "q0": f32(q0),
                "prev_bev": f32(prev_bev),
                "has_prev": f32(has_prev),
            }
            q, v, off, w = S.Post(enc, i)(sca_out, q1, q0, prev_bev, has_prev)
    return feeds


def run_pieces(sessions, enc, enc_in):
    """The split encoder with ORT pieces + msda_fused (the phone chain on the host)."""
    feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask = enc_in
    tsa_ref, ref_cam_h, vis = S.host_inputs(enc_in)
    f32 = lambda t: np.ascontiguousarray(np.asarray(t, np.float32))  # noqa: E731
    t = torch.from_numpy
    run = lambda p, feed: sessions[p].run(None, feed)  # noqa: E731
    sca_v, q0, v, off, w = run(
        "pre",
        {
            "feats": f32(feats),
            "prev_bev": f32(prev_bev),
            "has_prev": f32(has_prev),
            "can_bus": f32(can_bus),
        },
    )
    q = q0
    n = len(enc.layers)
    for i in range(n):
        tp, sp = enc.layers[i].tsa.p, enc.layers[i].sca.p
        tsa_out = S.msda_fused(
            t(v),
            (M.BEV_H, M.BEV_W),
            tsa_ref,
            t(off).reshape(M.NQ, M.HEADS, 2, tp, 2),
            t(w).reshape(M.NQ, M.HEADS, 2, tp),
        )
        q1, soff, sw = run(f"mid{i}", {"tsa_out": f32(tsa_out), "q": q})
        sca_out = S.msda_fused(
            t(sca_v[i]),
            (M.FH, M.FW),
            ref_cam_h,
            t(soff).reshape(M.NQ, M.HEADS, 1, sp, 2),
            t(sw).reshape(M.NQ, M.HEADS, 1, sp),
            vis,
        )
        if i == n - 1:
            return run(f"post{i}", {"sca_out": f32(sca_out), "q1": q1})[0]
        q, v, off, w = run(
            f"post{i}",
            {
                "sca_out": f32(sca_out),
                "q1": q1,
                "q0": q0,
                "prev_bev": f32(prev_bev),
                "has_prev": f32(has_prev),
            },
        )


def cmd_quant(a):
    import onnx
    import quantize as Q  # ../quantize.py: this checkout's onnxsim

    F = Q.load_onnxsim()
    _, enc, _ = M.load_official(a.ckpt)
    work = Path(a.work)
    feeds = [piece_feeds(enc, x) for x in S.calib_frames(work)]
    dtype, kw, off16 = POLICIES[a.policy]
    out = work / (
        f"msda_split_{a.policy}" + ("" if a.method == "minmax" else "." + a.method)
    )
    out.mkdir(exist_ok=True)
    for p in PIECES:
        m = onnx.load(str(work / "msda_split" / f"{p}.sim.onnx"))
        kw_ = dict(kw)
        outs = {o.name for o in m.graph.output}
        if off16:
            kw_["tensor_dtypes"] = {o: "uint16" for o in OFFSETS if o in outs}
        q = F.quantize_full_qdq(
            m, [f[p] for f in feeds], activation_dtype=dtype, method=a.method, **kw_
        )
        onnx.save(q, str(out / f"{p}.sim.onnx"))
        ops = {}
        for nd in q.graph.node:
            ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
        print(
            f"{out.name}/{p}: {len(q.graph.node)} nodes; Q {ops.get('QuantizeLinear', 0)} DQ {ops.get('DequantizeLinear', 0)}"
        )
    fq = work / "msda_split" / "feats_q.txt"
    if fq.exists():
        shutil.copy(fq, out / "feats_q.txt")


def cmd_eval(a):
    import onnxruntime as ort

    _, enc, _ = M.load_official(a.ckpt)
    work = Path(a.work)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sessions = {
        p: ort.InferenceSession(
            str(work / a.pieces / f"{p}.sim.onnx"),
            so,
            providers=["CPUExecutionProvider"],
        )
        for p in PIECES
    }
    sets = {
        "scene-0103": [f["enc_in"] for f in S.frames(work)],
        "calib": list(S.calib_frames(work)),
    }
    for name, ins in sets.items():
        cs = []
        for enc_in in ins:
            ref = enc(*enc_in).double().numpy().ravel()
            got = run_pieces(sessions, enc, enc_in).astype(np.float64).ravel()
            cs.append(float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got))))
        print(
            f"{a.pieces} {name}: bev cos vs fp32 min {min(cs):.6f} mean {np.mean(cs):.6f} ({len(cs)} frames)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quant", "eval"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--policy", default="all8", choices=sorted(POLICIES))
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--pieces", default="msda_split_all8")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"quant": cmd_quant, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
