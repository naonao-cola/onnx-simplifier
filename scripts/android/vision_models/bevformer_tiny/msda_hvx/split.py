#!/usr/bin/env python3
"""BEVFormer-tiny's encoder split around the HVX multi-scale deformable attention kernel
(../../../msda_hvx/, generic; this file is BEVFormer's glue).

The HTP runs every Linear/softmax/LayerNorm/FFN; the DSP runs, per TSA and per SCA, the whole
"sampling grid -> GridSample -> x attention weight -> sum over points (-> average over cameras)"
span as one call, `msda_fused` below. That span is 73% of an fp16 encoder layer on the HTP
(GridSample 56%, Mul + ReduceSum 17%; ../README.md "Encoder: exact rewrite first").

msda_fused(value, hw, ref, off, attw, vis) -> (Q, 256), the kernel's contract in BEVFormer's terms
(one level, mode "pix" of ../../../msda_hvx/msda_ref.py's msda_reference):
  value (NV, H*W, 256)      channels-last value maps (NV = 6 cameras for SCA, 2 queue frames for TSA)
  ref   (NV, Q, R, 2)       reference points in [0, 1] (x, y); point p uses ref[..., p % R, :]
  off   (Q, M, NO, P, 2)    raw sampling_offsets Linear output, in pixels of the value map; NO is
                            1 (SCA: shared by every camera) or NV (TSA: one set per queue frame)
  attw  (Q, M, NO, P)       softmaxed attention weights, same NO
  vis   (NV, Q) or None     1 where (value map, query) takes part (SCA: the camera sees any pillar
                            point); None = all
  out[q, 32h:32h+32] = sum_{nv visible} sum_p attw * bilinear(value[nv, :, 32h:32h+32],
                                                    ref * (W, H) + off - 0.5) / max(1, #visible)
  bilinear = grid_sample(mode=bilinear, padding_mode=zeros, align_corners=False).
For TSA (vis None, NV = 2) that is the mean over the 2-frame queue; for SCA the visibility average.

Pieces (fixed shapes, fp32 I/O, fp16 inside on the HTP) of the 3-layer encoder, with L = layer:
  pre       feats, prev_bev, has_prev, can_bus -> sca_v (3, 6, 375, 256), q0, tsa_v, tsa_off, tsa_w
  mid<L>    tsa_out, q                         -> q1, sca_off, sca_w
  post<L>   sca_out, q1, q0, prev_bev, has_prev -> q, tsa_v, tsa_off, tsa_w   (L = 0, 1)
  post2     sca_out, q1                         -> bev_embed
host per frame: tsa_ref (2, Q, 1, 2) = [ref_2d + shift * has_prev, ref_2d]; ref_cam; vis = any(bev_mask).

usage: split.py check  --ckpt <pth> --work <work>     torch split vs Encoder on the saved frames
       split.py dump   --ckpt <pth> --work <work>     kernel I/O of layer 0 of frame 1 -> <work>/msda_io
       split.py export --ckpt <pth> --work <work>     the pieces -> <work>/msda_split/*.onnx (+ onnxsim)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "msda_hvx"))
import model as M  # noqa: E402
import msda_ref  # noqa: E402

E, HD, NQ = M.EMBED, M.HEADS, M.NQ
D = E // HD


def _generic(value, hw, ref, off, attw):
    """BEVFormer's (off (Q, M, NO, P, 2), attw (Q, M, NO, P), ref (NV, Q, R, 2)) as the generic
    kernel's one-level layouts."""
    q, m, no, p, _ = off.shape
    nv, _, r, _ = ref.shape
    return (
        off.reshape(q, m, no, 1, p, 2),
        attw.reshape(q, m, no, 1, p),
        ref.reshape(nv, q, 1, r, 2),
    )


def msda_fused(value, hw, ref, off, attw, vis=None):
    """Reference semantics of the DSP kernel (see module docstring)."""
    loc, aw, rf = _generic(value, hw, ref, off, attw)
    return msda_ref.msda_reference(
        value, [tuple(hw)], loc, aw, mode="pix", ref=rf, vis=vis
    )


# ---- pieces ----------------------------------------------------------------------------------
def tsa_pre(tsa, q, pos, v_prev, v_cur):
    """TSA up to the sampling: value (2, Q, E), offsets (Q, M, 2, P, 2) flat, weights (Q, M, 2, P) flat."""
    qcat = torch.cat([v_prev, q + pos], -1)
    v = torch.stack([tsa.value_proj(v_prev), tsa.value_proj(v_cur)], 0)
    off = tsa.sampling_offsets(qcat)  # (Q, M*2*P*2), laid out (M, queue, P, xy)
    wl = tsa.attention_weights(qcat).reshape(NQ, HD * 2, tsa.p)
    return v, off, torch.softmax(wl, -1).reshape(NQ, HD * 2 * tsa.p)


def sca_pre(sca, q1):
    off = sca.sampling_offsets(q1)  # (Q, M*P*2), laid out (M, P, xy)
    w = torch.softmax(sca.attention_weights(q1).reshape(NQ, HD, sca.p), -1).reshape(
        NQ, HD * sca.p
    )
    return off, w


class Pre(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats, prev_bev, has_prev, can_bus):
        enc = self.enc
        q0 = enc.bev_embedding + enc.can_bus_mlp(can_bus[None])
        img_value = (
            feats.flatten(2).transpose(1, 2)
            + enc.cams_embeds[:, None]
            + enc.level_embeds[0]
        )
        sca_v = torch.stack(
            [ly.sca.value_proj(img_value) for ly in enc.layers], 0
        )  # (L, 6, HW, E)
        v_prev = prev_bev * has_prev + q0 * (1 - has_prev)
        v, off, w = tsa_pre(enc.layers[0].tsa, q0, enc.bev_pos(), v_prev, q0)
        return sca_v, q0, v, off, w


class Mid(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, tsa_out, q):
        ly = self.layer
        q1 = ly.norms[0](ly.tsa.output_proj(tsa_out) + q)
        off, w = sca_pre(ly.sca, q1)
        return q1, off, w


class Post(nn.Module):
    def __init__(self, enc, i):
        super().__init__()
        self.enc, self.i = enc, i

    def forward(self, sca_out, q1, q0=None, prev_bev=None, has_prev=None):
        ly = self.enc.layers[self.i]
        q = ly.norms[2](ly.ffn(ly.norms[1](ly.sca.output_proj(sca_out) + q1)))
        if self.i == len(self.enc.layers) - 1:
            return q
        v_prev = prev_bev * has_prev + q * (1 - has_prev)
        v_cur = q0 * has_prev + q * (1 - has_prev)
        v, off, w = tsa_pre(
            self.enc.layers[self.i + 1].tsa, q, self.enc.bev_pos(), v_prev, v_cur
        )
        return q, v, off, w


def host_inputs(enc_in):
    """Per-frame host tensors of the split encoder from the Encoder's inputs."""
    feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask = enc_in
    r2 = M.ref_2d()
    tsa_ref = torch.stack([r2 + shift * has_prev, r2], 0).reshape(2, NQ, 1, 2)
    vis = (bev_mask.sum(-1) > 0).to(torch.uint8)
    return tsa_ref, ref_cam, vis


def run_split(enc, enc_in, sampler=msda_fused, record=None):
    """The encoder as the phone runs it: pieces + sampler (msda_fused or the DSP)."""
    feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask = enc_in
    tsa_ref, ref_cam, vis = host_inputs(enc_in)
    n = len(enc.layers)
    sca_v, q0, v, off, w = Pre(enc)(feats, prev_bev, has_prev, can_bus)
    q = q0
    for i in range(n):
        tp, sp = enc.layers[i].tsa.p, enc.layers[i].sca.p
        tsa_args = (
            v,
            (M.BEV_H, M.BEV_W),
            tsa_ref,
            off.reshape(NQ, HD, 2, tp, 2),
            w.reshape(NQ, HD, 2, tp),
            None,
        )
        tsa_out = sampler(*tsa_args)
        q1, soff, sw = Mid(enc.layers[i])(tsa_out, q)
        sca_args = (
            sca_v[i],
            (M.FH, M.FW),
            ref_cam,
            soff.reshape(NQ, HD, 1, sp, 2),
            sw.reshape(NQ, HD, 1, sp),
            vis,
        )
        sca_out = sampler(*sca_args)
        if record is not None:
            record.append({"tsa": (tsa_args, tsa_out), "sca": (sca_args, sca_out)})
        if i == n - 1:
            q = Post(enc, i)(sca_out, q1)
        else:
            q, v, off, w = Post(enc, i)(sca_out, q1, q0, prev_bev, has_prev)
    return q


def frames(work):
    return [
        torch.load(p, weights_only=False)
        for p in sorted((Path(work) / "frames").glob("*.pt"))
    ]


def cmd_check(a):
    _, enc, _ = M.load_official(a.ckpt)
    for k, f in enumerate(frames(a.work)):
        ref = enc(*f["enc_in"])
        got = run_split(enc, f["enc_in"])
        d = (ref - got).abs().max().item()
        print(f"frame {k}: split encoder vs Encoder max abs {d:.2e}")
        assert d < 1e-4, d


def save_kernel_case(out: Path, name, args, y):
    """One kernel call's I/O as a msda_hvx case directory (msda_io.h reads it)."""
    value, hw, ref, off, attw, vis = args
    loc, aw, rf = _generic(value, hw, ref, off, attw)
    msda_ref.save_case(
        out / name, value, [tuple(hw)], loc, aw, y, mode="pix", ref=rf, vis=vis
    )
    nv, q = value.shape[0], off.shape[0]
    nvis = int((vis if vis is not None else torch.ones(nv, q)).sum())
    print(
        f"{name}: NV {nv} HW {tuple(hw)} Q {q} P {off.shape[3]}; visible (map, query) pairs {nvis} of {nv * q}"
    )


def cmd_dump(a):
    _, enc, _ = M.load_official(a.ckpt)
    f = frames(a.work)[min(1, len(frames(a.work)) - 1)]
    rec = []
    run_split(enc, f["enc_in"], record=rec)
    out = Path(a.work) / "msda_io"
    for i, r in enumerate(rec):
        for k in ("tsa", "sca"):
            save_kernel_case(out, f"l{i}_{k}", *r[k])


def cmd_export(a):
    import onnx
    import onnxruntime as ort

    import onnxsim

    _, enc, _ = M.load_official(a.ckpt)
    f = frames(a.work)[min(1, len(frames(a.work)) - 1)]
    feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask = f["enc_in"]
    rec = []
    run_split(enc, f["enc_in"], record=rec)
    out = Path(a.work) / "msda_split"
    out.mkdir(exist_ok=True)
    n = len(enc.layers)
    sca_v, q0, v, off, w = Pre(enc)(feats, prev_bev, has_prev, can_bus)
    pieces = [
        (
            "pre",
            Pre(enc),
            (feats, prev_bev, has_prev, can_bus),
            ["feats", "prev_bev", "has_prev", "can_bus"],
            ["sca_v", "q0", "tsa_v", "tsa_off", "tsa_w"],
        )
    ]
    q = q0
    for i in range(n):
        tsa_out = rec[i]["tsa"][1]
        q1, _, _ = Mid(enc.layers[i])(tsa_out, q)
        pieces.append(
            (
                f"mid{i}",
                Mid(enc.layers[i]),
                (tsa_out, q),
                ["tsa_out", "q"],
                ["q1", "sca_off", "sca_w"],
            )
        )
        sca_out = rec[i]["sca"][1]
        if i == n - 1:
            pieces.append(
                (
                    f"post{i}",
                    Post(enc, i),
                    (sca_out, q1),
                    ["sca_out", "q1"],
                    ["bev_embed"],
                )
            )
        else:
            pieces.append(
                (
                    f"post{i}",
                    Post(enc, i),
                    (sca_out, q1, q0, prev_bev, has_prev),
                    ["sca_out", "q1", "q0", "prev_bev", "has_prev"],
                    ["q", "tsa_v", "tsa_off", "tsa_w"],
                )
            )
            q = Post(enc, i)(sca_out, q1, q0, prev_bev, has_prev)[0]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name, mod, x, ins, outs in pieces:
        raw = out / f"{name}.onnx"
        torch.onnx.export(
            mod,
            x,
            str(raw),
            input_names=ins,
            output_names=outs,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
        sim, _ = onnxsim.simplify(str(raw), check_n=0)
        onnx.save(sim, str(out / f"{name}.sim.onnx"))
        ref = mod(*x)
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = ort.InferenceSession(
            str(out / f"{name}.sim.onnx"), so, providers=["CPUExecutionProvider"]
        ).run(None, {k: t.numpy() for k, t in zip(ins, x)})
        md = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
        ops = {}
        for nd in sim.graph.node:
            ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
        print(
            f"{name}: {len(sim.graph.node)} nodes, ORT CPU vs torch max abs {md:.2e}; "
            + " ".join(
                f"{k}:{c}" for k, c in sorted(ops.items(), key=lambda kv: -kv[1])
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "dump", "export"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"check": cmd_check, "dump": cmd_dump, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
