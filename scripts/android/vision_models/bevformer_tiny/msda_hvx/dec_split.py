#!/usr/bin/env python3
"""BEVFormer-tiny's decoder split around the HVX multi-scale deformable attention kernel, like
split.py does for the encoder.

Each decoder layer's cross-attention (CustomMSDeformableAttention over the BEV, one 50x50 level,
8 heads x 4 points, 900 queries) is "grid -> GridSample -> x weight -> sum over points" on the HTP,
~35-40% of the fp16 decoder (README). That span becomes one kernel call per layer (msda_fused with
NV = 1, R = 1, NO = 1), and the HTP keeps the Linears, self-attention, LayerNorms, FFN and the
reference-point refinement.

Pieces (fixed shapes, fp32 I/O), L = layer:
  dpre      bev_embed                    -> dv (6, 2500, 256): every layer's value_proj(bev)
  dmid<L>   dout, q1, refp               -> q1', doff, dw, refp', dref   (L = 0..4; layer L's
            output_proj + norm + FFN + norm + reg branch -> refined refp', then layer L+1's
            self-attention + norm and its sampling offsets / weights)
  dpost     dout, q1, refp               -> cls_scores, bbox_preds      (layer 5 + the heads)
Layer 0's inputs (query after self-attention, offsets, weights, reference) depend only on the
learned query embedding, so the host computes them once: dec_const/{q1,doff,dw,refp,dref}.f32.

usage: dec_split.py check  --ckpt <pth> --work <work>   torch split vs Decoder on the saved frames
       dec_split.py export --ckpt <pth> --work <work>   pieces -> <work>/msda_split/d*.onnx (+ onnxsim),
                                                        constants -> <work>/msda_split/dec_const/
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
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import split as S  # noqa: E402

E, HD, NQ, Q = M.EMBED, M.HEADS, M.NQ, M.NUM_QUERY


def self_attn(ly, q, qpos):
    qk = (q + qpos)[:, None]
    return ly.norms[0](
        q + ly.self_attn(qk, qk, q[:, None], need_weights=False)[0][:, 0]
    )


def cross_in(ly, q1, qpos):
    """Layer ly's sampling offsets (Q, M*P*2) and softmaxed weights (Q, M*P)."""
    ca = ly.cross_attn
    x = q1 + qpos
    w = torch.softmax(ca.attention_weights(x).reshape(Q, HD, ca.p), -1).reshape(
        Q, HD * ca.p
    )
    return ca.sampling_offsets(x), w


def refine(reg, q, refp):
    tmp = reg(q)
    xy = (tmp[:, 0:2] + M.inverse_sigmoid(refp[:, :2])).sigmoid()
    z = (tmp[:, 4:5] + M.inverse_sigmoid(refp[:, 2:3])).sigmoid()
    return tmp, torch.cat([xy, z], -1)


def layer_out(ly, dout, q1):
    q = ly.norms[1](ly.cross_attn.output_proj(dout) + q1)
    return ly.norms[2](ly.ffn(q))


class DPre(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, bev_embed):
        return torch.stack(
            [ly.cross_attn.value_proj(bev_embed) for ly in self.dec.layers], 0
        )


class DMid(nn.Module):
    def __init__(self, dec, i):
        super().__init__()
        self.dec, self.i = dec, i

    def forward(self, dout, q1, refp):
        dec, i = self.dec, self.i
        qpos = dec.query_embedding[:, :E]
        q = layer_out(dec.layers[i], dout, q1)
        _, refp = refine(dec.reg_branches[i], q, refp)
        nxt = dec.layers[i + 1]
        q1 = self_attn(nxt, q, qpos)
        off, w = cross_in(nxt, q1, qpos)
        return q1, off, w, refp, refp[:, :2].contiguous()


class DPost(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, dout, q1, refp):
        dec = self.dec
        n = len(dec.layers) - 1
        q = layer_out(dec.layers[n], dout, q1)
        tmp, _ = refine(dec.reg_branches[n], q, refp)
        cls = dec.cls_last(q)
        r = M.inverse_sigmoid(refp)
        xy = (tmp[:, 0:2] + r[:, 0:2]).sigmoid()
        z = (tmp[:, 4:5] + r[:, 2:3]).sigmoid()
        pc = M.PC_RANGE
        x = xy[:, 0:1] * (pc[3] - pc[0]) + pc[0]
        y = xy[:, 1:2] * (pc[4] - pc[1]) + pc[1]
        zz = z * (pc[5] - pc[2]) + pc[2]
        return cls, torch.cat([x, y, tmp[:, 2:4], zz, tmp[:, 5:]], -1)


def const(dec):
    """Layer 0's kernel inputs: fixed by the learned query embedding."""
    qpos, q = dec.query_embedding[:, :E], dec.query_embedding[:, E:]
    refp = dec.reference_points(qpos).sigmoid()
    q1 = self_attn(dec.layers[0], q, qpos)
    off, w = cross_in(dec.layers[0], q1, qpos)
    return {
        "q1": q1,
        "doff": off,
        "dw": w,
        "refp": refp,
        "dref": refp[:, :2].contiguous(),
    }


def sample(dv_l, doff, dw, dref, sampler=S.msda_fused):
    """One layer's cross-attention sampling in the kernel's terms (NV = 1, R = 1, NO = 1)."""
    p = doff.shape[1] // (HD * 2)
    return sampler(
        dv_l[None],
        (M.BEV_H, M.BEV_W),
        dref.reshape(1, Q, 1, 2),
        doff.reshape(Q, HD, 1, p, 2),
        dw.reshape(Q, HD, 1, p),
        None,
    )


def run_split(dec, bev, sampler=S.msda_fused, record=None):
    c = const(dec)
    dv = DPre(dec)(bev)
    q1, off, w, refp, dref = c["q1"], c["doff"], c["dw"], c["refp"], c["dref"]
    n = len(dec.layers)
    for i in range(n):
        dout = sample(dv[i], off, w, dref, sampler)
        if record is not None:
            record.append((dout, q1, refp))
        if i == n - 1:
            return DPost(dec)(dout, q1, refp)
        q1, off, w, refp, dref = DMid(dec, i)(dout, q1, refp)


def cmd_check(a):
    _, enc, dec = M.load_official(a.ckpt)
    for k, f in enumerate(S.frames(a.work)):
        bev = enc(*f["enc_in"])
        cls, bbox = dec(bev)
        c2, b2 = run_split(dec, bev)
        d = max((cls - c2).abs().max().item(), (bbox - b2).abs().max().item())
        print(f"frame {k}: split decoder vs Decoder max abs {d:.2e}")
        assert d < 1e-3, d


def cmd_export(a):
    import onnx
    import onnxruntime as ort

    import onnxsim

    _, enc, dec = M.load_official(a.ckpt)
    f = S.frames(a.work)[min(1, len(S.frames(a.work)) - 1)]
    bev = enc(*f["enc_in"])
    rec = []
    run_split(dec, bev, record=rec)
    out = Path(a.work) / "msda_split"
    cd = out / "dec_const"
    cd.mkdir(parents=True, exist_ok=True)
    for k, v in const(dec).items():
        np.ascontiguousarray(v.numpy(), np.float32).tofile(cd / f"{k}.f32")
    pieces = [("dpre", DPre(dec), (bev,), ["bev_embed"], ["dv"])]
    for i in range(len(dec.layers)):
        dout, q1, refp = rec[i]
        if i == len(dec.layers) - 1:
            pieces.append(
                (
                    "dpost",
                    DPost(dec),
                    (dout, q1, refp),
                    ["dout", "q1", "refp"],
                    ["cls_scores", "bbox_preds"],
                )
            )
        else:
            pieces.append(
                (
                    f"dmid{i}",
                    DMid(dec, i),
                    (dout, q1, refp),
                    ["dout", "q1", "refp"],
                    ["q1o", "doff", "dw", "refpo", "dref"],
                )
            )
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
        print(f"{name}: {len(sim.graph.node)} nodes, ORT CPU vs torch max abs {md:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "export"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"check": cmd_check, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
