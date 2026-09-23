#!/usr/bin/env python3
"""MapTR-tiny's encoder (and decoder) split around the HVX MSDA kernel (../../../msda_hvx/).

Same scheme as ../../bevformer_tiny/msda_hvx/split.py: the HTP runs the Linears / softmax /
LayerNorm / FFN, the DSP runs each deformable sampling ("locations -> bilinear taps -> x attention
weight -> sum over points (-> average over visible cameras)") as one kernel call. On the HTP that
span is what breaks MapTR's encoder: 20000 BEV queries x 6 cameras x 8 heads x 8 points of
broadcast grid math and GridSample, 2790 ms with bev cos 0.815 fp16 (README).

MapTR has no temporal state (prev_bev is always None), so TSA's 2-frame queue is [q, q]: both
frames sample the same value map at the same reference points. That is exactly ONE value map with
the two frames' 4 points each concatenated to 8 and every weight halved (the queue mean) -- half the
value traffic of the literal 2-map call.

Kernel calls (mode MSDA_REF_PIX: loc = ref + off / (W, H); one level, 8 heads x 32):
  tsa  value tsa_v (1, 20000, 256), ref ref_2d (1, Q, 1, 2), off (Q, 8, 1, 8, 2), attw (Q, 8, 1, 8)
  sca  value sca_v (6, 375, 256),   ref ref_cam (6, Q, 4, 2), off (Q, 8, 1, 8, 2), attw (Q, 8, 1, 8),
       vis (6, Q) -- point p uses pillar anchor p % 4; averaged over the visible cameras
  dec<i> (decoder, --dec) value dv<i> (1, 20000, 256), ref (1, 1000, 1, 2), off (1000, 8, 1, 4, 2),
       attw (1000, 8, 1, 4)

Encoder pieces (fp32 I/O, fp16 inside on the HTP):
  pre   feats (6, 256, 15, 25), can_bus (18) -> sca_v, q0, tsa_v, tsa_off, tsa_w
  mid   tsa_out, q0                           -> q1, sca_off, sca_w
  post  sca_out, q1                           -> bev (20000, 256)
Decoder pieces (--dec):
  dvals bev -> dv (6, 20000, 256) (every layer's value_proj of the BEV), q/ref init
  dpre<i>  q, ref -> q_sa (after self-attn + norm), off, attw     (i = 0..5)
  dpost<i> msda_out, q_sa, ref_in -> q, ref (refined)              (dpost5: -> cls, pts)

usage: split.py check  --ckpt <pth> --work <work>         torch split vs Encoder/Decoder on saved frames
       split.py export --ckpt <pth> --work <work>         pieces -> <work>/msda_split/*.onnx (+ onnxsim)
       split.py host   --work <work>                      per-frame host inputs -> <work>/msda_frames/<i>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "msda_hvx"))
sys.path.insert(0, str(HERE.parent))  # maptr/model.py first
import model as M  # noqa: E402
import msda_ref  # noqa: E402

E, HD, NQ, NQD = M.EMBED, M.HEADS, M.NQ, M.NUM_QUERY
D = E // HD


def msda_fused(value, hw, ref, off, attw, vis=None):
    """Reference semantics of the DSP call: value (NV, S, E), ref (NV, Q, R, 2),
    off (Q, M, NO, P, 2) raw pixels, attw (Q, M, NO, P), vis (NV, Q) or None -> (Q, E)."""
    q, m, no, p, _ = off.shape
    nv, _, r, _ = ref.shape
    return msda_ref.msda_reference(value, [tuple(hw)], off.reshape(q, m, no, 1, p, 2), attw.reshape(q, m, no, 1, p),
                                   mode="pix", ref=ref.reshape(nv, q, 1, r, 2), vis=vis)


# ---- encoder pieces --------------------------------------------------------------------------
class Pre(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats, can_bus):
        enc = self.enc
        q0 = enc.bev_embedding + enc.can_bus_mlp(can_bus[None])
        sca_v = enc.sca.value_proj(enc.img_value(feats))  # (6, 375, E)
        off, w = enc.tsa.offsets_weights(q0, enc.bev_pos())  # (Q, M, 2, 4, 2), (Q, M, 2, 4)
        tsa_off = off.reshape(NQ, HD, 1, 2 * M.TSA_POINTS, 2)  # frame-major: points 0-3 frame 0, 4-7 frame 1
        tsa_w = (w * 0.5).reshape(NQ, HD, 1, 2 * M.TSA_POINTS)  # the queue mean
        return sca_v, q0, enc.tsa.value_proj(q0), tsa_off, tsa_w


class Mid(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, tsa_out, q0):
        enc = self.enc
        q1 = enc.norms[0](enc.tsa.output_proj(tsa_out) + q0)
        off = enc.sca.sampling_offsets(q1).reshape(NQ, HD, 1, M.SCA_POINTS, 2)
        w = torch.softmax(enc.sca.attention_weights(q1).reshape(NQ, HD, M.SCA_POINTS), -1)
        return q1, off, w.reshape(NQ, HD, 1, M.SCA_POINTS)


class Post(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, sca_out, q1):
        enc = self.enc
        return enc.norms[2](enc.ffn(enc.norms[1](enc.sca.output_proj(sca_out) + q1)))


class PreV(nn.Module):
    """--cpu-tsa: the only HTP part of `pre` left, the SCA value maps."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats):
        return self.enc.sca.value_proj(self.enc.img_value(feats))


class MidC(nn.Module):
    """--cpu-tsa: `mid` recomputing q0 = bev_embedding + can_bus_mlp(can_bus) in-graph instead of
    reading it (20 MB fp32) from `pre`."""

    def __init__(self, enc):
        super().__init__()
        self.enc, self.mid = enc, Mid(enc)

    def forward(self, tsa_out, can_bus):
        return self.mid(tsa_out, self.enc.bev_embedding + self.enc.can_bus_mlp(can_bus[None]))


def tsa_consts(enc):
    """TSA's inputs are affine in c = can_bus_mlp(can_bus), one 256-vector per frame, because
    q0 = bev_embedding + c and the Linears are linear:
      tsa_v   = V0 + Wv c                 V0 = value_proj(bev_embedding)
      off     = A  + (Wo[:, :E] + Wo[:, E:]) c   (sampling_offsets on cat(q0, q0 + pos))
      logits  = A' + (Wa[:, :E] + Wa[:, E:]) c
    so the phone builds them on the CPU (cpu_tsa below, map_run's CPU_TSA path) from these
    constants: no HTP call, no 57 MB of fp32 graph outputs."""
    t, pos, be = enc.tsa, enc.bev_pos(), enc.bev_embedding
    qcat = torch.cat([be, be + pos], -1)
    wo, wa = t.sampling_offsets.weight, t.attention_weights.weight
    mlp = enc.can_bus_mlp
    return {
        "V0": t.value_proj(be), "Wv": t.value_proj.weight, "A": t.sampling_offsets(qcat),
        "Bo": wo[:, :E] + wo[:, E:], "Aw": t.attention_weights(qcat), "Bw": wa[:, :E] + wa[:, E:],
        "m0w": mlp[0].weight, "m0b": mlp[0].bias, "m2w": mlp[2].weight, "m2b": mlp[2].bias,
        "lnw": mlp[4].weight, "lnb": mlp[4].bias,
    }


def cpu_tsa(k, can_bus):
    """What map_run's CPU_TSA path computes (numpy mirror) -> tsa_v, tsa_off, tsa_w."""
    k = {n: v.detach().numpy().astype(np.float32) for n, v in k.items()}
    cb = can_bus.numpy().astype(np.float32)
    h = np.maximum(k["m0w"] @ cb + k["m0b"], 0)
    h = np.maximum(k["m2w"] @ h + k["m2b"], 0)
    c = (h - h.mean()) / np.sqrt(h.var() + 1e-5) * k["lnw"] + k["lnb"]
    v = k["V0"] + k["Wv"] @ c
    off = k["A"] + k["Bo"] @ c
    lg = (k["Aw"] + k["Bw"] @ c).reshape(NQ, HD * 2, M.TSA_POINTS)
    e = np.exp(lg - lg.max(-1, keepdims=True))
    w = 0.5 * e / e.sum(-1, keepdims=True)
    return (torch.from_numpy(v), torch.from_numpy(off).reshape(NQ, HD, 1, 2 * M.TSA_POINTS, 2),
            torch.from_numpy(w.astype(np.float32)).reshape(NQ, HD, 1, 2 * M.TSA_POINTS))


def host_inputs(f):
    """-> tsa_ref (1, Q, 1, 2), ref_cam (6, Q, 4, 2), vis (6, Q) uint8."""
    return M.ref_2d().reshape(1, NQ, 1, 2), f["ref_cam"], (f["bev_mask"].sum(-1) > 0).to(torch.uint8)


def run_encoder(enc, f, sampler=msda_fused):
    tsa_ref, ref_cam, vis = host_inputs(f)
    sca_v, q0, tsa_v, tsa_off, tsa_w = Pre(enc)(f["feats"], f["can_bus"])
    tsa_out = sampler(tsa_v[None], (M.BEV_H, M.BEV_W), tsa_ref, tsa_off, tsa_w, None)
    q1, sca_off, sca_w = Mid(enc)(tsa_out, q0)
    sca_out = sampler(sca_v, (M.FH, M.FW), ref_cam, sca_off, sca_w, vis)
    return Post(enc)(sca_out, q1)


# ---- decoder pieces --------------------------------------------------------------------------
class DVals(nn.Module):
    """Every decoder layer's value_proj of the BEV (they don't depend on the queries)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, bev):
        return torch.stack([ly.cross_attn.value_proj(bev) for ly in self.dec.layers], 0)


def dec_init(dec):
    emb = (dec.pts_embedding[None] + dec.instance_embedding[:, None]).reshape(NQD, 2 * E)
    qpos, q = emb[:, :E], emb[:, E:]
    return qpos, q, dec.reference_points(qpos).sigmoid()


class DPre(nn.Module):
    """Layer i up to the sampling: self-attn + norm, then the cross-attn's offsets / weights."""

    def __init__(self, dec, i):
        super().__init__()
        self.dec, self.i = dec, i
        qpos, _, _ = dec_init(dec)
        self.register_buffer("qpos", qpos.detach().clone())

    def forward(self, q):
        ly = self.dec.layers[self.i]
        qk = (q + self.qpos)[:, None]
        q = ly.norms[0](q + ly.self_attn(qk, qk, q[:, None], need_weights=False)[0][:, 0])
        qq = q + self.qpos
        ca = ly.cross_attn
        off = ca.sampling_offsets(qq).reshape(NQD, HD, 1, M.DEC_POINTS, 2)
        w = torch.softmax(ca.attention_weights(qq).reshape(NQD, HD, M.DEC_POINTS), -1)
        return q, off, w.reshape(NQD, HD, 1, M.DEC_POINTS)


class DPost(nn.Module):
    def __init__(self, dec, i):
        super().__init__()
        self.dec, self.i = dec, i

    def forward(self, msda_out, q_sa, ref):
        ly = self.dec.layers[self.i]
        q = ly.norms[2](ly.ffn(ly.norms[1](ly.cross_attn.output_proj(msda_out) + q_sa)))
        ref = (self.dec.reg_branches[self.i](q) + M.inverse_sigmoid(ref)).sigmoid()
        if self.i == len(self.dec.layers) - 1:
            return self.dec.cls_last(q.reshape(M.NUM_VEC, M.NUM_PTS, E).mean(1)), ref.reshape(M.NUM_VEC, M.NUM_PTS, 2)
        return q, ref


def run_decoder(dec, bev, sampler=msda_fused):
    dv = DVals(dec)(bev)
    _, q, ref = dec_init(dec)
    for i in range(len(dec.layers)):
        q_sa, off, w = DPre(dec, i)(q)
        out = sampler(dv[i:i + 1], (M.BEV_H, M.BEV_W), ref.reshape(1, NQD, 1, 2), off, w, None)
        q, ref = DPost(dec, i)(out, q_sa, ref)
    return q, ref  # (cls, pts) after the last layer


def frames(work):
    return [torch.load(p, weights_only=False) for p in sorted((Path(work) / "frames").glob("*.pt"))]


def cmd_check(a):
    _, enc, dec = M.load_official(a.ckpt)
    for k, f in enumerate(frames(a.work)):
        bev = run_encoder(enc, f)
        cls, pts = run_decoder(dec, f["bev"])
        _, _, tv, to, tw = Pre(enc)(f["feats"], f["can_bus"])
        cv, co, cw = cpu_tsa(tsa_consts(enc), f["can_bus"])
        print(f"frame {k}: cpu_tsa vs Pre: v {(cv - tv).abs().max():.2e} off {(co - to).abs().max():.2e} "
              f"w {(cw - tw).abs().max():.2e}")
        print(f"frame {k}: split encoder vs Encoder max abs {(bev - f['bev']).abs().max():.2e}; split decoder "
              f"cls {(cls - f['cls']).abs().max():.2e} pts {(pts - f['pts']).abs().max():.2e}")


def export_piece(mod, x, in_names, out_names, path):
    import onnx
    import onnxsim

    torch.onnx.export(mod, x, str(path), input_names=in_names, output_names=out_names, opset_version=17,
                      dynamo=False, do_constant_folding=True)
    sim, _ = onnxsim.simplify(str(path), check_n=0)
    onnx.save(sim, str(path))
    print(f"  {path.name}: {len(sim.graph.node)} nodes")


def cmd_export(a):
    _, enc, dec = M.load_official(a.ckpt)
    out = Path(a.work) / "msda_split"
    out.mkdir(parents=True, exist_ok=True)
    f = frames(a.work)[1]
    tsa_ref, ref_cam, vis = host_inputs(f)
    sca_v, q0, tsa_v, tsa_off, tsa_w = Pre(enc)(f["feats"], f["can_bus"])
    tsa_out = msda_fused(tsa_v[None], (M.BEV_H, M.BEV_W), tsa_ref, tsa_off, tsa_w, None)
    q1, sca_off, sca_w = Mid(enc)(tsa_out, q0)
    sca_out = msda_fused(sca_v, (M.FH, M.FW), ref_cam, sca_off, sca_w, vis)
    export_piece(Pre(enc), (f["feats"], f["can_bus"]), ["feats", "can_bus"],
                 ["sca_v", "q0", "tsa_v", "tsa_off", "tsa_w"], out / "pre.onnx")
    export_piece(Mid(enc), (tsa_out, q0), ["tsa_out", "q0"], ["q1", "sca_off", "sca_w"], out / "mid.onnx")
    export_piece(Post(enc), (sca_out, q1), ["sca_out", "q1"], ["bev"], out / "post.onnx")
    export_piece(PreV(enc), (f["feats"],), ["feats"], ["sca_v"], out / "prev.onnx")
    export_piece(MidC(enc), (tsa_out, f["can_bus"]), ["tsa_out", "can_bus"], ["q1", "sca_off", "sca_w"], out / "midc.onnx")
    for n, v in tsa_consts(enc).items():  # map_run CPU_TSA reads <piece dir>/tsa_<name>.f32
        v.detach().numpy().astype(np.float32).tofile(out / f"tsa_{n}.f32")
    if a.dec:
        bev = f["bev"]
        export_piece(DVals(dec), (bev,), ["bev"], ["dv"], out / "dvals.onnx")
        _, q, ref = dec_init(dec)
        for i in range(len(dec.layers)):
            q_sa, off, w = DPre(dec, i)(q)
            export_piece(DPre(dec, i), (q.detach(),), ["q"], ["q_sa", "doff", "dw"], out / f"dpre{i}.onnx")
            o = msda_fused(DVals(dec)(bev)[i:i + 1], (M.BEV_H, M.BEV_W), ref.reshape(1, NQD, 1, 2), off, w, None)
            names = ["cls", "pts"] if i == len(dec.layers) - 1 else ["q", "ref"]
            export_piece(DPost(dec, i), (o, q_sa, ref.detach()), ["msda_out", "q_sa", "ref_in"], names, out / f"dpost{i}.onnx")
            q, ref = DPost(dec, i)(o, q_sa, ref)
        qpos, q, ref = dec_init(dec)
        q.detach().numpy().astype(np.float32).tofile(out / "dq0.f32")
        ref.detach().numpy().astype(np.float32).tofile(out / "dref0.f32")


def cmd_host(a):
    """Per-frame inputs the phone chain reads: img (6,3,480,800), feats, can_bus, tsa_ref, ref_cam,
    vis (u8) and the fp32 torch references (bev, cls, pts)."""
    for k, f in enumerate(frames(a.work)):
        d = Path(a.work) / "msda_frames" / str(k)
        d.mkdir(parents=True, exist_ok=True)
        tsa_ref, ref_cam, vis = host_inputs(f)
        for n, t in (("img", f["img"]), ("feats", f["feats"]), ("can_bus", f["can_bus"]), ("tsa_ref", tsa_ref),
                     ("ref_cam", ref_cam), ("ref_bev", f["bev"]), ("ref_cls", f["cls"]), ("ref_pts", f["pts"])):
            t.detach().numpy().astype(np.float32).tofile(d / f"{n}.f32")
        vis.numpy().astype(np.uint8).tofile(d / "vis.u8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "export", "host"])
    ap.add_argument("--ckpt")
    ap.add_argument("--work", required=True)
    ap.add_argument("--dec", action="store_true", help="export: also the decoder split")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"check": cmd_check, "export": cmd_export, "host": cmd_host}[a.cmd](a)


if __name__ == "__main__":
    main()
