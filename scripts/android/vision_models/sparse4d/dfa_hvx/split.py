"""Sparse4D v3 split around the HVX DFA: HTP pieces for everything else, the DFA on the CDSP.

  python split.py export --ckpt <pth> --work <work>   # ONNX pieces -> <work>/split/
  python split.py check --ckpt <pth> --work <work>    # the split chain (torch pieces + dfa_core's
                                                      # math on uint8 maps) vs the fp32 model: GT

Per frame (T = the temporal variant, F = a scene's first frame):

  bb    rgb uint8 (6, 256, 704, 3) -> v0..v3 uint8 (6, H, W, 256): int8 ResNet-50 + FPN, uint8 outputs
  pre0  proj, proj_n -> pts (6, N, 16, 2), w (24, N, 8, 16): layer 0's keypoints / weights
  [dfa 0]
  midK (K = 0..4, F / T)  agg, feat, anchor, proj, proj_n [, dt, temp_feat, temp_anchor]
        -> feat, anchor, pts, w: finish layer K (output_proj, FFN, norm, refine [+ instance-bank
           update after layer 0]), then layer K+1's graph attention, norm and DFA inputs
  [dfa K+1]
  post  agg, feat, anchor [, dt] -> cls, box, quality, feat

The DFA's points / weights come out in dfa_core.h's layouts, padded 13 -> 16 points with zero
weights. temp_ae = anchor_encoder(anchor)[:600] everywhere in a temporal frame (after the bank
update the first 600 anchors *are* the projected cached ones), so no piece needs it as an input.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import match  # noqa: E402
from export import (  # noqa: E402
    FoldedConv1,
    normalized_proj,
    project_mm,
    u8_input_as_dq,
)
from model import (  # noqa: E402
    CAMS,
    GROUPS,
    LEVELS,
    NUM_ANCHOR,
    NUM_TEMP,
    OPS,
    PTS,
    InstanceBank,
    Sparse4D,
    decode,
    dfa_upstream,
)

DEFORM = [i for i, op in enumerate(OPS) if op == "deformable"]  # 0, 7, 14, 21, 28, 35
P16 = 16


W_LAYOUT = "v1"  # "v2": w as fp16 (N, 8, 6 * 4 * 16) straight from the Gemm + softmax (see dfa_weights_v2)
PAD_LOGIT = -30000.0  # exp() of it is 0 in fp16 and fp32: the 3 pad points get weight 0 exactly


def dfa_weights_v2(layer, feat, ae, cam):
    """The same softmaxed weights as v1, laid out (N, 8 groups, 6 cams * 4 levels * 16 points), fp16.

    weights_fc is linear, so W (feat + ae + cam_c) + b = W (feat + ae) + (W cam_c + b): one Gemm over
    the anchors plus a tiny per-camera term, broadcast-added. Its rows are reordered to (group, level,
    point) with 3 zero pad rows per 13 points, whose per-camera term is PAD_LOGIT. The softmax over
    (camera, level, point) is then the last axis, and nothing needs a transpose or a concat on the
    HTP (v1's permute + pad were ~40% of pre0 and ~30% of every mid piece)."""
    fc = layer.weights_fc
    n = feat.shape[0]
    wt = fc.weight.reshape(LEVELS, PTS, GROUPS, -1).permute(2, 0, 1, 3)  # (8, 4, 13, 256)
    wt = torch.cat([wt, torch.zeros(GROUPS, LEVELS, P16 - PTS, wt.shape[-1])], dim=2).reshape(GROUPS * LEVELS * P16, -1)
    b = fc.bias.reshape(LEVELS, PTS, GROUPS).permute(2, 0, 1)
    b = torch.cat([b, torch.full((GROUPS, LEVELS, P16 - PTS), PAD_LOGIT)], dim=2).reshape(-1)
    a = ((feat + ae) @ wt.t()).reshape(n, GROUPS, 1, LEVELS * P16)
    bc = (cam @ wt.t() + b).reshape(CAMS, GROUPS, LEVELS * P16).permute(1, 0, 2)[None]  # (1, 8, 6, 64)
    return (a + bc).reshape(n, GROUPS, CAMS * LEVELS * P16).softmax(dim=-1).half()


def w_v2_to_v1(w2):
    """(N, 8, 384) -> (24, N, 8, 16) fp32: the check chain's (and dfa_core.h v1's) layout."""
    n = w2.shape[0]
    return w2.float().reshape(n, GROUPS, CAMS * LEVELS, P16).permute(2, 0, 1, 3).contiguous()


def dfa_inputs(layer, feat, anchor, ae, proj, proj_n):
    """-> pts (6, N, 16, 2), w: (24, N, 8, 16) fp32 (v1) or (N, 8, 384) fp16 (v2), dfa_core.h's layouts."""
    n = feat.shape[0]
    pts = project_mm(layer.kps_generator(anchor, feat), proj_n)  # (6, N, 13, 2)
    pts = torch.cat([pts, torch.zeros(CAMS, n, P16 - PTS, 2)], dim=2)
    cam = layer.camera_encoder(proj[:, :3].reshape(CAMS, 12))
    if W_LAYOUT == "v2":
        return pts, dfa_weights_v2(layer, feat, ae, cam)
    f = (feat + ae)[:, None] + cam[None]
    w = layer.weights_fc(f).reshape(n, CAMS * LEVELS * PTS, GROUPS).softmax(dim=1)
    w = w.reshape(n, CAMS * LEVELS, PTS, GROUPS).permute(1, 0, 3, 2)  # (24, N, 8, 13)
    w = torch.cat([w, torch.zeros(CAMS * LEVELS, n, GROUPS, P16 - PTS)], dim=3)
    return pts, w


class BB(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.conv1 = FoldedConv1(m.backbone.img_backbone.conv1, m.backbone.img_backbone.bn1, (256, 704))

    def forward(self, rgb):
        bb = self.m.backbone
        b = bb.img_backbone
        x = rgb.float().permute(0, 3, 1, 2)
        x = b.maxpool(b.relu(self.conv1(x)))
        feats = []
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = layer(x)
            feats.append(x)
        lat = [c(f) for c, f in zip(bb.lateral, feats)]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        return tuple(c(t).permute(0, 2, 3, 1) for c, t in zip(bb.fpn, lat))  # (6, H, W, 256) channels-last


class Pre0(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, proj, proj_n):
        h = self.m.head
        feat, anchor = h.instance_bank.instance_feature, h.instance_bank.anchor
        return dfa_inputs(h.layers[0], feat, anchor, h.anchor_encoder(anchor), proj, proj_n)


class Seg(nn.Module):
    """ops DEFORM[k] (its finish) .. DEFORM[k+1] (its inputs), or to the end for k = 5."""

    def __init__(self, m, k, temporal):
        super().__init__()
        self.m, self.k, self.temporal = m, k, temporal

    def forward(self, agg, feat, anchor, proj=None, proj_n=None, dt=None, temp_feat=None, temp_anchor=None):
        h = self.m.head
        k, last_seg = self.k, self.k == 5
        dt = dt if dt is not None else torch.tensor(0.5)
        ae = h.anchor_encoder(anchor)
        feat = h.layers[DEFORM[k]].finish(agg, feat)
        end = len(OPS) if last_seg else DEFORM[k + 1]
        n_pred = k  # refines done before this segment
        for i in range(DEFORM[k] + 1, end):
            op, layer = OPS[i], h.layers[i]
            if op == "temp_gnn":
                if self.temporal:
                    feat = h.graph(i, feat, ae, temp_feat, ae[:NUM_TEMP], temp_feat)
                else:
                    feat = h.graph(i, feat, ae)
            elif op == "gnn":
                feat = h.graph(i, feat, ae, v=feat)
            elif op in ("norm", "ffn"):
                feat = layer(feat)
            elif op == "refine":
                last = i == len(OPS) - 1
                anchor, c, q = layer(feat, anchor, ae, dt, return_cls=n_pred == 0 or last)
                n_pred += 1
                if n_pred == 1 and self.temporal:
                    idx = torch.topk(c.max(dim=-1).values, NUM_ANCHOR - NUM_TEMP).indices
                    feat = torch.cat([temp_feat, feat[idx]])
                    anchor = torch.cat([temp_anchor, anchor[idx]])
                if last:
                    return c, anchor, q, feat
                ae = h.anchor_encoder(anchor)
        pts, w = dfa_inputs(h.layers[DEFORM[k + 1]], feat, anchor, ae, proj, proj_n)
        return feat, anchor, pts, w


class Named(nn.Module):
    """Exports a Seg with its inputs bound by name (the ONNX input order is `names`)."""

    def __init__(self, seg, names):
        super().__init__()
        self.seg, self.names = seg, names

    def forward(self, *args):
        return self.seg(**dict(zip(self.names, args)))


def seg_io(k, temporal):
    ins = ["agg", "feat", "anchor"]
    if k < 5:
        ins += ["proj", "proj_n"]
    if temporal:  # post has no temporal attention left: only the refine's time interval
        ins += ["dt"] if k == 5 else ["dt", "temp_feat"] + (["temp_anchor"] if k == 0 else [])
    outs = ["cls", "anchor_out", "quality", "feat_out"] if k == 5 else ["feat_out", "anchor_out", "pts", "w"]
    return ins, outs


def dfa_u8(vals, scales, zps, pts, w):
    """dfa_core.h's math in torch: vals uint8 (6, H, W, 256) per level -> (N, 256)."""
    fm = [((v.float() - z) * s).permute(0, 3, 1, 2) for v, s, z in zip(vals, scales, zps)]
    n = pts.shape[1]
    if w.dim() == 3:  # v2
        w = w_v2_to_v1(w)
    wk = w[..., :PTS].reshape(CAMS, LEVELS, n, GROUPS, PTS).permute(2, 0, 1, 4, 3)  # (N, 6, 4, 13, 8)
    return dfa_upstream(fm, pts[:, :, :PTS], wk)


def quant_u8(t):
    lo, hi = min(float(t.min()), 0.0), max(float(t.max()), 0.0)
    s = (hi - lo) / 255 or 1.0
    z = int(round(-lo / s))
    return torch.clamp(torch.round(t / s) + z, 0, 255).to(torch.uint8), s, z


class SplitChain:
    """The split pipeline in torch, frame after frame with the host instance bank: what the phone
    runs, piece for piece (DFA = dfa_core.h's math on per-frame minmax uint8 maps)."""

    def __init__(self, m):
        self.m, self.bank = m, InstanceBank()
        self.bb, self.pre0 = BB(m).eval(), Pre0(m).eval()
        self.seg = {(k, t): Seg(m, k, t).eval() for k in range(6) for t in (False, True)}

    @torch.no_grad()
    def frame(self, rgb, metas):
        proj, proj_n = metas["projection_mat"], normalized_proj(metas["projection_mat"])
        temp_feat, temp_anchor, dt = self.bank.get(metas)
        t = temp_feat is not None
        q = [quant_u8(v) for v in self.bb(torch.from_numpy(rgb))]
        vals, scales, zps = zip(*q)
        pts, w = self.pre0(proj, proj_n)
        feat, anchor = self.m.head.instance_bank.instance_feature, self.m.head.instance_bank.anchor
        for k in range(6):
            agg = dfa_u8(vals, scales, zps, pts, w)
            kw = (dict(dt=dt) if k == 5 else dict(dt=dt, temp_feat=temp_feat)) if t else {}
            if t and k == 0:
                kw["temp_anchor"] = temp_anchor
            if k < 5:
                feat, anchor, pts, w = self.seg[(k, t)](agg, feat, anchor, proj, proj_n, **kw)
            else:
                cls, anchor, qual, feat = self.seg[(k, t)](agg, feat, anchor, **kw)
        self.bank.cache(feat, anchor, cls, metas)
        return decode(cls, anchor, qual)


def load_frames(work):
    frames = sorted((Path(work) / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
    return [pickle.load(open(p, "rb")) for p in frames]


def check(m, work):
    ch = SplitChain(m)
    tot = {0.3: [0, 0, 0], 0.2: [0, 0, 0]}
    for k, fr in enumerate(load_frames(work)):
        b, s, lab = ch.frame(fr["rgb"], fr["metas"])
        for t in tot:
            tot[t] = [x + y for x, y in zip(tot[t], match(b, s, lab, fr["gt"], thr=t))]
        print(f"frame {k}: GT >= 0.3 {match(b, s, lab, fr['gt'])[0]}")
    for t, (tp, npred, ngt) in tot.items():
        print(f"split chain (torch, uint8 value maps), score >= {t}: GT {tp}/{ngt}, predictions {npred}")


def export(m, work, suffix="sim"):
    import onnx
    import onnxruntime as ort

    from onnxsim import simplify

    out = Path(work) / "split"
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(work)
    ch = SplitChain(m)
    # record one frame-0 (first) and one frame-1 (temporal) call of every piece: example inputs
    rec = {}

    def hook(name, mod):
        orig = mod.forward

        def f(*a, **kw):
            y = orig(*a, **kw)
            rec.setdefault(name, (a, kw))
            return y

        mod.forward = f

    hook("pre0", ch.pre0)
    for (k, t), s in ch.seg.items():
        hook(f"mid{k}{'T' if t else 'F'}" if k < 5 else f"post{'T' if t else 'F'}", s)
    for fr in frames[:2]:
        ch.frame(fr["rgb"], fr["metas"])
    pieces = {} if suffix != "sim" else {"bb": (ch.bb, ((torch.from_numpy(frames[0]["rgb"]),), {}), ["rgb"], ["v0", "v1", "v2", "v3"])}
    pieces["pre0"] = (ch.pre0, rec["pre0"], ["proj", "proj_n"], ["pts", "w"])
    for (k, t), s in ch.seg.items():
        name = f"mid{k}{'T' if t else 'F'}" if k < 5 else f"post{'T' if t else 'F'}"
        if suffix != "sim" and k == 5:
            continue  # post has no DFA inputs: only the sim variant
        a, kw = rec[name]
        ins, outs = seg_io(k, t)
        byname = dict(zip(["agg", "feat", "anchor", "proj", "proj_n"], a), **kw)
        pieces[name] = (Named(s, ins), (tuple(byname[n] for n in ins), {}), ins, outs)
    ib = m.head.instance_bank  # mid0 reads the initial instances as inputs (s4d_run loads these)
    ib.instance_feature.detach().numpy().astype(np.float32).tofile(out / "instance_feature.f32")
    ib.anchor.detach().numpy().astype(np.float32).tofile(out / "anchor.f32")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name, (mod, (args, _), ins, outs) in pieces.items():
        path = out / f"{name}.onnx"
        with torch.no_grad():
            ref = mod(*args)
        torch.onnx.export(mod, args, str(path), input_names=ins, output_names=outs, opset_version=17, dynamo=False)
        sm, ok = simplify(onnx.load(str(path)))
        assert ok, name
        if name == "bb":
            u8_input_as_dq(sm)
        onnx.save(sm, str(out / f"{name}.{suffix}.onnx"))
        path.unlink()
        feeds = {n: a.numpy() for n, a in zip(ins, args)}
        got = ort.InferenceSession(sm.SerializeToString(), so, providers=["CPUExecutionProvider"]).run(None, feeds)
        d = [float(np.abs(r.numpy().astype(np.float32) - g.astype(np.float32)).max()) for r, g in zip(ref, got)]
        print(f"{name}: {len(sm.graph.node)} nodes, ORT vs torch max abs {['%.1e' % x for x in d]}")


def quantize_bb(work, data_root):
    """int8 bb with uint8 channels-last outputs (onnxsim.full_qdq + quantized_io); the output
    scales / zero points go to the model metadata (v{l}_scale, v{l}_zero_point)."""
    import onnx
    from data import NuScenesMini
    from quantize import CALIB_SCENES

    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    out = Path(work) / "split"
    bb = onnx.load(str(out / "bb.sim.onnx"))
    ns = NuScenesMini(data_root)
    data = [{"rgb": ns.frame(tok)["rgb"]} for sc in CALIB_SCENES for tok in ns.scene_samples(sc)[:3]]
    q = quantize_full_qdq(bb, data, exclude_nodes=["rgb_dq"])
    q, info = quantized_io(q, inputs=[], outputs=["v0", "v1", "v2", "v3"])
    for name, qi in info.items():
        for key in ("scale", "zero_point"):
            e = q.metadata_props.add()
            e.key, e.value = f"{name}_{key}", str(qi[key])
    onnx.save(q, str(out / "bb.q8.onnx"))
    print("bb.q8.onnx:", {n: (qi["scale"], qi["zero_point"]) for n, qi in info.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check", "quantize"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", default=str(Path.home() / ".cache/onnxsim-bevformer/nuscenes-mini"))
    ap.add_argument("--w-layout", choices=["v1", "v2"], default="v1",
                    help="v2: pre0 / mid pieces emit w as fp16 (N, 8, 384), written as <piece>.w2.onnx")
    a = ap.parse_args()
    global W_LAYOUT
    W_LAYOUT = a.w_layout
    torch.set_grad_enabled(False)
    if a.cmd == "quantize":
        return quantize_bb(a.work, a.data)
    m = Sparse4D().load_official(a.ckpt).eval()
    if a.cmd == "export":
        export(m, a.work, "w2" if a.w_layout == "v2" else "sim")
    else:
        check(m, a.work)


if __name__ == "__main__":
    main()
