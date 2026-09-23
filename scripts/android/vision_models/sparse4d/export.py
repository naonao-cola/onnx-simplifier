"""Sparse4D v3 -> ONNX pieces for the phone's HTP.

  python export.py frame --ckpt <pth> --work <work>     # all-HTP baseline: the whole frame, 2 graphs
  python export.py inputs --ckpt <pth> --work <work>    # per-frame phone inputs + torch references

`frame` exports the whole model for one frame as one graph, twice:
  * frame_first.onnx: a scene's first frame (no cached instances: the temporal attention is plain
    self-attention and the instance-bank update is a no-op),
  * frame_temp.onnx: every later frame. Its extra inputs are the 600 cached instances (feature and
    anchor, already ego-motion projected on the host) and the time interval; the instance-bank
    update (top-300 of layer 0 + the 600 cached) runs in the graph (TopK + Gather + Concat).
Inputs: rgb uint8 (6, 256, 704, 3) (the normalization is folded into conv1), projection_mat
(6, 4, 4). Outputs: cls (900, 10), box (900, 11), quality (900, 2) and the last feature
(900, 256) the host caches. Every tensor is rank <= 4 (DFA as model.dfa_rank4; project_points
as a MatMul; the DFA weights' (cams, levels, pts, groups) split via rank-4 reshapes).
Each ONNX is checked against torch on ORT CPU, then simplified with onnxsim.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from data import MEAN, STD
from model import (
    CAMS,
    EMBED,
    GROUPS,
    LEVELS,
    NUM_ANCHOR,
    NUM_TEMP,
    OPS,
    PTS,
    Runner,
    Sparse4D,
    dfa_rank4,
)
from torch import nn


class FoldedConv1(nn.Module):
    """bn1(conv1((x - mean) / std)) for RGB x in [0, 255], exactly, without normalizing the image.

    conv1'(x) = conv(w / std, x) - sum(w * mean / std) is exact in the interior. At the border
    conv1 zero-pads the *normalized* image (i.e. pads x with the mean), conv1' pads x with 0: the
    difference only depends on the output position, so it's added back as a constant map. The
    (inference) BatchNorm after it is a per-channel affine map, folded into both."""

    def __init__(self, conv, bn, hw):
        super().__init__()
        a = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        b = (bn.bias - bn.running_mean * a).detach()
        w = conv.weight.detach() * a.view(-1, 1, 1, 1)
        std = torch.tensor(STD).view(1, 3, 1, 1)
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        self.conv = nn.Conv2d(3, w.shape[0], conv.kernel_size, conv.stride, conv.padding, bias=True)
        self.conv.weight.data = w / std
        self.conv.bias.data = b - (w * mean / std).sum(dim=(1, 2, 3))
        ones = torch.ones(1, 3, *hw)
        # true: conv(w/std, pad0(x - mean)); folded: conv(w/std, pad0(x)) - sum(w*mean/std).
        # With x = mean everywhere the true output is b (bn of a zero conv), so corr = b - folded(mean image).
        with torch.no_grad():
            self.register_buffer("corr", b.view(1, -1, 1, 1) - self.conv(ones * mean))

    def forward(self, x):
        return self.conv(x) + self.corr


DEPTH_MIN = float(__import__("os").environ.get("SPARSE4D_DEPTH_MIN", "1e-2"))


def project_mm(kp, proj_n):
    """project_points without Einsum and fp16-safe: kp (N, 13, 3), proj (6, 4, 4) -> (6, N, 13, 2).

    Upstream divides by max(depth, 1e-5), so a keypoint behind a camera lands at ~1e9: past
    fp16's 65504, which broke the fp16 HTP graph (cls cos 0.92 on frame 0). Here:
      * proj_n is lidar2img[:3] with rows 0 / 1 already divided by the image width / height
        (`normalized_proj`, host side), so the MatMul yields depth-scale values (<= ~60), not
        pixels x depth (~4e4);
      * the depth is clamped at 1 cm instead of 1e-5 m, so |x / depth| stays < ~4e3. A 10 cm
        floor is *not* exact (a few keypoints do sit that close to a camera plane: outputs moved
        by up to 0.35 on 3 of the 6 frames); 1 cm and 1 mm both match upstream on all 6 frames to
        <= 9e-4 (export.py checks every frame);
      * the normalized location is clamped to [-1.5, 2.5]: anything outside [0, 1] samples only
        grid_sample's zero padding at every level (>= 1 level-pixel outside), so this is exact."""
    n = kp.shape[0]
    ptsx = torch.cat([kp, torch.ones_like(kp[..., :1])], dim=-1).reshape(n * PTS, 4)
    p = ptsx @ proj_n.reshape(CAMS * 3, 4).transpose(0, 1)  # (N*13, 18)
    p = p.reshape(n * PTS, CAMS, 3).transpose(0, 1)  # (6, N*13, 3)
    xy = p[..., :2] / torch.clamp(p[..., 2:3], min=DEPTH_MIN)
    return torch.clamp(xy, -1.5, 2.5).reshape(CAMS, n, PTS, 2)


def dfa_weights_r4(layer, feat, ae, proj):
    """DFA.weights reshaped rank <= 4 straight into per-level (4, 6, 8, N*13) = (level, cam, group, anchor*pt)."""
    n = feat.shape[0]
    cam = layer.camera_encoder(proj[:, :3].reshape(CAMS, 12))
    f = (feat + ae)[:, None] + cam[None]
    w = layer.weights_fc(f).reshape(n, CAMS * LEVELS * PTS, GROUPS).softmax(dim=1)  # (N, 312, 8)
    w = w.reshape(n, CAMS * LEVELS, PTS, GROUPS).permute(1, 3, 0, 2)  # (24, 8, N, 13)
    return w.reshape(CAMS, LEVELS, GROUPS, n * PTS)


def normalized_proj(proj):
    """(6, 4, 4) lidar2img -> (6, 3, 4) with x / y rows divided by the image width / height."""
    return proj[:, :3] / torch.tensor([704.0, 256.0, 1.0]).view(1, 3, 1)


def dfa_graph(layer, fmaps, feat, anchor, ae, proj, proj_n):
    """DFA for the graph: rank <= 4 throughout; returns the 'cat' residual (N, 512)."""
    n = feat.shape[0]
    grid = project_mm(layer.kps_generator(anchor, feat), proj_n) * 2 - 1  # (6, N, 13, 2)
    wl = dfa_weights_r4(layer, feat, ae, proj)
    out = 0
    for lv, fm in enumerate(fmaps):
        s = F.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        s = s.reshape(CAMS, GROUPS, EMBED // GROUPS, n * PTS) * wl[:, lv][:, :, None]
        out = out + s.sum(dim=0).reshape(EMBED, n, PTS).sum(dim=-1)
    return layer.finish(out.transpose(0, 1), feat)


class FrameGraph(nn.Module):
    def __init__(self, m: Sparse4D, temporal: bool):
        super().__init__()
        self.m, self.temporal = m, temporal
        self.conv1 = FoldedConv1(m.backbone.img_backbone.conv1, m.backbone.img_backbone.bn1, (256, 704))

    def backbone(self, rgb):
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
        return [c(t) for c, t in zip(bb.fpn, lat)]

    def forward(self, rgb, proj, proj_n, dt=None, temp_feat=None, temp_anchor=None):
        head = self.m.head
        fmaps = self.backbone(rgb)
        feat = head.instance_bank.instance_feature
        anchor = head.instance_bank.anchor
        ae = head.anchor_encoder(anchor)
        temp_ae = head.anchor_encoder(temp_anchor) if self.temporal else None
        dt = dt if self.temporal else torch.tensor(0.5)
        n_pred = 0
        for i, op in enumerate(OPS):
            layer = head.layers[i]
            if op == "temp_gnn":
                feat = head.graph(i, feat, ae, temp_feat, temp_ae, temp_feat) if self.temporal else head.graph(i, feat, ae)
            elif op == "gnn":
                feat = head.graph(i, feat, ae, v=feat)
            elif op in ("norm", "ffn"):
                feat = layer(feat)
            elif op == "deformable":
                feat = dfa_graph(layer, fmaps, feat, anchor, ae, proj, proj_n)
            elif op == "refine":
                last = i == len(OPS) - 1
                anchor, c, q = layer(feat, anchor, ae, dt, return_cls=n_pred == 0 or last)
                n_pred += 1
                if n_pred == 1 and self.temporal:
                    idx = torch.topk(c.max(dim=-1).values, NUM_ANCHOR - NUM_TEMP).indices
                    feat = torch.cat([temp_feat, feat[idx]])
                    anchor = torch.cat([temp_anchor, anchor[idx]])
                if not last:
                    ae = head.anchor_encoder(anchor)
                if n_pred > 1 and self.temporal:
                    temp_ae = ae[:NUM_TEMP]
        return c, anchor, q, feat


def frame_inputs(m, ckpt, work):
    """Run the torch model through the saved frames (validate.py --work); per frame: the graph's
    inputs (with the temporal state the host holds) and its torch outputs."""
    run = Runner(m, dfa_rank4)
    frames = sorted((work / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
    out = []
    for p in frames:
        fr = pickle.load(open(p, "rb"))
        metas = fr["metas"]
        img = torch.from_numpy((fr["rgb"].astype(np.float32) - MEAN) / STD).permute(0, 3, 1, 2).contiguous()
        temp_feat, temp_anchor = run.bank.cached_feature, run.bank.cached_anchor
        prev = None if temp_feat is None else (run.bank.metas, temp_anchor.clone(), temp_feat.clone(), run.bank.confidence.clone())
        (_, _, _), raw = run.frame(img, metas)
        rec = {"rgb": fr["rgb"], "proj": metas["projection_mat"].numpy(),
               "proj_n": normalized_proj(metas["projection_mat"]).numpy(), "gt": fr["gt"], "metas": metas,
               "ref": [t.numpy() for t in raw]}
        if prev is not None:
            # redo the bank's get() on the saved state to capture the projected anchors + dt
            from model import InstanceBank
            b = InstanceBank()
            b.metas, b.cached_anchor, b.cached_feature, b.confidence = prev
            tf, ta, dt = b.get(metas)
            rec.update(temp_feat=tf.numpy(), temp_anchor=ta.numpy(), dt=np.array([float(dt)], np.float32))
        out.append(rec)
    return out


def export_frame(m, work, recs):
    import onnxruntime as ort

    # the graph's fp16-safe projection vs upstream's, through the whole model, on every frame
    for k, rec in enumerate(recs):
        temporal = "dt" in rec
        g = FrameGraph(m, temporal).eval()
        args = [torch.from_numpy(rec["rgb"]), torch.from_numpy(rec["proj"]), torch.from_numpy(rec["proj_n"])]
        if temporal:
            args += [torch.from_numpy(rec["dt"]), torch.from_numpy(rec["temp_feat"]), torch.from_numpy(rec["temp_anchor"])]
        with torch.no_grad():
            tout = g(*args)
        print(f"frame {k}: graph (torch) vs Runner max abs "
              f"{['%.1e' % float(np.abs(a.numpy() - b).max()) for a, b in zip(tout[:3], rec['ref'])]}")
    for name, temporal in (("frame_first", False), ("frame_temp", True)):
        g = FrameGraph(m, temporal).eval()
        rec = recs[1] if temporal else recs[0]
        args = [torch.from_numpy(rec["rgb"]), torch.from_numpy(rec["proj"]), torch.from_numpy(rec["proj_n"])]
        names = ["rgb", "proj", "proj_n"]
        if temporal:
            args += [torch.from_numpy(rec["dt"]), torch.from_numpy(rec["temp_feat"]), torch.from_numpy(rec["temp_anchor"])]
            names += ["dt", "temp_feat", "temp_anchor"]
        with torch.no_grad():
            tout = g(*args)
        path = work / f"{name}.onnx"
        torch.onnx.export(g, tuple(args), str(path), input_names=names, output_names=["cls", "box", "quality", "feat"],
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        oout = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})
        d = [float(np.abs(a.numpy() - b).max()) for a, b in zip(tout, oout)]
        r = [float(np.abs(a.numpy() - b).max()) for a, b in zip(tout[:3], rec["ref"])]
        print(f"{name}: ORT vs torch max abs {['%.1e' % x for x in d]}; graph vs Runner {['%.1e' % x for x in r]}")
        import onnx

        from onnxsim import simplify

        sm, ok = simplify(onnx.load(str(path)))
        assert ok
        u8_input_as_dq(sm)
        onnx.save(sm, str(work / f"{name}.sim.onnx"))
        sess = ort.InferenceSession(str(work / f"{name}.sim.onnx"), providers=["CPUExecutionProvider"])
        o2 = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})
        print(f"  simplified: {len(sm.graph.node)} nodes, vs torch {['%.1e' % float(np.abs(a.numpy() - b).max()) for a, b in zip(tout, o2)]}")


def u8_input_as_dq(model):
    """Cast(uint8 rgb -> float) -> DequantizeLinear(rgb, 1.0, 0): the same values, but QNN reads a
    uint8 graph input as a quantized tensor and needs its scale / zero point. With a plain Cast the
    fp16 HTP graph's FPN outputs came out at cos 0.03 - 0.15 vs ORT CPU."""
    from onnx import helper, numpy_helper

    for i, n in enumerate(model.graph.node):
        if n.op_type == "Cast" and n.input[0] == "rgb":
            model.graph.initializer.extend([numpy_helper.from_array(np.array(1.0, np.float32), "rgb_scale"),
                                            numpy_helper.from_array(np.array(0, np.uint8), "rgb_zp")])
            model.graph.node.remove(n)
            model.graph.node.insert(i, helper.make_node("DequantizeLinear", ["rgb", "rgb_scale", "rgb_zp"],
                                                        list(n.output), name="rgb_dq"))
            return
    raise ValueError("no Cast of the rgb input")


def write_inputs(work, recs):
    """<work>/in/<k>/: the graph inputs as .bin + a qnn_run_multi manifest, and the torch outputs."""
    for k, rec in enumerate(recs):
        d = work / "in" / str(k)
        d.mkdir(parents=True, exist_ok=True)
        items = [("rgb", "u8", rec["rgb"]), ("proj", "f32", rec["proj"]), ("proj_n", "f32", rec["proj_n"])]
        if "dt" in rec:
            items += [("dt", "f32", rec["dt"]), ("temp_feat", "f32", rec["temp_feat"]),
                      ("temp_anchor", "f32", rec["temp_anchor"])]
        with open(d / "manifest.txt", "w") as fh:
            for n, dt, a in items:
                a.tofile(d / f"{n}.bin")
                fh.write(f"{n} {dt} {d / (n + '.bin')} {','.join(map(str, a.shape))}\n")
        pickle.dump({"ref": rec["ref"], "gt": rec["gt"], "metas": rec["metas"]}, open(d / "ref.pkl", "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=["frame", "inputs"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    m = Sparse4D().load_official(a.ckpt).eval()
    recs = frame_inputs(m, a.ckpt, work)
    if a.piece == "frame":
        export_frame(m, work, recs)
    else:
        write_inputs(work, recs)


if __name__ == "__main__":
    main()
