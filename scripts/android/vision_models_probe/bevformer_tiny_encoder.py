#!/usr/bin/env python3
"""Export one BEVFormer-tiny encoder layer at its real dimensions, for a QNN partition report.

There is no public, deployable BEVFormer ONNX on the Hub, and exporting the official model needs
mmdet3d + mmcv + the BEVFormer repo. The encoder is the part of BEVFormer that nothing in this
project has run yet (the image backbone is a ResNet-50, the same kind of network the Mask R-CNN
work already runs on the HTP), so this script rebuilds one encoder layer with BEVFormer-tiny's
real sizes, random weights, and the deploy-style static formulation:

  * BEV grid 50x50 (2500 queries), embed 256, 8 heads, one image level (stride 32 of 480x800 ->
    15x25), 6 cameras, 4 pillar points x 2 = 8 sampling points for spatial cross-attention,
    4 points for temporal self-attention (bevformer_tiny.py config values).
  * Deformable attention goes through this repo's own path: the mmdeploy custom op
    `MMCVMultiScaleDeformableAttention` (tests/_bev_torch_custom_ops.py), then onnxsim's
    `rewrite_msdeformattn_to_gridsample` pass, which is what a real mmdeploy export would need.
  * Upstream BEVFormer selects, per camera, only the BEV queries that project into it
    (`bev_mask ... .nonzero()`), which gives a data-dependent query count. Deployment exports
    replace that with all queries per camera plus a mask; this script does the same, with
    `bev_mask` and the projected reference points as inputs (they're cheap host-side geometry
    from lidar2img).

Writes `bevformer_tiny_enc<N>.onnx` (N layers, default 1) and its input `.bin` files + a
`manifest.txt` for `htp_exploration/qnn_shell/run_multi.sh`.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings

import numpy as np
import onnx
import torch
from torch import nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests"))
from _bev_torch_custom_ops import MSDeformAttnFunction  # noqa: E402

NUM_CAMS, EMBED, HEADS = 6, 256, 8
FH, FW = 15, 25  # stride-32 level of a 480x800 image
BEV_H = BEV_W = 50
NQ = BEV_H * BEV_W
Z_ANCHORS, SCA_POINTS, TSA_POINTS = 4, 8, 4
FFN = 512


class EncoderLayer(nn.Module):
    """TSA -> norm -> SCA -> norm -> FFN -> norm (BEVFormerLayer operation_order)."""

    def __init__(self):
        super().__init__()
        m, d = HEADS, EMBED // HEADS
        self.m, self.d = m, d
        # temporal self-attention: queries attend to [prev_bev, current] (2 "frames")
        self.tsa_value = nn.Linear(EMBED, EMBED)
        self.tsa_offsets = nn.Linear(2 * EMBED, 2 * m * TSA_POINTS * 2)
        self.tsa_weights = nn.Linear(2 * EMBED, 2 * m * TSA_POINTS)
        self.tsa_out = nn.Linear(EMBED, EMBED)
        self.norm1 = nn.LayerNorm(EMBED)
        # spatial cross-attention over 6 cameras, one level
        self.sca_value = nn.Linear(EMBED, EMBED)
        self.sca_offsets = nn.Linear(EMBED, m * SCA_POINTS * 2)
        self.sca_weights = nn.Linear(EMBED, m * SCA_POINTS)
        self.sca_out = nn.Linear(EMBED, EMBED)
        self.norm2 = nn.LayerNorm(EMBED)
        self.ffn1 = nn.Linear(EMBED, FFN)
        self.ffn2 = nn.Linear(FFN, EMBED)
        self.norm3 = nn.LayerNorm(EMBED)
        self.register_buffer("ss_img", torch.tensor([[FH, FW]], dtype=torch.int64))
        self.register_buffer("ls_img", torch.tensor([0], dtype=torch.int64))
        self.register_buffer("ss_bev", torch.tensor([[BEV_H, BEV_W]], dtype=torch.int64))
        self.register_buffer("ls_bev", torch.tensor([0], dtype=torch.int64))

    def forward(self, q, prev_bev, img_value, ref_2d, ref_cam, bev_mask):
        m, d = self.m, self.d
        # --- temporal self-attention (bs folded to 2: prev + current) ---
        v = self.tsa_value(torch.cat([prev_bev, q], 0)).reshape(2, NQ, m, d)
        qq = torch.cat([prev_bev, q], -1)  # (1, NQ, 512)
        off = self.tsa_offsets(qq).reshape(NQ, m, 2, 1, TSA_POINTS, 2).permute(2, 0, 1, 3, 4, 5)
        w = torch.softmax(self.tsa_weights(qq).reshape(NQ, m, 2, TSA_POINTS), -1)
        w = w.permute(2, 0, 1, 3).reshape(2, NQ, m, 1, TSA_POINTS)
        loc = ref_2d.reshape(1, NQ, 1, 1, 1, 2) + off / torch.tensor([BEV_W, BEV_H], dtype=off.dtype)
        t = MSDeformAttnFunction.apply(v, self.ss_bev, self.ls_bev, loc, w)  # (2, NQ, 256)
        q = self.norm1(q + self.tsa_out(t.mean(0, keepdim=True)))
        # --- spatial cross-attention, all queries per camera + mask (deploy formulation) ---
        v = self.sca_value(img_value).reshape(NUM_CAMS, FH * FW, m, d)
        off = self.sca_offsets(q).reshape(1, NQ, m, 1, SCA_POINTS, 2).expand(NUM_CAMS, -1, -1, -1, -1, -1)
        w = torch.softmax(self.sca_weights(q).reshape(1, NQ, m, SCA_POINTS), -1)
        w = w.reshape(1, NQ, m, 1, SCA_POINTS).expand(NUM_CAMS, -1, -1, -1, -1)
        # each of the 4 pillar reference points gets SCA_POINTS/Z_ANCHORS offsets
        ref = ref_cam.reshape(NUM_CAMS, NQ, 1, 1, Z_ANCHORS, 1, 2).expand(
            -1, -1, m, 1, -1, SCA_POINTS // Z_ANCHORS, -1).reshape(NUM_CAMS, NQ, m, 1, SCA_POINTS, 2)
        loc = ref + off / torch.tensor([FW, FH], dtype=off.dtype)
        s = MSDeformAttnFunction.apply(v, self.ss_img, self.ls_img, loc, w)  # (6, NQ, 256)
        msk = bev_mask.reshape(NUM_CAMS, NQ, 1)
        s = (s * msk).sum(0, keepdim=True) / msk.sum(0, keepdim=True).clamp(min=1.0)
        q = self.norm2(q + self.sca_out(s))
        q = self.norm3(q + self.ffn2(torch.relu(self.ffn1(q))))
        return q


class Encoder(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(layers))

    def forward(self, bev_queries, prev_bev, img_feats, ref_2d, ref_cam, bev_mask):
        img_value = img_feats.flatten(2).transpose(1, 2)  # (6, 375, 256)
        q = bev_queries
        for layer in self.layers:
            q = layer(q, prev_bev, img_value, ref_2d, ref_cam, bev_mask)
        return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = Encoder(a.layers).eval()
    rng = np.random.default_rng(0)
    ins = {
        "bev_queries": rng.standard_normal((1, NQ, EMBED)).astype(np.float32) * 0.1,
        "prev_bev": rng.standard_normal((1, NQ, EMBED)).astype(np.float32) * 0.1,
        "img_feats": rng.standard_normal((NUM_CAMS, EMBED, FH, FW)).astype(np.float32),
        "ref_2d": rng.uniform(0, 1, (1, NQ, 1, 2)).astype(np.float32),
        "ref_cam": rng.uniform(0, 1, (NUM_CAMS, NQ, Z_ANCHORS, 2)).astype(np.float32),
        # each BEV cell is visible from ~1-2 cameras, like the real 6-camera ring
        "bev_mask": (rng.uniform(0, 1, (NUM_CAMS, NQ)) < 0.25).astype(np.float32),
    }
    tins = tuple(torch.from_numpy(v) for v in ins.values())
    with torch.no_grad():
        ref = model(*tins).numpy()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, tins, buf, opset_version=17, dynamo=False,
                          input_names=list(ins), output_names=["bev_embed"])
    raw = onnx.load_model_from_string(buf.getvalue())
    import onnxsim

    sim, ok = onnxsim.simplify(raw, extra_optimizers=["rewrite_msdeformattn_to_gridsample"])
    assert ok
    name = os.path.join(a.out, f"bevformer_tiny_enc{a.layers}.onnx")
    onnx.save(sim, name)
    import onnxruntime as ort

    got = ort.InferenceSession(name, providers=["CPUExecutionProvider"]).run(None, ins)[0]
    print(f"{name}: {len(sim.graph.node)} nodes, max|ORT-torch|={np.abs(got - ref).max():.3e}")
    with open(os.path.join(a.out, "manifest.txt"), "w") as f:
        for k, v in ins.items():
            p = os.path.join(a.out, f"in_{k}.bin")
            v.tofile(p)
            f.write(f"{k} f32 {p} {','.join(map(str, v.shape))}\n")
    from collections import Counter

    print(sorted(Counter(n.op_type for n in sim.graph.node).items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
