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


def msda_l1_rank5(value, hw, loc, w):
    """Single-level multi-scale deformable attention using only rank<=5 tensors.

    value (bs, H*W, m, d); loc (bs, nq, m, P, 2) in [0,1]; w (bs, nq, m, P).
    Same math as mmcv's multi_scale_deformable_attn_pytorch for L=1, but it never materializes
    the (bs, nq, m, L, P, 2) rank-6 layout that the HTP rejects."""
    h, wd = hw
    bs, _, m, d = value.shape
    nq, p = loc.shape[1], loc.shape[3]
    v = value.permute(0, 2, 3, 1).reshape(bs * m, d, h, wd)
    grid = (2 * loc - 1).permute(0, 2, 1, 3, 4).reshape(bs * m, nq, p, 2)
    s = torch.nn.functional.grid_sample(v, grid, mode="bilinear", padding_mode="zeros",
                                        align_corners=False)  # (bs*m, d, nq, p)
    a = w.permute(0, 2, 1, 3).reshape(bs * m, 1, nq, p)
    return (s * a).sum(-1).reshape(bs, m * d, nq).transpose(1, 2)


class EncoderLayer(nn.Module):
    """TSA -> norm -> SCA -> norm -> FFN -> norm (BEVFormerLayer operation_order)."""

    def __init__(self, rank5: bool = False):
        super().__init__()
        self.rank5 = rank5
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
        if self.rank5:
            return self.forward_rank5(q, prev_bev, img_value, ref_2d, ref_cam, bev_mask)
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

    def forward_rank5(self, q, prev_bev, img_value, ref_2d, ref_cam, bev_mask):
        """Same layer, every intermediate tensor at rank <= 5 (HTP's limit)."""
        m, d, pt, ps = self.m, self.d, TSA_POINTS, SCA_POINTS
        v = self.tsa_value(torch.cat([prev_bev, q], 0)).reshape(2, NQ, m, d)
        qq = torch.cat([prev_bev, q], -1)
        off = self.tsa_offsets(qq).reshape(NQ, m, 2, pt * 2).permute(2, 0, 1, 3).reshape(2, NQ, m, pt, 2)
        w = torch.softmax(self.tsa_weights(qq).reshape(NQ, m, 2, pt), -1).permute(2, 0, 1, 3)
        loc = ref_2d.reshape(1, NQ, 1, 1, 2) + off / torch.tensor([BEV_W, BEV_H], dtype=off.dtype)
        t = msda_l1_rank5(v, (BEV_H, BEV_W), loc, w)
        q = self.norm1(q + self.tsa_out(t.mean(0, keepdim=True)))
        v = self.sca_value(img_value).reshape(NUM_CAMS, FH * FW, m, d)
        off = self.sca_offsets(q).reshape(1, NQ, m, ps, 2)
        w = torch.softmax(self.sca_weights(q).reshape(1, NQ, m, ps), -1).expand(NUM_CAMS, -1, -1, -1)
        r = ref_cam.reshape(NUM_CAMS, NQ, 1, Z_ANCHORS, 2).expand(-1, -1, m, -1, -1)
        # anchor z's reference feeds points [z*k, z*k+k) (k = ps // Z_ANCHORS), as in the
        # custom-op variant; an index gather keeps this at rank 5.
        idx = torch.arange(ps) // (ps // Z_ANCHORS)
        r = r[:, :, :, idx]  # (C, NQ, m, ps, 2)
        loc = r + off / torch.tensor([FW, FH], dtype=off.dtype)
        s = msda_l1_rank5(v, (FH, FW), loc, w)
        msk = bev_mask.reshape(NUM_CAMS, NQ, 1)
        s = (s * msk).sum(0, keepdim=True) / msk.sum(0, keepdim=True).clamp(min=1.0)
        q = self.norm2(q + self.sca_out(s))
        q = self.norm3(q + self.ffn2(torch.relu(self.ffn1(q))))
        return q


class Encoder(nn.Module):
    def __init__(self, layers: int, rank5: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(rank5) for _ in range(layers))

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
    ap.add_argument("--rank5", action="store_true",
                    help="rank<=5 deformable attention (HTP) instead of the mmdeploy custom op path")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = Encoder(a.layers, a.rank5).eval()
    if a.rank5:  # same weights as the custom-op variant, to cross-check the two formulations
        torch.manual_seed(0)
        ref_model = Encoder(a.layers, False).eval()
        model.load_state_dict(ref_model.state_dict())
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
        if a.rank5:
            ref_cop = ref_model(*tins).numpy()
            print(f"rank5 vs custom-op formulation (torch): max abs diff {np.abs(ref - ref_cop).max():.3e}")
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, tins, buf, opset_version=17, dynamo=False,
                          input_names=list(ins), output_names=["bev_embed"])
    raw = onnx.load_model_from_string(buf.getvalue())
    import onnxsim

    sim, ok = onnxsim.simplify(raw, extra_optimizers=["rewrite_msdeformattn_to_gridsample"])
    assert ok
    name = os.path.join(a.out, f"bevformer_tiny_enc{a.layers}{'_rank5' if a.rank5 else ''}.onnx")
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
