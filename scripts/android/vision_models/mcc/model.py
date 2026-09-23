"""MCC (Multiview Compressive Coding, Wu et al. CVPR 2023) split for deployment.

The upstream model (facebookresearch/MCC, CC BY-NC 4.0 -- not vendored here; point
``MCC_REPO`` at a clone) runs its decoder on ``[197 seen tokens ; Q query points]`` with
a mask that lets seen tokens attend only to seen tokens and every query attend only to
the seen tokens and itself. So, exactly:

* the seen-token stream through the 8 decoder blocks does not depend on the queries:
  run it once per image and keep each block's seen K/V ("KV cache"), and
* each query's decoder pass is attention over ``[K_seen ; k_self]`` -- a static
  query-chunk graph whose cost is linear in the chunk size (upstream builds an
  (197+Q)^2 mask and attends over all of it).

Pieces (all rank <= 4, static shapes):
  ``Encoder``      img (1,3,224,224) normalized, xyz windows (196,64,3), valid (196,64)
                   -> kv (16, 8*? ...) see ``Encoder.forward``
  ``QueryDecoder`` xyz (1,Q,3), k (8,16,197,32), v (8,16,197,32) -> occ logit (1,Q), rgb (1,Q,3)
The XYZ encoder's 8x8 window partition (a rank-6 view upstream) is done on the host.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WIN = 8  # XYZPosEmbed window
SEEN = 197  # 1 cls + 14*14 patches
HEADS = 16
HEAD_DIM = 32
LAYERS = 8
TEMPERATURE = 0.1  # demo.py's color temperature


def upstream(repo=None):
    repo = repo or os.environ.get(
        "MCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/MCC")
    )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if not hasattr(
        np, "float"
    ):  # upstream util/pos_embed.py uses np.float (removed in numpy 1.24)
        np.float = float
    import mcc_model  # noqa: E402

    return mcc_model


def load_mcc(ckpt, repo=None):
    mcc_model = upstream(repo)
    args = SimpleNamespace(drop_path=0.0, regress_color=False, shrink_threshold=10.0)
    model = mcc_model.get_mcc_model(occupancy_weight=1.0, rgb_weight=0.01, args=args)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, missing
    return model.eval()


def shrink(xyz, threshold=10.0):
    """upstream shrink_points_beyond_threshold, torch-traceable (no boolean indexing)."""
    dist = (xyz**2).sum(-1, keepdim=True).sqrt()
    f = threshold * (2.0 - threshold / dist) / dist
    return torch.where(dist > threshold, xyz * f, xyz)


def xyz_windows(seen_xyz):
    """(112,112,3) with non-finite = invalid -> windows (196,64,3), valid (196,64) float.
    Invalid points are set to -100 and shrunk, as upstream prepare_data + forward do."""
    xyz = torch.as_tensor(seen_xyz, dtype=torch.float32).clone()
    valid = torch.isfinite(xyz.sum(-1))
    xyz[~valid] = -100.0
    xyz = shrink(xyz)
    h, w = xyz.shape[:2]
    win = (
        xyz.view(h // WIN, WIN, w // WIN, WIN, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, WIN * WIN, 3)
    )
    val = (
        valid.view(h // WIN, WIN, w // WIN, WIN)
        .permute(0, 2, 1, 3)
        .reshape(-1, WIN * WIN)
    )
    return win.contiguous(), val.float().contiguous()


class Encoder(nn.Module):
    """RGB + XYZ encoders and the decoder's seen-token stream -> per-block seen K/V."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img, xyz_win, valid):
        m = self.m
        # E_RGB
        x = m.patch_embed(img) + m.pos_embed[:, 1:, :]
        x = torch.cat([m.cls_token + m.pos_embed[:, :1, :], x], dim=1)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        # XYZPosEmbed on host-partitioned windows (upstream: boolean-indexed invalid token)
        pe = m.xyz_pos_embed
        emb = pe.pos_embed(xyz_win)
        emb = torch.where(valid[..., None] > 0.5, emb, pe.invalid_xyz_token)
        emb = emb + pe.two_d_pos_embed[:, 1:, :]
        cls = (pe.cls_token + pe.two_d_pos_embed[:, :1, :]).expand(emb.shape[0], -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        for blk in pe.blocks:
            emb = blk(emb)
        y = emb[:, 0][None]  # (1,196,C)
        # E_XYZ
        y = torch.cat([m.cls_token_xyz, y], dim=1)
        for blk in m.blocks_xyz:
            y = blk(y)
        y = m.norm_xyz(y)
        lat = torch.cat([x, y], dim=2)
        # decoder seen stream: seen tokens attend only to seen tokens
        s = m.decoder_embed(lat) + m.decoder_pos_embed
        ks, vs = [], []
        for blk in m.decoder_blocks:
            a = blk.attn
            qkv = (
                a.qkv(blk.norm1(s))
                .reshape(1, SEEN, 3, HEADS, HEAD_DIM)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # (1,16,197,32)
            ks.append(k[0])
            vs.append(v[0])
            att = torch.softmax((q @ k.transpose(-2, -1)) * a.scale, dim=-1)
            o = (att @ v).transpose(1, 2).reshape(1, SEEN, HEADS * HEAD_DIM)
            s = s + a.proj(o)
            s = s + blk.mlp(blk.norm2(s))
        return torch.stack(ks), torch.stack(vs)  # (8,16,197,32) each


class QueryDecoder(nn.Module):
    """A fixed-size chunk of query points against the cached seen K/V."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("levels", torch.linspace(0, 1, 256), persistent=False)

    def forward(self, xyz, k_all, v_all):
        m = self.m
        n = xyz.shape[1]
        x = m.decoder_xyz_pos_embed(shrink(xyz))  # (1,Q,512)
        for i, blk in enumerate(m.decoder_blocks):
            a = blk.attn
            qkv = (
                a.qkv(blk.norm1(x))
                .reshape(1, n, 3, HEADS, HEAD_DIM)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # (1,16,Q,32)
            s_seen = (q @ k_all[i][None].transpose(-2, -1)) * a.scale  # (1,16,Q,197)
            s_self = (q * k).sum(-1, keepdim=True) * a.scale  # (1,16,Q,1)
            p = torch.softmax(torch.cat([s_seen, s_self], dim=-1), dim=-1)
            o = p[..., :SEEN] @ v_all[i][None] + p[..., SEEN:] * v
            o = o.transpose(1, 2).reshape(1, n, HEADS * HEAD_DIM)
            x = x + a.proj(o)
            x = x + blk.mlp(blk.norm2(x))
        pred = m.decoder_pred(m.decoder_norm(x))  # (1,Q,769)
        occ = pred[..., 0]
        logits = pred[..., 1:].reshape(1, n, 3, 256) / TEMPERATURE
        rgb = (torch.softmax(logits, dim=-1) * self.levels).sum(-1)
        return occ, rgb


def grid(granularity, world=3.0):
    """upstream engine_mcc.get_grid (co3d_world_size 3)."""
    n = int(np.ceil(2 * world / granularity))
    ii = torch.arange(n, dtype=torch.float32)
    g = torch.stack(torch.meshgrid(ii, ii, ii, indexing="ij"), dim=-1)
    g = (g - n / 2.0) / ((n / 2.0) / world)
    return g.reshape(1, -1, 3)


def load_demo(repo, name="quest2"):
    """demo.py's main() preprocessing, without pytorch3d: -> img224 (1,3,224,224) normalized,
    seen_xyz (112,112,3) (non-finite = invalid)."""
    import cv2

    d = os.path.join(repo, "demo")
    rgb = cv2.imread(os.path.join(d, f"{name}.jpg"))
    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    h, w = seen_rgb.shape[:2]
    verts = []
    with open(os.path.join(d, f"{name}.obj")) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
    seen_xyz = torch.tensor(verts, dtype=torch.float32).reshape(h, w, 3)
    seg = cv2.imread(os.path.join(d, f"{name}_seg.png"), cv2.IMREAD_UNCHANGED)
    mask = torch.tensor(cv2.resize(seg, (w, h))).bool()
    return prep(seen_rgb, seen_xyz, mask)


def prep(seen_rgb, seen_xyz, mask):
    """seen_rgb (H,W,3) in [0,1], seen_xyz (H,W,3), mask (H,W) bool -> (img224, xyz112)."""
    seen_xyz = seen_xyz.clone()
    seen_xyz[~mask] = float("inf")
    fin = torch.isfinite(seen_xyz.sum(-1))
    seen_xyz = seen_xyz / (seen_xyz[fin].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[fin].mean(axis=0)
    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]
    bottom, right = bottom + 40, right + 40
    top, left = max(top - 40, 0), max(left - 40, 0)
    seen_xyz = seen_xyz[top : bottom + 1, left : right + 1]
    seen_rgb = seen_rgb[top : bottom + 1, left : right + 1]

    def pad(im, value):
        a, b = im.shape[:2]
        if a > b:
            return torch.cat([im, torch.zeros((a, a - b, im.shape[2])) + value], dim=1)
        return torch.cat([im, torch.zeros((b - a, b, im.shape[2])) + value], dim=0)

    seen_xyz, seen_rgb = pad(seen_xyz, float("inf")), pad(seen_rgb, 0)
    img = F.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )
    xyz = F.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    )
    img = F.interpolate(img, scale_factor=224.0 / 800.0, mode="bilinear")
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return (img - mean) / std, xyz[0].permute(1, 2, 0)
