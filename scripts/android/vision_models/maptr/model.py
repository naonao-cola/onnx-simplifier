"""MapTR-tiny (R50, BEVFormer encoder) in plain PyTorch, loadable from the official checkpoint.

Rebuilt from hustvl/MapTR (branch maptrv2, projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py,
projects/mmdet3d_plugin/{maptr,bevformer}/) without mmcv/mmdet/mmdet3d. MapTRv2's own nuScenes
checkpoints are not released (README: "WIP"), so this is MapTR (v1) with the BEVFormer-style
encoder -- the published variant whose encoder and decoder attention map onto the HVX MSDA kernel
(../../msda_hvx/). The pieces follow ../bevformer_tiny/model.py (same encoder/decoder code upstream,
MapTR shapes); they are copied, not imported, because that file's shapes are module constants.

Differences from BEVFormer-tiny, all from the config / MapTR code:
  * BEV grid 200 (y) x 100 (x) over pc_range x [-15, 15] m, y [-30, 30] m, z [-2, 2] m (20000
    queries); 1 encoder layer; SCA: 8 points over 4 pillar anchors (Z = 4 m).
  * video_test_mode=False: prev_bev is always None, so the TSA value queue is [q, q] and
    MapTR.forward_test zeroes can_bus[:3] and can_bus[-1]: no temporal state at all.
  * decoder: 50 map instances x 20 points = 1000 queries, query = pts_embedding + instance_embedding
    ("instance_pts"); 2-D reference points refined by every layer's reg branch (code_size 2).
  * head: class logits (3: divider, ped_crossing, boundary) from the mean over each instance's 20
    point features; the points are the last layer's refined references (sigmoid), denormalized
    to metres by `decode()`.

Pieces (plain tensor inputs, each exports to ONNX on its own):
  Backbone  ResNet-50 + FPN (1 level): img (6, 3, 480, 800) normalized -> feats (6, 256, 15, 25)
  Encoder   feats, can_bus (18,), ref_cam (6, 20000, 4, 2), bev_mask (6, 20000, 4) -> bev (20000, 256)
  Decoder   bev -> cls (50, 3) logits, pts (50, 20, 2) normalized [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

EMBED, HEADS, FFN = 256, 8, 512
BEV_H, BEV_W = 200, 100
NQ = BEV_H * BEV_W
NUM_CAMS = 6
FH, FW = 15, 25
Z_ANCHORS = 4
SCA_POINTS, TSA_POINTS, DEC_POINTS = 8, 4, 4
PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
NUM_VEC, NUM_PTS, NUM_CLASSES = 50, 20, 3
NUM_QUERY = NUM_VEC * NUM_PTS
CLASSES = ["divider", "ped_crossing", "boundary"]
POST_CENTER_RANGE = [-20, -35, -20, -35, 20, 35, 20, 35]
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]


def msda_rank5(value, hw, loc, w):
    """Single-level MSDA. value (B, H*W, M, D); loc (B, Q, M, P, 2) in [0, 1]; w (B, Q, M, P)."""
    h, wd = hw
    b, _, m, d = value.shape
    q, p = loc.shape[1], loc.shape[3]
    v = value.reshape(b, h * wd, m * d).transpose(1, 2).reshape(b * m, d, h, wd)
    grid = (2 * loc - 1).permute(0, 2, 1, 3, 4).reshape(b * m, q, p, 2)
    s = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    a = w.permute(0, 2, 1, 3).reshape(b * m, 1, q, p)
    return (s * a).sum(-1).reshape(b, m * d, q).transpose(1, 2)


def msda_mmcv_reference(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Verbatim port of mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision

        r = torchvision.models.resnet50(weights=None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.lateral = nn.Conv2d(2048, EMBED, 1)
        self.fpn = nn.Conv2d(EMBED, EMBED, 3, padding=1)

    def forward(self, img):
        return self.fpn(self.lateral(self.body(img)))


class TemporalSelfAttention(nn.Module):
    """prev_bev is always None in MapTR's test mode: the value queue is [q, q]."""

    def __init__(self):
        super().__init__()
        p = TSA_POINTS
        self.sampling_offsets = nn.Linear(2 * EMBED, 2 * HEADS * p * 2)
        self.attention_weights = nn.Linear(2 * EMBED, 2 * HEADS * p)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def offsets_weights(self, query, bev_pos):
        """-> off (Q, M, 2, P, 2) in BEV cells, attw (Q, M, 2, P) softmaxed per queue frame."""
        m, p = HEADS, TSA_POINTS
        qcat = torch.cat([query, query + bev_pos], -1)  # upstream: cat([value[:bs], query + pos])
        off = self.sampling_offsets(qcat).reshape(NQ, m, 2, p, 2)
        w = torch.softmax(self.attention_weights(qcat).reshape(NQ, m, 2, p), -1)
        return off, w

    def forward(self, query, bev_pos, ref, reference=False):
        m, p = HEADS, TSA_POINTS
        off, w = self.offsets_weights(query, bev_pos)
        v = self.value_proj(query).reshape(1, NQ, m, EMBED // m).expand(2, -1, -1, -1)
        loc = ref.reshape(1, NQ, 1, 1, 2) + off.permute(2, 0, 1, 3, 4) / torch.tensor([BEV_W, BEV_H], dtype=off.dtype)
        wq = w.permute(2, 0, 1, 3)
        if reference:
            out = msda_mmcv_reference(v, [(BEV_H, BEV_W)], loc.reshape(2, NQ, m, 1, p, 2), wq.reshape(2, NQ, m, 1, p))
        else:
            out = msda_rank5(v, (BEV_H, BEV_W), loc, wq)
        return self.output_proj(out.mean(0)) + query


class SpatialCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        p = SCA_POINTS
        self.sampling_offsets = nn.Linear(EMBED, HEADS * p * 2)
        self.attention_weights = nn.Linear(EMBED, HEADS * p)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def forward(self, query, img_value, ref_cam, bev_mask, reference=False):
        """query (Q, E); img_value (C, HW, E); ref_cam (C, Q, Z, 2); bev_mask (C, Q, Z)."""
        m, p = HEADS, SCA_POINTS
        v = self.value_proj(img_value).reshape(NUM_CAMS, FH * FW, m, EMBED // m)
        off = self.sampling_offsets(query).reshape(NQ, m, p, 2) / torch.tensor([FW, FH], dtype=query.dtype)
        w = torch.softmax(self.attention_weights(query).reshape(NQ, m, p), -1)
        vis = (bev_mask.sum(-1) > 0).to(query.dtype)  # (C, Q)
        slots = torch.zeros_like(query)
        if reference:  # upstream: per camera, rebatch the visible queries (nonzero), MSDeformableAttention3D
            for c in range(NUM_CAMS):
                idx = vis[c].nonzero().squeeze(-1)
                if idx.numel() == 0:
                    continue
                n = idx.numel()
                r = ref_cam[c, idx][:, None, None, None, :, :]
                o = off[idx].reshape(n, m, 1, p // Z_ANCHORS, Z_ANCHORS, 2)
                loc = (r + o).reshape(1, n, m, 1, p, 2)
                out = msda_mmcv_reference(v[c:c + 1], [(FH, FW)], loc, w[idx].reshape(1, n, m, 1, p))
                slots[idx] += out[0]
        else:  # all queries per camera, masked -- same outputs (point p uses anchor p % Z)
            for c in range(NUM_CAMS):  # per camera: the (M, Q, P) sample tensor is 400 MB at Q=20000
                ref = ref_cam[c].repeat(1, p // Z_ANCHORS, 1)  # (Q, P, 2)
                loc = (ref[:, None] + off)[None]  # (1, Q, M, P, 2)
                out = msda_rank5(v[c:c + 1], (FH, FW), loc, w[None])[0]
                slots += out * vis[c, :, None]
        return self.output_proj(slots / vis.sum(0).clamp(min=1.0)[:, None]) + query


class FFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED, FFN)
        self.fc2 = nn.Linear(FFN, EMBED)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))


def ref_2d() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(0.5, BEV_H - 0.5, BEV_H), torch.linspace(0.5, BEV_W - 0.5, BEV_W),
                            indexing="ij")
    return torch.stack([xs.reshape(-1) / BEV_W, ys.reshape(-1) / BEV_H], -1)


class Encoder(nn.Module):
    """1 BEVFormer layer: TSA -> norm -> SCA -> norm -> FFN -> norm."""

    def __init__(self):
        super().__init__()
        self.tsa = TemporalSelfAttention()
        self.sca = SpatialCrossAttention()
        self.ffn = FFNBlock()
        self.norms = nn.ModuleList(nn.LayerNorm(EMBED) for _ in range(3))
        self.bev_embedding = nn.Parameter(torch.zeros(NQ, EMBED))
        self.row_embed = nn.Parameter(torch.zeros(BEV_H, EMBED // 2))
        self.col_embed = nn.Parameter(torch.zeros(BEV_W, EMBED // 2))
        self.level_embeds = nn.Parameter(torch.zeros(4, EMBED))
        self.cams_embeds = nn.Parameter(torch.zeros(NUM_CAMS, EMBED))
        self.can_bus_mlp = nn.Sequential(nn.Linear(18, EMBED // 2), nn.ReLU(), nn.Linear(EMBED // 2, EMBED), nn.ReLU(),
                                         nn.LayerNorm(EMBED))
        self.register_buffer("ref_2d", ref_2d(), persistent=False)

    def bev_pos(self):
        x = self.col_embed[None].expand(BEV_H, -1, -1)
        y = self.row_embed[:, None].expand(-1, BEV_W, -1)
        return torch.cat([x, y], -1).reshape(NQ, EMBED)

    def img_value(self, feats):
        return feats.flatten(2).transpose(1, 2) + self.cams_embeds[:, None] + self.level_embeds[0]

    def forward(self, feats, can_bus, ref_cam, bev_mask, reference=False):
        q = self.bev_embedding + self.can_bus_mlp(can_bus[None])
        q = self.norms[0](self.tsa(q, self.bev_pos(), self.ref_2d, reference))
        q = self.norms[1](self.sca(q, self.img_value(feats), ref_cam, bev_mask, reference))
        return self.norms[2](self.ffn(q))


class DecoderMSDA(nn.Module):
    def __init__(self):
        super().__init__()
        p = DEC_POINTS
        self.sampling_offsets = nn.Linear(EMBED, HEADS * p * 2)
        self.attention_weights = nn.Linear(EMBED, HEADS * p)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def forward(self, query, query_pos, value, ref, reference=False):
        m, p = HEADS, DEC_POINTS
        q = query + query_pos
        v = self.value_proj(value).reshape(1, NQ, m, EMBED // m)
        off = self.sampling_offsets(q).reshape(1, NUM_QUERY, m, p, 2)
        w = torch.softmax(self.attention_weights(q).reshape(1, NUM_QUERY, m, p), -1)
        loc = ref.reshape(1, NUM_QUERY, 1, 1, 2) + off / torch.tensor([BEV_W, BEV_H], dtype=off.dtype)
        if reference:
            out = msda_mmcv_reference(v, [(BEV_H, BEV_W)], loc.reshape(1, NUM_QUERY, m, 1, p, 2),
                                      w.reshape(1, NUM_QUERY, m, 1, p))
        else:
            out = msda_rank5(v, (BEV_H, BEV_W), loc, w)
        return self.output_proj(out[0]) + query


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(EMBED, HEADS)
        self.cross_attn = DecoderMSDA()
        self.ffn = FFNBlock()
        self.norms = nn.ModuleList(nn.LayerNorm(EMBED) for _ in range(3))

    def forward(self, q, qpos, bev, ref, reference=False):
        qk = (q + qpos)[:, None]
        q = self.norms[0](q + self.self_attn(qk, qk, q[:, None], need_weights=False)[0][:, 0])
        q = self.norms[1](self.cross_attn(q, qpos, bev, ref, reference))
        return self.norms[2](self.ffn(q))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class Decoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(layers))
        self.instance_embedding = nn.Parameter(torch.zeros(NUM_VEC, 2 * EMBED))
        self.pts_embedding = nn.Parameter(torch.zeros(NUM_PTS, 2 * EMBED))
        self.reference_points = nn.Linear(EMBED, 2)
        self.reg_branches = nn.ModuleList(
            nn.Sequential(nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, 2))
            for _ in range(layers))
        self.cls_last = nn.Sequential(nn.Linear(EMBED, EMBED), nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED),
                                      nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, NUM_CLASSES))

    def forward(self, bev, reference=False):
        emb = (self.pts_embedding[None] + self.instance_embedding[:, None]).reshape(NUM_QUERY, 2 * EMBED)
        qpos, q = emb[:, :EMBED], emb[:, EMBED:]
        ref = self.reference_points(qpos).sigmoid()
        for i, layer in enumerate(self.layers):
            q = layer(q, qpos, bev, ref, reference)
            ref = (self.reg_branches[i](q) + inverse_sigmoid(ref)).sigmoid()
        cls = self.cls_last(q.reshape(NUM_VEC, NUM_PTS, EMBED).mean(1))
        return cls, ref.reshape(NUM_VEC, NUM_PTS, 2)


def decode(cls, pts, max_num=50):
    """MapTRNMSFreeCoder.decode_single (score_threshold None): top-50 over 50x3 sigmoid scores ->
    (pts (K, 20, 2) in metres (x right, y forward), scores (K,), labels (K,))."""
    scores, idx = cls.sigmoid().reshape(-1).topk(max_num)
    labels = idx % NUM_CLASSES
    p = pts[idx // NUM_CLASSES].clone()
    p[..., 0] = p[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    p[..., 1] = p[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    # post_center_range filter on the points' min/max box (always true for points inside pc_range)
    box = torch.cat([p.min(1).values, p.max(1).values], -1)
    lo, hi = torch.tensor(POST_CENTER_RANGE[:4], dtype=p.dtype), torch.tensor(POST_CENTER_RANGE[4:], dtype=p.dtype)
    keep = ((box >= lo) & (box <= hi)).all(1)
    return p[keep], scores[keep], labels[keep]


def reference_points_cam(lidar2img: torch.Tensor, img_hw=(480, 800), clamp=(-1.0, 2.0)):
    """BEVFormerEncoder.get_reference_points('3d') + point_sampling on the host, MapTR's pc_range
    (Z = 4 m). -> ref_cam (6, Q, Z, 2), bev_mask (6, Q, Z). The clamp to [-1, 2] keeps points
    behind a camera (|xy| ~ 1e7) out of fp16 overflow; it is exact because such points are
    outside the image and SCA offsets move a point by < 1 image width (validate.py checks)."""
    zd = PC_RANGE[5] - PC_RANGE[2]
    zs = torch.linspace(0.5, zd - 0.5, Z_ANCHORS).view(-1, 1, 1).expand(Z_ANCHORS, BEV_H, BEV_W) / zd
    xs = torch.linspace(0.5, BEV_W - 0.5, BEV_W).view(1, 1, BEV_W).expand(Z_ANCHORS, BEV_H, BEV_W) / BEV_W
    ys = torch.linspace(0.5, BEV_H - 0.5, BEV_H).view(1, BEV_H, 1).expand(Z_ANCHORS, BEV_H, BEV_W) / BEV_H
    r = torch.stack((xs, ys, zs), -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1).clone()
    for i in range(3):
        r[..., i] = r[..., i] * (PC_RANGE[i + 3] - PC_RANGE[i]) + PC_RANGE[i]
    r = torch.cat([r, torch.ones_like(r[..., :1])], -1)
    cam = torch.einsum("cij,zqj->czqi", lidar2img.float(), r)
    eps = 1e-5
    mask = cam[..., 2:3] > eps
    xy = cam[..., 0:2] / torch.maximum(cam[..., 2:3], torch.ones_like(cam[..., 2:3]) * eps)
    xy[..., 0] /= img_hw[1]
    xy[..., 1] /= img_hw[0]
    mask = mask & (xy[..., 1:2] > 0) & (xy[..., 1:2] < 1) & (xy[..., 0:1] < 1) & (xy[..., 0:1] > 0)
    mask = torch.nan_to_num(mask.float())
    if clamp is not None:
        xy = xy.clamp(*clamp)
    return xy.permute(0, 2, 1, 3).contiguous(), mask[..., 0].permute(0, 2, 1).contiguous()


def test_can_bus(can_bus_abs) -> torch.Tensor:
    """MapTR.forward_test with prev_bev None: can_bus[:3] = 0, can_bus[-1] = 0."""
    c = torch.tensor(can_bus_abs, dtype=torch.float32).clone()
    c[:3] = 0
    c[-1] = 0
    return c


# ---- loading ---------------------------------------------------------------------------------
def name_map(sd: dict) -> dict:
    m = {}
    tv = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    for k in sd:
        if k.startswith("img_backbone."):
            head, _, tail = k[len("img_backbone."):].partition(".")
            m[k] = ("backbone", f"body.{tv.index(head)}.{tail}")
    for a, b in (("lateral_convs", "lateral"), ("fpn_convs", "fpn")):
        for x in ("weight", "bias"):
            m[f"img_neck.{a}.0.conv.{x}"] = ("backbone", f"{b}.{x}")
    h, t = "pts_bbox_head", "pts_bbox_head.transformer"
    e = f"{t}.encoder.layers.0"
    for x in ("weight", "bias"):
        for a in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
            m[f"{e}.attentions.0.{a}.{x}"] = ("encoder", f"tsa.{a}.{x}")
        for a in ("sampling_offsets", "attention_weights", "value_proj"):
            m[f"{e}.attentions.1.deformable_attention.{a}.{x}"] = ("encoder", f"sca.{a}.{x}")
        m[f"{e}.attentions.1.output_proj.{x}"] = ("encoder", f"sca.output_proj.{x}")
        m[f"{e}.ffns.0.layers.0.0.{x}"] = ("encoder", f"ffn.fc1.{x}")
        m[f"{e}.ffns.0.layers.1.{x}"] = ("encoder", f"ffn.fc2.{x}")
        for i in range(3):
            m[f"{e}.norms.{i}.{x}"] = ("encoder", f"norms.{i}.{x}")
        for s, d in {"0": "0", "2": "2", "norm": "4"}.items():
            m[f"{t}.can_bus_mlp.{s}.{x}"] = ("encoder", f"can_bus_mlp.{d}.{x}")
        m[f"{t}.reference_points.{x}"] = ("decoder", f"reference_points.{x}")
    m[f"{h}.bev_embedding.weight"] = ("encoder", "bev_embedding")
    m[f"{h}.positional_encoding.row_embed.weight"] = ("encoder", "row_embed")
    m[f"{h}.positional_encoding.col_embed.weight"] = ("encoder", "col_embed")
    m[f"{t}.level_embeds"] = ("encoder", "level_embeds")
    m[f"{t}.cams_embeds"] = ("encoder", "cams_embeds")
    m[f"{h}.instance_embedding.weight"] = ("decoder", "instance_embedding")
    m[f"{h}.pts_embedding.weight"] = ("decoder", "pts_embedding")
    for i in range(6):
        d = f"{t}.decoder.layers.{i}"
        m[f"{d}.attentions.0.attn.in_proj_weight"] = ("decoder", f"layers.{i}.self_attn.in_proj_weight")
        m[f"{d}.attentions.0.attn.in_proj_bias"] = ("decoder", f"layers.{i}.self_attn.in_proj_bias")
        for x in ("weight", "bias"):
            m[f"{d}.attentions.0.attn.out_proj.{x}"] = ("decoder", f"layers.{i}.self_attn.out_proj.{x}")
            for a in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
                m[f"{d}.attentions.1.{a}.{x}"] = ("decoder", f"layers.{i}.cross_attn.{a}.{x}")
            m[f"{d}.ffns.0.layers.0.0.{x}"] = ("decoder", f"layers.{i}.ffn.fc1.{x}")
            m[f"{d}.ffns.0.layers.1.{x}"] = ("decoder", f"layers.{i}.ffn.fc2.{x}")
            for j in range(3):
                m[f"{d}.norms.{j}.{x}"] = ("decoder", f"layers.{i}.norms.{j}.{x}")
            for j in (0, 2, 4):
                m[f"{h}.reg_branches.{i}.{j}.{x}"] = ("decoder", f"reg_branches.{i}.{j}.{x}")
    for j in (0, 1, 3, 4, 6):
        for x in ("weight", "bias"):
            m[f"{h}.cls_branches.5.{j}.{x}"] = ("decoder", f"cls_last.{j}.{x}")
    return m


def unused(k: str) -> bool:
    """Checkpoint keys inference never reads: loss weights, aux classifiers of layers 0-4."""
    return (k == "pts_bbox_head.code_weights" or k.endswith("num_batches_tracked")
            or any(k.startswith(f"pts_bbox_head.cls_branches.{i}.") for i in range(5)))


def load_official(path):
    """-> (backbone, encoder, decoder) with the official MapTR-tiny (bevformer encoder) weights."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    pieces = {"backbone": Backbone(), "encoder": Encoder(), "decoder": Decoder()}
    nm = name_map(sd)
    per = {k: {} for k in pieces}
    bad = []
    for k, v in sd.items():
        if k in nm:
            p, d = nm[k]
            per[p][d] = v
        elif not unused(k):
            bad.append(k)
    if bad:
        raise RuntimeError(f"unmapped checkpoint keys: {bad[:10]} ({len(bad)})")
    for name, mod in pieces.items():
        r = mod.load_state_dict(per[name], strict=False)
        missing = [k for k in r.missing_keys if not k.endswith("num_batches_tracked")]
        if missing or r.unexpected_keys:
            raise RuntimeError(f"{name}: missing {missing[:10]} unexpected {r.unexpected_keys[:10]}")
        mod.eval()
    return pieces["backbone"], pieces["encoder"], pieces["decoder"]
