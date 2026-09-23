"""BEVFormer-tiny in plain PyTorch, loadable from the official checkpoint, exportable in pieces.

Rebuilt from the upstream code (fundamentalvision/BEVFormer, projects/configs/bevformer/
bevformer_tiny.py and projects/mmdet3d_plugin/bevformer/) without mmcv/mmdet/mmdet3d: the
checkpoint's state_dict loads through `load_official()` with an explicit name map, and every
deformable-attention call goes through `msda_rank5`, a single-level multi-scale deformable
attention that never builds a tensor of rank > 5 (the HTP rejects rank-6 ops; see
../../vision_models_plan.md). `msda_mmcv_reference` is a verbatim copy of mmcv's pure-PyTorch
`multi_scale_deformable_attn_pytorch` (the CPU path upstream itself uses), kept to check
`msda_rank5` against.

Pieces (each an nn.Module with plain tensor inputs, so each exports to ONNX on its own):
  Backbone   ResNet-50 (torchvision layout = mmdet's) + FPN (1 level, 2048 -> 256)
             img (N, 3, 480, 800), normalized -> feats (N, 256, 15, 25)
  Encoder    3 BEVFormer layers: temporal self-attn -> norm -> spatial cross-attn -> norm -> FFN
             -> norm. Inputs are per-frame tensors the host computes from calibration/ego motion
             (upstream computes the same things inline from img_metas with numpy):
               feats (6, 256, 15, 25), prev_bev (2500, 256), has_prev (1,), shift (2,),
               can_bus (18,), ref_cam (6, 2500, 4, 2), bev_mask (6, 2500, 4)
             -> bev_embed (2500, 256)   (the next frame's prev_bev, kept on the host)
  Decoder    6 layers (MHA self-attn, deformable cross-attn on the BEV grid, FFN) + the
             last layer's cls/reg branches -> cls_scores (900, 10) logits, bbox_preds (900, 10)
             (only the last layer is decoded at inference, so the other five heads are skipped)
The NMS-free decode (TopK 300 over 900x10 sigmoid scores, denormalize, center-range filter) is
`decode()`, cheap host-side post-processing.

Upstream behaviour kept, on purpose:
  * spatial cross-attention: upstream gathers, per camera, only the BEV queries that project into
    it (`.nonzero()`, a data-dependent count). Here every camera attends with all 2500 queries and
    the per-camera visibility mask (any pillar point visible) zeroes the rest before the same
    average; the outputs are identical (checked by `reference=True`, which follows upstream's
    rebatch literally).
  * temporal value queue: [prev_bev, encoder-input BEV query] for all 3 layers when prev_bev
    exists; [query, query] of the current layer's query on the first frame (prev_bev None).
    `has_prev` = 0 reproduces the latter inside the fixed graph (and zeroes the shift).
  * prev_bev rotation (torchvision rotate by can_bus[-1] about (100, 100)) stays on the host,
    like the ego-motion shift; `rotate_prev_bev()` below is the same call.
  * point order in MSDeformableAttention3D is (point, anchor) with the anchor fastest, so
    sampling point p uses pillar anchor p % 4.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

EMBED, HEADS, FFN = 256, 8, 512
BEV_H = BEV_W = 50
NQ = BEV_H * BEV_W
NUM_CAMS = 6
FH, FW = 15, 25  # stride-32 level of the 480x800 (padded 450x800) input
Z_ANCHORS = 4
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
NUM_QUERY, NUM_CLASSES, CODE_SIZE = 900, 10, 10
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]


# ---- deformable attention --------------------------------------------------------------------
def msda_rank5(value, hw, loc, w):
    """Single-level MSDA. value (B, H*W, M, D); loc (B, Q, M, P, 2) in [0, 1]; w (B, Q, M, P)
    -> (B, Q, M*D). Same math as mmcv's multi_scale_deformable_attn_pytorch at L=1."""
    h, wd = hw
    b, _, m, d = value.shape
    q, p = loc.shape[1], loc.shape[3]
    v = value.permute(0, 2, 3, 1).reshape(b * m, d, h, wd)
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


# ---- backbone --------------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision

        r = torchvision.models.resnet50(weights=None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.lateral = nn.Conv2d(2048, EMBED, 1)
        self.fpn = nn.Conv2d(EMBED, EMBED, 3, padding=1)

    def forward(self, img):  # (N, 3, 480, 800) normalized
        return self.fpn(self.lateral(self.body(img)))


# ---- encoder ---------------------------------------------------------------------------------
class TemporalSelfAttention(nn.Module):
    def __init__(self, points=4):
        super().__init__()
        self.p = points
        self.sampling_offsets = nn.Linear(2 * EMBED, 2 * HEADS * points * 2)
        self.attention_weights = nn.Linear(2 * EMBED, 2 * HEADS * points)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def forward(self, query, bev_pos, v_prev, v_cur, ref_prev, ref_cur, reference=False):
        """query (Q, E); the 2-frame value queue is [v_prev, v_cur] (Q, E) each; ref_* (Q, 2)."""
        identity = query
        qp = query + bev_pos
        qcat = torch.cat([v_prev, qp], -1)  # upstream: cat([value[:bs], query + pos])
        v = self.value_proj(torch.stack([v_prev, v_cur], 0)).reshape(2, NQ, HEADS, EMBED // HEADS)
        m, p = HEADS, self.p
        # linear outputs are laid out (head, queue, level=1, point, xy) / (head, queue, point)
        off = self.sampling_offsets(qcat).reshape(NQ, m, 2, p * 2).permute(2, 0, 1, 3).reshape(2, NQ, m, p, 2)
        w = torch.softmax(self.attention_weights(qcat).reshape(NQ, m, 2, p), -1).permute(2, 0, 1, 3)
        ref = torch.stack([ref_prev, ref_cur], 0).reshape(2, NQ, 1, 1, 2)
        loc = ref + off / torch.tensor([BEV_W, BEV_H], dtype=off.dtype)
        if reference:
            out = msda_mmcv_reference(v, [(BEV_H, BEV_W)], loc.reshape(2, NQ, m, 1, p, 2), w.reshape(2, NQ, m, 1, p))
        else:
            out = msda_rank5(v, (BEV_H, BEV_W), loc, w)
        return self.output_proj(out.mean(0)) + identity  # mean over the 2-frame queue


class SpatialCrossAttention(nn.Module):
    def __init__(self, points=8):
        super().__init__()
        self.p = points
        self.sampling_offsets = nn.Linear(EMBED, HEADS * points * 2)
        self.attention_weights = nn.Linear(EMBED, HEADS * points)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def forward(self, query, img_value, ref_cam, bev_mask, reference=False):
        """query (Q, E); img_value (C, HW, E); ref_cam (C, Q, Z, 2); bev_mask (C, Q, Z) in {0, 1}."""
        m, p = HEADS, self.p
        v = self.value_proj(img_value).reshape(NUM_CAMS, FH * FW, m, EMBED // m)
        off = self.sampling_offsets(query).reshape(1, NQ, m, p, 2) / torch.tensor([FW, FH], dtype=query.dtype)
        w = torch.softmax(self.attention_weights(query).reshape(1, NQ, m, p), -1)
        vis = (bev_mask.sum(-1) > 0).to(query.dtype)  # (C, Q): camera sees any pillar point
        if reference:
            return self._upstream(query, v, off, w, ref_cam, vis)
        # point p uses pillar anchor p % Z (offsets are laid out (point, anchor), anchor fastest)
        idx = torch.arange(p) % Z_ANCHORS
        ref = ref_cam.reshape(NUM_CAMS, NQ, 1, Z_ANCHORS, 2).expand(-1, -1, m, -1, -1)[:, :, :, idx]
        out = msda_rank5(v, (FH, FW), ref + off, w.expand(NUM_CAMS, -1, -1, -1))  # (C, Q, E)
        vis = vis.reshape(NUM_CAMS, NQ, 1)
        slots = (out * vis).sum(0) / vis.sum(0).clamp(min=1.0)
        return self.output_proj(slots) + query

    def _upstream(self, query, v, off, w, ref_cam, vis):
        """Upstream's literal formulation: per camera, gather the visible queries (nonzero), run
        MSDeformableAttention3D on the rebatched set, scatter back, average."""
        m, p = HEADS, self.p
        slots = torch.zeros_like(query)
        for c in range(NUM_CAMS):
            idx = vis[c].nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            n = idx.numel()
            r = ref_cam[c, idx][:, None, None, None, :, :]  # (n,1,1,1,Z,2)
            o = off[0, idx].reshape(n, m, 1, p // Z_ANCHORS, Z_ANCHORS, 2)
            loc = (r + o).reshape(1, n, m, 1, p, 2)
            out = msda_mmcv_reference(v[c:c + 1], [(FH, FW)], loc, w[0, idx].reshape(1, n, m, 1, p))
            slots[idx] += out[0]
        count = vis.sum(0).clamp(min=1.0)
        return self.output_proj(slots / count[:, None]) + query


class FFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED, FFN)
        self.fc2 = nn.Linear(FFN, EMBED)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tsa = TemporalSelfAttention()
        self.sca = SpatialCrossAttention()
        self.ffn = FFNBlock()
        self.norms = nn.ModuleList(nn.LayerNorm(EMBED) for _ in range(3))

    def forward(self, q, bev_pos, v_prev, v_cur, ref_prev, ref_cur, img_value, ref_cam, bev_mask, reference=False):
        q = self.norms[0](self.tsa(q, bev_pos, v_prev, v_cur, ref_prev, ref_cur, reference))
        q = self.norms[1](self.sca(q, img_value, ref_cam, bev_mask, reference))
        return self.norms[2](self.ffn(q))


def ref_2d() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(0.5, BEV_H - 0.5, BEV_H), torch.linspace(0.5, BEV_W - 0.5, BEV_W),
                            indexing="ij")
    return torch.stack([xs.reshape(-1) / BEV_W, ys.reshape(-1) / BEV_H], -1)  # (Q, 2)


class Encoder(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(layers))
        self.bev_embedding = nn.Parameter(torch.zeros(NQ, EMBED))
        self.row_embed = nn.Parameter(torch.zeros(BEV_H, EMBED // 2))
        self.col_embed = nn.Parameter(torch.zeros(BEV_W, EMBED // 2))
        self.level_embeds = nn.Parameter(torch.zeros(4, EMBED))
        self.cams_embeds = nn.Parameter(torch.zeros(NUM_CAMS, EMBED))
        self.can_bus_mlp = nn.Sequential(nn.Linear(18, EMBED // 2), nn.ReLU(), nn.Linear(EMBED // 2, EMBED), nn.ReLU(),
                                         nn.LayerNorm(EMBED))
        self.register_buffer("ref_2d", ref_2d(), persistent=False)

    def bev_pos(self):
        # mmdet LearnedPositionalEncoding: cat(col_embed(x), row_embed(y)) over (h, w)
        x = self.col_embed[None].expand(BEV_H, -1, -1)
        y = self.row_embed[:, None].expand(-1, BEV_W, -1)
        return torch.cat([x, y], -1).reshape(NQ, EMBED)

    def forward(self, feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask, reference=False):
        q0 = self.bev_embedding + self.can_bus_mlp(can_bus[None])
        img_value = feats.flatten(2).transpose(1, 2) + self.cams_embeds[:, None] + self.level_embeds[0]
        pos = self.bev_pos()
        ref_prev = self.ref_2d + shift * has_prev
        q = q0
        for layer in self.layers:
            # upstream: the value queue is [prev_bev, encoder input query] for every layer when
            # prev_bev exists, and [query, query] of the current layer's query on the first frame
            v_prev = prev_bev * has_prev + q * (1 - has_prev)
            v_cur = q0 * has_prev + q * (1 - has_prev)
            q = layer(q, pos, v_prev, v_cur, ref_prev, self.ref_2d, img_value, ref_cam, bev_mask, reference)
        return q


# ---- decoder + head --------------------------------------------------------------------------
class DecoderMSDA(nn.Module):
    def __init__(self, points=4):
        super().__init__()
        self.p = points
        self.sampling_offsets = nn.Linear(EMBED, HEADS * points * 2)
        self.attention_weights = nn.Linear(EMBED, HEADS * points)
        self.value_proj = nn.Linear(EMBED, EMBED)
        self.output_proj = nn.Linear(EMBED, EMBED)

    def forward(self, query, query_pos, value, ref, reference=False):
        """query (Q, E); value (NQ, E) = bev_embed; ref (Q, 2) in [0, 1]."""
        m, p = HEADS, self.p
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
        qk = (q + qpos)[:, None]  # (Q, 1, E): batch_first=False, bs=1
        q = self.norms[0](q + self.self_attn(qk, qk, q[:, None], need_weights=False)[0][:, 0])
        q = self.norms[1](self.cross_attn(q, qpos, bev, ref, reference))
        return self.norms[2](self.ffn(q))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def cls_branch():
    return nn.Sequential(nn.Linear(EMBED, EMBED), nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED),
                         nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, NUM_CLASSES))


def reg_branch():
    return nn.Sequential(nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED), nn.ReLU(),
                         nn.Linear(EMBED, CODE_SIZE))


class Decoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(layers))
        self.query_embedding = nn.Parameter(torch.zeros(NUM_QUERY, 2 * EMBED))
        self.reference_points = nn.Linear(EMBED, 3)
        # every layer refines the reference points with its own reg branch (with_box_refine)
        self.reg_branches = nn.ModuleList(reg_branch() for _ in range(layers))
        self.cls_last = cls_branch()

    def forward(self, bev_embed, reference=False):
        qpos, q = self.query_embedding[:, :EMBED], self.query_embedding[:, EMBED:]
        refp = self.reference_points(qpos).sigmoid()  # (Q, 3)
        prev_ref = refp
        for i, layer in enumerate(self.layers):
            q = layer(q, qpos, bev_embed, refp[:, :2], reference)
            tmp = self.reg_branches[i](q)
            prev_ref = refp
            xy = (tmp[:, 0:2] + inverse_sigmoid(refp[:, :2])).sigmoid()
            z = (tmp[:, 4:5] + inverse_sigmoid(refp[:, 2:3])).sigmoid()
            refp = torch.cat([xy, z], -1)
        # head on the last layer's output (upstream: reference = inter_references[lvl - 1])
        cls = self.cls_last(q)
        r = inverse_sigmoid(prev_ref)
        xy = (tmp[:, 0:2] + r[:, 0:2]).sigmoid()
        z = (tmp[:, 4:5] + r[:, 2:3]).sigmoid()
        x = xy[:, 0:1] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
        y = xy[:, 1:2] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
        zz = z * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
        bbox = torch.cat([x, y, tmp[:, 2:4], zz, tmp[:, 5:]], -1)
        return cls, bbox


def decode(cls, bbox, max_num=300):
    """NMSFreeCoder.decode_single + BEVFormerHead.get_bboxes (bottom-center z): host side."""
    scores, idx = cls.sigmoid().reshape(-1).topk(max_num)
    labels = idx % NUM_CLASSES
    b = bbox[idx // NUM_CLASSES]
    rot = torch.atan2(b[:, 6:7], b[:, 7:8])
    out = torch.cat([b[:, 0:2], b[:, 4:5], b[:, 2:3].exp(), b[:, 3:4].exp(), b[:, 5:6].exp(), rot, b[:, 8:10]], -1)
    lo, hi = torch.tensor(POST_CENTER_RANGE[:3]), torch.tensor(POST_CENTER_RANGE[3:])
    keep = ((out[:, :3] >= lo) & (out[:, :3] <= hi)).all(1)
    out, scores, labels = out[keep], scores[keep], labels[keep]
    out[:, 2] = out[:, 2] - out[:, 5] * 0.5
    return out, scores, labels


# ---- geometry (host side; upstream BEVFormerEncoder.get_reference_points/point_sampling) ------
def reference_points_cam(lidar2img: torch.Tensor, img_hw=(480, 800)):
    """lidar2img (6, 4, 4) -> ref_cam (6, Q, Z, 2) in [0, 1], bev_mask (6, Q, Z) float."""
    zs = torch.linspace(0.5, 8 - 0.5, Z_ANCHORS).view(-1, 1, 1).expand(Z_ANCHORS, BEV_H, BEV_W) / 8
    xs = torch.linspace(0.5, BEV_W - 0.5, BEV_W).view(1, 1, BEV_W).expand(Z_ANCHORS, BEV_H, BEV_W) / BEV_W
    ys = torch.linspace(0.5, BEV_H - 0.5, BEV_H).view(1, BEV_H, 1).expand(Z_ANCHORS, BEV_H, BEV_W) / BEV_H
    r = torch.stack((xs, ys, zs), -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (Z, Q, 3)
    r = r.clone()
    for i in range(3):
        r[..., i] = r[..., i] * (PC_RANGE[i + 3] - PC_RANGE[i]) + PC_RANGE[i]
    r = torch.cat([r, torch.ones_like(r[..., :1])], -1)  # (Z, Q, 4)
    cam = torch.einsum("cij,zqj->czqi", lidar2img.float(), r)  # (C, Z, Q, 4)
    eps = 1e-5
    mask = cam[..., 2:3] > eps
    xy = cam[..., 0:2] / torch.maximum(cam[..., 2:3], torch.ones_like(cam[..., 2:3]) * eps)
    xy[..., 0] /= img_hw[1]
    xy[..., 1] /= img_hw[0]
    mask = mask & (xy[..., 1:2] > 0) & (xy[..., 1:2] < 1) & (xy[..., 0:1] < 1) & (xy[..., 0:1] > 0)
    mask = torch.nan_to_num(mask.float())
    return xy.permute(0, 2, 1, 3).contiguous(), mask[..., 0].permute(0, 2, 1).contiguous()


def can_bus_shift(can_bus: torch.Tensor, grid=(102.4 / BEV_H, 102.4 / BEV_W)) -> torch.Tensor:
    """PerceptionTransformer.get_bev_features' ego-motion shift (x, y) in BEV-grid units."""
    dx, dy = float(can_bus[0]), float(can_bus[1])
    ego_angle = float(can_bus[-2]) / math.pi * 180
    tl = math.hypot(dx, dy)
    ta = math.atan2(dy, dx) / math.pi * 180
    bev_angle = ego_angle - ta
    sy = tl * math.cos(bev_angle / 180 * math.pi) / grid[0] / BEV_H
    sx = tl * math.sin(bev_angle / 180 * math.pi) / grid[1] / BEV_W
    return torch.tensor([sx, sy])


def rotate_prev_bev(prev_bev: torch.Tensor, can_bus: torch.Tensor) -> torch.Tensor:
    """PerceptionTransformer.get_bev_features: rotate prev_bev (Q, E) by can_bus[-1] degrees."""
    from torchvision.transforms.functional import rotate

    t = prev_bev.reshape(BEV_H, BEV_W, EMBED).permute(2, 0, 1)
    t = rotate(t, float(can_bus[-1]), center=[100, 100])
    return t.permute(1, 2, 0).reshape(NQ, EMBED)


# ---- loading ---------------------------------------------------------------------------------
UNUSED = {
    "pts_bbox_head.code_weights": "loss weights for the 10 box-code terms (training only)",
}


def _layer_map(prefix_src, prefix_dst, kind):
    m = {}
    if kind == "enc":
        for a in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
            for t in ("weight", "bias"):
                m[f"{prefix_src}.attentions.0.{a}.{t}"] = f"{prefix_dst}.tsa.{a}.{t}"
        for a in ("sampling_offsets", "attention_weights", "value_proj"):
            for t in ("weight", "bias"):
                m[f"{prefix_src}.attentions.1.deformable_attention.{a}.{t}"] = f"{prefix_dst}.sca.{a}.{t}"
        for t in ("weight", "bias"):
            m[f"{prefix_src}.attentions.1.output_proj.{t}"] = f"{prefix_dst}.sca.output_proj.{t}"
    else:
        m[f"{prefix_src}.attentions.0.attn.in_proj_weight"] = f"{prefix_dst}.self_attn.in_proj_weight"
        m[f"{prefix_src}.attentions.0.attn.in_proj_bias"] = f"{prefix_dst}.self_attn.in_proj_bias"
        for t in ("weight", "bias"):
            m[f"{prefix_src}.attentions.0.attn.out_proj.{t}"] = f"{prefix_dst}.self_attn.out_proj.{t}"
        for a in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
            for t in ("weight", "bias"):
                m[f"{prefix_src}.attentions.1.{a}.{t}"] = f"{prefix_dst}.cross_attn.{a}.{t}"
    for t in ("weight", "bias"):
        m[f"{prefix_src}.ffns.0.layers.0.0.{t}"] = f"{prefix_dst}.ffn.fc1.{t}"
        m[f"{prefix_src}.ffns.0.layers.1.{t}"] = f"{prefix_dst}.ffn.fc2.{t}"
        for i in range(3):
            m[f"{prefix_src}.norms.{i}.{t}"] = f"{prefix_dst}.norms.{i}.{t}"
    return m


def name_map(sd: dict) -> dict:
    """official checkpoint key -> (piece, key in that piece). Every key is mapped or listed in
    UNUSED; load_official() fails on anything else."""
    m = {}
    tv = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    for k in sd:
        if k.startswith("img_backbone."):
            rest = k[len("img_backbone."):]
            head, _, tail = rest.partition(".")
            m[k] = ("backbone", f"body.{tv.index(head)}.{tail}")
    m["img_neck.lateral_convs.0.conv.weight"] = ("backbone", "lateral.weight")
    m["img_neck.lateral_convs.0.conv.bias"] = ("backbone", "lateral.bias")
    m["img_neck.fpn_convs.0.conv.weight"] = ("backbone", "fpn.weight")
    m["img_neck.fpn_convs.0.conv.bias"] = ("backbone", "fpn.bias")
    h, t = "pts_bbox_head", "pts_bbox_head.transformer"
    for i in range(3):
        for s, d in _layer_map(f"{t}.encoder.layers.{i}", f"layers.{i}", "enc").items():
            m[s] = ("encoder", d)
    for i in range(6):
        for s, d in _layer_map(f"{t}.decoder.layers.{i}", f"layers.{i}", "dec").items():
            m[s] = ("decoder", d)
    m[f"{h}.bev_embedding.weight"] = ("encoder", "bev_embedding")
    m[f"{h}.positional_encoding.row_embed.weight"] = ("encoder", "row_embed")
    m[f"{h}.positional_encoding.col_embed.weight"] = ("encoder", "col_embed")
    m[f"{t}.level_embeds"] = ("encoder", "level_embeds")
    m[f"{t}.cams_embeds"] = ("encoder", "cams_embeds")
    for s, d in {"0": "0", "2": "2", "norm": "4"}.items():
        for x in ("weight", "bias"):
            m[f"{t}.can_bus_mlp.{s}.{x}"] = ("encoder", f"can_bus_mlp.{d}.{x}")
    m[f"{h}.query_embedding.weight"] = ("decoder", "query_embedding")
    for x in ("weight", "bias"):
        m[f"{t}.reference_points.{x}"] = ("decoder", f"reference_points.{x}")
    for i in range(6):
        for j in (0, 2, 4):
            for x in ("weight", "bias"):
                m[f"{h}.reg_branches.{i}.{j}.{x}"] = ("decoder", f"reg_branches.{i}.{j}.{x}")
    for j in (0, 1, 3, 4, 6):
        for x in ("weight", "bias"):
            m[f"{h}.cls_branches.5.{j}.{x}"] = ("decoder", f"cls_last.{j}.{x}")
    return m


# cls branches 0..4 score the 5 intermediate decoder layers; only the last layer is decoded at
# inference (NMSFreeCoder.decode takes all_cls_scores[-1]), so they are not needed.
for _i in range(5):
    for _j in (0, 1, 3, 4, 6):
        for _x in ("weight", "bias"):
            UNUSED[f"pts_bbox_head.cls_branches.{_i}.{_j}.{_x}"] = "classifies intermediate decoder layers (training aux loss only)"


def load_official(path):
    """-> (backbone, encoder, decoder) with the official BEVFormer-tiny weights."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    pieces = {"backbone": Backbone(), "encoder": Encoder(), "decoder": Decoder()}
    nm = name_map(sd)
    per = {k: {} for k in pieces}
    unexpected = []
    for k, v in sd.items():
        if k in nm:
            p, d = nm[k]
            per[p][d] = v
        elif k not in UNUSED and not k.endswith("num_batches_tracked"):
            unexpected.append(k)
    if unexpected:
        raise RuntimeError(f"unmapped checkpoint keys: {unexpected[:10]} (+{len(unexpected) - 10})")
    for name, mod in pieces.items():
        r = mod.load_state_dict(per[name], strict=False)
        missing = [k for k in r.missing_keys if not k.endswith("num_batches_tracked")]
        if missing or r.unexpected_keys:
            raise RuntimeError(f"{name}: missing {missing[:10]} unexpected {r.unexpected_keys[:10]}")
        mod.eval()
    return pieces["backbone"], pieces["encoder"], pieces["decoder"]
