"""Sparse4D v3 (HorizonRobotics/Sparse4D, sparse4dv3_temporal_r50_1x8_bs6_256x704) in plain PyTorch.

No mmcv / mmdet / mmdet3d: the modules below are the upstream ones restated with the same
parameter names, so the official checkpoint loads with `load_official()` unchanged (strict).
Inference only (no dropout / grid mask / denoising / depth branch).

Deformable 4D aggregation (DFA) has two implementations:
  * `dfa_upstream`: verbatim upstream `DeformableFeatureAggregation.feature_sampling` +
    `multi_view_level_fusion` (the non-CUDA path of blocks.py), rank 6;
  * `dfa_rank4`: the same math with every tensor rank <= 4 (the HTP rejects rank-5/6 at execute
    time), one grid_sample per level over the 6 cameras. `validate.py` checks the two agree.

The temporal instance bank (top-k caching, ego-motion anchor projection, confidence decay) is
host-side state in `InstanceBank`, exactly as upstream's `instance_bank.py` for batch 1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import Bottleneck, ResNet

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = range(11)
CNS = 0
EMBED, GROUPS, LEVELS, CAMS, PTS = 256, 8, 4, 6, 13
NUM_ANCHOR, NUM_TEMP = 900, 600
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]
IMG_HW = (256, 704)
STRIDES = (4, 8, 16, 32)
LEVEL_HW = [(IMG_HW[0] // s, IMG_HW[1] // s) for s in STRIDES]
FIX_SCALE = [[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]]
# operation_order[2:] of the config: layer 0 has no graph attention (no instances to attend yet)
OPS = (["deformable", "ffn", "norm", "refine"]
       + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"] * 5)


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    input_dims = embed_dims if input_dims is None else input_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers += [nn.Linear(input_dims, embed_dims), nn.ReLU(inplace=True)]
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class Backbone(nn.Module):
    """ResNet-50 (mmdet style='pytorch' == torchvision) + FPN(256, num_outs=4): 4 levels, NCHW."""

    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        del self.img_backbone.fc, self.img_backbone.avgpool
        ins = [256, 512, 1024, 2048]
        self.lateral = nn.ModuleList(nn.Conv2d(c, EMBED, 1) for c in ins)
        self.fpn = nn.ModuleList(nn.Conv2d(EMBED, EMBED, 3, padding=1) for _ in ins)

    def forward(self, img):  # (N, 3, 256, 704) normalized
        b = self.img_backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(img))))
        feats = []
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = layer(x)
            feats.append(x)
        lat = [conv(f) for conv, f in zip(self.lateral, feats)]
        for i in range(3, 0, -1):  # mmdet FPN: nearest upsample to the finer level's size
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        return [conv(t) for conv, t in zip(self.fpn, lat)]


class AnchorEncoder(nn.Module):
    """SparseBox3DEncoder(embed_dims=[128, 32, 32, 64], mode='cat', output_fc=False, out_loops=4)."""

    def __init__(self):
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(128, 1, 4, 3))
        self.size_fc = nn.Sequential(*linear_relu_ln(32, 1, 4, 3))
        self.yaw_fc = nn.Sequential(*linear_relu_ln(32, 1, 4, 2))
        self.vel_fc = nn.Sequential(*linear_relu_ln(64, 1, 4, 3))

    def forward(self, a):
        return torch.cat([self.pos_fc(a[..., [X, Y, Z]]), self.size_fc(a[..., [W, L, H]]),
                          self.yaw_fc(a[..., [SIN_YAW, COS_YAW]]), self.vel_fc(a[..., VX:VX + 3])], dim=-1)


class KeyPoints(nn.Module):
    """SparseBox3DKeyPointsGenerator(num_learnable_pts=6, fix_scale=FIX_SCALE)."""

    def __init__(self):
        super().__init__()
        self.fix_scale = nn.Parameter(torch.tensor(FIX_SCALE, dtype=torch.float32), requires_grad=False)
        self.learnable_fc = nn.Linear(EMBED, 6 * 3)

    def forward(self, anchor, feat):  # (N, 11), (N, 256) -> (N, 13, 3)
        size = anchor[:, None, [W, L, H]].exp()
        kp = torch.cat([self.fix_scale * size,
                        (self.learnable_fc(feat).reshape(-1, 6, 3).sigmoid() - 0.5) * size], dim=1)
        c, s = anchor[:, COS_YAW, None], anchor[:, SIN_YAW, None]
        # rotation about z, written out (the upstream 3x3 matmul, without a rank-4 matmul broadcast)
        x = c * kp[..., 0] - s * kp[..., 1]
        y = s * kp[..., 0] + c * kp[..., 1]
        return torch.stack([x, y, kp[..., 2]], dim=-1) + anchor[:, None, [X, Y, Z]]


def project_points(kp, proj, image_wh):
    """upstream project_points for batch 1: kp (N, P, 3), proj (6, 4, 4), image_wh (6, 2)
    -> normalized [0, 1] image coordinates (6, N, P, 2)."""
    ptsx = torch.cat([kp, torch.ones_like(kp[..., :1])], dim=-1)  # (N, P, 4)
    p = torch.einsum("cij,npj->cnpi", proj[:, :3], ptsx)  # (6, N, P, 3)
    xy = p[..., :2] / torch.clamp(p[..., 2:3], min=1e-5)
    return xy / image_wh[:, None, None]


class DFA(nn.Module):
    """DeformableFeatureAggregation(use_camera_embed=True, residual_mode='cat')."""

    def __init__(self):
        super().__init__()
        self.kps_generator = KeyPoints()
        self.output_proj = nn.Linear(EMBED, EMBED)
        self.camera_encoder = nn.Sequential(*linear_relu_ln(EMBED, 1, 2, 12))
        self.weights_fc = nn.Linear(EMBED, GROUPS * LEVELS * PTS)

    def weights(self, feat, anchor_embed, proj):
        """-> (N, 6, 4, 13, 8) softmaxed jointly over (cams, levels, pts) per group."""
        cam = self.camera_encoder(proj[:, :3].reshape(CAMS, 12))  # (6, 256)
        f = (feat + anchor_embed)[:, None] + cam[None]  # (N, 6, 256)
        n = f.shape[0]
        w = self.weights_fc(f).reshape(n, CAMS * LEVELS * PTS, GROUPS).softmax(dim=-2)
        return w.reshape(n, CAMS, LEVELS, PTS, GROUPS)

    def sample_inputs(self, feat, anchor, anchor_embed, proj, image_wh):
        """Everything the aggregation needs: (points (6, N, 13, 2) in [0, 1], weights (N, 6, 4, 13, 8))."""
        kp = self.kps_generator(anchor, feat)
        return project_points(kp, proj, image_wh), self.weights(feat, anchor_embed, proj)

    def finish(self, agg, feat):
        return torch.cat([self.output_proj(agg), feat], dim=-1)


def dfa_upstream(fmaps, pts, w, layer=None):
    """Verbatim upstream feature_sampling + multi_view_level_fusion + sum over points (batch 1).
    fmaps: 4 x (6, 256, H, W); pts (6, N, 13, 2); w (N, 6, 4, 13, 8) -> (N, 256)."""
    n = pts.shape[1]
    points_2d = (pts * 2 - 1).reshape(CAMS, n, PTS, 2)
    feats = [F.grid_sample(fm, points_2d) for fm in fmaps]  # 4 x (6, 256, N, 13)
    feats = torch.stack(feats, dim=1)  # (6, 4, 256, N, 13)
    feats = feats.reshape(1, CAMS, LEVELS, -1, n, PTS).permute(0, 4, 1, 2, 5, 3)  # bs,N,cams,lv,pts,C
    f = w[None, ..., None] * feats.reshape(feats.shape[:-1] + (GROUPS, EMBED // GROUPS))
    f = f.sum(dim=2).sum(dim=2).reshape(1, n, PTS, EMBED)
    return f.sum(dim=2)[0]


def dfa_rank4(fmaps, pts, w, layer=None):
    """dfa_upstream with every tensor rank <= 4: per level, one grid_sample over the 6 cameras,
    then the group-weighted sum over cameras and points."""
    n = pts.shape[1]
    grid = pts * 2 - 1  # (6, N, 13, 2)
    # (N, 6, 4, 13, 8) -> per level (6, 8, N*13): cam, group, anchor x point
    wl = w.permute(2, 1, 4, 0, 3).reshape(LEVELS, CAMS, GROUPS, n * PTS)
    out = 0
    for lv, fm in enumerate(fmaps):
        s = F.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        s = s.reshape(CAMS, GROUPS, EMBED // GROUPS, n * PTS) * wl[lv][:, :, None]  # (6, 8, 32, N*13)
        s = s.sum(dim=0).reshape(EMBED, n, PTS).sum(dim=-1)  # (256, N)
        out = out + s
    return out.transpose(0, 1)


class MHA(nn.Module):
    """mmcv MultiheadAttention(embed_dims=512, num_heads=8, batch_first=True) at inference:
    identity (the query, *before* query_pos is added) + attention output."""

    def __init__(self, dims=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dims, heads, batch_first=True)

    def forward(self, q, k, v):  # (N, 512), (M, 512), (M, 512)
        return q + self.attn(q[None], k[None], v[None], need_weights=False)[0][0]


class AsymmetricFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = nn.LayerNorm(2 * EMBED)
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(2 * EMBED, 4 * EMBED), nn.ReLU(inplace=True)),
                                    nn.Linear(4 * EMBED, EMBED))
        self.identity_fc = nn.Linear(2 * EMBED, EMBED)

    def forward(self, x):
        x = self.pre_norm(x)
        return self.identity_fc(x) + self.layers(x)


class Scale(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * self.scale


class Refine(nn.Module):
    """SparseBox3DRefinementModule(refine_yaw=True, with_quality_estimation=True, output_dim=11)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*linear_relu_ln(EMBED, 2, 2), nn.Linear(EMBED, 11), Scale(11))
        self.cls_layers = nn.Sequential(*linear_relu_ln(EMBED, 1, 2), nn.Linear(EMBED, 10))
        self.quality_layers = nn.Sequential(*linear_relu_ln(EMBED, 1, 2), nn.Linear(EMBED, 2))

    def forward(self, feat, anchor, anchor_embed, time_interval, return_cls):
        f = feat + anchor_embed
        out = self.layers(f)
        # refine_state [X..COS_YAW] += anchor; velocity = out / time_interval + anchor velocity
        box = torch.cat([out[:, :VX] + anchor[:, :VX], out[:, VX:] / time_interval + anchor[:, VX:]], dim=-1)
        if not return_cls:
            return box, None, None
        return box, self.cls_layers(feat), self.quality_layers(f)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_bank = nn.Module()
        self.instance_bank.anchor = nn.Parameter(torch.zeros(NUM_ANCHOR, 11))
        self.instance_bank.instance_feature = nn.Parameter(torch.zeros(NUM_ANCHOR, EMBED))
        self.anchor_encoder = AnchorEncoder()
        mk = {"deformable": DFA, "ffn": AsymmetricFFN, "norm": lambda: nn.LayerNorm(EMBED), "refine": Refine,
              "temp_gnn": MHA, "gnn": MHA}
        self.layers = nn.ModuleList(mk[op]() for op in OPS)
        self.fc_before = nn.Linear(EMBED, 2 * EMBED, bias=False)
        self.fc_after = nn.Linear(2 * EMBED, EMBED, bias=False)

    def graph(self, i, q, qpos, k=None, kpos=None, v=None):
        """Sparse4DHead.graph_model with decouple_attn: cat [feature, pos] for query/key; the value
        goes through fc_before; mmcv MHA's key/value default to the query."""
        q = torch.cat([q, qpos], dim=-1)
        k = q if k is None else torch.cat([k, kpos], dim=-1)
        v = k if v is None else self.fc_before(v)
        return self.fc_after(self.layers[i](q, k, v))


class Sparse4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head()

    def load_official(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        out = {}
        for k, v in sd.items():
            if k.startswith("depth_branch.") or k == "head.instance_bank.anchor_handler.fix_scale":
                continue  # auxiliary depth supervision / the anchor handler's unused buffer
            k = k.replace("img_backbone.", "backbone.img_backbone.")
            k = k.replace("img_neck.lateral_convs.", "backbone.lateral.").replace("img_neck.fpn_convs.", "backbone.fpn.")
            k = k.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias") if "backbone.lateral" in k or "backbone.fpn" in k else k
            out[k] = v
        self.load_state_dict(out, strict=True)
        return self


# ---------------------------------------------------------------------------------------------
# host-side instance bank (upstream instance_bank.py, batch 1) and decoder


def anchor_projection(anchor, T, time_interval):
    """SparseBox3DKeyPointsGenerator.anchor_projection for one T (4, 4), dt = -time_interval."""
    vel = anchor[:, VX:]
    center = anchor[:, [X, Y, Z]] - vel * (-time_interval)
    center = center @ T[:3, :3].T + T[:3, 3]
    yaw = anchor[:, [COS_YAW, SIN_YAW]] @ T[:2, :2].T
    vel = vel @ T[:3, :3].T
    # upstream's "TODO: Fix bug": the projected yaw is written as [cos, sin] into the [sin, cos] slots
    return torch.cat([center, anchor[:, [W, L, H]], yaw, vel], dim=-1)


class InstanceBank:
    def __init__(self, decay=0.6, max_dt=2.0, default_dt=0.5):
        self.decay, self.max_dt, self.default_dt = decay, max_dt, default_dt
        self.reset()

    def reset(self):
        self.cached_feature = self.cached_anchor = self.confidence = self.metas = None

    def get(self, metas):
        """-> (temp_feature | None, temp_anchor | None, time_interval)."""
        if self.cached_anchor is None:
            return None, None, torch.tensor(self.default_dt)
        dt = torch.tensor(metas["timestamp"] - self.metas["timestamp"], dtype=torch.float32)
        self.mask = bool(abs(dt) <= self.max_dt)
        T = torch.tensor(metas["T_global_inv"] @ self.metas["T_global"], dtype=torch.float32)
        self.cached_anchor = anchor_projection(self.cached_anchor, T, dt)
        if not self.mask:
            raise NotImplementedError("gap > max_time_interval: upstream keeps the cache but drops it in update()")
        dt = dt if float(dt) != 0 else torch.tensor(self.default_dt)
        return self.cached_feature, self.cached_anchor, dt

    def update(self, feat, anchor, cls):
        """after layer 0: keep the 600 cached instances + the current top-300."""
        if self.cached_feature is None:
            return feat, anchor
        idx = torch.topk(cls.max(dim=-1).values, NUM_ANCHOR - NUM_TEMP).indices
        return torch.cat([self.cached_feature, feat[idx]]), torch.cat([self.cached_anchor, anchor[idx]])

    def cache(self, feat, anchor, cls, metas):
        conf = cls.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            conf[:NUM_TEMP] = torch.maximum(self.confidence * self.decay, conf[:NUM_TEMP])
        self.metas = metas
        self.confidence, idx = torch.topk(conf, NUM_TEMP)
        self.cached_feature, self.cached_anchor = feat[idx], anchor[idx]


def decode(cls, box, quality, num_output=300):
    """SparseBox3DDecoder.decode with instance ids (tracking_test: max over classes first)."""
    scores, cls_ids = cls.sigmoid().max(dim=-1)
    scores, idx = scores.topk(num_output)
    scores = scores * quality[idx, CNS].sigmoid()
    scores, order = torch.sort(scores, descending=True)
    idx = idx[order]
    b = box[idx]
    yaw = torch.atan2(b[:, SIN_YAW], b[:, COS_YAW])
    boxes = torch.cat([b[:, [X, Y, Z]], b[:, [W, L, H]].exp(), yaw[:, None], b[:, VX:]], dim=-1)
    return boxes, scores, cls_ids[idx]


# ---------------------------------------------------------------------------------------------
# frame driver: the whole model, piece boundaries marked (they become the exported graphs)


class Runner:
    """Runs frames in order with the temporal state. `dfa(fmaps, pts, w, layer) -> (N, 256)` is
    dfa_upstream, dfa_rank4 or another implementation of the same contract."""

    def __init__(self, model, dfa=dfa_rank4):
        self.m, self.dfa, self.bank = model, dfa, InstanceBank()

    @torch.no_grad()
    def frame(self, img, metas, fmaps=None):
        m, head = self.m, self.m.head
        if fmaps is None:
            fmaps = m.backbone(img)
        proj, image_wh = metas["projection_mat"], metas["image_wh"]
        temp_feat, temp_anchor, dt = self.bank.get(metas)
        feat = head.instance_bank.instance_feature.clone()
        anchor = head.instance_bank.anchor.clone()
        ae = head.anchor_encoder(anchor)
        temp_ae = head.anchor_encoder(temp_anchor) if temp_anchor is not None else None
        n_pred, cls = 0, None
        for i, op in enumerate(OPS):
            layer = head.layers[i]
            if op == "temp_gnn":
                if temp_feat is None:
                    feat = head.graph(i, feat, ae)  # no cache: plain self-attention
                else:
                    feat = head.graph(i, feat, ae, temp_feat, temp_ae, temp_feat)
            elif op == "gnn":
                feat = head.graph(i, feat, ae, v=feat)
            elif op in ("norm", "ffn"):
                feat = layer(feat)
            elif op == "deformable":
                pts, w = layer.sample_inputs(feat, anchor, ae, proj, image_wh)
                agg = self.dfa(fmaps, pts, w, i)
                feat = layer.finish(agg, feat)
            elif op == "refine":
                last = i == len(OPS) - 1
                anchor, c, q = layer(feat, anchor, ae, dt, return_cls=n_pred == 0 or last)
                n_pred += 1
                if n_pred == 1:
                    feat, anchor = self.bank.update(feat, anchor, c)
                if not last:
                    ae = head.anchor_encoder(anchor)
                if n_pred > 1 and temp_ae is not None:
                    temp_ae = ae[:NUM_TEMP]
                cls, quality = c, q
        self.bank.cache(feat, anchor, cls, metas)
        return decode(cls, anchor, quality), (cls, anchor, quality)
