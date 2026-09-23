"""StreamPETR (R50, 428 queries, 256x704) in plain PyTorch, without mmcv / mmdet / mmdet3d.

Official checkpoint stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth (exiawsh/StreamPETR). Test
path only: Petr3D.simple_test -> ResNet-50 (C4, C5) -> CPFPN (level 0, stride 16) ->
StreamPETRHead.forward -> NMSFreeCoder. The 2D FocalHead is aux-only at test time (aux_2d_only)
and is not built.

Two head implementations:
  * ``UpstreamHead``: a literal transcription of StreamPETRHead.forward + pre/post_update_memory +
    position_embeding + temporal_alignment + PETRTemporalTransformer (all 6 decoder levels,
    float64 timestamps, the upstream tensor ops in the upstream order) -- the reference.
  * the deployment split: ``HeadCore`` (what runs on the HTP: memory_embed, spatial-alignment MLN,
    featurized PE, the 6-layer decoder, last-level cls/reg branches) fed by ``HostState``
    (everything that is trigonometric, rig-static or bookkeeping: the 3D position embedding, the
    memory queue with its ego-motion transforms, the nerf / sine encodings, top-k propagation,
    box decode). Constant query-side terms are folded into HeadCore buffers.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
POS_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
NUM_QUERY, NUM_PROP, MEMORY_LEN, TOPK = 300, 128, 512, 128
EMBED, HEADS, STRIDE, DEPTH_NUM = 256, 8, 16, 64


# ----------------------------------------------------------------------------- image branch
class CPFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(1024, 256, 1), nn.Conv2d(2048, 256, 1)])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1)])

    def forward(self, c4, c5):
        l0, l1 = self.lateral_convs[0](c4), self.lateral_convs[1](c5)
        l0 = l0 + F.interpolate(l1, size=l0.shape[2:], mode="nearest")
        return self.fpn_convs[0](l0)  # outs[position_level=0]


class ImageBranch(nn.Module):
    """(N, 3, 256, 704) normalized -> (N, 256, 16, 44)."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50()
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.neck = CPFPN()

    def forward(self, img):
        c3 = self.backbone(img)
        c4 = self.layer3(c3)
        return self.neck(c4, self.layer4(c4))


# ----------------------------------------------------------------------------- shared pieces
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    pos = pos * (2 * math.pi)
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    out = []
    for i in (1, 0, 2):  # (y, x, z)
        p = pos[..., i, None] / dim_t
        out.append(torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2))
    return torch.cat(out, dim=-1)


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    pos = pos * (2 * math.pi)
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    p = pos[..., 0, None] / dim_t
    return torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2)


def nerf_positional_encoding(t, n=6):
    bands = 2.0 ** torch.linspace(0.0, n - 1, n, dtype=t.dtype, device=t.device)
    return torch.cat([f(t * b) for b in bands for f in (torch.sin, torch.cos)], dim=-1)


class MLN(nn.Module):
    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(c_dim, f_dim), nn.ReLU())
        self.gamma, self.beta = nn.Linear(f_dim, f_dim), nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)

    def forward(self, x, c):
        c = self.reduce(c)
        return self.gamma(c) * self.ln(x) + self.beta(c)


class SELayerLinear(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_reduce, self.conv_expand = nn.Linear(c, c), nn.Linear(c, c)

    def forward(self, x, x_se):
        return x * torch.sigmoid(self.conv_expand(F.relu(self.conv_reduce(x_se))))


class MHA(nn.Module):
    """mmcv MultiheadAttention / FlashMHA with identity add (packed in_proj, 1/sqrt(d) scale)."""

    def __init__(self):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * EMBED, EMBED))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * EMBED))
        self.out_proj = nn.Linear(EMBED, EMBED)

    transposed = False  # export variants, set per instance: see forward
    fold_v = False

    def forward(self, q, k, v):
        """q (Lq, C), k/v (Lk, C) -> (Lq, C); ranks <= 3 throughout.

        ``transposed``: the same math as scores^T = K Q^T (heads, Lk, Lq), softmax over the key axis,
        out^T = V^T A^T (heads, d, Lq) -- the HTP runs A V with a 32-wide output (the head dim) far
        below its matmul throughput; this puts the query count (428) on the output width instead."""
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3)
        d = EMBED // HEADS
        if self.transposed:
            qt = (F.linear(q, wq, bq) * (d ** -0.5)).view(-1, HEADS, d).permute(1, 2, 0)  # (h, d, Lq)
            kt = F.linear(k, wk, bk).view(-1, HEADS, d).transpose(0, 1)  # (h, Lk, d)
            vt = F.linear(v, wv, bv).view(-1, HEADS, d).permute(1, 2, 0)  # (h, d, Lk)
            a = torch.softmax(torch.matmul(kt, qt), dim=1)  # (h, Lk, Lq)
            return self.out_proj(torch.matmul(vt, a).permute(2, 0, 1).reshape(-1, EMBED))
        q = (F.linear(q, wq, bq) * (d ** -0.5)).view(-1, HEADS, d).transpose(0, 1)
        k = F.linear(k, wk, bk).view(-1, HEADS, d).permute(1, 2, 0)
        v = F.linear(v, wv, bv).view(-1, HEADS, d).transpose(0, 1)
        a = torch.softmax(torch.matmul(q, k), dim=-1)
        if self.fold_v:
            # out_proj folded into each head's values: sum_h A_h (V_h W_o,h^T) + b_o, a 256-wide batched
            # matmul output instead of 32 (8x the MACs, no transpose of the (h, Lq, Lk) attention)
            wo = self.out_proj.weight.view(EMBED, HEADS, d).permute(1, 2, 0)  # (h, d, C)
            return torch.matmul(a, torch.matmul(v, wo)).sum(0) + self.out_proj.bias
        return self.out_proj(torch.matmul(a, v).transpose(0, 1).reshape(-1, EMBED))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(EMBED, 2048)), nn.Linear(2048, EMBED)])

    def forward(self, x):
        return x + self.layers[1](F.relu(self.layers[0](x)))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([nn.Module(), nn.Module()])
        self.attentions[0].attn, self.attentions[1].attn = MHA(), MHA()
        self.norms = nn.ModuleList([nn.LayerNorm(EMBED) for _ in range(3)])
        self.ffns = nn.ModuleList([FFN()])

    def forward(self, query, query_pos, temp_memory, temp_pos, key, key_pos):
        """query (Q, C); temp_memory (M, C); key (T, C). Post-norm: SA, N, CA, N, FFN, N."""
        tk = torch.cat([query, temp_memory], 0)
        tp = torch.cat([query_pos, temp_pos], 0)
        query = self.norms[0](query + self.attentions[0].attn(query + query_pos, tk + tp, tk))
        query = self.norms[1](query + self.attentions[1].attn(query + query_pos, key + key_pos, key))
        return self.norms[2](self.ffns[0](query))


def _branches():
    cls = nn.Sequential(nn.Linear(EMBED, EMBED), nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED),
                        nn.LayerNorm(EMBED), nn.ReLU(), nn.Linear(EMBED, 10))
    reg = nn.Sequential(nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, 10))
    return cls, reg


class HeadParams(nn.Module):
    """StreamPETRHead's learned modules, checkpoint names without the ``pts_bbox_head.`` prefix."""

    def __init__(self):
        super().__init__()
        cls, reg = _branches()
        self.cls_branches, self.reg_branches = nn.ModuleList([cls]), nn.ModuleList([reg])  # 6 shared copies
        self.position_encoder = nn.Sequential(nn.Linear(DEPTH_NUM * 3, EMBED * 4), nn.ReLU(), nn.Linear(EMBED * 4, EMBED))
        self.memory_embed = nn.Sequential(nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED))
        self.featurized_pe = SELayerLinear(EMBED)
        self.reference_points = nn.Embedding(NUM_QUERY, 3)
        self.pseudo_reference_points = nn.Embedding(NUM_PROP, 3)
        self.query_embedding = nn.Sequential(nn.Linear(EMBED * 3 // 2, EMBED), nn.ReLU(), nn.Linear(EMBED, EMBED))
        self.spatial_alignment = MLN(8)
        self.time_embedding = nn.Sequential(nn.Linear(EMBED, EMBED), nn.LayerNorm(EMBED))
        self.ego_pose_pe, self.ego_pose_memory = MLN(180), MLN(180)
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(6)])
        self.post_norm = nn.LayerNorm(EMBED)
        self.register_buffer("coords_d", torch.zeros(DEPTH_NUM))


def load_official(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    img, head = ImageBranch(), HeadParams()
    isd, hsd = {}, {}
    for k, v in sd.items():
        if k.startswith("img_backbone."):
            k = k[len("img_backbone."):]
            top = k.split(".")[0]
            if top in ("layer3", "layer4"):
                isd[k] = v
            else:
                idx = {"conv1": 0, "bn1": 1, "layer1": 4, "layer2": 5}[top]
                isd[f"backbone.{idx}" + k[len(top):]] = v
        elif k.startswith("img_neck."):
            isd["neck." + k[len("img_neck."):].replace(".conv.", ".")] = v
        elif k.startswith("pts_bbox_head."):
            k = k[len("pts_bbox_head."):]
            if k.startswith(("cls_branches.", "reg_branches.")):
                if k.split(".")[1] != "0":
                    continue
            elif k.startswith("transformer.decoder.layers."):
                k = "decoder_layers." + k[len("transformer.decoder.layers."):]
            elif k.startswith("transformer.decoder.post_norm."):
                k = "post_norm." + k[len("transformer.decoder.post_norm."):]
            elif k in ("code_weights", "match_costs", "pc_range", "position_range"):
                continue
            hsd[k] = v
    img.load_state_dict(isd)
    head.load_state_dict(hsd)
    return img.eval(), head.eval()


def locations(h, w, pad_h=256, pad_w=704):
    xs = (torch.arange(0, STRIDE * w, STRIDE, dtype=torch.float32) + STRIDE // 2) / pad_w
    ys = (torch.arange(0, STRIDE * h, STRIDE, dtype=torch.float32) + STRIDE // 2) / pad_h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), 1).reshape(h, w, 2)


def position_embedding(head, lidar2img, intrinsics, h=16, w=44, pad_h=256, pad_w=704):
    """StreamPETRHead.position_embeding (topk_indexes None): -> pe_in (T, 192), cone (T, 8).

    Depends only on the camera rig (lidar2img, intrinsics), not on the images."""
    eps = 1e-5
    n = lidar2img.shape[0]
    intr = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], -1).abs() / 1e3  # (N, 2)
    intr = intr[None].repeat(1, h * w, 1).view(1, -1, 2)  # upstream's repeat: token t -> camera t % N
    length = intr.shape[1]
    centers = locations(h, w, pad_h, pad_w)[None].repeat(n, 1, 1, 1).clone()
    centers[..., 0] *= pad_w
    centers[..., 1] *= pad_h
    d = head.coords_d.shape[0]
    centers = centers.view(1, length, 1, 2).repeat(1, 1, d, 1)
    coords = torch.cat([centers, head.coords_d.view(1, 1, d, 1).repeat(1, length, 1, 1)], -1)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
    img2lidars = lidar2img.inverse().view(n, 1, 1, 4, 4).repeat(1, h * w, d, 1, 1).view(1, length, d, 4, 4)
    coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]
    pr = torch.tensor(POS_RANGE)
    coords3d = (coords3d - pr[:3]) / (pr[3:] - pr[:3])
    coords3d = coords3d.reshape(1, -1, d * 3)
    cone = torch.cat([intr, coords3d[..., -3:], coords3d[..., -90:-87]], -1)
    return inverse_sigmoid(coords3d)[0], cone[0]


def transform_reference_points(ref, pose):
    ref = torch.cat([ref, torch.ones_like(ref[..., :1])], -1)
    return (pose.unsqueeze(1) @ ref.unsqueeze(-1)).squeeze(-1)[..., :3]


def denormalize(bbox):
    """(K, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy] -> (K, 9) with log-sizes exp'ed, yaw."""
    rot = torch.atan2(bbox[:, 6:7], bbox[:, 7:8])
    return torch.cat([bbox[:, 0:3], bbox[:, 3:6].exp(), rot, bbox[:, 8:10]], -1)


def decode(cls_scores, bbox_preds, max_num=300):
    """NMSFreeCoder.decode_single + StreamPETRHead.get_bboxes (bottom-center z)."""
    scores, idx = cls_scores.sigmoid().view(-1).topk(max_num)
    labels = idx % 10
    boxes = denormalize(bbox_preds[torch.div(idx, 10, rounding_mode="floor")])
    pcr = torch.tensor(POS_RANGE)
    keep = ((boxes[:, :3] >= pcr[:3]).all(1) & (boxes[:, :3] <= pcr[3:]).all(1))
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    boxes[:, 2] = boxes[:, 2] - boxes[:, 5] * 0.5
    return boxes, scores, labels


# ----------------------------------------------------------------------------- upstream reference
class UpstreamHead:
    """Literal StreamPETRHead test-time forward with its memory queue (B = 1)."""

    def __init__(self, head):
        self.h = head
        self.pc_range = torch.tensor(PC_RANGE)
        self.memory_embedding = None

    @torch.no_grad()
    def __call__(self, feats, data, prev_exists):
        h, pc = self.h, self.pc_range
        x = torch.tensor([float(prev_exists)])
        b = 1
        # pre_update_memory
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(b, MEMORY_LEN, EMBED)
            self.memory_reference_point = x.new_zeros(b, MEMORY_LEN, 3)
            self.memory_timestamp = x.new_zeros(b, MEMORY_LEN, 1)
            self.memory_egopose = x.new_zeros(b, MEMORY_LEN, 4, 4)
            self.memory_velo = x.new_zeros(b, MEMORY_LEN, 2)
        else:
            self.memory_timestamp = self.memory_timestamp + data["timestamp"].view(1, 1, 1)
            self.memory_egopose = data["ego_pose_inv"].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data["ego_pose_inv"])
            refresh = lambda m: m[:, :MEMORY_LEN] * x.view(-1, *([1] * (m.dim() - 1)))  # noqa: E731
            self.memory_timestamp = refresh(self.memory_timestamp)
            self.memory_reference_point = refresh(self.memory_reference_point)
            self.memory_embedding = refresh(self.memory_embedding)
            self.memory_egopose = refresh(self.memory_egopose)
            self.memory_velo = refresh(self.memory_velo)
        pseudo = h.pseudo_reference_points.weight * (pc[3:6] - pc[0:3]) + pc[0:3]
        self.memory_reference_point[:, :NUM_PROP] = self.memory_reference_point[:, :NUM_PROP] + (1 - x).view(b, 1, 1) * pseudo
        self.memory_egopose[:, :NUM_PROP] = self.memory_egopose[:, :NUM_PROP] + (1 - x).view(b, 1, 1, 1) * torch.eye(4)

        n, c, fh, fw = feats.shape
        memory = feats[None].permute(0, 1, 3, 4, 2).reshape(b, n * fh * fw, c)
        pe_in, cone = position_embedding(h, data["lidar2img"], data["intrinsics"], fh, fw)
        pos_embed = h.position_encoder(pe_in)[None]
        memory = h.memory_embed(memory)
        memory = h.spatial_alignment(memory, cone[None])
        pos_embed = h.featurized_pe(pos_embed, memory)

        reference_points = h.reference_points.weight.unsqueeze(0)
        query_pos = h.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        # temporal_alignment
        temp_reference_point = (self.memory_reference_point - pc[:3]) / (pc[3:6] - pc[0:3])
        temp_pos = h.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(b, query_pos.size(1), 1, 1)
        rec_ego_motion = torch.cat([torch.zeros_like(reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], -1)
        rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
        tgt = h.ego_pose_memory(tgt, rec_ego_motion)
        query_pos = h.ego_pose_pe(query_pos, rec_ego_motion)
        memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp,
                                       self.memory_egopose[..., :3, :].flatten(-2)], -1).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
        temp_pos = h.ego_pose_pe(temp_pos, memory_ego_motion)
        temp_memory = h.ego_pose_memory(temp_memory, memory_ego_motion)
        query_pos = query_pos + h.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos = temp_pos + h.time_embedding(pos2posemb1d(self.memory_timestamp).float())
        tgt = torch.cat([tgt, temp_memory[:, :NUM_PROP]], 1)
        query_pos = torch.cat([query_pos, temp_pos[:, :NUM_PROP]], 1)
        reference_points = torch.cat([reference_points, temp_reference_point[:, :NUM_PROP]], 1)
        rec_ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(b, query_pos.shape[1] + NUM_PROP, 1, 1)
        temp_memory, temp_pos = temp_memory[:, NUM_PROP:], temp_pos[:, NUM_PROP:]

        # PETRTemporalTransformer, return_intermediate with post_norm
        q, outs = tgt[0], []
        for layer in h.decoder_layers:
            q = layer(q, query_pos[0], temp_memory[0], temp_pos[0], memory[0], pos_embed[0])
            outs.append(h.post_norm(q))
        outs_dec = torch.nan_to_num(torch.stack(outs))[:, None]
        cls_l, reg_l = [], []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            cls_l.append(h.cls_branches[0](outs_dec[lvl]))
            tmp = h.reg_branches[0](outs_dec[lvl])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            reg_l.append(tmp)
        all_cls, all_bbox = torch.stack(cls_l), torch.stack(reg_l)
        all_bbox[..., 0:3] = all_bbox[..., 0:3] * (pc[3:6] - pc[0:3]) + pc[0:3]

        # post_update_memory
        rec_reference_points = all_bbox[..., :3][-1]
        rec_velo = all_bbox[..., -2:][-1]
        rec_memory = outs_dec[-1]
        rec_score = all_cls[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        _, topk = torch.topk(rec_score, TOPK, dim=1)
        g = lambda f: torch.gather(f, 1, topk.view(b, TOPK, *([1] * (f.dim() - 2))).expand(-1, -1, *f.shape[2:]))  # noqa: E731
        self.memory_embedding = torch.cat([g(rec_memory), self.memory_embedding], 1)
        self.memory_timestamp = torch.cat([g(rec_timestamp), self.memory_timestamp], 1)
        self.memory_egopose = torch.cat([g(rec_ego_pose), self.memory_egopose], 1)
        self.memory_reference_point = torch.cat([g(rec_reference_points), self.memory_reference_point], 1)
        self.memory_velo = torch.cat([g(rec_velo), self.memory_velo], 1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data["ego_pose"])
        self.memory_timestamp = self.memory_timestamp - data["timestamp"].view(1, 1, 1)
        self.memory_egopose = data["ego_pose"].unsqueeze(1) @ self.memory_egopose
        return all_cls[-1, 0], all_bbox[-1, 0]


# ----------------------------------------------------------------------------- deployment split
class HeadCore(nn.Module):
    """The HTP piece: image tokens + host encodings -> last decoder level's (cls, reg, dec).

    inputs  feat     (N*16*44, 256) image tokens (N, h, w order), e.g. the image piece's NHWC output
            pe       (T, 256)   position_encoder(inverse_sigmoid(coords3d))  } host, once per camera rig
            sa_gamma (T, 256)   spatial_alignment's gamma / beta of the cone  }
            sa_beta  (T, 256)                                                }
            mem_emb  (512, 256) memory embeddings                            } host, per frame
            mem_pe3d (512, 384) pos2posemb3d(normalized memory reference)    }
            mem_time (512, 256) pos2posemb1d(memory timestamp)               }
            mem_motion (512, 180) nerf(velo, timestamp, egopose[:3])         }
    outputs cls (428, 10) logits, reg (428, 10) raw regression (before + inverse_sigmoid(ref)),
            dec (428, 256) last post-normed decoder output (the propagated memory)."""

    def __init__(self, head):
        super().__init__()
        self.h = head
        with torch.no_grad():
            ref = head.reference_points.weight[None]
            qpos = head.query_embedding(pos2posemb3d(ref))
            motion = nerf_positional_encoding(torch.cat([torch.zeros_like(ref), torch.eye(4)[:3].flatten()
                                                         .expand(1, NUM_QUERY, 12)], -1))
            tgt0 = head.ego_pose_memory(torch.zeros_like(qpos), motion)
            qpos0 = head.ego_pose_pe(qpos, motion) + head.time_embedding(pos2posemb1d(torch.zeros_like(ref[..., :1])))
        self.register_buffer("tgt0", tgt0[0].clone())
        self.register_buffer("qpos0", qpos0[0].clone())

    def forward(self, feat, pe, sa_gamma, sa_beta, mem_emb, mem_pe3d, mem_time, mem_motion):
        h = self.h
        memory = h.memory_embed(feat)
        memory = sa_gamma * h.spatial_alignment.ln(memory) + sa_beta
        pos_embed = h.featurized_pe(pe, memory)
        temp_pos = h.ego_pose_pe(h.query_embedding(mem_pe3d), mem_motion) + h.time_embedding(mem_time)
        temp_memory = h.ego_pose_memory(mem_emb, mem_motion)
        q = torch.cat([self.tgt0, temp_memory[:NUM_PROP]], 0)
        qpos = torch.cat([self.qpos0, temp_pos[:NUM_PROP]], 0)
        tm, tp = temp_memory[NUM_PROP:], temp_pos[NUM_PROP:]
        for layer in h.decoder_layers:
            q = layer(q, qpos, tm, tp, memory, pos_embed)
        dec = h.post_norm(q)
        return h.cls_branches[0](dec), h.reg_branches[0](dec), dec


class HostState:
    """StreamPETR's memory queue and the host-side encodings around ``HeadCore`` (numpy/torch CPU)."""

    def __init__(self, head):
        self.h = head
        self.pc = torch.tensor(PC_RANGE)
        self.rig_key, self.rig = None, None
        self.reset()

    def reset(self):
        self.emb = torch.zeros(MEMORY_LEN, EMBED)
        self.ref = torch.zeros(MEMORY_LEN, 3)
        self.ts = torch.zeros(MEMORY_LEN, 1, dtype=torch.float64)
        self.pose = torch.zeros(MEMORY_LEN, 4, 4)
        self.velo = torch.zeros(MEMORY_LEN, 2)

    @torch.no_grad()
    def rig_inputs(self, data):
        key = (data["lidar2img"].numpy().tobytes(), data["intrinsics"].numpy().tobytes())
        if key != self.rig_key:
            self.rig_key = key
            pe_in, cone = position_embedding(self.h, data["lidar2img"], data["intrinsics"])
            sa = self.h.spatial_alignment
            c = sa.reduce(cone)
            self.rig = {"pe": self.h.position_encoder(pe_in), "sa_gamma": sa.gamma(c), "sa_beta": sa.beta(c)}
        return self.rig

    @torch.no_grad()
    def pre(self, data, prev_exists):
        """pre_update_memory + temporal_alignment's host part -> HeadCore memory inputs."""
        if not prev_exists:
            self.reset()
        else:
            inv = data["ego_pose_inv"][0]
            self.ts = (self.ts + data["timestamp"])[:MEMORY_LEN]
            self.pose = (inv @ self.pose)[:MEMORY_LEN]
            self.ref = transform_reference_points(self.ref[None], inv[None])[0][:MEMORY_LEN]
            self.emb, self.velo = self.emb[:MEMORY_LEN], self.velo[:MEMORY_LEN]
            self.pose, self.ts, self.ref = self.pose[:MEMORY_LEN], self.ts[:MEMORY_LEN], self.ref[:MEMORY_LEN]
        if not prev_exists:
            pc = self.pc
            self.ref[:NUM_PROP] += self.h.pseudo_reference_points.weight * (pc[3:6] - pc[0:3]) + pc[0:3]
            self.pose[:NUM_PROP] += torch.eye(4)
        refn = (self.ref - self.pc[:3]) / (self.pc[3:6] - self.pc[0:3])
        motion = torch.cat([self.velo.double(), self.ts, self.pose[:, :3, :].flatten(-2).double()], -1).float()
        self.refn_prop = refn[:NUM_PROP]
        return {"mem_emb": self.emb, "mem_pe3d": pos2posemb3d(refn), "mem_time": pos2posemb1d(self.ts).float(),
                "mem_motion": nerf_positional_encoding(motion)}

    @torch.no_grad()
    def post(self, data, cls, reg, dec):
        """Box coordinates + post_update_memory. -> (cls (428, 10), bbox (428, 10)) like the reference."""
        ref = torch.cat([self.h.reference_points.weight, self.refn_prop], 0)
        bbox = reg.clone()
        bbox[:, 0:3] = (bbox[:, 0:3] + inverse_sigmoid(ref)).sigmoid() * (self.pc[3:6] - self.pc[0:3]) + self.pc[0:3]
        score = cls.sigmoid().max(-1).values
        top = torch.topk(score, TOPK).indices
        self.emb = torch.cat([dec[top], self.emb], 0)
        self.ts = torch.cat([torch.zeros(TOPK, 1, dtype=torch.float64), self.ts], 0) - data["timestamp"]
        pose = data["ego_pose"][0]
        self.pose = pose @ torch.cat([torch.eye(4).expand(TOPK, 4, 4), self.pose], 0)
        self.ref = transform_reference_points(torch.cat([bbox[top, :3], self.ref], 0)[None], pose[None])[0]
        self.velo = torch.cat([bbox[top, 8:10], self.velo], 0)
        return cls, bbox


def to_torch(frame):
    """data.py frame -> the tensors both head paths take."""
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"lidar2img": f32(frame["lidar2img"]), "intrinsics": f32(frame["intrinsics"]),
            "ego_pose": f32(frame["ego_pose"])[None], "ego_pose_inv": f32(frame["ego_pose_inv"])[None],
            "timestamp": torch.tensor(frame["timestamp"], dtype=torch.float64)}
