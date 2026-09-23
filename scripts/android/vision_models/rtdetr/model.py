"""RT-DETR-r18vd (HF transformers `PekingU/rtdetr_r18vd`) with an HTP-friendly, exact rewrite of the
decoder's multi-scale deformable attention (MSDA).

HF's MSDA builds (batch, queries, heads, levels, points, 2) sampling locations -- rank 6 -- and QNN
refuses every op touching them (39 ops, see ../../vision_models_plan.md item 3). `msda_forward`
computes the same thing with every tensor rank <= 4, one level at a time, heads as the batch axis:

  offsets_l = x @ Woff_l + boff_l            (H, Q, P*2)   per-head MatMul, no 6-D view
  grid_l    = (2 ref_xy - 1) + offsets_l * ref_wh / P  (H, Q, P, 2)   [== 2 * loc - 1]
  logits    = x @ Wattn + battn                (H, Q, L*P) -> softmax over L*P
  value_l   = value[level rows]^T              (H, D, h_l, w_l)
  out      += sum_P grid_sample(value_l, grid_l) * attn_l    (H, D, Q)

The per-level weights are the original Linear weights, re-sliced (see `_split`). `validate.py`
checks the patched model against HF's own on real images (max abs diff ~1e-5).

`patch(model)` also replaces the anchors' "invalid" marker (float32 max, which is inf in fp16) by a
finite 1e4: both saturate sigmoid to exactly 1 in fp32, so the model is unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

MODEL_ID = "PekingU/rtdetr_r18vd"
MODEL_REV = "ac77a11ff0170a41b771c03264987f8ce2b0d753"
MODEL_SHA256 = "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093"  # model.safetensors
SIZE = 640
SHAPES = [(80, 80), (40, 40), (20, 20)]  # strides 8/16/32 at 640x640
INVALID_ANCHOR = 1e4


def _split(m):
    """Per-level, per-head MatMul weights out of sampling_offsets / attention_weights Linears.

    sampling_offsets: out features ordered (H, L, P, 2); attention_weights: (H, L*P).
    Returns woff (L, H, C, P*2), boff (L, H, 1, P*2), wat (H, C, L*P), bat (H, 1, L*P).
    """
    H, L, P = m.n_heads, m.n_levels, m.n_points
    C = m.d_model
    w = m.sampling_offsets.weight.detach().view(H, L, P * 2, C)
    b = m.sampling_offsets.bias.detach().view(H, L, 1, P * 2)
    woff = w.permute(1, 0, 3, 2).contiguous()  # L, H, C, P*2
    boff = b.permute(1, 0, 2, 3).contiguous()  # L, H, 1, P*2
    wa = (
        m.attention_weights.weight.detach()
        .view(H, L * P, C)
        .permute(0, 2, 1)
        .contiguous()
    )
    ba = m.attention_weights.bias.detach().view(H, 1, L * P)
    return woff, boff, wa, ba


def msda_sampling(m, x, ref):
    """x (1, Q, C) queries (+pos), ref (1, Q, 4) cx,cy,w,h in [0,1].
    Returns grids [L x (H, Q, P, 2)] in grid_sample's [-1, 1] and attn (H, Q, L*P)."""
    H, L, P = m.n_heads, m.n_levels, m.n_points
    woff, boff, wa, ba = m._rk4
    Q = x.shape[1]
    xy = (2 * ref[..., :2] - 1).reshape(1, Q, 1, 2)
    wh = (ref[..., 2:] * (1.0 / P)).reshape(1, Q, 1, 2)  # 2 * 0.5 / P
    grids = []
    for lvl in range(L):
        off = (torch.matmul(x, woff[lvl]) + boff[lvl]).reshape(H, Q, P, 2)
        grids.append(xy + off * wh)
    attn = torch.softmax(torch.matmul(x, wa) + ba, dim=-1)  # H, Q, L*P
    return grids, attn


def msda_gather(m, value, grids, attn, shapes=SHAPES):
    """value (1, S, C) after value_proj; returns (1, Q, C) before output_proj."""
    H, P = m.n_heads, m.n_points
    C = m.d_model
    D = C // H
    Q = attn.shape[1]
    out = None
    start = 0
    for lvl, (h, w) in enumerate(shapes):
        v = value[0, start : start + h * w].transpose(0, 1).reshape(H, D, h, w)
        start += h * w
        s = F.grid_sample(
            v, grids[lvl], mode="bilinear", padding_mode="zeros", align_corners=False
        )
        a = attn[:, :, lvl * P : (lvl + 1) * P].reshape(H, 1, Q, P)
        o = (s * a).sum(-1)  # H, D, Q
        out = o if out is None else out + o
    return out.reshape(C, Q).transpose(0, 1).reshape(1, Q, C)


def _msda_forward(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    position_embeddings=None,
    reference_points=None,
    spatial_shapes=None,
    spatial_shapes_list=None,
    level_start_index=None,
    **kw,
):
    assert attention_mask is None
    if position_embeddings is not None:
        hidden_states = hidden_states + position_embeddings
    value = self.value_proj(encoder_hidden_states)
    ref = reference_points[:, :, 0, :]  # (1, Q, 4): one reference box for every level
    grids, attn = msda_sampling(self, hidden_states, ref)
    out = msda_gather(
        self,
        value,
        grids,
        attn,
        [tuple(int(v) for v in s) for s in spatial_shapes_list],
    )
    return self.output_proj(out), None


def patch(model):
    """In place: rank-<=4 MSDA in every decoder layer, finite invalid-anchor marker."""
    inner = model.model if hasattr(model, "model") else model
    for layer in inner.decoder.layers:
        m = layer.encoder_attn
        m._rk4 = _split(m)
        m.forward = types.MethodType(_msda_forward, m)
    orig = (
        inner._cached_generate_anchors
        if hasattr(inner, "_cached_generate_anchors")
        else None
    )

    def gen(
        self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32
    ):
        if spatial_shapes is None:
            spatial_shapes = tuple(SHAPES)
        a, v = orig(
            tuple(tuple(int(t) for t in s) for s in spatial_shapes),
            grid_size,
            device,
            dtype,
        )
        return torch.where(v, a, torch.full((), INVALID_ANCHOR, dtype=dtype)), v

    inner.generate_anchors = types.MethodType(gen, inner)
    return model


def load(patched=True):
    from transformers import RTDetrForObjectDetection

    m = RTDetrForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REV).eval()
    return patch(m) if patched else m


class Full(torch.nn.Module):
    """pixels (1, 3, 640, 640) in [0, 1] -> logits (1, 300, 80), boxes (1, 300, 4) cxcywh in [0, 1]."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits, o.pred_boxes
