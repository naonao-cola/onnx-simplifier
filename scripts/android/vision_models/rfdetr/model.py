"""RF-DETR (roboflow/rf-detr, Apache-2.0) loaded through the rfdetr package, in its export mode
(forward_export(pixels) -> (boxes cxcywh in [0, 1], logits)), with one HTP-oriented patch:

- `MSDeformAttn.forward` computes the same deformable attention with every tensor at rank <= 4
  (batch 1, heads as the GridSample batch), like ../rtdetr/model.py did for RT-DETR. The library's
  export path builds (B, Q, heads, levels*points, 2) = rank-5 sampling locations, which QNN refuses.

The DINOv2 window partition/merge (also rank 5) is rewritten on the ONNX graph instead, see
export.py `fold_rank5_transposes`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def _msda_forward_r4(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
    input_spatial_shapes_hw=None,
):
    assert query.shape[0] == 1 and input_padding_mask is None
    H, L, P, C = self.n_heads, self.n_levels, self.n_points, self.d_model
    D = C // H
    Q = query.shape[1]
    q = query[0]  # (Q, C)
    value = self.value_proj(input_flatten[0])  # (S, C)
    value = value.reshape(-1, H, D).permute(1, 2, 0)  # (H, D, S)
    off = (
        self.sampling_offsets(q).reshape(Q, H, L * P, 2).permute(1, 0, 2, 3)
    )  # (H, Q, LP, 2)
    attn = (
        self.attention_weights(q).reshape(Q, H, L * P).softmax(-1).permute(1, 0, 2)
    )  # (H, Q, LP)
    ref = reference_points[0]  # (Q, Lr, 2|4), Lr in (1, L)
    out = None
    start = 0
    for lvl, (h, w) in enumerate(input_spatial_shapes_hw):
        r = ref[:, lvl if ref.shape[1] > 1 else 0]  # (Q, 2|4)
        o = off[:, :, lvl * P : (lvl + 1) * P]  # (H, Q, P, 2)
        if r.shape[-1] == 2:
            norm = torch.tensor([float(w), float(h)], dtype=o.dtype)
            loc = r[None, :, None, :] + o / norm
        else:
            loc = r[None, :, None, :2] + o / P * r[None, :, None, 2:] * 0.5
        v = value[:, :, start : start + h * w].reshape(H, D, h, w)
        start += h * w
        s = F.grid_sample(
            v, 2 * loc - 1, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # (H, D, Q, P)
        a = attn[:, :, lvl * P : (lvl + 1) * P].reshape(H, 1, Q, P)
        t = (s * a).sum(-1)  # (H, D, Q)
        out = t if out is None else out + t
    out = out.reshape(C, Q).transpose(0, 1)  # (Q, C)
    return self.output_proj(out)[None]


def load(variant: str, patched: bool = True):
    """Returns (core module in export mode, resolution). `variant` is nano | small | medium,
    optionally `@<res>` for another square resolution (RF-DETR's weight-sharing NAS: the same
    weights run at any multiple of patch_size * num_windows, positional encodings interpolated)."""
    import rfdetr
    from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

    name, _, res = variant.partition("@")
    m = getattr(rfdetr, VARIANTS[name])(**({"resolution": int(res)} if res else {}))
    core = m.model.model.eval()
    core.export()
    if patched:
        for mod in core.modules():
            if isinstance(mod, MSDeformAttn):
                mod.forward = _msda_forward_r4.__get__(mod)
    return core, m.model.resolution


class Wrapped(torch.nn.Module):
    """pixels (1,3,R,R) normalized -> (logits, boxes); the output order phone_eval expects."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, pixels):
        boxes, logits = self.core(pixels)[:2]
        return logits, boxes
