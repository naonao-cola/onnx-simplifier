"""Segment-Anything variants as two exportable pieces with one interface.

Every variant becomes
  encoder: pixels  float32 [1, 3, S, S], raw RGB 0..255, resized longest side -> S and padded
           (bottom/right) with the mean pixel             -> image_embeddings [1, 256, 64, 64]
  decoder: image_embeddings, point_coords [1, 2, 2] (x, y in the 1024 prompt frame),
           point_labels [1, 2] (1 fg, 0 bg, 2/3 box corners, -1 padding)
                                                           -> iou_predictions [1, 4],
                                                              low_res_masks   [1, 4, 256, 256]
Normalization lives in the encoder graph, so a quantized encoder's input Q has scale 1, zero
point 0: the uint8 input *is* the padded RGB image. The decoder has no mask input (no-mask
embedding) and no resize to the original image size -- both would be dynamic; the host upsamples
the chosen 256x256 mask. Two prompt slots cover a point (+ padding) and a box (two corners).

The SAM-family decoders (SAM, MobileSAM, EdgeSAM, EfficientViT-SAM) share segment_anything's
PromptEncoder/MaskDecoder, so one wrapper (SamDecoder, the same math as segment_anything's
SamOnnxModel / EdgeSAM's SamCoreMLModel) serves all of them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

SRC = Path(os.environ.get("SAM_SRC", Path.home() / ".cache/sam-src"))
WEIGHTS = Path(os.environ.get("SAM_WEIGHTS", Path.home() / ".cache/sam-weights"))
SAM_MEAN = [123.675, 116.28, 103.53]
SAM_STD = [58.395, 57.12, 57.375]


@dataclass
class Parts:
    encoder: nn.Module  # normalized image [1, 3, S, S] -> [1, 256, 64, 64]
    prompt_encoder: nn.Module
    mask_decoder: nn.Module
    size: int  # encoder input side S
    mean: list
    std: list
    source: str  # weights provenance for the README


class Encoder(nn.Module):
    def __init__(self, p: Parts):
        super().__init__()
        self.enc = p.encoder
        self.register_buffer("mean", torch.tensor(p.mean).view(1, 3, 1, 1))
        self.register_buffer("inv_std", 1.0 / torch.tensor(p.std).view(1, 3, 1, 1))

    def forward(self, pixels):
        return self.enc((pixels - self.mean) * self.inv_std)


class SamDecoder(nn.Module):
    def __init__(self, p: Parts, prompt_size: int = 1024):
        super().__init__()
        self.pe, self.md, self.prompt_size = p.prompt_encoder, p.mask_decoder, prompt_size

    def forward(self, image_embeddings, point_coords, point_labels):
        pe = self.pe
        coords = (point_coords + 0.5) / self.prompt_size
        emb = pe.pe_layer._pe_encoding(coords)
        lab = point_labels.unsqueeze(-1)
        emb = emb * (lab != -1) + pe.not_a_point_embed.weight * (lab == -1)
        for i in range(pe.num_point_embeddings):
            emb = emb + pe.point_embeddings[i].weight * (lab == i)
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, *pe.image_embedding_size)
        return self.predict_masks(image_embeddings, pe.get_dense_pe(), emb, dense)

    def forward_upstream(self, image_embeddings, point_coords, point_labels):
        """The same prompt embedding, then upstream MaskDecoder.predict_masks."""
        try:
            pm = self.md.predict_masks
            self.predict_masks = lambda e, pe, sp, de: tuple(reversed(pm(
                image_embeddings=e, image_pe=pe, sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=de)))
            return self.forward(image_embeddings, point_coords, point_labels)
        finally:
            del self.predict_masks  # back to the class method

    def predict_masks(self, image_embeddings, image_pe, sparse, dense):
        """MaskDecoder.predict_masks for one image and one prompt set. Upstream's
        repeat_interleave(image_embeddings, n_prompts) exports as a rank-5 Unsqueeze + Tile,
        which crashes QNN's HTP graph compile (a segfault, not a refusal); with batch 1 it is the
        identity, so it is dropped."""
        md = self.md
        out_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat((out_tokens, sparse), dim=1)
        src = image_embeddings + dense
        b, c, h, w = src.shape
        hs, src = md.transformer(src, image_pe, tokens)
        iou_out, mask_out = hs[:, 0, :], hs[:, 1:(1 + md.num_mask_tokens), :]
        up = md.output_upscaling(src.transpose(1, 2).reshape(b, c, h, w))
        hyper = torch.stack([md.output_hypernetworks_mlps[i](mask_out[:, i, :])
                             for i in range(md.num_mask_tokens)], dim=1)
        b, c, h, w = up.shape
        masks = (hyper @ up.reshape(b, c, h * w)).reshape(b, -1, h, w)
        return md.iou_prediction_head(iou_out), masks


class SigmoidGelu(nn.Module):
    """x * sigmoid(1.702 x): the cheap GELU approximation (Hendrycks & Gimpel)."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TanhGelu(nn.Module):
    """GELU's tanh approximation spelled out as elementwise ops (Mul/Add/Tanh). QNN's Gelu op
    ignores approximate="tanh" (same time, same output as erf GELU), this form does not."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def set_gelu(module: nn.Module, mode: str) -> int:
    """Swap every nn.GELU under `module`: "exact" (erf, upstream), "tanh" (the Gelu op's
    approximate attribute), "tanh_ops" (TanhGelu), "sigmoid" (SigmoidGelu)."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU):
            if mode == "tanh":
                child.approximate = "tanh"
            elif mode == "sigmoid":
                setattr(module, name, SigmoidGelu())
            elif mode == "tanh_ops":
                setattr(module, name, TanhGelu())
            n += 1
        else:
            n += set_gelu(child, mode)
    return n


def set_upsample(module: nn.Module, mode: str) -> int:
    """Set the interpolation mode of every upsampling layer with a `mode` attribute (EfficientViT's
    UpSampleLayer: bicubic upstream)."""
    n = 0
    for m in module.modules():
        if type(m).__name__ == "UpSampleLayer" and getattr(m, "mode", None) != mode:
            m.mode = mode
            n += 1
    return n


def _sam_parts(sam, size, source, mean=SAM_MEAN, std=SAM_STD):
    return Parts(sam.image_encoder, sam.prompt_encoder, sam.mask_decoder, size, mean, std, source)


def _tinyvit_block_rank4(self, x):
    """TinyViTBlock.forward with the window partition/reverse in rank <= 4 (batch 1 only).

    Upstream views (B, nH, ws, nW, ws, C) and swaps axes 2/3: a rank-6 Reshape/Transpose, which
    the HTP refuses (the whole encoder then falls off strict all-HTP). With B = 1 the same
    permutation is (nH, ws, nW, ws*C) -> swap axes 1/2: each window row's ws*C values stay
    contiguous, so it is exact.
    """
    import torch.nn.functional as F

    H, W = self.input_resolution
    B, L, C = x.shape
    ws = self.window_size
    res_x = x
    if H == ws and W == ws:
        x = self.attn(x)
    else:
        assert B == 1
        x = x.view(H, W, C)
        pad_b, pad_r = (ws - H % ws) % ws, (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // ws, pW // ws
        x = x.reshape(nH, ws, nW, ws * C).transpose(1, 2).reshape(nH * nW, ws * ws, C)
        x = self.attn(x)
        x = x.reshape(nH, nW, ws, ws * C).transpose(1, 2).reshape(pH, pW, C)
        if pad_b or pad_r:
            x = x[:H, :W]
        x = x.reshape(1, L, C)
    x = res_x + self.drop_path(x)
    x = x.transpose(1, 2).reshape(B, C, H, W)
    x = self.local_conv(x)
    x = x.view(B, C, L).transpose(1, 2)
    return x + self.drop_path(self.mlp(x))


def _layernorm2d_channels_last(self, x):
    """SAM's LayerNorm2d (normalize over channels per pixel) as one LayerNormalization on the
    channels-last view. Upstream spells it out as mean / (x - u)^2 / sqrt: in fp16 on the HTP the
    square of the neck's ~1e3 activations overflows (max 65504) and the embedding is garbage.
    Same formula (biased variance, eps inside the sqrt); only float rounding differs."""
    import torch.nn.functional as F

    y = F.layer_norm(x.permute(0, 2, 3, 1), tuple(self.weight.shape), self.weight, self.bias,
                     self.eps)
    return y.permute(0, 3, 1, 2)


PATCHES: list = []  # (class, attribute, upstream, ours): export uses ours, upstream() swaps back


def _patch(cls, attr, fn):
    PATCHES.append((cls, attr, getattr(cls, attr), fn))
    setattr(cls, attr, fn)


class upstream:
    """`with upstream():` runs the unpatched modules (for the exactness check in export)."""

    def __enter__(self):
        for cls, attr, up, _ in PATCHES:
            setattr(cls, attr, up)

    def __exit__(self, *exc):
        for cls, attr, _, ours in PATCHES:
            setattr(cls, attr, ours)


def mobilesam():
    from mobile_sam import sam_model_registry
    from mobile_sam.modeling import common, tiny_vit_sam

    _patch(tiny_vit_sam.TinyViTBlock, "forward", _tinyvit_block_rank4)
    _patch(common.LayerNorm2d, "forward", _layernorm2d_channels_last)
    _patch(tiny_vit_sam.LayerNorm2d, "forward", _layernorm2d_channels_last)
    ck = SRC / "MobileSAM/weights/mobile_sam.pt"
    return _sam_parts(sam_model_registry["vit_t"](checkpoint=str(ck)).eval(), 1024,
                      "ChaoningZhang/MobileSAM weights/mobile_sam.pt")


def reparam(net: nn.Module) -> int:
    """RepViT-style structural reparameterization: every child with a fuse() (Conv2d_BN,
    Residual, RepVGGDW, BN_Linear) becomes its fused single conv/linear. Exact up to rounding."""
    n = 0
    for name, child in list(net.named_children()):
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, name, fused)
            n += 1 + reparam(fused)
        else:
            n += reparam(child)
    return n


def edgesam():
    from edge_sam import sam_model_registry
    from edge_sam.modeling import common

    _patch(common.LayerNorm2d, "forward", _layernorm2d_channels_last)
    ck = WEIGHTS / "edge_sam_3x.pth"
    # bilinear instead of bicubic in the encoder's FPN upsample, as EdgeSAM's own ONNX export does
    sam = sam_model_registry["edge_sam"](checkpoint=str(ck), upsample_mode="bilinear").eval()
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        ref = sam.image_encoder(x)
        n = reparam(sam.image_encoder)
        err = float((sam.image_encoder(x) - ref).abs().max())
    print(f"edgesam: {n} RepViT modules fused, max abs change {err:.2e}")
    return _sam_parts(sam, 1024, "huggingface chongzhou/EdgeSAM edge_sam_3x.pth")


def efficientvit_sam_l0():
    from efficientvit.models.nn import norm
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    from segment_anything.modeling import common

    _patch(norm.LayerNorm2d, "forward", _layernorm2d_channels_last)
    _patch(common.LayerNorm2d, "forward", _layernorm2d_channels_last)
    ck = WEIGHTS / "efficientvit_sam_l0.pt"
    sam = create_efficientvit_sam_model("efficientvit-sam-l0", weight_url=str(ck)).eval()
    # 512x512 encoder input; prompts stay in the 1024 frame (its prompt encoder's input size)
    return _sam_parts(sam, 512, "huggingface han-cai/efficientvit-sam efficientvit_sam_l0.pt")


def _window_partition_rank4(x, ws):
    """segment_anything window_partition for B = 1 without its rank-6 view/permute: (Hp/ws, ws,
    Wp/ws, ws*C) swap axes 1/2 keeps each window row's ws*C values together -- exact."""
    import torch.nn.functional as F

    B, H, W, C = x.shape
    assert B == 1
    pad_h, pad_w = (ws - H % ws) % ws, (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(Hp // ws, ws, Wp // ws, ws * C).transpose(1, 2)
    return x.reshape(-1, ws, ws, C), (Hp, Wp)


def _window_unpartition_rank4(windows, ws, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    C = windows.shape[-1]
    x = windows.reshape(Hp // ws, Wp // ws, ws, ws * C).transpose(1, 2).reshape(1, Hp, Wp, C)
    return x[:, :H, :W, :] if Hp > H or Wp > W else x


def _rel_pos_rank4(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """add_decomposed_rel_pos without rank-5 tensors: the two einsums as batched matmuls over
    the row (column) axis, the broadcast add on an (B*qh*qw, kh, kw) view -- exact."""
    from segment_anything.modeling.image_encoder import get_rel_pos

    (q_h, q_w), (k_h, k_w) = q_size, k_size
    Rh, Rw = get_rel_pos(q_h, k_h, rel_pos_h), get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = (r_q.permute(1, 0, 2, 3).reshape(q_h, B * q_w, dim) @ Rh.transpose(1, 2))
    rel_h = rel_h.reshape(q_h, B, q_w, k_h).permute(1, 0, 2, 3)
    rel_w = (r_q.permute(2, 0, 1, 3).reshape(q_w, B * q_h, dim) @ Rw.transpose(1, 2))
    rel_w = rel_w.reshape(q_w, B, q_h, k_w).permute(1, 2, 0, 3)
    n = B * q_h * q_w
    attn = (attn.reshape(n, k_h, k_w) + rel_h.reshape(n, k_h, 1) + rel_w.reshape(n, 1, k_w))
    return attn.reshape(B, q_h * q_w, k_h * k_w)


def _vit_attention_rank4(self, x):
    """segment_anything Attention.forward with rank <= 4 qkv split / head merge (exact)."""
    from segment_anything.modeling import image_encoder as ie

    B, H, W, _ = x.shape
    nh = self.num_heads
    qkv = self.qkv(x).reshape(B, H * W, 3 * nh, -1).permute(0, 2, 1, 3)
    hd = qkv.shape[-1]
    q, k, v = (t.reshape(B * nh, H * W, hd) for t in qkv.split(nh, dim=1))
    attn = (q * self.scale) @ k.transpose(-2, -1)
    if self.use_rel_pos:
        attn = ie.add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
    attn = attn.softmax(dim=-1)
    x = (attn @ v).reshape(B, nh, H * W, hd).permute(0, 2, 1, 3).reshape(B, H, W, nh * hd)
    return self.proj(x)


def sam_vit_b():
    from segment_anything import sam_model_registry
    from segment_anything.modeling import common
    from segment_anything.modeling import image_encoder as ie

    _patch(common.LayerNorm2d, "forward", _layernorm2d_channels_last)
    _patch(ie, "window_partition", _window_partition_rank4)
    _patch(ie, "window_unpartition", _window_unpartition_rank4)
    _patch(ie, "add_decomposed_rel_pos", _rel_pos_rank4)
    _patch(ie.Attention, "forward", _vit_attention_rank4)
    ck = WEIGHTS / "sam_vit_b_01ec64.pth"
    return _sam_parts(sam_model_registry["vit_b"](checkpoint=str(ck)).eval(), 1024,
                      "dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")


VARIANTS = {
    "mobilesam": mobilesam,
    "edgesam": edgesam,
    "efficientvit_sam_l0": efficientvit_sam_l0,
    "sam_vit_b": sam_vit_b,
}


def load(name: str) -> Parts:
    return VARIANTS[name]()
