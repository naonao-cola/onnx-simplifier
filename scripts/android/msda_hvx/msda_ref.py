#!/usr/bin/env python3
"""Torch reference of the HVX multi-scale deformable attention kernel (msda_kernel.h), case files
for its C checks, and synthetic cases.

`msda_reference(value, levels, loc, attw, mode=..., ref=..., vis=...)` is the kernel's contract
(README.md): for NV = 1 and mode "loc" it is mmcv's multi_scale_deformable_attn_pytorch (checked
against a verbatim copy, `mmcv_msda_pytorch`), with mmcv's (Q, M, L, P) layouts plus an NO axis:
  value (NV, S, M*D)          channels-last value maps, level l = rows [start_l, start_l + H_l*W_l)
  loc   (Q, M, NO, L, P, 2)   normalized locations (mode "loc") or raw offsets ("pix", "box")
  attw  (Q, M, NO, L, P)      attention weights (softmaxed over L*P, as the model does)
  ref   (NVR, Q, RL, R, RD)   reference points for "pix" (loc = ref.xy + off / (W_l, H_l)) and
                              "box" (loc = ref.xy + off / P * ref.wh * 0.5, RD = 4); point p uses
                              entry p % R; NVR in {1, NV}, RL in {1, L}
  vis   (NV, Q) uint8 or None visibility; the output averages over the visible maps
  -> out (Q, M*D)

usage: msda_ref.py synth <kind> <out dir> [--q N] [--seed S]   kind: rtdetr_decoder, bevformer_tsa,
                                                                bevformer_sca, loc_small
       msda_ref.py mmcv-check                                 msda_reference vs mmcv's code
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODES = {"loc": 0, "pix": 1, "box": 2}


def level_starts(levels):
    starts, s = [], 0
    for h, w in levels:
        starts.append(s)
        s += h * w
    return starts


def locations(loc, levels, v, mode="loc", ref=None):
    """Normalized sampling locations (Q, M, L, P, 2) of value map v (for o = v when NO > 1)."""
    q, m, no, nl, p, _ = loc.shape
    off = loc[:, :, v if no > 1 else 0]  # (Q, M, L, P, 2)
    if mode == "loc":
        return off
    nvr, _, rl, r, rd = ref.shape
    rv = ref[v if nvr > 1 else 0]  # (Q, RL, R, RD)
    rp = rv[:, :, torch.arange(p) % r]  # (Q, RL, P, RD)
    rp = rp.expand(q, nl, p, rd)[:, None]  # (Q, 1, L, P, RD)
    if mode == "pix":
        norm = torch.tensor([[w, h] for h, w in levels], dtype=loc.dtype)[None, None, :, None]  # (1,1,L,1,2)
        return rp[..., :2] + off / norm
    assert mode == "box" and rd == 4
    return rp[..., :2] + off / p * rp[..., 2:] * 0.5


def msda_reference(value, levels, loc, attw, mode="loc", ref=None, vis=None, starts=None):
    nv, _, c = value.shape
    q, m, no, nl, p, _ = loc.shape
    d = c // m
    starts = starts or level_starts(levels)
    acc = torch.zeros(q, c, dtype=value.dtype)
    cnt = torch.zeros(q, 1, dtype=value.dtype)
    for v in range(nv):
        ln = locations(loc, levels, v, mode, ref)  # (Q, M, L, P, 2)
        a = attw[:, :, v if no > 1 else 0]  # (Q, M, L, P)
        out = torch.zeros(m, d, q, dtype=value.dtype)
        for l_, ((h, w), s0) in enumerate(zip(levels, starts)):
            vl = value[v, s0 : s0 + h * w].reshape(h, w, m, d).permute(2, 3, 0, 1)  # (M, D, H, W)
            grid = (2 * ln[:, :, l_] - 1).permute(1, 0, 2, 3)  # (M, Q, P, 2)
            smp = F.grid_sample(vl, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # (M, D, Q, P)
            out += (smp * a[:, :, l_].permute(1, 0, 2)[:, None]).sum(-1)
        vc = torch.ones(q, 1, dtype=value.dtype) if vis is None else vis[v].reshape(q, 1).to(value.dtype)
        acc += out.permute(2, 0, 1).reshape(q, c) * vc
        cnt += vc
    return acc / cnt.clamp(min=1.0)


def mmcv_msda_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Verbatim port of mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros",
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


def save_case(d: Path, value, levels, loc, attw, out, mode="loc", ref=None, vis=None, starts=None):
    """One kernel call as msda_io.h reads it."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    nv, s, c = value.shape
    q, m, no, nl, p, _ = loc.shape
    starts = starts or level_starts(levels)
    nvr, rl, r, rd = (1, 1, 1, 2) if ref is None else (ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[4])
    f32 = lambda t: np.ascontiguousarray(t.detach().numpy(), np.float32)  # noqa: E731
    f32(value).tofile(d / "value.f32")
    f32(loc).tofile(d / "loc.f32")
    f32(attw).tofile(d / "attw.f32")
    f32(out).tofile(d / "ref_out.f32")
    if ref is not None and mode != "loc":
        f32(ref).tofile(d / "ref.f32")
    if vis is not None:
        np.ascontiguousarray(vis.numpy(), np.uint8).tofile(d / "vis.u8")
    lines = [f"{nv} {nl} {s} {m} {c // m} {p} {q} {no} {MODES[mode]} {nvr} {rl} {r} {rd} {int(vis is not None)}"]
    lines += [f"{h} {w} {s0}" for (h, w), s0 in zip(levels, starts)]
    (d / "meta.txt").write_text("\n".join(lines) + "\n")


def synthetic(kind: str, q: int | None = None, seed: int = 0):
    """(value, levels, loc, attw, mode, ref, vis) for a model-shaped call, with edge cases mixed in:
    points off the map, on pixel centers and borders, invisible maps, a query no map sees."""
    g = torch.Generator().manual_seed(seed)
    if kind == "rtdetr_decoder":  # RT-DETR-r18 decoder cross-attention: 300 box queries, 3 levels
        q = q or 300
        levels, m, d, nl, p, nv, no, mode = [(80, 80), (40, 40), (20, 20)], 8, 32, 3, 4, 1, 1, "box"
        cxcy = torch.rand(1, q, 1, 1, 2, generator=g) * 0.9 + 0.05
        wh = torch.rand(1, q, 1, 1, 2, generator=g) * 0.5 + 0.02
        ref = torch.cat([cxcy, wh], -1)
        loc = torch.randn(q, m, no, nl, p, 2, generator=g) * 2.0
        ref[0, 0, 0, 0] = torch.tensor([0.0, 1.0, 0.1, 0.1])  # corners: zero-padding taps
    elif kind in ("bevformer_tsa", "bevformer_sca"):
        tsa = kind == "bevformer_tsa"
        q = q or 96
        levels, m, d, nl = ([(50, 50)] if tsa else [(15, 25)]), 8, 32, 1
        nv, no, p, r, mode = (2, 2, 4, 1, "pix") if tsa else (6, 1, 8, 4, "pix")
        h, w = levels[0]
        ref = torch.rand(nv, q, 1, r, 2, generator=g) * 1.4 - 0.2
        loc = torch.randn(q, m, no, nl, p, 2, generator=g) * 2.0
        ref[:, :4] = 0.0  # exact pixel centers / borders / half pixels
        loc[:4] = torch.tensor([-0.5, 0.0, 0.5, float(w) - 0.5]).reshape(4, 1, 1, 1, 1, 1)
        loc[4, :, :, :, :, 0] = float(w) + 0.5 - ref[0, 4, 0, 0, 0] * w  # x = w: just past the right edge
    elif kind == "loc_small":  # mmcv's own interface: normalized locations, 2 levels
        q = q or 64
        levels, m, d, nl, p, nv, no, mode, ref = [(12, 16), (6, 8)], 4, 32, 2, 4, 1, 1, "loc", None
        loc = torch.rand(q, m, no, nl, p, 2, generator=g) * 1.2 - 0.1
    else:
        raise SystemExit(f"unknown kind {kind}")
    s = sum(h * w for h, w in levels)
    value = torch.randn(nv, s, m * d, generator=g)
    attw = torch.softmax(torch.randn(q, m, no, nl * p, generator=g), -1).reshape(q, m, no, nl, p)
    vis = None
    if kind == "bevformer_sca":
        vis = (torch.rand(nv, q, generator=g) < 0.3).to(torch.uint8)
        vis[:, 5] = 0  # no camera sees it -> zeros
        vis[:, 6] = 1  # every camera sees it
    return value, levels, loc, attw, mode, ref, vis


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("kind")
    s.add_argument("out")
    s.add_argument("--q", type=int)
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("mmcv-check")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.cmd == "synth":
        value, levels, loc, attw, mode, ref, vis = synthetic(a.kind, a.q, a.seed)
        out = msda_reference(value, levels, loc, attw, mode, ref, vis)
        save_case(Path(a.out), value, levels, loc, attw, out, mode, ref, vis)
        print(f"{a.kind}: Q {loc.shape[0]} levels {levels} -> {a.out}")
    else:
        value, levels, loc, attw, _, _, _ = synthetic("loc_small")
        m = loc.shape[1]
        ref = mmcv_msda_pytorch(value.reshape(1, -1, m, value.shape[-1] // m), levels, loc[:, :, 0][None],
                                attw[:, :, 0][None])[0]
        print("max abs vs mmcv:", float((msda_reference(value, levels, loc, attw) - ref).abs().max()))


if __name__ == "__main__":
    main()
