"""The HVX cross-attention kernel's integer contract (attn_kernel.h), numpy only.

q (LQ, 256), k and v (LK, 256) uint8 with zero points zq / zk / zv; per head h (32 dims) and row:
  s_j = (q - zq) . (k_j - zk)        exact int (the kernel drops a per-row constant softmax ignores)
  p_j = exp_u8(max s - s_j)          round(255 * e^-(d * step)) via 2^-t, t in Q11, cubic in Q15
  out = div_round(sum p_j v_j, sum p_j)   uint8 in V's own (scale, zero point)"""
from __future__ import annotations

from pathlib import Path

import numpy as np

C1, C2, C3 = 22663, 7582, 1307  # 2^-x ~ 1 - C1 x + C2 x^2 - C3 x^3 on [0, 1), Q15 (max rel err 3.6e-4)
T_MAX = 9 * 2048 + 2047  # t in Q11 of log2 units; beyond 9 bits 255 * 2^-t rounds to 0


def exp_params(step):
    """The kernel's two ints for scores whose unit is ``step`` nats (sq * sk)."""
    m16 = max(1, int(round(step * np.log2(np.e) * (1 << 15))))
    return m16, -(-(T_MAX + 1) * 16 // m16)  # d is clamped to dcl first so d * m16 fits in 32 bits


def mulq15(a, b):
    return (a * b + (1 << 14)) >> 15


def exp_u8(d, m16, dcl):
    """p = round(255 * 2^-(d * step * log2 e)) for d = max - s >= 0, bit-exact with the kernel."""
    t = np.minimum((np.minimum(d, dcl) * m16) >> 4, T_MAX)
    n, x = t >> 11, (t & 2047) << 4
    x2 = mulq15(x, x)
    x3 = mulq15(x2, x)
    y = (32767 - mulq15(C1, x) + mulq15(C2, x2) - mulq15(C3, x3)) >> n
    return (y * 255 + (1 << 14)) >> 15


def div_round(acc, sp):
    """round(acc / sp) as the kernel computes it: (acc + sp / 2) * R >> 32, R = 2^32 / sp + 1."""
    r = (np.int64(1) << 32) // sp + 1
    return ((acc + (sp >> 1)) * r) >> 32


def attn_u8(qu, ku, vu, zq, zk, m16, dcl):
    """The kernel's output (LQ, 256) uint8 from uint8 q / k / v."""
    qu, ku, vu = (np.asarray(x, np.int64) for x in (qu, ku, vu))
    out = np.empty(qu.shape, np.int64)
    for h in range(8):
        sl = slice(32 * h, 32 * h + 32)
        s = (qu[:, sl] - zq) @ (ku[:, sl] - zk).T
        p = exp_u8(s.max(1, keepdims=True) - s, m16, dcl)
        out[:, sl] = div_round(p @ vu[:, sl], p.sum(1, keepdims=True))
    return out.astype(np.uint8)


def write_case(d, qu, ku, vu, zq, zk, m16, dcl):
    """A case directory for attn_host_check / attn_sim / attn_client."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for n, x in (("q", qu), ("k", ku), ("v", vu)):
        np.asarray(x, np.uint8).tofile(d / f"{n}.bin")
    (d / "params.txt").write_text(f"{len(qu)} {len(ku)} {zq} {m16} {dcl}\n")
    attn_u8(qu, ku, vu, zq, zk, m16, dcl).tofile(d / "ref_out.bin")


def synth_case(d, lq=20, lk=256, seed=0, step=0.06):
    """A synthetic case (uint8 q / k / v around their zero points) for phone-free checks."""
    rng = np.random.default_rng(seed)
    zq, zk = 120, 125
    qu = np.clip(rng.normal(zq, 30, (lq, 256)), 0, 255).astype(np.uint8)
    ku = np.clip(rng.normal(zk, 40, (lk, 256)), 0, 255).astype(np.uint8)
    vu = rng.integers(0, 256, (lk, 256), dtype=np.uint8)
    write_case(d, qu, ku, vu, zq, zk, *exp_params(step))
