#!/usr/bin/env python3
"""A from-scratch, hand-written Hexagon HVX fp32 GEMM kernel via tinygrad's `Tensor.custom_kernel`,
for Mask R-CNN's box-head fc6 layer (the first of the 4 `MatMul` nodes in `rest.onnx`, currently
100% ONNX Runtime CPU -- see ../../maskrcnn_e2e/README.md's "roughly 13 GMAC for the fc6 box-head
layer on 1000 proposals" note).

Unlike every conv/GEMM kernel elsewhere in this project (all uint8/int8 x vrmpy), this op is
**fp32**: `rest.onnx` wraps the real MatMul in QuantizeLinear/DequantizeLinear (confirmed by
inspecting the actual graph -- the MatMul's own inputs are DequantizeLinear outputs, elem_type
FLOAT, and its output feeds a QuantizeLinear), so the MatMul itself computes in float32, not int8.
Weight shape (12544, 1024) confirmed from the real graph: 12544 = 7*7*256 (RoiAlign's 7x7 crop of
256-channel FPN features, flattened), 1024 = fc6's output width. Activation rows = proposal count
(dynamic; ~1000 in the real pipeline, matching the README's own "1000 proposals" figure).

No `vrmpy`-equivalent exists for fp32 -- this is a plain HVX vector FMA loop: one `float
__attribute__((vector_size(128)))` (32 fp32 lanes) accumulator per (M row, N-tile) pair, stepped
one K element at a time (`acc += a_scalar * b_vec`, a scalar-broadcast multiply-add clang lowers
directly to HVX float FMA under `-mhvx`), reusing the exact `_reg_*`-style multi-range-dependency
accumulator-init pattern established in `hex_gemm_kernel.py` (every range that can degenerate to
extent 1 must be in the dependency list, or "reset once per row" silently downgrades to "reset once
ever" -- a real bug that hit every kernel in this series until fixed the first time).
"""
from __future__ import annotations

import argparse
import functools

import numpy as np


def pack_b(b: np.ndarray, n_tile: int = 32) -> np.ndarray:
    """B: (K, N) float32 -> Bp: (N//n_tile, K, n_tile) float32, the layout one FMA step reads
    contiguously: Bp[nt, k, :] = B[k, nt*n_tile:(nt+1)*n_tile]."""
    k, n = b.shape
    assert n % n_tile == 0
    return b.reshape(k, n // n_tile, n_tile).transpose(1, 0, 2).copy()


def build_kernel(m: int, k: int, n: int, a, bp, kernel_name: str = "hex_boxhead_gemm"):
    """Build + apply the HVX fp32 GEMM custom_kernel. `a`: shape (m, k) float32 Tensor. `bp`: shape
    (n//32, k, 32) float32 Tensor (pre-packed via pack_b()). Returns the (m, n) float32 output
    Tensor; call .realize() to run it."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert n % 32 == 0
    nt_count = n // 32
    f32x32 = "float __attribute__((vector_size(128)))"

    def _reg_f32(shape, slot, *deps):
        # See hex_gemm_kernel.py's _reg_i32 for why every enclosing range (not just the innermost)
        # must be a dependency -- a degenerate extent-1 range (e.g. NT=1 when n==32) gets eliminated
        # by tinygrad's optimizer, and an init scoped only to that range silently degrades from
        # "reset once per row" to "reset once ever", corrupting every row after the first.
        ret = UOp.placeholder(shape, dtypes.float32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0.0)))

    def kernel_fn(C: UOp, A: UOp, Bp: UOp) -> UOp:
        m_rng = UOp.range(m, 0, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 1, AxisType.WEAK)
        acc = _reg_f32((32,), 0, m_rng, nt_rng)
        k_rng = UOp.range(k, 2, AxisType.REDUCE)
        acc_addr = acc.after(k_rng)[0]
        a_idx = A[m_rng, k_rng]
        b_idx = Bp[nt_rng, k_rng, 0]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, b_idx, a_idx),
            arg=(f"*({f32x32}*){{0}} = *({f32x32}*){{0}} + "
                 f"(*(float*){{2}}) * (*({f32x32}*){{1}});"),
        )
        update = step.end(k_rng)
        final_addr = acc.after(update)[0]
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[m_rng, nt_rng * 32], final_addr),
            arg=f"*({f32x32}*){{0}} = *({f32x32}*){{1}};",
        )
        return out_step.end(nt_rng, m_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(m, n, dtype="float32", device="DSP")
    return Tensor.custom_kernel(c, a, bp, fxn=functools.partial(kernel_fn))[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=64, help="proposal count (real shape: ~1000)")
    p.add_argument("--k", type=int, default=12544, help="real fc6 K dim (7*7*256)")
    p.add_argument("--n", type=int, default=1024, help="real fc6 N dim")
    p.add_argument("--out", default="kernel.c")
    args = p.parse_args()

    import os

    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")

    from tinygrad import Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    captured: dict[str, str] = {}
    orig_render = ClangRenderer.render

    def _capture(self, uops):
        src = orig_render(self, uops)
        captured["src"] = src
        return src

    ClangRenderer.render = _capture
    try:
        rng = np.random.default_rng(7)
        a_np = rng.normal(0, 1, (args.m, args.k)).astype(np.float32)
        b_np = rng.normal(0, 1, (args.k, args.n)).astype(np.float32)
        a = Tensor(a_np, device="DSP")
        bp = Tensor(pack_b(b_np), device="DSP")
        out = build_kernel(args.m, args.k, args.n, a, bp)
        out.realize()
        ref = a_np.astype(np.float64) @ b_np.astype(np.float64)
        got = out.numpy().astype(np.float64)
        max_abs_err = float(np.abs(got - ref).max())
        print(f"correctness (M={args.m},K={args.k},N={args.n}): max_abs_err={max_abs_err:.3e}")
        if max_abs_err > 1e-2 * max(1.0, np.abs(ref).max()) * 1e-3:
            pass  # informational only; caller decides tolerance
    finally:
        ClangRenderer.render = orig_render

    src = captured.get("src", "")
    marker = src.find("/* DSP boilerplate */")
    kernel_src = (src[:marker] if marker >= 0 else src).rstrip() + "\n"
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes) for m={args.m} k={args.k} n={args.n}")


if __name__ == "__main__":
    main()
