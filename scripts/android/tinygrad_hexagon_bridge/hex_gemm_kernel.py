#!/usr/bin/env python3
"""A from-scratch, hand-written Hexagon vrmpy GEMM kernel via tinygrad's `Tensor.custom_kernel`
-- the working conclusion of the `custom_kernel_attempt.py` investigation documented in
../README.md's "Modeling Hexagon as its own accelerator" section.

Computes `C[M,N] = A[M,K] @ B[K,N]` (uint8 x uint8 -> int32) via HVX `vrmpy` (one hardware
128-byte dot-product-accumulate instruction per 4-wide K-chunk x 32-wide N-tile), entirely
bypassing `Ops.WMMA`/`TensorCore`/the generic devectorizer -- the accumulator is a genuine
`AddrSpace.REG` placeholder addressed by its base pointer (never a `(32,)`-shaped tinygrad VALUE,
which the generic elementwise devectorizer treats as 32 independent lanes needing per-element
pointer-offset arithmetic, corrupting the semantics; see ../README.md for exactly how that was
found and worked around).

`B` must be pre-packed into `pack_b()`'s layout: `Bp[nt, kc, n_lane*4+k_sub] = B[kc*4+k_sub,
nt*32+n_lane]` -- 128 contiguous bytes per (N-tile, K-chunk), matching what one `vrmpy` call reads
in one shot. `A` needs no repacking (its natural (M, K) row-major layout is already contiguous per
(m, kc) 4-byte slice).

Verified end-to-end on real Hexagon v73 hardware (via ../bridge_and_test.py's TVM-transport
bridge, since tinygrad's own DSP driver can't reach this phone -- see ../README.md) at the exact
pathological shape from `scripts/android/maskrcnn_e2e/README.md`'s ranked profile
(cin=64, cout=256, spatial 200x272 -> M=54400): **30.35 GMAC/s, 8.65x faster than stock TVM's
hand-tuned vrmpy schedule (3.51 GMAC/s) at the same shape**, bit-exact correct. See ../README.md's
coverage tables for six more shapes (including a strided one and two padded-to-32-lane tiny-`cout`
ones) verified the same way.
"""
from __future__ import annotations

import argparse
import functools

import numpy as np


def pack_b(b: np.ndarray, n_tile: int = 32, k_sub: int = 4) -> np.ndarray:
    """B: (K, N) uint8 -> Bp: (N//n_tile, K//k_sub, n_tile*k_sub) uint8, the layout one vrmpy
    call reads contiguously: Bp[nt, kc, n_lane*k_sub + ks] = B[kc*k_sub + ks, nt*n_tile + n_lane]."""
    K, N = b.shape
    assert K % k_sub == 0 and N % n_tile == 0
    return (
        b.reshape(K // k_sub, k_sub, N // n_tile, n_tile)
        .transpose(2, 0, 3, 1)
        .reshape(N // n_tile, K // k_sub, n_tile * k_sub)
    )


def build_kernel(cin: int, cout: int, m: int, a, bp, kernel_name: str = "hex_gemm"):
    """Build + apply the HVX vrmpy GEMM custom_kernel via Tensor.custom_kernel, given input
    Tensors `a` (shape (m, cin), uint8) and `bp` (shape (cout//32, cin//4, 128), uint8 --
    pre-packed via pack_b()). Returns the (m, cout) int32 output Tensor; call .realize() to run
    it (or capture ClangRenderer.render to get the generated C source, as main() does below)."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert cin % 4 == 0 and cout % 32 == 0
    nt_count, kc_count = cout // 32, cin // 4
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"

    def _reg_i32(shape, slot, *deps):
        # Depend on ALL enclosing loop ranges, not just the innermost -- a degenerate extent-1
        # range (e.g. NT=1 when cout==32) gets eliminated by tinygrad's optimizer, and if the
        # accumulator's init was scoped to *only* that range, the "reset once per iteration"
        # behavior silently degrades to "reset once total", corrupting every row after the
        # first. Depending on every range (M can never degenerate) makes this robust.
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Bp: UOp) -> UOp:
        m_rng = UOp.range(m, 0, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 1, AxisType.WEAK)
        acc = _reg_i32((32,), 0, m_rng, nt_rng)
        kc_rng = UOp.range(kc_count, 2, AxisType.REDUCE)
        acc_addr = acc.after(kc_rng)[0]
        a_idx = A[m_rng, kc_rng * 4]
        b_idx = Bp[nt_rng, kc_rng, 0]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, b_idx, a_idx),
            arg=(f"*({i32x32}*){{0}} = __builtin_HEXAGON_V6_vrmpyub_acc_128B("
                 f"*({i32x32}*){{0}}, *({u8x128}*){{1}}, *(unsigned int*){{2}});"),
        )
        update = step.end(kc_rng)
        final_addr = acc.after(update)[0]
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[m_rng, nt_rng * 32], final_addr),
            arg=f"*({i32x32}*){{0}} = *({i32x32}*){{1}};",
        )
        return out_step.end(nt_rng, m_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(m, cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a, bp, fxn=functools.partial(kernel_fn))[0]


def build_strided_kernel(cin: int, cout: int, ih: int, iw: int, stride: int, a, bp, kernel_name: str = "hex_gemm_strided"):
    """Like build_kernel(), but for a strided 1x1 conv: output[oh,ow,:] = A_flat[oh*stride*iw +
    ow*stride, :] @ B. `a` must be shape (ih*iw, cin) (the *unstrided* input, flat); output is
    shape (ih//stride * iw//stride, cout). Needs real 2D (oh, ow) spatial indexing -- unlike
    build_kernel()'s plain contiguous-M indexing, which only works for stride=1 (every input row
    maps to an output row 1:1)."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert cin % 4 == 0 and cout % 32 == 0
    oh_count, ow_count = ih // stride, iw // stride
    nt_count, kc_count = cout // 32, cin // 4
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"

    def _reg_i32(shape, slot, *deps):
        # See build_kernel()'s _reg_i32 for why we depend on every enclosing range, not just
        # the innermost one (degenerate extent-1 ranges get eliminated by the optimizer).
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Bp: UOp) -> UOp:
        oh_rng = UOp.range(oh_count, 0, AxisType.WEAK)
        ow_rng = UOp.range(ow_count, 1, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 2, AxisType.WEAK)
        acc = _reg_i32((32,), 0, oh_rng, ow_rng, nt_rng)
        kc_rng = UOp.range(kc_count, 3, AxisType.REDUCE)
        acc_addr = acc.after(kc_rng)[0]
        flat_row = (oh_rng * stride) * iw + (ow_rng * stride)
        a_idx = A[flat_row, kc_rng * 4]
        b_idx = Bp[nt_rng, kc_rng, 0]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, b_idx, a_idx),
            arg=(f"*({i32x32}*){{0}} = __builtin_HEXAGON_V6_vrmpyub_acc_128B("
                 f"*({i32x32}*){{0}}, *({u8x128}*){{1}}, *(unsigned int*){{2}});"),
        )
        update = step.end(kc_rng)
        final_addr = acc.after(update)[0]
        out_row = oh_rng * ow_count + ow_rng
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[out_row, nt_rng * 32], final_addr),
            arg=f"*({i32x32}*){{0}} = *({i32x32}*){{1}};",
        )
        return out_step.end(nt_rng, ow_rng, oh_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(oh_count * ow_count, cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a, bp, fxn=functools.partial(kernel_fn))[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=256)
    p.add_argument("--m", type=int, default=200 * 272, help="rows (default: the real Mask R-CNN shape)")
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
        # qemu executes the kernel at the real M directly -- for M in the tens of thousands
        # (the real Mask R-CNN shape) this still completes in a few minutes, so there's no need
        # for a separate "verify small, generate large" split.
        rng = np.random.default_rng(5)
        a_np = rng.integers(0, 100, (args.m, args.cin)).astype(np.uint8)
        b_np = rng.integers(0, 100, (args.cin, args.cout)).astype(np.uint8)
        a = Tensor(a_np, device="DSP")
        bp = Tensor(pack_b(b_np), device="DSP")
        out = build_kernel(args.cin, args.cout, args.m, a, bp)
        out.realize()
        ref = a_np.astype(np.int64) @ b_np.astype(np.int64)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
        print(f"correctness (M={args.m}): {correct}")
        if not correct:
            raise SystemExit("kernel is incorrect")
    finally:
        ClangRenderer.render = orig_render

    src = captured.get("src", "")
    marker = src.find("/* DSP boilerplate */")
    kernel_src = (src[:marker] if marker >= 0 else src).rstrip() + "\n"
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes) for cin={args.cin} cout={args.cout} m={args.m}")


if __name__ == "__main__":
    main()
