#!/usr/bin/env python3
"""A from-scratch, hand-written Hexagon vrmpybusv 7x7-conv kernel via tinygrad's
`Tensor.custom_kernel` -- covers the ResNet stem conv, the last uncovered *conv* shape in
`scripts/android/maskrcnn_e2e/profile_data/conv_profile.json` (308.5 ms, `ishape=[1,3,800,1088]`,
`wshape=[64,3,7,7]`, `stride=[2,2]`, `pad=[3,3]`). Generalizes ../hex_conv3x3_kernel.py's
9-position pattern to 49 positions and adds real stride (output spatial != input spatial, unlike
the 3x3 kernels' stride=1 "same" case -- needs `build_strided_kernel()`-style separate (oh, ow)
output ranges from the (ih, iw) input ones, see ../hex_gemm_kernel.py).

The one genuinely new wrinkle this shape has that no prior kernel in this project needed: the
stem conv's `cin=3` is *not* a multiple of 4, the K-chunk width every `vrmpybusv` call in this
project assumes. Handled the same way ../hex_gemm_kernel.py's "tiny RPN/mask-head convs" coverage
handled `cout` not being a multiple of 32 (see ../README.md's "Coverage: the tiny RPN/mask-head
convs" section) -- zero-pad, but on the *reduction* (K/cin) axis instead of the N/cout axis this
time: `cin` is padded up to 4 with an extra all-zero weight channel, so the 4th lane of every
K-chunk always contributes `activation * 0 == 0` to the dot product regardless of what's in the
padding's activation byte. Unlike N-axis padding (which wastes output lanes but costs nothing
extra since vrmpy's width is fixed at 32 regardless), K-axis padding here is free for the same
reason: `vrmpybusv` always consumes a full 4-byte K-chunk per lane-group, so cin=3 already pays
for a 4-wide reduction step; padding to cin=4 with a zero weight just makes that width explicit
instead of undefined behavior on to a byte that shouldn't matter.

Weights are signed int8 (matching the real backbone's QNN quantization, like ../hex_conv3x3_kernel.py),
so this uses `vrmpybusv_acc_128B`.
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

K = 7
STRIDE = 2
PAD = 3


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def pack_weight_7x7(w: np.ndarray, cin_pad: int, n_tile: int = 32, k_sub: int = 4) -> np.ndarray:
    """w: (cout, cin, 7, 7) int8 (real cin, e.g. 3) -> Wp: (49, cout//n_tile, cin_pad//k_sub,
    n_tile*k_sub) uint8, cin zero-padded to cin_pad first (extra channels get a zero weight, so
    they contribute nothing regardless of the corresponding activation padding byte), then packed
    per kernel position exactly like ../hex_conv3x3_kernel.py's pack_weight_3x3():
    Wp[kh*7+kw, nt, kc, n_lane*k_sub+ks] = w_padded[nt*n_tile+n_lane, kc*k_sub+ks, kh, kw]."""
    cout, cin, kh, kw = w.shape
    assert (kh, kw) == (K, K)
    assert cin <= cin_pad and cin_pad % k_sub == 0 and cout % n_tile == 0
    w_pad = np.zeros((cout, cin_pad, K, K), dtype=np.int8)
    w_pad[:, :cin] = w
    w_u8 = w_pad.view(np.uint8)
    out = np.empty((K * K, cout // n_tile, cin_pad // k_sub, n_tile * k_sub), dtype=np.uint8)
    for pos in range(K * K):
        i, j = pos // K, pos % K
        b = w_u8[:, :, i, j].T  # (cin_pad, cout)
        out[pos] = (
            b.reshape(cin_pad // k_sub, k_sub, cout // n_tile, n_tile)
            .transpose(2, 0, 3, 1)
            .reshape(cout // n_tile, cin_pad // k_sub, n_tile * k_sub)
        )
    return out


def pad_input(a: np.ndarray, cin_pad: int) -> np.ndarray:
    """a: (ih, iw, cin) uint8 -> (ih+2*PAD, iw+2*PAD, cin_pad) uint8: channel-padded up to
    cin_pad with zero (paired with pack_weight_7x7's zero weight padding, so the padding
    contributes nothing to the dot product no matter its value -- zero is just the simplest
    choice), then zero-padded spatially by PAD on each side. Same real-vs-literal-zero caveat as
    ../hex_conv3x3_kernel.py's pad_input(): a real integration needs the input's actual
    quantization zero-point for the *spatial* border, not literal 0, to match qnn.conv2d exactly;
    this kernel itself doesn't care what's in the border, only this file's own correctness check
    (which references the same literal-0-padded array) does."""
    ih, iw, cin = a.shape
    assert cin <= cin_pad
    a_cpad = np.zeros((ih, iw, cin_pad), dtype=np.uint8)
    a_cpad[:, :, :cin] = a
    return np.pad(a_cpad, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant", constant_values=0)


def out_hw(ih: int, iw: int) -> tuple[int, int]:
    oh = (ih + 2 * PAD - K) // STRIDE + 1
    ow = (iw + 2 * PAD - K) // STRIDE + 1
    return oh, ow


def build_kernel(cin: int, cout: int, ih: int, iw: int, a_pad, wp, kernel_name: str = "hex_stem7x7"):
    """Build + apply the HVX vrmpybusv 7x7-stride-2-conv custom_kernel. `a_pad`: flat
    (ih+2*PAD)*(iw+2*PAD)*cin_pad uint8 Tensor (pre-padded, see pad_input()). `wp`: flat
    49*(cout//32)*(cin_pad//4)*128 uint8 Tensor (pre-packed via pack_weight_7x7(), flattened to
    1D). `cin` is the *real* channel count (e.g. 3); the kernel internally works in cin_pad =
    round_up(cin, 4) -- callers must have built `a_pad`/`wp` with that same cin_pad. Returns the
    (oh*ow, cout) int32 output Tensor; call .realize() to run it."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    cin_pad = _round_up(cin, 4)
    assert cout % 32 == 0
    oh_count, ow_count = out_hw(ih, iw)
    nt_count, kc_count = cout // 32, cin_pad // 4
    r_count = K * K * kc_count
    iw_pad = iw + 2 * PAD
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"

    def _reg_i32(shape, slot, *deps):
        # See ../hex_gemm_kernel.py's _reg_i32: depend on every enclosing range (oh, ow, nt), not
        # just the innermost one -- a degenerate extent-1 range (nt when cout==32, or the
        # reduction range if this ever ran with cin_pad==4 and K*K==1, which it never does here
        # but the pattern is applied uniformly per this project's established convention) gets
        # eliminated by tinygrad's optimizer, silently downgrading "reset per output pixel" to
        # "reset once total".
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Wp: UOp) -> UOp:
        oh_rng = UOp.range(oh_count, 0, AxisType.WEAK)
        ow_rng = UOp.range(ow_count, 1, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 2, AxisType.WEAK)
        acc = _reg_i32((32,), 0, oh_rng, ow_rng, nt_rng)
        r_rng = UOp.range(r_count, 3, AxisType.REDUCE)
        acc_addr = acc.after(r_rng)[0]

        pos = r_rng // kc_count
        kc = r_rng % kc_count
        kh = pos // K
        kw = pos % K

        a_row = oh_rng * STRIDE + kh
        a_col = ow_rng * STRIDE + kw
        a_flat_idx = a_row * (iw_pad * cin_pad) + a_col * cin_pad + kc * 4
        wp_flat_idx = pos * (nt_count * kc_count * 128) + nt_rng * (kc_count * 128) + kc * 128

        a_idx = A[a_flat_idx]
        w_idx = Wp[wp_flat_idx]
        broadcast32 = ",".join(["*(unsigned int*){2}"] * 32)
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, w_idx, a_idx),
            arg=(f"*({i32x32}*){{0}} = __builtin_HEXAGON_V6_vrmpybusv_acc_128B("
                 f"*({i32x32}*){{0}}, ({i32x32}){{{{{broadcast32}}}}}, *({u8x128}*){{1}});"),
        )
        update = step.end(r_rng)
        final_addr = acc.after(update)[0]
        out_row = oh_rng * ow_count + ow_rng
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[out_row, nt_rng * 32], final_addr),
            arg=f"*({i32x32}*){{0}} = *({i32x32}*){{1}};",
        )
        return out_step.end(nt_rng, ow_rng, oh_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(oh_count * ow_count, cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a_pad, wp, fxn=functools.partial(kernel_fn))[0]


def reference_conv(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a: (ih, iw, cin) uint8, w: (cout, cin, 7, 7) int8 -> (oh*ow, cout) int64, stride=2 pad=3,
    matching build_kernel()'s semantics exactly (same literal-0 spatial padding, real cin -- no
    channel padding here, this is the ground-truth math, not the padded-representation kernel)."""
    ih, iw, cin = a.shape
    cout = w.shape[0]
    oh, ow = out_hw(ih, iw)
    a_pad = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="constant", constant_values=0).astype(np.int64)
    w64 = w.astype(np.int64)
    out = np.zeros((oh, ow, cout), dtype=np.int64)
    for kh in range(K):
        for kw in range(K):
            patch = a_pad[kh:kh + STRIDE * oh:STRIDE, kw:kw + STRIDE * ow:STRIDE, :]  # (oh, ow, cin)
            out += np.einsum("hwc,oc->hwo", patch, w64[:, :, kh, kw])
    return out.reshape(oh * ow, cout)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cin", type=int, default=3, help="real backbone stem conv has cin=3")
    p.add_argument("--cout", type=int, default=64)
    p.add_argument("--ih", type=int, default=20)
    p.add_argument("--iw", type=int, default=20)
    p.add_argument("--out", default="stem7x7_kernel.c")
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
        cin_pad = _round_up(args.cin, 4)
        rng = np.random.default_rng(11)
        a_np = rng.integers(0, 100, (args.ih, args.iw, args.cin)).astype(np.uint8)
        w_np = rng.integers(-40, 40, (args.cout, args.cin, K, K)).astype(np.int8)
        a_pad_np = pad_input(a_np, cin_pad).reshape(-1)
        wp_np = pack_weight_7x7(w_np, cin_pad).reshape(-1)

        a_pad_t = Tensor(a_pad_np, device="DSP")
        wp_t = Tensor(wp_np, device="DSP")
        out = build_kernel(args.cin, args.cout, args.ih, args.iw, a_pad_t, wp_t)
        out.realize()

        ref = reference_conv(a_np, w_np)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
        oh, ow = out_hw(args.ih, args.iw)
        print(f"correctness (cin={args.cin} cout={args.cout} ih={args.ih} iw={args.iw} -> {oh}x{ow}): {correct}")
        if not correct:
            got = out.numpy().astype(np.int64)
            diff = got - ref
            print(f"max abs diff {np.abs(diff).max()} mismatched {np.count_nonzero(diff)}/{diff.size}")
            raise SystemExit("kernel is incorrect")
    finally:
        ClangRenderer.render = orig_render

    src = captured.get("src", "")
    marker = src.find("/* DSP boilerplate */")
    kernel_src = (src[:marker] if marker >= 0 else src).rstrip() + "\n"
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes) for cin={args.cin} cout={args.cout} ih={args.ih} iw={args.iw}")


if __name__ == "__main__":
    main()
