#!/usr/bin/env python3
"""A hand-written Hexagon `custom_kernel` for the real backbone's 3x3/stride=2/pad=1 max-pool
(`scripts/android/maskrcnn_e2e/profile_data/noncon_profile.json`'s `maxpool` entries), unblocking
the "no wide contiguous vectorization axis" finding in PR #1791 (see
../README.md's "`maxpool`/`sigmoid`: investigated, not covered" section) by building on TVM's own
*real* reference layout for this op, not the plain-NCHW layout that blocked the first attempt.

Confirmed empirically (not assumed) by compiling a real `qnn.conv2d -> nn.max_pool2d ->
qnn.conv2d` graph -- the real backbone's actual sandwich, stem-conv -> pool -> layer1 -- through
`relay.build()` for `target=hexagon("v73")` and inspecting the compiled graph JSON: TVM's own
`AlterOpLayout` pass keeps the whole subgraph in the **packed NCHWc layout** (`ic_bn=32`, the same
convention `chunked_kernel_test.py`/`backbone_splice/gen_chunked_prod.py` already validated for
conv kernels in this project) across the pooling op too -- the compiled `nn.max_pool2d` node's
shape is `[1, ic_chunks, oh, ow, 32]`, not plain NCHW `[1, C, oh, ow]`. TVM never registers a
Hexagon-specific pooling schedule (`topi/hexagon/pooling.py` is a generic, layout-agnostic
`AutoInlineInjective` schedule); the *packing* comes entirely from `topi.nn.pool2d`'s generic
layout-string parametrization (`cpp.nn.pool2d(data, ..., layout="NCHW32c")`) combined with
`AlterOpLayout` choosing to keep the packed layout across the whole graph rather than repacking
back to NCHW around the pool. So the "no contiguous axis" blocker was specific to the *plain-NCHW*
approach PR #1791 tried -- in the packed layout TVM itself actually uses for this exact real
subgraph, the channel-block axis (32 contiguous uint8 bytes) is right there, the same axis every
conv kernel in this project already vectorizes across.

uint8 max is dtype-safe to zero-pad (unlike a sum/accumulate): 0 is uint8's true minimum, so a
padded lane can never win a max against any real activation value (uint8's range is [0,255], so
max(real_value, 0) == real_value whenever real_value >= 0, always true) -- exactly the same
"padding is free" argument `hex_stem7x7_kernel.py` made for its cin-axis zero-padding, applied
here to the spatial pad instead.

Not yet done: the ~4x throughput a full 128-byte HVX vector could offer (this kernel's natural
data width is 32 bytes -- one channel-block -- since TVM's stride=2 pooling makes neighboring
*output* positions' input windows non-contiguous, unlike the stride=1 conv kernels' ow_tile-style
column batching; grouping multiple output positions into one 128-byte op here needs a real
gather/deinterleave step, the same complexity PR #1791 flagged and this kernel deliberately avoids
by working at native channel-block width instead). See ../README.md's coverage section for the
real-hardware result.
"""
from __future__ import annotations

import argparse
import functools

import numpy as np


IC_BN = 32  # TVM's fixed Hexagon NCHWc channel-block width, confirmed via relay.build() above.


def pack_nchwc(a: np.ndarray) -> np.ndarray:
    """a: (C, H, W) uint8, C % 32 == 0 -> flat (C//32 * H * W * 32,) uint8, TVM's own packed
    NCHWc layout: packed[ic_chunk, h, w, ic_block] = a[ic_chunk*32 + ic_block, h, w]."""
    C, H, W = a.shape
    assert C % IC_BN == 0
    return a.reshape(C // IC_BN, IC_BN, H, W).transpose(0, 2, 3, 1).reshape(-1)


def unpack_nchwc(flat: np.ndarray, ic_chunks: int, H: int, W: int) -> np.ndarray:
    """Inverse of pack_nchwc, for reading a kernel's packed-layout output back to (C, H, W)."""
    return flat.reshape(ic_chunks, H, W, IC_BN).transpose(0, 3, 1, 2).reshape(ic_chunks * IC_BN, H, W)


def pad_nchwc(packed: np.ndarray, ic_chunks: int, H: int, W: int, pad: int) -> np.ndarray:
    """Zero-pad the H/W axes of a packed-layout flat tensor by `pad` on each side -- safe for
    uint8 max (see module docstring): a padded lane can never win a max against a real value."""
    a = packed.reshape(ic_chunks, H, W, IC_BN)
    a_pad = np.pad(a, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    return a_pad.reshape(-1)


def build_kernel(ic_chunks: int, ih: int, iw: int, kernel_size: int, stride: int, pad: int,
                  a_pad, kernel_name: str = "hex_maxpool"):
    """Build + apply the HVX max-pool custom_kernel via Tensor.custom_kernel. `a_pad`: flat
    (ic_chunks*(ih+2*pad)*(iw+2*pad)*32,) uint8 Tensor, TVM's own packed-NCHWc layout, pre-padded
    via pad_nchwc(). Returns the (ic_chunks*oh*ow, 32) uint8 output Tensor -- reshape/unpack via
    unpack_nchwc() to get back to (C, oh, ow). call .realize() to run it."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    ih_pad, iw_pad = ih + 2 * pad, iw + 2 * pad
    oh_count = (ih_pad - kernel_size) // stride + 1
    ow_count = (iw_pad - kernel_size) // stride + 1
    kk_count = kernel_size * kernel_size
    u8x32 = "unsigned char __attribute__((vector_size(32)))"

    def _reg_u8x32(shape, slot, *deps):
        # See hex_gemm_kernel.py's _reg_i32 for why every enclosing range must be a dependency,
        # not just the innermost: a degenerate extent-1 range gets eliminated by tinygrad's
        # optimizer, and an init scoped only to it would silently stop resetting per iteration.
        ret = UOp.placeholder(shape, dtypes.uint8, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp) -> UOp:
        ic_rng = UOp.range(ic_chunks, 0, AxisType.WEAK)
        oh_rng = UOp.range(oh_count, 1, AxisType.WEAK)
        ow_rng = UOp.range(ow_count, 2, AxisType.WEAK)
        acc = _reg_u8x32((32,), 0, ic_rng, oh_rng, ow_rng)
        kk_rng = UOp.range(kk_count, 3, AxisType.REDUCE)
        acc_addr = acc.after(kk_rng)[0]
        kh, kw = kk_rng // kernel_size, kk_rng % kernel_size
        a_row = oh_rng * stride + kh
        a_col = ow_rng * stride + kw
        a_flat = ic_rng * (ih_pad * iw_pad * 32) + (a_row * iw_pad + a_col) * 32
        a_idx = A[a_flat]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (acc_addr, a_idx),
            arg=(f"*({u8x32}*){{0}} = __builtin_elementwise_max("
                 f"*({u8x32}*){{0}}, *({u8x32}*){{1}});"),
        )
        update = step.end(kk_rng)
        final_addr = acc.after(update)[0]
        c_flat = (ic_rng * oh_count + oh_rng) * ow_count + ow_rng
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (C[c_flat, 0], final_addr),
            arg=f"*({u8x32}*){{0}} = *({u8x32}*){{1}};",
        )
        return out_step.end(ow_rng, oh_rng, ic_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(ic_chunks * oh_count * ow_count, 32, dtype="uint8", device="DSP")
    return Tensor.custom_kernel(c, a_pad, fxn=functools.partial(kernel_fn))[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--ih", type=int, default=400, help="input height (default: the real stem-pool shape)")
    p.add_argument("--iw", type=int, default=544, help="input width (default: the real stem-pool shape)")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--pad", type=int, default=1)
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
        ic_chunks = args.cin // IC_BN
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (args.cin, args.ih, args.iw)).astype(np.uint8)
        packed = pack_nchwc(img)
        padded = pad_nchwc(packed, ic_chunks, args.ih, args.iw, args.pad)
        a = Tensor(padded, device="DSP")
        out = build_kernel(ic_chunks, args.ih, args.iw, args.kernel_size, args.stride, args.pad, a)
        out.realize()

        oh = (args.ih + 2 * args.pad - args.kernel_size) // args.stride + 1
        ow = (args.iw + 2 * args.pad - args.kernel_size) // args.stride + 1
        out_np = unpack_nchwc(out.numpy(), ic_chunks, oh, ow)

        # numpy reference: real MaxPool semantics (pad, then windowed max), independent of layout.
        img_pad = np.pad(img, ((0, 0), (args.pad, args.pad), (args.pad, args.pad)), constant_values=0)
        ref = np.zeros((args.cin, oh, ow), dtype=np.uint8)
        for i in range(oh):
            for j in range(ow):
                window = img_pad[:, i * args.stride:i * args.stride + args.kernel_size,
                                  j * args.stride:j * args.stride + args.kernel_size]
                ref[:, i, j] = window.max(axis=(1, 2))

        correct = bool(np.array_equal(out_np, ref))
        print(f"correctness (cin={args.cin}, {args.ih}x{args.iw}): {correct}")
        if not correct:
            diff = out_np.astype(np.int64) - ref.astype(np.int64)
            print("max abs diff", np.abs(diff).max(), "mismatched", np.count_nonzero(diff), "/", diff.size)
            raise SystemExit("kernel is incorrect")
    finally:
        ClangRenderer.render = orig_render

    src = captured.get("src", "")
    marker = src.find("/* DSP boilerplate */")
    kernel_src = (src[:marker] if marker >= 0 else src).rstrip() + "\n"
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes) for cin={args.cin} {args.ih}x{args.iw}")


if __name__ == "__main__":
    main()
