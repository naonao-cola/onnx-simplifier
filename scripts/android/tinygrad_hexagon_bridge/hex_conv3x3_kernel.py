#!/usr/bin/env python3
"""A from-scratch, hand-written Hexagon vrmpy 3x3-conv kernel via tinygrad's `Tensor.custom_kernel`
-- extends ../hex_gemm_kernel.py's (1x1-conv / plain GEMM) approach to a genuine spatial 3x3
convolution, the next entry in `scripts/android/maskrcnn_e2e/README.md`'s ranked profile after
the 1x1 convs `hex_gemm_kernel.py` already covers.

Computes `C[oh,ow,cout] = sum_{kh,kw,cin} A_pad[oh+kh, ow+kw, cin] * W[cout, cin, kh, kw]`
(uint8 activation x int8 weight -> int32), stride=1, "same" output size via 1-pixel
zero-padding applied to the input *before* calling this kernel (im2col-style: push the
boundary-condition complexity to a cheap host-side pad, keep the hot loop branch-free -- the
same tradeoff `build_strided_kernel()` in ../hex_gemm_kernel.py makes for stride).

Weights are genuinely signed int8 in the real backbone (unlike this repo's synthetic uint8 x
uint8 GEMM tests), so this uses `vrmpybusv_acc_128B` (unsigned x signed, vector-broadcast
calling convention: the *activation* scalar is broadcast to a full 32-wide vector via a
compound literal, the *weight* is the plain 128-byte vector operand) instead of
`vrmpyub_acc_128B`'s plain-scalar convention -- see this repo's own `chunked_kernel_test.py`
(referenced in ../README.md) for where that calling-convention split was first established.

The reduction axis is a single combined range of extent `9 * (cin//4)`, decomposed inside the
kernel into `(kh, kw, kc)` via integer div/mod -- same `_reg_i32` multi-range-dependency
accumulator-init fix as ../hex_gemm_kernel.py (any range that can ever be extent-1, including
this reduction range when cin==4, needs every enclosing range in its dependency list or
tinygrad's optimizer silently drops a degenerate range and corrupts the "reset per output
pixel" behavior).
"""
from __future__ import annotations

import argparse
import functools

import numpy as np


def pack_weight_3x3(w: np.ndarray, n_tile: int = 32, k_sub: int = 4) -> np.ndarray:
    """w: (cout, cin, 3, 3) int8 (standard conv weight layout) -> Wp: (9, cout//n_tile,
    cin//k_sub, n_tile*k_sub) uint8 (bit-pattern-preserving view of the signed bytes), packed
    per kernel position exactly like ../hex_gemm_kernel.py's pack_b() packs a GEMM's B:
    Wp[kh*3+kw, nt, kc, n_lane*k_sub+ks] = w[nt*n_tile+n_lane, kc*k_sub+ks, kh, kw]."""
    cout, cin, kh, kw = w.shape
    assert (kh, kw) == (3, 3)
    assert cin % k_sub == 0 and cout % n_tile == 0
    w_u8 = w.view(np.uint8)  # preserve the signed bit pattern; vrmpybusv reads it as signed
    out = np.empty((9, cout // n_tile, cin // k_sub, n_tile * k_sub), dtype=np.uint8)
    for pos in range(9):
        i, j = pos // 3, pos % 3
        b = w_u8[:, :, i, j].T  # (cin, cout), matching pack_b()'s (K, N) convention
        out[pos] = (
            b.reshape(cin // k_sub, k_sub, cout // n_tile, n_tile)
            .transpose(2, 0, 3, 1)
            .reshape(cout // n_tile, cin // k_sub, n_tile * k_sub)
        )
    return out


def pad_input(a: np.ndarray) -> np.ndarray:
    """a: (ih, iw, cin) uint8 -> (ih+2, iw+2, cin) uint8, zero-padded by 1 on each spatial side.
    NOTE: real backbone activations are asymmetrically quantized (uint8 zero-point != 0), so a
    real integration must pad with the input's actual zero-point, not literal 0, to match TVM's
    qnn.conv2d semantics exactly -- this kernel itself is agnostic to that; it just convolves
    whatever's in the padded border. Kept as literal-0 padding here since this file's own
    correctness check computes its numpy reference against the same padded array."""
    return np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)


def build_kernel(cin: int, cout: int, ih: int, iw: int, a_pad, wp, kernel_name: str = "hex_conv3x3", ow_tile: int = 1):
    """Build + apply the HVX vrmpybusv 3x3-conv custom_kernel. `a_pad`: flat (ih+2)*(iw+2)*cin
    uint8 Tensor (pre-padded, see pad_input()). `wp`: flat 9*(cout//32)*(cin//4)*128 uint8
    Tensor (pre-packed via pack_weight_3x3(), viewed/flattened to 1D). Returns the (ih*iw, cout)
    int32 output Tensor; call .realize() to run it.

    `ow_tile`: number of adjacent output columns processed per accumulator group (default 1 =
    the original untiled kernel). For each reduction step, the packed weight slice for that
    (kh, kw, kc) position depends only on `nt`/`kc`, not on the output column -- with ow_tile>1,
    it's loaded into a local vector once and reused across `ow_tile` separate vrmpybusv_acc
    calls (one per tiled output column, each with its own accumulator) before the next reduction
    step, instead of being re-fetched from memory once per output pixel. This is the fix for the
    real-hardware slowdown at cin=cout=256 documented in ../README.md's "Coverage: a real 3x3
    conv" section: that shape's packed weight (589824 bytes) blows past L1, so with ow_tile=1
    every output pixel independently re-streams the whole 9-position weight set from L2/memory;
    tiling amortizes each weight fetch across ow_tile pixels' worth of MACs."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert cin % 4 == 0 and cout % 32 == 0
    assert iw % ow_tile == 0, f"iw={iw} must be divisible by ow_tile={ow_tile}"
    nt_count, kc_count = cout // 32, cin // 4
    r_count = 9 * kc_count
    owt_count = iw // ow_tile
    iw_pad = iw + 2
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"

    def _reg_i32(shape, slot, *deps):
        # See ../hex_gemm_kernel.py's _reg_i32: depend on every enclosing range (oh, owt, nt),
        # not just the innermost one, so a degenerate extent-1 range never silently downgrades
        # the "reset once per output pixel" init to "reset once total".
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Wp: UOp) -> UOp:
        oh_rng = UOp.range(ih, 0, AxisType.WEAK)
        owt_rng = UOp.range(owt_count, 1, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 2, AxisType.WEAK)
        # One accumulator per tiled output column, each its own REG slot (physical register).
        accs = [_reg_i32((32,), t, oh_rng, owt_rng, nt_rng) for t in range(ow_tile)]
        r_rng = UOp.range(r_count, 3, AxisType.REDUCE)
        acc_addrs = [acc.after(r_rng)[0] for acc in accs]

        pos = r_rng // kc_count
        kc = r_rng % kc_count
        kh = pos // 3
        kw = pos % 3

        a_row = oh_rng + kh
        wp_flat_idx = pos * (nt_count * kc_count * 128) + nt_rng * (kc_count * 128) + kc * 128
        w_idx = Wp[wp_flat_idx]

        a_idxs = []
        for t in range(ow_tile):
            a_col = owt_rng * ow_tile + t + kw
            a_flat_idx = a_row * (iw_pad * cin) + a_col * cin + kc * 4
            a_idxs.append(A[a_flat_idx])

        # Fuse all ow_tile accumulate updates for this reduction step into one CUSTOM statement:
        # load the (kh,kw,kc) weight slice (the operand that's *shared* across tiled output
        # columns) into a local vector once (__wv), then reuse it across ow_tile vrmpybusv_acc
        # calls -- the compiler keeps __wv register-resident across the statements below (no
        # repeated memory load per tiled column). vrmpybusv_acc_128B's calling convention (see
        # ../hex_gemm_kernel.py/../chunked_kernel_test.py): 2nd arg is the *activation*
        # broadcast to a 32-wide vector (one scalar per tiled column, differs per t), 3rd arg is
        # the raw 128-byte *weight* vector (shared -- this is __wv).
        acc_lines = []
        for t in range(ow_tile):
            broadcast = ",".join([f"*(unsigned int*){{{ow_tile + 1 + t}}}"] * 32)
            acc_lines.append(
                f"*({i32x32}*){{{t}}} = __builtin_HEXAGON_V6_vrmpybusv_acc_128B("
                f"*({i32x32}*){{{t}}}, ({i32x32}){{{{{broadcast}}}}}, __wv);"
            )
        arg = f"{u8x128} __wv = *({u8x128}*){{{ow_tile}}}; " + " ".join(acc_lines)
        step = UOp(Ops.CUSTOM, dtypes.void, (*acc_addrs, w_idx, *a_idxs), arg=arg)
        update = step.end(r_rng)
        final_addrs = [acc.after(update)[0] for acc in accs]

        # Fuse all ow_tile output writes into one CUSTOM statement too -- chaining separate
        # per-column CUSTOM statements via .after() looked right (each later write's source
        # embeds a dependency on the earlier write) but silently dropped every write except the
        # last from the rendered output: .after() only orders two nodes relative to each other
        # when both are already reachable from the sink, it does not itself make an
        # otherwise-unreferenced void statement reachable. Confirmed by inspecting the generated
        # C for ow_tile=2: only the t=1 write appeared, so half the output columns (t=0) were
        # simply never written -- exactly the 50% mismatch this bug produced. One CUSTOM with
        # all ow_tile writes as one statement list sidesteps this the same way the fused
        # accumulate step above does.
        out_cptrs = [C[oh_rng * iw + (owt_rng * ow_tile + t), nt_rng * 32] for t in range(ow_tile)]
        out_lines = [f"*({i32x32}*){{{t}}} = *({i32x32}*){{{ow_tile + t}}};" for t in range(ow_tile)]
        out_step = UOp(
            Ops.CUSTOM, dtypes.void, (*out_cptrs, *final_addrs),
            arg=" ".join(out_lines),
        )
        return out_step.end(nt_rng, owt_rng, oh_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(ih * iw, cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a_pad, wp, fxn=functools.partial(kernel_fn))[0]


def reference_conv3x3(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a: (ih, iw, cin) uint8, w: (cout, cin, 3, 3) int8 -> (ih*iw, cout) int64, stride=1
    pad=1, matching build_kernel()'s semantics exactly (same zero-padding convention)."""
    ih, iw, cin = a.shape
    cout = w.shape[0]
    a_pad = pad_input(a).astype(np.int64)
    w64 = w.astype(np.int64)
    out = np.zeros((ih, iw, cout), dtype=np.int64)
    for kh in range(3):
        for kw in range(3):
            patch = a_pad[kh:kh + ih, kw:kw + iw, :]  # (ih, iw, cin)
            out += np.einsum("hwc,oc->hwo", patch, w64[:, :, kh, kw])
    return out.reshape(ih * iw, cout)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=64)
    p.add_argument("--ih", type=int, default=6)
    p.add_argument("--iw", type=int, default=6)
    p.add_argument("--ow-tile", type=int, default=1)
    p.add_argument("--out", default="conv3x3_kernel.c")
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
        a_np = rng.integers(0, 100, (args.ih, args.iw, args.cin)).astype(np.uint8)
        w_np = rng.integers(-40, 40, (args.cout, args.cin, 3, 3)).astype(np.int8)
        a_pad_np = pad_input(a_np).reshape(-1)
        wp_np = pack_weight_3x3(w_np).reshape(-1)

        a_pad_t = Tensor(a_pad_np, device="DSP")
        wp_t = Tensor(wp_np, device="DSP")
        out = build_kernel(args.cin, args.cout, args.ih, args.iw, a_pad_t, wp_t, ow_tile=args.ow_tile)
        out.realize()

        ref = reference_conv3x3(a_np, w_np)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
        print(f"correctness (cin={args.cin} cout={args.cout} ih={args.ih} iw={args.iw}): {correct}")
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
