#!/usr/bin/env python3
"""Coverage for the real Mask R-CNN backbone's FPN `resize2d` upsample -- the last op in the
profile that had no tinygrad-generated kernel at all (every conv shape, `add`, `maxpool` already
do; see ../README.md). Read `scripts/android/maskrcnn_e2e/README.md`'s "Fixed: FPN resize2d, via
an exact-2x integer fast path" section first: this project already has a real, shipped fix for
this op (`scripts/android/hexagon_resize2x.py`), a TVM `te.compute` schedule swap, not a
`custom_kernel` -- that's the baseline this file is trying to beat, not stock TVM's original
(much slower, per-pixel float `ceil`/`floor`/`round`) schedule.

FPN's resize is always exactly 2x per axis, nearest-neighbor, half_pixel, round_prefer_floor --
which collapses to plain integer replication with no float math at all:
`out[c, oh, ow] = in[c, oh // 2, ow // 2]` (verified/derived in `hexagon_resize2x.py`'s own
docstring). dtype is `int8` (`scripts/android/maskrcnn_e2e/profile_noncon_ops.py`'s
`build_specs()`). No accumulator, no reduction -- pure data movement: every input pixel is
replicated into a 2x2 output block, per channel, independently.

`build_kernel()`'s approach: NCHW's innermost axis (W) is what makes the horizontal doubling
vectorizable -- each output row is built from the corresponding input row via a compile-time
`__builtin_shufflevector` byte-duplication mask over 128-byte input chunks (`n_chunks =
ceil(w/128)` per row; `pad_input_row()` zero-pads each row's width up to a multiple of 128
host-side so every 128-byte chunk read is always within allocated memory, matching this project's
established pre-padding convention from `hex_conv3x3_kernel.py`). The vertical doubling (output
rows `2r` and `2r+1` are byte-identical) is done by building the doubled row once into a LOCAL
stack buffer, then writing that buffer to both output row addresses -- not by writing row `2r` to
the output array and reading it back to duplicate into row `2r+1`. That direct approach was tried
first and produced a real, silent bug: row `2r` itself was bit-exact under the real Hexagon
toolchain, but row `2r+1` diverged in exactly the byte range covered by a preceding *partial*
(non-full-128-byte) shuffle write -- the compiler doesn't reliably order a later scalar read
against an earlier HVX vector store to the same output-array region across what are, at the C
level, separate statements, even inside one function body. A local buffer has no such ambiguity:
it's never aliased by anything else, so the two final output writes are trivially independent.

Verified bit-exact correct on real hardware (device `239dbd8f`) at all three real FPN shapes.
Result is a genuinely mixed one, reported honestly: this kernel is dramatically faster than
*stock* TVM's original resize2d schedule (5.6x-17.3x) but, after correcting for a real
session-level measurement artifact in this file's own bridge script (comparing the fast path in
the same process as stock caused a compilation-cache collision that silently made the fast-path
number read out identical to stock's -- confirmed by rebuilding the fast path in total process
isolation, which reproduced the originally-established ~4/6.7/27.5 ms numbers from
`maskrcnn_e2e/README.md`), it's actually *slower* than the already-fixed fast path at the two
smaller shapes and only marginally faster at the largest one. See ../README.md's "Coverage: FPN
resize2d" section for the full numbers and what's not yet been tried to close that gap.
"""
from __future__ import annotations

import argparse
import functools
import os

import numpy as np


def pad_input_row(a: np.ndarray) -> np.ndarray:
    """a: (C, H, W) int8. Pads W up to a multiple of 128 (zero-fill) so every 128-byte chunk read
    inside the kernel is always within allocated memory. Returns (C, H, W_padded)."""
    c, h, w = a.shape
    w_padded = ((w + 127) // 128) * 128
    if w_padded == w:
        return a
    return np.pad(a, ((0, 0), (0, 0), (0, w_padded - w)), mode="constant", constant_values=0)


def build_kernel(c: int, h: int, w: int, a, kernel_name: str = "hex_resize2x"):
    """Build + apply the exact-2x nearest-neighbor resize `custom_kernel`. `a`: flat
    (c*h*w_padded,) int8 Tensor (pre-padded via pad_input_row(), W padded up to a multiple of
    128). Returns the (c*2h*2w,) int8 output Tensor; call .realize() to run it.

    `out[c, oh, ow] = in[c, oh // 2, ow // 2]` -- exact-2x nearest-neighbor, half_pixel,
    round_prefer_floor (the only resize mode Mask R-CNN's FPN ever uses; see
    ../hexagon_resize2x.py and ../../maskrcnn_e2e/README.md's "Fixed: FPN resize2d" section for
    the algebraic derivation of `out_x // 2` as the closed-form nearest index for this exact case).
    """
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    n_chunks = (w + 127) // 128
    w_padded = n_chunks * 128
    i8x128 = "signed char __attribute__((vector_size(128)))"

    # __builtin_shufflevector byte-duplication masks over one 128-byte input chunk: the low mask
    # produces output bytes [0,127] (input columns 0-63, each read twice), the high mask produces
    # output bytes [128,255] (input columns 64-127, each read twice) -- together, one 128-byte
    # input chunk's worth of columns doubles into up to 256 bytes of output (two output vectors).
    low_mask = ",".join(str(i // 2) for i in range(128))
    high_mask = ",".join(str(64 + i // 2) for i in range(128))

    def kernel_fn(C: UOp, A: UOp) -> UOp:
        c_rng = UOp.range(c, 0, AxisType.WEAK)
        r_rng = UOp.range(h, 1, AxisType.WEAK)

        in_row_off = c_rng * (h * w_padded) + r_rng * w_padded
        out_row_off = c_rng * (2 * h * 2 * w) + (r_rng * 2) * (2 * w)

        srcs: list[UOp] = []
        # Build the doubled row entirely in a local stack buffer first, then write it to BOTH
        # output rows (2r, 2r+1) -- never reading the output array back (see module docstring for
        # the real bug this avoids).
        buf_bytes = n_chunks * 256

        def add_src(idx_uop: UOp) -> int:
            srcs.append(idx_uop)
            return len(srcs) - 1

        stmt_parts = [f"{i8x128} _row[{(buf_bytes + 127) // 128}];"]

        for k in range(n_chunks):
            v_slot = add_src(A[in_row_off + k * 128])
            valid_in = min(128, w - k * 128)
            valid_out = 2 * valid_in

            if valid_out >= 128:
                stmt_parts.append(
                    f"*({i8x128}*)((signed char*)_row + {k * 256}) = __builtin_shufflevector("
                    f"*({i8x128}*){{{v_slot}}}, *({i8x128}*){{{v_slot}}}, {low_mask});"
                )
                remaining = valid_out - 128
                if remaining > 0:
                    if remaining >= 128:
                        stmt_parts.append(
                            f"*({i8x128}*)((signed char*)_row + {k * 256 + 128}) = __builtin_shufflevector("
                            f"*({i8x128}*){{{v_slot}}}, *({i8x128}*){{{v_slot}}}, {high_mask});"
                        )
                    else:
                        stmt_parts.append(
                            f"{{{{ {i8x128} _t{k} = __builtin_shufflevector(*({i8x128}*){{{v_slot}}}, "
                            f"*({i8x128}*){{{v_slot}}}, {high_mask}); "
                            f"for (int _i{k} = 0; _i{k} < {remaining}; _i{k}++) "
                            f"((signed char*)_row)[{k * 256 + 128} + _i{k}] = ((signed char*)&_t{k})[_i{k}]; }}}}"
                        )
            else:
                stmt_parts.append(
                    f"{{{{ {i8x128} _t{k} = __builtin_shufflevector(*({i8x128}*){{{v_slot}}}, "
                    f"*({i8x128}*){{{v_slot}}}, {low_mask}); "
                    f"for (int _i{k} = 0; _i{k} < {valid_out}; _i{k}++) "
                    f"((signed char*)_row)[{k * 256} + _i{k}] = ((signed char*)&_t{k})[_i{k}]; }}}}"
                )

        row0_slot = add_src(C[out_row_off])
        row1_slot = add_src(C[out_row_off + 2 * w])
        stmt_parts.append(
            f"for (int _j = 0; _j < {2 * w}; _j++) {{{{ signed char _b = ((signed char*)_row)[_j]; "
            f"((signed char*){{{row0_slot}}})[_j] = _b; ((signed char*){{{row1_slot}}})[_j] = _b; }}}}"
        )

        body = " ".join(stmt_parts)
        step = UOp(Ops.CUSTOM, dtypes.void, tuple(srcs), arg=body)
        return step.end(r_rng, c_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    out = Tensor.empty(c * 2 * h * 2 * w, dtype="int8", device="DSP")
    return Tensor.custom_kernel(out, a, fxn=functools.partial(kernel_fn))[0]


def extract_kernel_only(src: str) -> str:
    marker = src.find("/* DSP boilerplate */")
    return (src[:marker] if marker >= 0 else src).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c", type=int, default=256)
    p.add_argument("--h", type=int, default=100)
    p.add_argument("--w", type=int, default=136, help="default: the real biggest profile shape")
    p.add_argument("--out", default="resize2x_kernel.c")
    args = p.parse_args()

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
        rng = np.random.default_rng(5)
        a_np = rng.integers(-100, 100, (args.c, args.h, args.w)).astype(np.int8)
        a_pad = pad_input_row(a_np)
        a = Tensor(a_pad.reshape(-1), dtype="int8", device="DSP")
        out = build_kernel(args.c, args.h, args.w, a)
        out.realize()
        out_np = out.numpy().reshape(args.c, 2 * args.h, 2 * args.w)
        ref = np.repeat(np.repeat(a_np, 2, axis=1), 2, axis=2)
        correct = bool(np.array_equal(out_np, ref))
        print(f"correctness (c={args.c} h={args.h} w={args.w}): {correct}")
        if not correct:
            raise SystemExit("kernel is incorrect")
    finally:
        ClangRenderer.render = orig_render

    kernel_src = extract_kernel_only(captured.get("src", ""))
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes)")


if __name__ == "__main__":
    main()
