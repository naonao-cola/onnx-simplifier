#!/usr/bin/env python3
"""Coverage for the real Mask R-CNN backbone's elementwise `add` (FPN lateral + top-down merge,
`int32 + int32 -> int32`, pre-requantization accumulators -- see
`scripts/android/maskrcnn_e2e/profile_noncon_ops.py`'s `build_specs()`), the biggest remaining
uncovered item in `noncon_profile.json` (102.3 ms across 6 shapes) besides `resize` (already
handled separately, not via `custom_kernel`) and `maxpool`/`sigmoid` (smaller, not attempted here).

Two approaches, both here, in the order they were actually tried:

1. `capture_source()`: tinygrad's *normal* (non-`custom_kernel`) codegen path (`a + b`) with BEAM
   search -- the cheaper option `hex_gemm_kernel.py`'s docstring says to prefer when no specific
   hardware intrinsic is needed. At BEAM up to 6, the `ClangRenderer`/Hexagon backend found no
   vectorized candidate at the real full-scale shape (13,926,400 elements) -- every candidate
   explored was the same plain scalar `for` loop. Bridged to real hardware anyway to get an actual
   number rather than assume: **17.6x slower than stock TVM's `relay.add`** (292.8 ms / 0.048
   G-elem/s vs TVM's 16.6 ms / 0.838 G-elem/s) -- not remotely competitive, a real negative result,
   not a search failure to route around silently.
2. `build_vector_kernel()`: given (1)'s result, a hand-written `Tensor.custom_kernel` *is*
   warranted here after all (unlike the docstring's general default advice) -- not because a
   specific HVX intrinsic is needed (there's no `vrmpy`-style instruction for plain addition), but
   because tinygrad's Hexagon backend apparently doesn't auto-vectorize even a trivial elementwise
   loop to HVX width on its own. The kernel is deliberately the simplest possible `custom_kernel`
   in this project: no accumulator, no reduction, no `_reg_i32`-style degenerate-range dependency
   tracking needed (every other kernel here needed that because of a persistent REG accumulator
   across a reduction loop; this op has neither) -- just a `(32,)`-wide `int __attribute__((
   vector_size(128)))` load/add/store per iteration, one HVX vector (32 int32 lanes) at a time,
   using plain C vector-extension `+` (not an HVX builtin -- clang lowers vector-extension
   arithmetic to the matching HVX instruction directly under `-mhvx`, unlike `vrmpy`'s
   accumulate-into-place semantics, which needed an explicit intrinsic).
"""
from __future__ import annotations

import argparse
import functools
import os


def capture_source(shape: tuple[int, ...], beam: int, seed: int = 5) -> tuple[str, bool]:
    """Run `a + b` (int32, matching the real backbone's pre-requantization accumulator add)
    through DEV=DSP MOCKDSP=1, capture the rendered C source, and verify correctness against a
    numpy reference. Returns (source, correct)."""
    import numpy as np
    from tinygrad import Context, Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    captured: dict[str, str] = {}
    orig_render = ClangRenderer.render

    def _capture(self, uops):
        src = orig_render(self, uops)
        captured["src"] = src
        return src

    ClangRenderer.render = _capture
    try:
        rng = np.random.default_rng(seed)
        a_np = rng.integers(-1000, 1000, shape).astype(np.int32)
        b_np = rng.integers(-1000, 1000, shape).astype(np.int32)
        with Context(BEAM=beam):
            a, b = Tensor(a_np, dtype="int32"), Tensor(b_np, dtype="int32")
            out = (a + b).realize()
        ref = a_np.astype(np.int64) + b_np.astype(np.int64)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
    finally:
        ClangRenderer.render = orig_render
    return captured["src"], correct


def build_vector_kernel(n: int, a, b, kernel_name: str = "hex_add"):
    """Build + apply a plain HVX-vectorized elementwise-add `custom_kernel`. `n` must be a
    multiple of 32 (one HVX 128-byte vector = 32 int32 lanes). `a`, `b`: shape (n,) int32
    Tensors. Returns the (n,) int32 output Tensor; call .realize() to run it."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert n % 32 == 0
    i32x32 = "int __attribute__((vector_size(128)))"

    def kernel_fn(C: UOp, A: UOp, B: UOp) -> UOp:
        v_rng = UOp.range(n // 32, 0, AxisType.WEAK)
        a_idx, b_idx, c_idx = A[v_rng * 32], B[v_rng * 32], C[v_rng * 32]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (c_idx, a_idx, b_idx),
            arg=f"*({i32x32}*){{0}} = *({i32x32}*){{1}} + *({i32x32}*){{2}};",
        )
        return step.end(v_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(n, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, a, b, fxn=functools.partial(kernel_fn))[0]


def extract_kernel_only(src: str) -> str:
    marker = src.find("/* DSP boilerplate */")
    return (src[:marker] if marker >= 0 else src).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs="+", default=[1, 256, 200, 272],
                    help="tensor shape (default: the real biggest profile `add` shape)")
    p.add_argument("--beam", type=int, default=2, help="tinygrad BEAM search width for the "
                    "normal-codegen path (0 = off); ignored with --vector")
    p.add_argument("--vector", action="store_true",
                    help="use the hand-vectorized custom_kernel instead of normal codegen+BEAM")
    p.add_argument("--out", default="add_kernel.c")
    args = p.parse_args()

    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")

    n = 1
    for d in args.shape:
        n *= d

    if args.vector:
        import numpy as np
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
            a_np = rng.integers(-1000, 1000, n).astype(np.int32)
            b_np = rng.integers(-1000, 1000, n).astype(np.int32)
            a, b = Tensor(a_np, dtype="int32", device="DSP"), Tensor(b_np, dtype="int32", device="DSP")
            out = build_vector_kernel(n, a, b)
            out.realize()
            ref = a_np.astype(np.int64) + b_np.astype(np.int64)
            correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
        finally:
            ClangRenderer.render = orig_render
        print(f"correctness (vector, n={n}): {correct}")
        if not correct:
            raise SystemExit("vector kernel is incorrect")
        src = captured.get("src", "")
    else:
        src, correct = capture_source(tuple(args.shape), args.beam)
        print(f"correctness (shape={tuple(args.shape)}): {correct}")
        if not correct:
            raise SystemExit("codegen produced incorrect results under qemu -- not safe to bridge")

    kernel_src = extract_kernel_only(src)
    with open(args.out, "w") as f:
        f.write(kernel_src)
    print(f"wrote {args.out} ({len(kernel_src)} bytes)")


if __name__ == "__main__":
    main()
