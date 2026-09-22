#!/usr/bin/env python3
"""Coverage for the per-output-channel int32 bias-add every real `qnn.conv2d` in the Mask R-CNN
backbone needs before `qnn.requantize` -- a gap Stage 2's chained-subgraph work (see
../README.md's "Chaining a real subgraph" section) found while trying to compose
`hex_stem7x7_kernel.py`'s raw accumulate (no bias) with `hex_requantize_kernel.py`: the real
backbone's ONNX QDQ graph carries a real int32 bias (`ConvAddFusion_Add_B_*_quantized`), already
pre-scaled to the conv's `input_scale * weight_scale` so it adds directly to the int32 accumulator
-- and for the very first conv (the stem, whose input is the raw asymmetric-quantized image,
`zero_point=114`, not 0 like every later activation in this network), that same bias slot is also
where the activation zero-point's cross term gets folded in host-side (`b' = b - x_zp *
sum_{ic,kh,kw}(weight)`, precomputed once since weights are static), so this one simple op is what
makes zero-point handling a solved problem without needing any real on-device subtraction logic.

Operates on `hex_stem7x7_kernel.py`'s own output layout directly: `(pos, cout)`, NHWC-flat
(position-major, channel-minor) -- `out[pos, c] = acc[pos, c] + bias[c]`. No accumulator, no
reduction -- the simplest kind of kernel in this project (same class as `hex_add_kernel.py`), just
a broadcast instead of an elementwise same-shape add.
"""
from __future__ import annotations

import argparse
import functools
import os


def build_kernel(pos_count: int, cout: int, acc, bias, kernel_name: str = "hex_bias_add"):
    """Build + apply the bias-add custom_kernel. `acc`: flat (pos_count*cout,) int32 Tensor
    (conv's raw accumulator, `(pos, cout)` NHWC-flat). `bias`: (cout,) int32 Tensor. Returns the
    (pos_count*cout,) int32 output Tensor; call .realize() to run it."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    def kernel_fn(C: UOp, A: UOp, B: UOp) -> UOp:
        p_rng = UOp.range(pos_count, 0, AxisType.WEAK)
        c_rng = UOp.range(cout, 1, AxisType.WEAK)
        flat = p_rng * cout + c_rng
        a_idx, b_idx, c_idx = A[flat], B[c_rng], C[flat]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (c_idx, a_idx, b_idx),
            arg="*(int*){0} = *(int*){1} + *(int*){2};",
        )
        return step.end(c_rng, p_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(pos_count * cout, dtype="int32", device="DSP")
    return Tensor.custom_kernel(c, acc, bias, fxn=functools.partial(kernel_fn))[0]


def extract_kernel_only(src: str) -> str:
    marker = src.find("/* DSP boilerplate */")
    return (src[:marker] if marker >= 0 else src).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pos", type=int, default=400 * 544, help="spatial positions (default: real stem-conv output, oh=400,ow=544)")
    p.add_argument("--cout", type=int, default=64)
    p.add_argument("--out", default="bias_add_kernel.c")
    args = p.parse_args()

    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")

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
        rng = np.random.default_rng(11)
        acc_np = rng.integers(-1000, 1000, (args.pos, args.cout)).astype(np.int32)
        bias_np = rng.integers(-500, 500, args.cout).astype(np.int32)
        acc = Tensor(acc_np.reshape(-1), device="DSP")
        bias = Tensor(bias_np, device="DSP")
        out = build_kernel(args.pos, args.cout, acc, bias)
        out.realize()
        ref = (acc_np.astype(np.int64) + bias_np.astype(np.int64)[None, :]).reshape(-1)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
        print(f"correctness (pos={args.pos}, cout={args.cout}): {correct}")
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
