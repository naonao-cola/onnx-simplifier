#!/usr/bin/env python3
"""Coverage for `qnn.requantize` -- the int32-accumulator-to-uint8 rescale+clamp step between
every conv layer in the real Mask R-CNN backbone -- the one op this project's kernel coverage was
still missing (every conv/GEMM/add/maxpool/resize op already has a real, hand-written
`Tensor.custom_kernel`; see ../README.md). Needed to chain any two covered conv kernels together
into a real subgraph.

Formula, read directly out of TVM's own source (`src/relay/qnn/op/requantize.cc` +
`src/target/intrin_rule.cc`'s `QMultiplyShift`, the `tir.q_multiply_shift` intrinsic
`fixed_point_multiply` lowers to) rather than assumed, to guarantee bit-exact agreement with the
"UPWARD" rounding mode `scripts/android/maskrcnn_e2e/README.md` confirms is the one the real
FakeQuantizationToInteger pass actually uses (its finding #3: "`requantize` with `TONEAREST`
rounding mismatched the host on a synthetic test (72% of elements); the default `UPWARD` mode ...
is correct"):

    tensor = int32(x) - input_zero_point                    # usually 0 for a fresh accumulator
    left_shift, right_shift = max(shift, 0), max(-shift, 0)  # shift from GetFixedPointMultiplierShift
    prod = (int64(tensor) << left_shift) * int64(multiplier) # multiplier is a Q31 fixed-point int32
    total_shift = right_shift + 31                            # q=31 (Q31 format)
    scaled = (prod + (1 << (total_shift - 1))) >> total_shift # UPWARD: round-half-up via bias-then-shift
    out = clip(scaled + output_zero_point, 0, 255)             # uint8 output range

`(multiplier, shift)` are derived from `input_scale / output_scale` via
`GetFixedPointMultiplierShift` (frexp-based Q31 significand + exponent) -- computed host-side in
Python (`compute_multiplier_shift()` below, a direct port) and baked into the kernel as constants,
matching how every other kernel in this project bakes shapes in at trace time. Not vectorized to
HVX width yet (a plain per-element `long long` scalar loop) -- correctness first, matching this
project's established precedent (see `hex_gemm_kernel.py`'s docstring on `custom_kernel` vs.
normal codegen, and every kernel's own two-stage qemu-then-hardware verification order); the int64
multiply-and-variable-shift this op needs has no direct single HVX vector instruction the way
`vrmpy`/plain add do, so vectorizing it well is a real follow-up, not attempted here.
"""
from __future__ import annotations

import argparse
import math
import os


def compute_multiplier_shift(double_multiplier: float) -> tuple[int, int]:
    """Direct port of TVM's `GetFixedPointMultiplierShift` (src/relay/qnn/utils.cc): decompose
    `double_multiplier` into a Q31 fixed-point `(multiplier, shift)` pair via `math.frexp`."""
    if double_multiplier == 0.0:
        return 0, 0
    significand, exponent = math.frexp(double_multiplier)
    significand_int = round(significand * (1 << 31))
    if significand_int == (1 << 31):
        significand_int //= 2
        exponent += 1
    assert significand_int <= 0x7FFFFFFF
    return significand_int, exponent


def requantize_ref(x, input_zero_point: int, multiplier: int, shift: int, output_zero_point: int):
    """Bit-exact numpy reference matching TVM's int64 fixed-point arithmetic exactly (not a float
    approximation) -- used to verify the kernel below."""
    import numpy as np

    tensor = x.astype(np.int64) - np.int64(input_zero_point)
    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)
    prod = (tensor << left_shift) * np.int64(multiplier)
    total_shift = right_shift + 31
    rounding = np.int64(1) << (total_shift - 1)
    scaled = (prod + rounding) >> np.int64(total_shift)
    out = scaled + np.int64(output_zero_point)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_kernel(n: int, multiplier: int, shift: int, input_zero_point: int, output_zero_point: int,
                  a, kernel_name: str = "hex_requantize"):
    """Build + apply a `Tensor.custom_kernel` computing `requantize_ref` above on-device.
    `a`: shape (n,) int32 Tensor (the conv accumulator). Returns the (n,) uint8 output Tensor;
    call `.realize()` to run it. `multiplier`/`shift`/`input_zero_point`/`output_zero_point` are
    baked in as compile-time constants (from `compute_multiplier_shift()` plus the QNN node's own
    zero points), matching every other kernel in this project."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)
    total_shift = right_shift + 31
    rounding = 1 << (total_shift - 1)

    def kernel_fn(C: UOp, A: UOp) -> UOp:
        v_rng = UOp.range(n, 0, AxisType.WEAK)
        a_idx, c_idx = A[v_rng], C[v_rng]
        step = UOp(
            Ops.CUSTOM, dtypes.void, (c_idx, a_idx),
            arg=(
                f"{{{{ long long t = (long long)(*(int*){{1}}) - {input_zero_point}; "
                f"t = (t << {left_shift}) * {multiplier}LL; "
                f"t = (t + {rounding}LL) >> {total_shift}; "
                f"t += {output_zero_point}; "
                f"if (t < 0) t = 0; if (t > 255) t = 255; "
                f"*(unsigned char*){{0}} = (unsigned char)t; }}}}"
            ),
        )
        return step.end(v_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(n, dtype="uint8", device="DSP")
    return Tensor.custom_kernel(c, a, fxn=lambda C, A: kernel_fn(C, A))[0]


def extract_kernel_only(src: str) -> str:
    marker = src.find("/* DSP boilerplate */")
    return (src[:marker] if marker >= 0 else src).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=64 * 400 * 544,
                    help="element count (default: the real stem-conv output size, cin=64,H=400,W=544)")
    p.add_argument("--in-scale", type=float, default=0.02)
    p.add_argument("--out-scale", type=float, default=0.05)
    p.add_argument("--in-zp", type=int, default=0)
    p.add_argument("--out-zp", type=int, default=114)
    p.add_argument("--out", default="requantize_kernel.c")
    args = p.parse_args()

    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")

    import numpy as np
    from tinygrad import Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    multiplier, shift = compute_multiplier_shift(args.in_scale / args.out_scale)
    print(f"multiplier={multiplier} shift={shift}")

    captured: dict[str, str] = {}
    orig_render = ClangRenderer.render

    def _capture(self, uops):
        src = orig_render(self, uops)
        captured["src"] = src
        return src

    ClangRenderer.render = _capture
    try:
        rng = np.random.default_rng(7)
        x_np = rng.integers(-2_000_000, 2_000_000, args.n).astype(np.int32)
        a = Tensor(x_np, dtype="int32", device="DSP")
        out = build_kernel(args.n, multiplier, shift, args.in_zp, args.out_zp, a)
        out.realize()
        ref = requantize_ref(x_np, args.in_zp, multiplier, shift, args.out_zp)
        correct = bool(np.array_equal(out.numpy(), ref))
    finally:
        ClangRenderer.render = orig_render

    print(f"correctness (n={args.n}): {correct}")
    if not correct:
        raise SystemExit("requantize kernel is incorrect")

    src = extract_kernel_only(captured.get("src", ""))
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out} ({len(src)} bytes)")


if __name__ == "__main__":
    main()
