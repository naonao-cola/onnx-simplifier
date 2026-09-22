#!/usr/bin/env python3
"""Capture tinygrad's generated Hexagon C kernel for a uint8 GEMM, and split it into the
compute function alone (dropping the MOCKDSP-specific `_start`/syscall boilerplate).

tinygrad's real (non-mock) Hexagon DSP backend needs raw `/dev/adsprpc-smd` access via its own
FastRPC ioctl protocol, which production Android builds don't grant to an unprivileged process
(confirmed on-device: `/vendor/dsp/cdsp/fastrpc_shell_3` exists but is permission-denied to the
`shell` user, `adb root` is refused -- "adbd cannot run as root in production builds"). tinygrad
links `libcdsprpc.so` for that -- unlike TVM's `tvm_rpc_android`, which uses the vendor's official
user-space FastRPC library and reaches the DSP fine as a plain `adb shell`-launched process. So
this script runs the *codegen* under `MOCKDSP=1` (qemu emulation) purely to capture correct,
qemu-verified source and extract the kernel function; `bridge_and_test.py` compiles that function
with the real (non-mock) Hexagon toolchain and runs it on the phone via TVM's existing, already-
permitted RPC transport instead.

Requires a tinygrad checkout with the vrmpy TensorCore patch applied (../vrmpy_tensorcore.patch)
for the WMMA/vrmpy path; without it this still runs, just with tinygrad's default (scalar/auto-
vectorized) Hexagon codegen.
"""
from __future__ import annotations

import argparse
import os

os.environ["DEV"] = "DSP"
os.environ["MOCKDSP"] = "1"

import numpy as np  # noqa: E402
from tinygrad import Context, Tensor  # noqa: E402
from tinygrad.renderer.cstyle import ClangRenderer  # noqa: E402


def capture_source(hw: int, cin: int, cout: int, beam: int, seed: int = 5) -> tuple[str, bool]:
    """Run a (hw,cin) x (cin,cout) uint8 GEMM through DEV=DSP MOCKDSP=1, capture the rendered C
    source, and verify correctness against a numpy reference. Returns (source, correct)."""
    captured: dict[str, str] = {}
    orig_render = ClangRenderer.render

    def _capture(self, uops):
        src = orig_render(self, uops)
        captured["src"] = src
        return src

    ClangRenderer.render = _capture
    try:
        rng = np.random.default_rng(seed)
        a_np = rng.integers(0, 100, (hw, cin)).astype(np.uint8)
        b_np = rng.integers(0, 100, (cin, cout)).astype(np.uint8)
        with Context(BEAM=beam):
            a, b = Tensor(a_np, dtype="uint8"), Tensor(b_np, dtype="uint8")
            # a.dot(b, dtype=...) keeps the multiply's operands at uint8 (only the accumulation
            # is int32) -- required for the vrmpy TensorCore match, which checks the multiply's
            # *operand* dtype directly. `(a.cast(int32) @ b.cast(int32))` casts before the
            # multiply and will never match any int8/uint8 TensorCore.
            out = a.dot(b, dtype="int32").realize()
        ref = a_np.astype(np.int64) @ b_np.astype(np.int64)
        correct = bool(np.array_equal(out.numpy().astype(np.int64), ref))
    finally:
        ClangRenderer.render = orig_render
    return captured["src"], correct


def extract_kernel_only(src: str) -> str:
    """Drop everything from the MOCKDSP `/* DSP boilerplate */` marker onward, keeping just the
    typedefs + the compute kernel function itself (portable to the real Hexagon toolchain)."""
    marker = src.index("/* DSP boilerplate */")
    return src[:marker].rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hw", type=int, default=512, help="M dimension (rows)")
    p.add_argument("--cin", type=int, default=64, help="K dimension (reduction)")
    p.add_argument("--cout", type=int, default=256, help="N dimension (columns)")
    p.add_argument("--beam", type=int, default=2, help="tinygrad BEAM search width (0 = off)")
    p.add_argument("--out", default="kernel.c", help="where to write the extracted kernel C source")
    args = p.parse_args()

    src, correct = capture_source(args.hw, args.cin, args.cout, args.beam)
    print(f"correctness match (qemu): {correct}")
    if not correct:
        raise SystemExit("codegen produced incorrect results under qemu -- not safe to bridge")
    kernel_src = extract_kernel_only(src)
    with open(args.out, "w") as f:
        f.write(kernel_src)
    uses_vrmpy = "vrmpy" in kernel_src
    print(f"wrote {args.out} ({len(kernel_src)} bytes); uses vrmpy: {uses_vrmpy}")


if __name__ == "__main__":
    main()
