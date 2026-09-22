#!/usr/bin/env python3
"""Run tinygrad's *Hexagon code generation* on backbone-shaped convolutions under QEMU.

tinygrad's DSP backend has a mock mode (`DEV=DSP MOCKDSP=1`) that compiles the generated C with
`clang --target=hexagon -mcpu=hexagonv65 -mhvx=v65` and executes it with `qemu-hexagon-static`,
returning the instruction count. Needs a clang with the Hexagon target and lld on PATH plus
`qemu-hexagon-static` (`apt install qemu-user-static`). This checks numerics and instruction cost
of tinygrad's codegen; it does not use the phone (tinygrad's real DSP runtime needs raw access to
`/dev/adsprpc-smd` and `/dev/ion`, which Android denies to the ADB shell).
"""

from __future__ import annotations

import os

os.environ.update({"DEV": "DSP", "MOCKDSP": "1"})

import numpy as np  # noqa: E402
from tinygrad import Tensor  # noqa: E402
from tinygrad.helpers import GlobalCounters  # noqa: E402

CASES = [
    ("1x1 64->64 @28x28", 64, 64, 1, 28, 0),
    ("3x3 64->64 @14x14", 64, 64, 3, 14, 1),
    ("3x3 128->128 @14x14", 128, 128, 3, 14, 1),
]


def reference(x, w, pad):
    k, hw = w.shape[-1], x.shape[-1]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((1, w.shape[0], hw, hw), dtype="float64")
    for ky in range(k):
        for kx in range(k):
            patch = xp[:, :, ky : ky + hw, kx : kx + hw].astype("float64")
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, ky, kx].astype("float64"))
    return out


def main():
    rng = np.random.default_rng(2)
    for name, cin, cout, k, hw, pad in CASES:
        x = rng.normal(size=(1, cin, hw, hw)).astype("float32")
        w = (rng.normal(size=(cout, cin, k, k)) * 0.05).astype("float32")
        GlobalCounters.reset()
        out = Tensor(x).conv2d(Tensor(w), padding=pad).numpy()
        macs = cout * cin * k * k * hw * hw
        instructions = GlobalCounters.time_sum_s * 1e9  # the mock reports inscount / 1e9
        error = np.abs(out - reference(x, w, pad)).max()
        print(
            f"{name:22} max_abs_err={error:.1e} MACs={macs / 1e6:.1f}M "
            f"insns={instructions / 1e6:.1f}M insn/MAC={instructions / macs:.2f}"
        )


if __name__ == "__main__":
    main()
