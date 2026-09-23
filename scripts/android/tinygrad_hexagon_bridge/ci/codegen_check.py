#!/usr/bin/env python3
"""CI checks for tinygrad's own Hexagon DSP codegen (the onnxsim/tinygrad fork's HVX codegen), no phone needed.

Run as a subprocess so tinygrad's DEV/MOCKDSP/HEXSIM/NOOPT environment never leaks into other tests:

  PYTHONPATH=<tinygrad checkout> CC=clang-19 python codegen_check.py ops       # MOCKDSP=1 (qemu-hexagon)
  PYTHONPATH=<tinygrad checkout> CC=clang-19 python codegen_check.py maxchain  # MOCKDSP=1
  PYTHONPATH=<tinygrad checkout> HEXAGON_TOOLS=<Tools> python codegen_check.py hexsim   # HEXSIM=1 (hexagon-sim)

Prints one JSON object on the last line of stdout; tests/test_hexagon_tinygrad.py asserts on it.

- ops: add, 3x3/s2 maxpool (packed NCHWc) and a Q31 requantize written as plain Tensor code, run under qemu and
  compared byte-exact with NumPy, plus the vector types in the rendered C (the codegen's reason to exist).
- maxchain: rendered source size for chains of N float `max` ops. tinygrad commit fccd84cfa fixed an exponential
  blow-up here (each float MAX re-evaluated its operands, so the source doubled per op); it OOM-killed a dev box.
- hexsim: hexagon-sim --timing cycles for the same add, generated normally vs with NOOPT=1 (scalar). Only the
  ratio is checked, so different toolchain/simulator versions don't make it flaky.
"""

import json
import os
import re
import sys

MODE = sys.argv[1]
os.environ["DEV"] = "DSP"
if MODE == "hexsim":
    os.environ["HEXSIM"] = "1"
else:
    os.environ["MOCKDSP"] = "1"

import numpy as np  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.renderer.cstyle import ClangRenderer  # noqa: E402

captured: list[str] = []
_render = ClangRenderer.render


def _capture(self, uops):
    src = _render(self, uops)
    captured.append(src)
    return src


ClangRenderer.render = _capture
VEC = re.compile(
    r"typedef\s+([\w ]+?)\s+\w+\s+__attribute__\(\(aligned\(\d+\),ext_vector_type\((\d+)\)\)\)"
)
SIZES = {
    "char": 1,
    "unsigned char": 1,
    "short": 2,
    "unsigned short": 2,
    "int": 4,
    "unsigned int": 4,
    "float": 4,
    "long long": 8,
    "unsigned long long": 8,
}


def kernel_src() -> str:
    assert len(captured) == 1, f"expected one kernel, got {len(captured)}"
    return captured[0].split("/* DSP boilerplate */")[0]


def max_vector_bytes(src: str) -> int:
    return max(
        (int(n) * SIZES.get(t.strip(), 0) for t, n in VEC.findall(src)), default=0
    )


def pool(x, maximum):
    oh, ow, out = (x.shape[2] - 2) // 2, (x.shape[3] - 2) // 2, None
    for kh in range(3):
        for kw in range(3):
            v = x[:, :, kh : kh + 2 * oh : 2, kw : kw + 2 * ow : 2, :]
            out = v if out is None else maximum(out, v)
    return out


def run_ops() -> dict:
    res = {}
    rng = np.random.default_rng(0)
    a, b = (
        rng.integers(-(2**20), 2**20, (1, 64, 25, 34)).astype(np.int32)
        for _ in range(2)
    )
    captured.clear()
    out = (Tensor(a) + Tensor(b)).realize().numpy()
    res["add"] = {
        "exact": bool(np.array_equal(out, a + b)),
        "vector_bytes": max_vector_bytes(kernel_src()),
    }

    x = rng.integers(0, 256, (1, 2, 52, 70, 32)).astype(
        np.uint8
    )  # packed NCHWc, pre-padded by 1
    x[:, :, 0], x[:, :, -1], x[:, :, :, 0], x[:, :, :, -1] = 0, 0, 0, 0
    captured.clear()
    out = pool(Tensor(x), lambda p, q: p.maximum(q)).realize().numpy()
    res["maxpool"] = {
        "exact": bool(np.array_equal(out, pool(x, np.maximum))),
        "vector_bytes": max_vector_bytes(kernel_src()),
    }

    q = rng.integers(-(2**24), 2**24, (1 << 14,)).astype(np.int32)
    mult, ts, zp = 1518500250, 8 + 31, 3  # a real Q31 multiplier, shift -8
    captured.clear()
    t = Tensor(q).cast(dtypes.int64)
    out = (
        (((t * mult + (1 << (ts - 1))) >> ts) + zp)
        .clip(0, 255)
        .cast(dtypes.uint8)
        .realize()
        .numpy()
    )
    ref = np.clip(
        ((q.astype(np.int64) * mult + (np.int64(1) << (ts - 1))) >> ts) + zp, 0, 255
    ).astype(np.uint8)
    res["requant"] = {
        "exact": bool(np.array_equal(out, ref)),
        "vector_bytes": max_vector_bytes(kernel_src()),
    }
    return res


def run_maxchain() -> dict:
    x = np.random.default_rng(3).standard_normal((1024,)).astype(np.float32)
    sizes = {}
    for n in (4, 8, 16):
        a, z = Tensor(x), None
        z = a
        for i in range(n):
            z = z.maximum(a * float(i + 1) - 1.0)
        captured.clear()
        out = z.realize().numpy()
        ref = x.copy()
        for i in range(n):
            ref = np.maximum(ref, x * np.float32(i + 1) - np.float32(1.0))
        assert np.array_equal(out, ref), f"max chain n={n} is not exact"
        sizes[n] = len(kernel_src())
    return {"sizes": sizes}


def run_hexsim() -> dict:
    import tinygrad.runtime.ops_dsp as dsp

    times: list[float] = []
    call = dsp.HexagonSimProgram.__call__

    def timed(self, *args, **kwargs):
        t = call(self, *args, **kwargs)
        times.append(t)
        return t

    dsp.HexagonSimProgram.__call__ = timed
    a = Tensor.empty(1, 64, 25, 34, dtype="int32")
    b = Tensor.empty(1, 64, 25, 34, dtype="int32")
    (a + b).realize()
    assert len(times) == 1, times
    return {"seconds_at_1ghz": times[0], "noopt": os.environ.get("NOOPT", "0")}


print(
    json.dumps({"ops": run_ops, "maxchain": run_maxchain, "hexsim": run_hexsim}[MODE]())
)
