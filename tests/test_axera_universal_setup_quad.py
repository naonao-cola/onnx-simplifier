"""A third quad-shaped field near site A/the output quad -- but this one
is pure boilerplate, not a decoded value.

Spotted by a systematic scan for the same general shape as the two
decoded quads (`test_axera_site_a_generalizes.py`,
`test_axera_output_scale_quad_generalizes.py`: N constant bytes
repeated 4 times at a fixed stride, with an incrementing single-byte
"id" in the last position -- `20, 30, 40, 50` here) over the whole Conv
mcode stream, since that is exactly how the README's own prior Conv
work found its still-undecoded periodic 3-byte field. This one is real
structure -- `1c 00 00 ff ff a1 00 <id>` x4 at stride 8, the same `a1
00` tag byte pair the two decoded quads also end their payload with --
but **the 5-byte payload itself (`1c 00 00 ff ff`) is identical, byte
for byte, in every one of four builds checked (Mul, Gemm at two
different K/N, Conv, MatMul), regardless of any of those builds' own
scales.** Confirmed constant, not merely "not yet correlated with
anything" -- this is boilerplate (`scripts/axera/README.md`'s own
"What's op-specific vs. boilerplate" framing), not a fifth per-build
field waiting to be decoded. Recorded so a future scan doesn't spend
time trying to correlate this specific pattern with scale, zero-point,
or shape data again.
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")

CONSTANT_QUAD_BODY = bytes.fromhex("1c0000ffffa100")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestUniversalSetupQuadIsBoilerplate(unittest.TestCase):
    CASES = [
        "mul_1x8.mcode.gz",
        "gemm_1x8x8_tb0.mcode.gz",
        "gemm_4x16x8_tb0.mcode.gz",
        "conv_1c1c_8x8_k3.mcode.gz",
        "matmul_4x8x8.mcode.gz",
    ]

    def test_present_and_byte_identical_across_ops_and_scales(self):
        for name in self.CASES:
            data = load(name)
            found = hits(data, CONSTANT_QUAD_BODY)
            self.assertEqual(len(found), 4, f"{name}: quad count")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: quad stride")
            # The single trailing byte after this constant body is the
            # only thing that varies -- an incrementing id, same shape as
            # the two decoded quads' own trailing id byte.
            ids = [data[i + len(CONSTANT_QUAD_BODY)] for i in found]
            self.assertEqual(
                len(set(ids)), 4, f"{name}: expected 4 distinct ids, got {ids}"
            )


if __name__ == "__main__":
    unittest.main()
