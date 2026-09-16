"""A systematic search for a Gemm analog of MatMul's shape-dependent
"var" tag byte, applied for the first time this session -- negative,
with an explanation: Gemm's mcode changes pervasively with `M`/`K`/`N`,
not sparsely, so the search method that found MatMul's `var` byte
doesn't directly transfer.

`tests/test_axera_matmul_batched_var_byte.py` (PR #1510) and its many
follow-ups found batched MatMul's short-form `A` quad has a "var" tag
byte whose value depends on `batch`/`M`/`K` in rich, reproducible ways
(period-32 in `K`, a clean mod-4 rule in part of `M`'s range) -- found
by diffing same-length builds that vary one shape dimension, excluding
already-known fields, and looking for whatever small, isolated byte
remained. Nobody had tried this same systematic search on Gemm's own
mcode.

## Method

Built 16 small `Gemm(A,B,C)` models (transB=0, `AxQuantizedFullyConnected`),
holding two of `M`/`K`/`N` fixed at a small baseline (4/8/8) and
sweeping the third across `{1,2,4,8,16,32}` (or `{1,2,4,16,32}` for `K`
and `N`, `4` already being the baseline). Grouped same-length builds and
diffed each pair, excluding the already-decoded output-scale quad
(`tests/test_axera_gemm_output_quad.py`) and the confirmed `~295-330`
noise zone. This is the exact method that found MatMul's `var` byte.

## Result: every same-length pair shows a pervasive diff, not a
sparse one -- `M`'s cleanest, most controlled case makes it decisive

`m1` (`M=1,K=8,N=8`) and `m2` (`M=2,K=8,N=8`) are the cleanest natural
experiment available: `B`/`C` are drawn from the identical RNG seed at
the identical `(K,N)=(8,8)` shape, so their weight content is
**byte-identical** -- only `M` (and, as a consequence of MinMax
calibration over an `M`-shaped array, the input/output scales) differs.
Despite that, and despite both builds landing on the identical
2,568-byte mcode length, **395 bytes differ** (net of the output-scale
quad and the noise zone), spanning offset 516 through 2520. That span
covers 2 of `mcode.segments()`'s 5 segments -- the 416-byte segment at
344 and the 960-byte segment at 760 -- with 148 and 243 differing bytes
respectively; the other 3 segments (a 64-byte one at offset 280, and
the two smallest tail segments at 1720/1752) are completely untouched.
So the diff is real and large, but concentrated in Gemm's two biggest
segments, not literally the entire stream.

**This is confirmed to be real, `M`-driven content, not ordinary
noise.** An independent rebuild of `m1` alone (identical config,
nothing varied) differs from the original `m1` build at only **6**
bytes, all inside the already-known `~295-330` noise zone -- two orders
of magnitude smaller than, and structurally unlike, the 395-byte
`m1`-vs-`m2` diff. `M=1` vs `M=2` is not a coincidence of rebuild
noise; `M` genuinely restructures large parts of Gemm's mcode stream.

Every other same-length group checked (`K` varying at fixed `M,N`; `N`
varying at fixed `M,K`; larger `M` groups) showed the same qualitative
picture -- diffs in the hundreds of bytes, spanning large spans of the
stream, not a small isolated candidate. Not committed as additional
fixtures here (the `m1`/`m2`/`m1_rebuild` trio above already makes the
point decisively and with the least added repo weight); the pattern
held for `k4` vs `k16`/`k32`, `k4` vs `n1`/`n4`/`n16`, and `m16` vs
`m32` as well.

## Why the MatMul method doesn't transfer, and what's still open

MatMul has no weight tensor at all -- both its operands are runtime
activations, so its mcode stream stays comparatively sparse and
structurally stable as shape changes, which is exactly what let a
same-length-pair byte diff isolate one small `var` byte cleanly. Gemm
has a real weight tensor (`B`) with **per-output-channel weight
scales** (`tests/test_axera_gemm_output_quad.py`'s own discovery) baked
into its mcode as tiling/requantization content -- and evidently `M`
(the output row count, which does not even touch `B`'s own values)
still reshapes how that content -- or some other, comparably large
part of the instruction stream -- is laid out throughout the whole
program, not just at one small field. A raw same-length byte diff
cannot distinguish "the field we're looking for" from "ordinary,
legitimate M/K/N-driven restructuring" when the latter is this
pervasive.

**Not established here**: whether an isolated Gemm selector byte
exists at all, buried inside this pervasive difference and only
findable via a `mcode.decode()`/`segments()`-level structural diff
(comparing record-for-record rather than byte-for-byte, the technique
that resolved Conv's own scaling questions) rather than a raw byte
diff; or whether Gemm genuinely has no such compact selector and
everything shape-dependent beyond the already-decoded fields is
diffuse by nature. This file only establishes that the *direct* search
method fails here, and explains precisely why -- a real, useful
negative result narrowing what's left to try, not a decode.
"""

import gzip
import os
import struct
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


def quad_bytes(data, scale):
    found = hits(data, struct.pack("<f", scale))
    return {j for i in found for j in range(i, i + 4)}


NOISE_ZONE = set(range(290, 335))


class TestM1RebuildIsOrdinaryNoise(unittest.TestCase):
    """An independent rebuild of the identical M=1 config differs at
    only 6 bytes, all inside the already-known noise zone -- the
    ordinary floor this project has confirmed repeatedly."""

    def test_rebuild_diff_is_small_and_in_the_known_noise_zone(self):
        d0 = load("gemm_1x8x8_m1.mcode.gz")
        d1 = load("gemm_1x8x8_m1_rebuild.mcode.gz")
        self.assertEqual(len(d0), len(d1))
        diffs = [i for i in range(len(d0)) if d0[i] != d1[i]]
        self.assertLessEqual(
            len(diffs), 10, f"unexpectedly large rebuild diff: {diffs}"
        )
        for i in diffs:
            self.assertIn(
                i, NOISE_ZONE, f"@{i}: rebuild diff outside the known noise zone"
            )


class TestM1VsM2IsPervasiveNotSparse(unittest.TestCase):
    """M=1 vs M=2 (byte-identical B/C, same mcode length) differs at
    395 bytes spanning most of the stream (offset 516-2520) -- two
    orders of magnitude more than ordinary rebuild noise, concentrated
    in Gemm's two biggest segments rather than one small field."""

    M1_SCALE = 0.018186496570706367
    M2_SCALE = 0.020946303382515907

    def _unexplained_diffs(self, d0, d1):
        quad0 = quad_bytes(d0, self.M1_SCALE)
        quad1 = quad_bytes(d1, self.M2_SCALE)
        return [
            i
            for i in range(len(d0))
            if d0[i] != d1[i]
            and i not in quad0
            and i not in quad1
            and i not in NOISE_ZONE
        ]

    def test_diff_is_large_and_spans_most_of_the_stream(self):
        d0 = load("gemm_1x8x8_m1.mcode.gz")
        d1 = load("gemm_2x8x8_m2.mcode.gz")
        self.assertEqual(len(d0), len(d1))
        diffs = self._unexplained_diffs(d0, d1)
        self.assertGreater(
            len(diffs),
            300,
            "expected a pervasive diff, not a sparse selector-sized one",
        )
        self.assertEqual(min(diffs), 516)
        self.assertEqual(max(diffs), 2520)

    def test_diff_concentrates_in_the_two_biggest_segments_only(self):
        """Segments layout is identical between M=1 and M=2 (same 5
        boundaries); the diff touches only the 416-byte and 960-byte
        segments, leaving the other 3 (including the smallest, at 280)
        completely untouched -- real signal, but not scattered
        everywhere."""
        d0 = load("gemm_1x8x8_m1.mcode.gz")
        d1 = load("gemm_2x8x8_m2.mcode.gz")
        _, segs0 = mcode.segments(d0)
        _, segs1 = mcode.segments(d1)
        self.assertEqual(
            [(pos, length) for pos, length, _ in segs0],
            [(pos, length) for pos, length, _ in segs1],
            "segment layout itself should be identical between M=1 and M=2",
        )
        diffs = set(self._unexplained_diffs(d0, d1))
        touched = {}
        for pos, length, _ in segs0:
            count = sum(1 for i in diffs if pos <= i < pos + length)
            touched[pos] = count
        self.assertEqual(touched[280], 0, "smallest segment should be untouched")
        self.assertEqual(touched[1720], 0, "small tail segment should be untouched")
        self.assertEqual(touched[1752], 0, "final tail segment should be untouched")
        self.assertGreater(touched[344], 0, "416-byte segment should carry real diffs")
        self.assertGreater(touched[760], 0, "960-byte segment should carry real diffs")


if __name__ == "__main__":
    unittest.main()
