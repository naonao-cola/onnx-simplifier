"""Gemm's `8*M-1`/`8*floor(M/2)-1` field is not a pure `M`-function:
it also depends on `K` -- the true formula is `K*M-1` (pre-break) and
`K*floor(M/2)-1` (post-break), which happen to reduce to the
already-decoded `8*M-1`/`8*floor(M/2)-1` forms only because every
prior test held `K=8` fixed. `N` genuinely does not affect it.

`tests/test_axera_gemm_m_field_is_8m_minus_1.py` (PR #1538) and
`tests/test_axera_gemm_8m1_field_breaks_at_m5.py` (PR #1540) decoded a
Gemm S-unit record (`p=2, tag=130, reg=70`, payload `d0 0c <XX>`) whose
trailing byte tracks `M`: `8*M-1` for `M<=4`, `8*floor(M/2)-1` for
`M>=5`. Every build in both files held `K=8, N=8` fixed -- the field
had never been checked against `K` or `N` at all, so whether "8" in
those formulas was really `K` in disguise (since `K=8` was the only
value ever tested) was an open question neither file raised explicitly
but never ruled out either.

## Result: `N` is confirmed irrelevant; `K` is the real second variable

Holding `M=4` fixed (the clean pre-break case) and sweeping `N` over
`{1,4,32}` (`K=8` fixed) leaves the field at `31` every time -- `N`
genuinely does not move it, matching `B`'s and `C`'s own output-column
dimension having no reason to enter a row/reduction-oriented field.

Holding `M=4` fixed and sweeping `K` over `{1,2,4,16,32}` instead
changes the field every time: `K=2 -> 7`, `K=4 -> 15`, `K=16 -> 63`,
`K=32 -> 127` -- exactly `4*K - 1` at this `M=4` baseline, i.e. the
"8" in PR #1538's own `8*M-1` formula was really `K` (which just
happened to be fixed at `8` in every one of that file's own builds).
**`K=1` is a genuine edge case**: the record is entirely absent from
that build's mcode (not merely a different value) -- consistent with
`K=1` degenerating into some other code path this record's context
doesn't apply to; not chased further here.

## The unified formula: `field = K*M - 1` (pre-break), `K*floor(M/2) - 1` (post-break)

Combining `K`'s own `M=4`-fixed sweep with `M`'s own `K=8`-fixed sweep
(from PR #1538/#1540) already fits `K*M-1` for every pre-break point
tested by either file. This file adds five new cross points, varying
BOTH `M` and `K` together, to confirm the two-variable formula
directly rather than inferring it from two separate single-variable
sweeps:

| M | K | field | `K*M-1` | `K*floor(M/2)-1` |
| --- | --- | --- | --- | --- |
| 4 | 2 | 7 | 7 | -- |
| 4 | 16 | 63 | 63 | -- |
| 5 | 4 | 7 | 19 | **7** |
| 8 | 2 | 7 | 15 | **7** |
| 8 | 16 | 63 | 127 | **63** |
| 16 | 4 | 31 | 63 | **31** |

Every pre-break point (`M<=4`) matches `K*M-1` and NOT
`K*floor(M/2)-1`; every post-break point (`M>=5`) matches
`K*floor(M/2)-1` and NOT `K*M-1`. This is the same `M=4`/`M=5`
threshold PR #1540 already found, now confirmed to hold with `K`
varying too, not just at the single `K=8` slice that formula was
originally decoded from.

## Confirmed above the noise floor

An independent rebuild of `M=8,K=16` reproduces the field exactly
(`63` both times) at the identical offset; the two builds' mcode
streams are otherwise not checked byte-for-byte here (this file only
verifies the field itself, the same scope as PR #1538/#1540's own
rebuild checks).

## What remains open

*Why* the formula is `K*M` (a natural "total input-A element count"
interpretation, `M` rows times `K` columns) rather than some other
combination is not decoded -- only that it fits every point tested
across both `M`-only and `K`-only single-variable sweeps and this
file's five new cross points, with zero exceptions. The `K=1` absence
and the `M=4`/`M=5` regime boundary's own cause remain undecoded, as
PR #1540 already noted for the latter.
"""

import gzip
import os
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


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def find_field(data):
    """Locate the same p=2 short unit (tag=130/reg=70, payload
    d0 0c <XX>) PR #1538/#1540 decoded, and return its trailing
    payload byte, or None if absent."""
    _, segs = mcode.segments(data)
    for pos, length, _ in segs:
        for r in decode_segment(data, pos, length):
            if (
                r.get("kind") == "S"
                and r.get("tag") == 130
                and r.get("reg") == 70
                and r.get("p") == 2
                and r["payload"][:2] == b"\xd0\x0c"
            ):
                return r["payload"][2]
    return None


class TestNDoesNotAffectTheField(unittest.TestCase):
    """M=4, K=8 fixed; N varied over {1,4,32} -- field stays at 31
    (8*4-1 == 4*8-1, the shared value both old and new formulas
    predict at this baseline) every time."""

    CASES = [
        "gemm_4x8x1_n1.mcode.gz",
        "gemm_4x8x4_n4.mcode.gz",
        "gemm_4x8x32_n32.mcode.gz",
    ]

    def test_field_is_31_regardless_of_n(self):
        for fname in self.CASES:
            data = load(fname)
            self.assertEqual(find_field(data), 31, fname)


class TestKChangesTheFieldAtFixedM(unittest.TestCase):
    """M=4 fixed; K varied over {2,4,16,32} -- field tracks 4*K-1
    exactly (the "8" in PR #1538's 8*M-1 was really K, fixed at 8 in
    every one of that file's own builds)."""

    CASES = {
        "gemm_4x2x8_k2.mcode.gz": 2,
        "gemm_4x4x8_k4.mcode.gz": 4,
        "gemm_4x16x8_k16.mcode.gz": 16,
        "gemm_4x32x8_k32.mcode.gz": 32,
    }

    def test_field_is_4k_minus_1(self):
        for fname, k in self.CASES.items():
            data = load(fname)
            self.assertEqual(find_field(data), 4 * k - 1, fname)


class TestK1IsAGenuineEdgeCase(unittest.TestCase):
    """K=1 (M=4) has no matching record at all -- the field is
    absent, not merely a different value."""

    def test_field_is_absent_at_k1(self):
        data = load("gemm_4x1x8_k1.mcode.gz")
        self.assertIsNone(find_field(data))


class TestUnifiedFormulaAcrossMAndKTogether(unittest.TestCase):
    """Five new cross points varying both M and K confirm the unified
    two-variable formula directly: K*M-1 pre-break (M<=4),
    K*floor(M/2)-1 post-break (M>=5)."""

    # fixture: (M, K)
    CASES = {
        "gemm_8x2x8_m8k2.mcode.gz": (8, 2),
        "gemm_8x16x8_m8k16.mcode.gz": (8, 16),
        "gemm_16x4x8_m16k4.mcode.gz": (16, 4),
        "gemm_5x4x8_m5k4.mcode.gz": (5, 4),
    }

    def test_post_break_points_match_k_floor_m_over_2_minus_1(self):
        for fname, (m, k) in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertEqual(value, k * (m // 2) - 1, fname)

    def test_post_break_points_do_not_match_the_pre_break_formula(self):
        for fname, (m, k) in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertNotEqual(
                value, k * m - 1, f"{fname}: unexpectedly matched K*M-1"
            )

    def test_pre_break_cross_points_match_k_m_minus_1(self):
        # M=4 (pre-break) cross points, reusing the K-sweep fixtures
        # above at K=2 and K=16 -- already covered by
        # TestKChangesTheFieldAtFixedM, re-asserted here explicitly
        # against the K*M-1 form for symmetry with the post-break test.
        for fname, k in [
            ("gemm_4x2x8_k2.mcode.gz", 2),
            ("gemm_4x16x8_k16.mcode.gz", 16),
        ]:
            data = load(fname)
            value = find_field(data)
            self.assertEqual(value, k * 4 - 1, fname)


class TestM8K16FieldSurvivesAnIndependentRebuild(unittest.TestCase):
    def test_field_unchanged_across_rebuild(self):
        original = load("gemm_8x16x8_m8k16.mcode.gz")
        rebuild = load("gemm_8x16x8_m8k16_rebuild.mcode.gz")
        self.assertEqual(find_field(original), 63)
        self.assertEqual(find_field(rebuild), 63)


if __name__ == "__main__":
    unittest.main()
