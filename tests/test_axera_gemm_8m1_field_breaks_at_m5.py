"""Stress-testing PR #1538's `8*M - 1` Gemm field past its original
`M ∈ {1,2,4}` range: the formula holds through `M=4`, then breaks at
`M=5` into a different, also-clean formula -- `8*floor(M/2) - 1` --
which then holds robustly through `M=32`.

`tests/test_axera_gemm_m_field_is_8m_minus_1.py` (PR #1538) decoded a
Gemm S-unit record (`p=2, tag=130, reg=70`, payload `d0 0c <XX>`,
found in the op-program segment at offset 344 relative to the segment
start) whose trailing payload byte equals `8*M - 1` at `M ∈ {1,2,4}`
(all powers of two: 7, 15, 31). It explicitly left open whether this
holds for non-power-of-two `M` or breaks down at larger values, citing
this session's own MatMul `var`-byte work as precedent for "clean"
rules turning out to hide more structure once tested further.

## Result: it does break down -- cleanly, at M=5

Building `M = 3, 5, 6, 7, 8, 16, 32` (holding `K=8, N=8, transB=0`
fixed, PR #1538's own baseline; `M=8/16/32` reuse builds already made
while investigating PR #1536, `M=3/5/6/7`, and an independent rebuild
of `M=8`, are new here) and locating the same record gives:

| M | field | `8*M - 1` | `8*floor(M/2) - 1` |
| --- | --- | --- | --- |
| 1 | 7 | 7 | -1 |
| 2 | 15 | 15 | 7 |
| 3 | 23 | 23 | 7 |
| 4 | 31 | 31 | 15 |
| 5 | **15** | 39 | **15** |
| 6 | **23** | 47 | **23** |
| 7 | **23** | 55 | **23** |
| 8 | **31** | 63 | **31** |
| 16 | **63** | 127 | **63** |
| 32 | **127** | 255 | **127** |

`M ∈ {1,2,3,4}` matches `8*M - 1` exactly (extending PR #1538's
`{1,2,4}` result to `M=3` too, closing that specific gap). `M ∈
{5,6,7,8,16,32}` -- every value tested past 4 -- matches a *different*
formula instead, `8*floor(M/2) - 1`, exactly. Both formulas are clean
and exact within their own range; there is a hard switch between them
at `M=4`/`M=5`, not a gradual drift or a third, messier structure the
way some of this session's other stress-tests found (e.g. MatMul's
`var`-byte `K` map, which looked like period-16 before turning out to
be period-32).

## The switch aligns with a real mcode length-class boundary

`M=1,2,3` all serialize to the identical 2568-byte length; `M=4` alone
is 2600 bytes; `M=5,6,16,32` are 2984 bytes; `M=7,8` are 3016 bytes --
non-monotonic, this project's now-familiar pattern for mcode length
vs. a growing shape parameter. The formula switch happens exactly at
the `M=4`-to-`M=5` boundary, which is also where the length class
first moves off its original 2568/2600 baseline into the 2984+ range --
consistent with (not proof of) the compiler switching to a materially
different code-generation strategy for `M` once it stops fitting
whatever single-tile/single-pass capacity `M<=4` represents here, with
the field's own encoded quantity changing meaning (or referring to a
different derived count, such as processing `M` in pairs) along with
that strategy shift. *Why* the post-switch formula is specifically
`floor(M/2)` rather than some other function of `M` is not decoded
here -- only that the switch is real, clean, and precisely located.

## Confirmed above the noise floor -- and a second, previously
## unremarked noise zone spotted along the way

An independent rebuild of `M=8` reproduces the field exactly (`31`
both times) at the identical offset (449). The rebuild diff itself is
20 bytes, all at offsets 677-698 -- **outside this project's
previously-documented `~295-335` noise zone**, a location not
remarked on before. Not chased further here (out of scope for this
file's own question), but noted so a future search at this Gemm shape
knows to expect noise there too, not just in the `~295-335` zone.
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
    """Locate the p=2 short unit tag=130/reg=70 payload d0 0c <XX>,
    the same record PR #1538 decoded, and return its trailing payload
    byte, or None if absent."""
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


class TestFormulaHoldsThroughM4(unittest.TestCase):
    """Extends PR #1538's {1,2,4} to include M=3: 8*M-1 holds for
    every M in [1,4]."""

    CASES = {
        "gemm_1x8x8_m1.mcode.gz": 1,
        "gemm_2x8x8_m2.mcode.gz": 2,
        "gemm_3x8x8_m3.mcode.gz": 3,
        "gemm_4x8x8_m4.mcode.gz": 4,
    }

    def test_field_is_8m_minus_1(self):
        for fname, m in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertIsNotNone(value, f"{fname}: field not found")
            self.assertEqual(value, 8 * m - 1, f"{fname}: M={m}")


class TestFormulaBreaksAtM5AndBecomes8FloorM2Minus1(unittest.TestCase):
    """Every M in {5,6,7,8,16,32} matches 8*floor(M/2)-1 exactly, and
    none of them match the original 8*M-1 rule anymore."""

    CASES = {
        "gemm_5x8x8_m5.mcode.gz": 5,
        "gemm_6x8x8_m6.mcode.gz": 6,
        "gemm_7x8x8_m7.mcode.gz": 7,
        "gemm_8x8x8_m8.mcode.gz": 8,
        "gemm_16x8x8_m16.mcode.gz": 16,
        "gemm_32x8x8_m32.mcode.gz": 32,
    }

    def test_field_is_8_floor_m_over_2_minus_1(self):
        for fname, m in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertIsNotNone(value, f"{fname}: field not found")
            self.assertEqual(value, 8 * (m // 2) - 1, f"{fname}: M={m}")

    def test_field_no_longer_matches_the_original_8m_minus_1_rule(self):
        for fname, m in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertNotEqual(
                value, 8 * m - 1, f"{fname}: M={m} unexpectedly matched the old rule"
            )


class TestM8FieldSurvivesAnIndependentRebuild(unittest.TestCase):
    def test_field_unchanged_across_rebuild(self):
        original = load("gemm_8x8x8_m8.mcode.gz")
        rebuild = load("gemm_8x8x8_m8_rebuild.mcode.gz")
        self.assertEqual(find_field(original), 31)
        self.assertEqual(find_field(rebuild), 31)

    def test_rebuild_diff_is_a_previously_unremarked_noise_zone(self):
        """The M=8 rebuild's own diff sits at offsets 677-698, entirely
        outside this project's previously-documented ~295-335 noise
        zone and nowhere near the field's own offset (449) -- a real,
        reproducible finding worth recording even though not chased
        further here."""
        original = load("gemm_8x8x8_m8.mcode.gz")
        rebuild = load("gemm_8x8x8_m8_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertGreater(len(diffs), 0)
        self.assertLessEqual(len(diffs), 30)
        for i in diffs:
            self.assertTrue(670 <= i <= 700, f"@{i}: outside the expected zone")
            self.assertFalse(295 <= i <= 335, f"@{i}: unexpectedly in the OLD zone")


if __name__ == "__main__":
    unittest.main()
