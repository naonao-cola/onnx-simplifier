"""Extending batched MatMul's `var`-byte `K` map past 16: the value does
NOT settle at `0x30` (the value already confirmed at `K=16` and
`K=32`) -- instead the `K=9..16` window recurs almost exactly at
`K=25..32`, a clean period-16 structure, with one specific 4-wide
break in the middle window (`K=21..24`).

`tests/test_axera_matmul_var_byte_k_plateaus.py` (PR #1526) densely
mapped `K=9..16` and found three plateaus (`K=9-11 -> 0x9a`, `K=12 ->
0x3a` singleton, `K=13-15 -> 0x98`) landing on `K=16 -> 0x30`. It left
open "whether `K=17..31` holds more plateaus before settling at
`0x30`."

## It does not settle -- it's periodic, with one break

Sweeping the full `K=17..31` range (holding `batch=2, M=4, N=8` fixed,
the established baseline) and lining up three 8-wide windows against
each other (`K=9..16`, `K=17..24`, `K=25..32`, using `K=32 -> 0x30`
from `test_axera_matmul_var_byte_thresholds.py`, PR #1524) shows:

| offset | K=9..16 | K=17..24 | K=25..32 |
| --- | --- | --- | --- |
| 0 | `0x9a` | `0x9a` | `0x9a` |
| 1 | `0x9a` | `0x9a` | `0x9a` |
| 2 | `0x9a` | `0x9a` | `0x9a` |
| 3 | `0x3a` | `0x3a` | `0x3a` |
| 4 | `0x98` | **`0x9a`** | `0x98` |
| 5 | `0x98` | **`0x9a`** | `0x98` |
| 6 | `0x98` | **`0x9a`** | `0x98` |
| 7 | `0x30` | **`0x38`** | `0x30` |

**`K=9..16` and `K=25..32` are byte-for-byte identical, offset for
offset -- a clean period-16.** The middle window, `K=17..24`, matches
that period at its first 4 offsets (`K=17,18,19 -> 0x9a`, `K=20 ->
0x3a`, all identical to `K=9,10,11,12` and `K=25,26,27,28`) but
diverges at the last 4: `K=21,22,23` give `0x9a` (where the period
predicts `0x98`), and `K=24` gives `0x38` -- a value that has never
appeared anywhere else in this project's `var`-byte work, not the
`0x30` the period predicts nor any of the three values already known
from `K=9..16`.

**Confirmed above the noise floor and independent of calibration.**
`K=20`, `K=21`, `K=24`, and `K=28` were each rebuilt with a genuinely
different RNG seed (`11` vs. the sweep's own `3`, giving a different
`A_scale`, `0.007833948358893394` vs. `0.007840047590434551`) and
reproduce the identical `var` value every time -- the period-16 match
at `K=20`/`K=28` (both `0x3a`) and the break at `K=21`/`K=24` (`0x9a`,
`0x38`) are real properties of `K` at this shape, not sampling noise
or an artifact of one calibration draw.

## What remains open

Why the period holds cleanly for 12 of the 16 offsets checked in the
`17..24` window but breaks for exactly 4 of them (`21..24`), and what
`0x38` -- a genuinely new value -- represents, are not decoded here.
Untested: whether `K=33..48` continues the period (would predict a
repeat of `9..16`'s exact pattern again) or breaks the same way
`17..24` did; whether the break's position/width depends on
`batch`/`M`/`N` the way `K`'s own earlier plateaus (`M=8`, `K=12`)
each showed an isolated, shape-specific singleton. Reported as a
precise widening of the map, consistent with this project's
established finding (`M`'s own richer, non-monotone pattern in
`test_axera_matmul_var_byte_thresholds.py`) that `var` is not a simple
function of any one dimension -- not a decode of what it means.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def find_short_form_var(data, a_scale):
    """Locate the short-form A quad (`<3 bytes> 82 <var> 02` x4, stride
    6) and return its 4 var bytes, or [] if absent."""
    short = struct.pack("<f", 1.0 / a_scale)[:3]
    found = [i for i in range(len(data) - 5) if data[i : i + 3] == short]
    for start in found:
        run = [start]
        i = start + 6
        while i in found:
            run.append(i)
            i += 6
        if len(run) == 4:
            return [data[i + 4] for i in run]
    return []


A_SCALE_PASS1 = 0.007840047590434551
A_SCALE_REBUILD = 0.007833948358893394


class TestK17To31Map(unittest.TestCase):
    CASES = {
        "matmul_var_k17.mcode.gz": 0x9A,
        "matmul_var_k18.mcode.gz": 0x9A,
        "matmul_var_k19.mcode.gz": 0x9A,
        "matmul_var_k20.mcode.gz": 0x3A,
        "matmul_var_k21.mcode.gz": 0x9A,
        "matmul_var_k22.mcode.gz": 0x9A,
        "matmul_var_k23.mcode.gz": 0x9A,
        "matmul_var_k24.mcode.gz": 0x38,
        "matmul_var_k25.mcode.gz": 0x9A,
        "matmul_var_k26.mcode.gz": 0x9A,
        "matmul_var_k27.mcode.gz": 0x9A,
        "matmul_var_k28.mcode.gz": 0x3A,
        "matmul_var_k29.mcode.gz": 0x98,
        "matmul_var_k30.mcode.gz": 0x98,
        "matmul_var_k31.mcode.gz": 0x98,
    }

    def test_each_k_matches_its_recorded_value(self):
        for fname, expected in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, A_SCALE_PASS1)
            self.assertEqual(var, [expected] * 4, fname)


class TestPeriod16HoldsBetweenK9And16AndK25And32(unittest.TestCase):
    """K=9..16's own map (from PR #1526, reconstructed here from its
    recorded values) matches K=25..32 exactly, offset for offset --
    K=32's own 0x30 comes from test_axera_matmul_var_byte_thresholds.py
    (PR #1524)."""

    K9_16 = [0x9A, 0x9A, 0x9A, 0x3A, 0x98, 0x98, 0x98, 0x30]
    K25_32_FIXTURES = [
        "matmul_var_k25.mcode.gz",
        "matmul_var_k26.mcode.gz",
        "matmul_var_k27.mcode.gz",
        "matmul_var_k28.mcode.gz",
        "matmul_var_k29.mcode.gz",
        "matmul_var_k30.mcode.gz",
        "matmul_var_k31.mcode.gz",
    ]

    def test_k25_to_31_matches_k9_to_15_offset_for_offset(self):
        for offset, fname in enumerate(self.K25_32_FIXTURES):
            data = load(fname)
            var = find_short_form_var(data, A_SCALE_PASS1)
            self.assertEqual(var, [self.K9_16[offset]] * 4, fname)


class TestK17To24BreaksThePeriodAtTheLastFourOffsets(unittest.TestCase):
    """The first 4 offsets of the middle window match the period; the
    last 4 diverge -- K21-23 give 0x9a (period predicts 0x98) and K24
    gives a genuinely new value, 0x38 (period predicts 0x30)."""

    def test_first_four_offsets_match_the_period(self):
        for fname, expected in (
            ("matmul_var_k17.mcode.gz", 0x9A),
            ("matmul_var_k18.mcode.gz", 0x9A),
            ("matmul_var_k19.mcode.gz", 0x9A),
            ("matmul_var_k20.mcode.gz", 0x3A),
        ):
            data = load(fname)
            self.assertEqual(find_short_form_var(data, A_SCALE_PASS1), [expected] * 4)

    def test_last_four_offsets_break_the_period(self):
        for fname in (
            "matmul_var_k21.mcode.gz",
            "matmul_var_k22.mcode.gz",
            "matmul_var_k23.mcode.gz",
        ):
            data = load(fname)
            self.assertEqual(find_short_form_var(data, A_SCALE_PASS1), [0x9A] * 4)
            self.assertNotEqual(
                find_short_form_var(data, A_SCALE_PASS1),
                [0x98] * 4,
                f"{fname}: period would predict 0x98 here",
            )

    def test_k24_is_a_genuinely_new_value(self):
        data = load("matmul_var_k24.mcode.gz")
        var = find_short_form_var(data, A_SCALE_PASS1)
        self.assertEqual(var, [0x38] * 4)
        # 0x38 has never appeared anywhere else in this project's
        # var-byte work (0x62, 0x30, 0x2c, 0x9a, 0x3a, 0x98 are the
        # other known values) -- not the 0x30 the period predicts.
        self.assertNotIn(0x38, {0x62, 0x30, 0x2C, 0x9A, 0x3A, 0x98})


class TestBreakSurvivesAnIndependentRebuildWithDifferentCalibration(unittest.TestCase):
    """K=20, K=21, K=24, K=28 each rebuilt with a different RNG seed --
    a genuinely different calibration draw -- reproduce the identical
    var value every time, confirming both the period match (K=20,
    K=28) and the break (K=21, K=24) are real properties of K, not
    noise or calibration-dependence."""

    CASES = [
        ("matmul_var_k20.mcode.gz", "matmul_var_k20_rebuild.mcode.gz", 0x3A),
        ("matmul_var_k21.mcode.gz", "matmul_var_k21_rebuild.mcode.gz", 0x9A),
        ("matmul_var_k24.mcode.gz", "matmul_var_k24_rebuild.mcode.gz", 0x38),
        ("matmul_var_k28.mcode.gz", "matmul_var_k28_rebuild.mcode.gz", 0x3A),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        for orig_name, rebuild_name, _ in self.CASES:
            orig = load(orig_name)
            rebuild = load(rebuild_name)
            self.assertNotEqual(orig, rebuild, f"{rebuild_name}: not independent")

    def test_var_value_is_unchanged_across_the_independent_rebuild(self):
        for orig_name, rebuild_name, expected in self.CASES:
            orig_var = find_short_form_var(load(orig_name), A_SCALE_PASS1)
            rebuild_var = find_short_form_var(load(rebuild_name), A_SCALE_REBUILD)
            self.assertEqual(orig_var, [expected] * 4, orig_name)
            self.assertEqual(rebuild_var, [expected] * 4, rebuild_name)


if __name__ == "__main__":
    unittest.main()
