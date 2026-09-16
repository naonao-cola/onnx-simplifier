"""Batched MatMul `var`-byte `K` map: the period is 32, not 16 -- the
"broken" window (`K=17..24`) recurs exactly at `K=33..40`, alternating
with the "clean" window (`K=9..16`, `K=25..32`).

`tests/test_axera_matmul_var_byte_k_extend.py` (PR #1529) found `K=9..16`
matches `K=25..32` byte-for-byte (a clean period-16), but the *middle*
window `K=17..24` breaks that period at its last 4 offsets: `K=21,22,23`
give `0x9a` (period-16 predicts `0x98`) and `K=24` gives a genuinely new
value, `0x38` (period-16 predicts `0x30`). It left open two possibilities
for what `K=33..40` would show: a repeat of the *clean* pattern (making
the break at `17..24` a one-off anomaly), or a repeat of the *broken*
pattern (making the break itself periodic).

## Result: it's the broken pattern, exactly

Building `K=33..40` (holding `batch=2, M=4, N=8` fixed, the established
baseline) gives `var` = `0x9a, 0x9a, 0x9a, 0x3a, 0x9a, 0x9a, 0x9a, 0x38`
-- **identical, offset for offset, to `K=17..24`'s own broken pattern**,
not `K=9..16`'s clean one (which differs at the last 3 offsets: `0x98,
0x98, 0x98, 0x30` vs. the actual `0x9a, 0x9a, 0x9a, 0x38`).

So the true structure is **period 32**, built from two alternating
8-wide sub-windows: a "clean" one (`K=9..16`, `K=25..32`) and a "broken"
one (`K=17..24`, `K=33..40`) that share their first 4 offsets
(`0x9a, 0x9a, 0x9a, 0x3a`) but diverge at the last 4 (`0x98,0x98,0x98,0x30`
vs. `0x9a,0x9a,0x9a,0x38`). What looked, from `test_axera_matmul_var_byte_k_extend.py`'s
data alone, like "period 16 with one exceptional break" is actually a
clean, exactly-recurring period-32 signal -- the "break" is not an
anomaly, it is one of the two states the period cycles through.

**Confirmed above the noise floor and independent of calibration.**
`K=37` and `K=40` were each rebuilt with a genuinely different RNG seed
(`17` vs. the sweep's own `3`, giving a different `A_scale`,
`0.007842199876904488` vs. `0.007840401493012905`) and reproduce the
identical `var` value both times (`K=37 -> 0x9a`, `K=40 -> 0x38`) --
the period-32 match is a real property of `K` at this shape, not
sampling noise or an artifact of one calibration draw.

## What remains open

The exact mechanism producing this period-32 alternation, and what
`0x38`/`0x9a`-vs-`0x98`/`0x30`-vs-`0x38` actually encode, are not
decoded here. Untested: whether `K=41..48` (predicted, by this period,
to repeat the clean pattern again) confirms the period continues, or
whether it eventually breaks down at larger `K`; whether this same
period-32 (clean/broken alternation) structure appears in `M`'s own
richer, differently-shaped gap (`test_axera_matmul_var_byte_m_plateaus.py`)
if swept far enough past its own already-tested range. Reported as a
precise resolution of `test_axera_matmul_var_byte_k_extend.py`'s own
open question (period continues past 32, and specifically repeats the
broken sub-window) -- not a decode of what `var` means.
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


A_SCALE_K33_35 = 0.007840047590434551
A_SCALE_K36_40 = 0.007840401493012905
A_SCALE_REBUILD = 0.007842199876904488


class TestK33To40MatchesTheBrokenPatternExactly(unittest.TestCase):
    CASES = {
        "matmul_var_k33.mcode.gz": (A_SCALE_K33_35, 0x9A),
        "matmul_var_k34.mcode.gz": (A_SCALE_K33_35, 0x9A),
        "matmul_var_k35.mcode.gz": (A_SCALE_K33_35, 0x9A),
        "matmul_var_k36.mcode.gz": (A_SCALE_K36_40, 0x3A),
        "matmul_var_k37.mcode.gz": (A_SCALE_K36_40, 0x9A),
        "matmul_var_k38.mcode.gz": (A_SCALE_K36_40, 0x9A),
        "matmul_var_k39.mcode.gz": (A_SCALE_K36_40, 0x9A),
        "matmul_var_k40.mcode.gz": (A_SCALE_K36_40, 0x38),
    }

    def test_each_k_matches_its_recorded_value(self):
        for fname, (a_scale, expected) in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, fname)

    def test_k33_to_40_equals_k17_to_24_offset_for_offset(self):
        """K17..24's own recorded values (from PR #1529), reconstructed
        here -- confirms K33..40 is the SAME broken pattern, not just
        superficially similar."""
        k17_24 = [0x9A, 0x9A, 0x9A, 0x3A, 0x9A, 0x9A, 0x9A, 0x38]
        k33_40 = [
            find_short_form_var(load(f), a)[0] for f, (a, _) in self.CASES.items()
        ]
        self.assertEqual(k33_40, k17_24)

    def test_k33_to_40_does_not_equal_k9_to_16s_clean_pattern(self):
        """K9..16's own recorded values (from PR #1526) -- the clean
        pattern K33..40 does NOT match, ruling out a simple period-16
        explanation."""
        k9_16 = [0x9A, 0x9A, 0x9A, 0x3A, 0x98, 0x98, 0x98, 0x30]
        k33_40 = [
            find_short_form_var(load(f), a)[0] for f, (a, _) in self.CASES.items()
        ]
        self.assertNotEqual(k33_40, k9_16)
        # They DO agree on the first 4 offsets -- only the last 4 differ.
        self.assertEqual(k33_40[:4], k9_16[:4])
        self.assertNotEqual(k33_40[4:], k9_16[4:])


class TestBreakRecurrenceSurvivesAnIndependentRebuild(unittest.TestCase):
    """K=37 and K=40 each rebuilt with a different RNG seed (a genuinely
    different calibration draw) reproduce the identical var value,
    confirming the period-32 match is real, not noise or an artifact of
    one calibration draw."""

    CASES = [
        ("matmul_var_k37.mcode.gz", "matmul_var_k37_rebuild.mcode.gz", 0x9A),
        ("matmul_var_k40.mcode.gz", "matmul_var_k40_rebuild.mcode.gz", 0x38),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        for orig_name, rebuild_name, _ in self.CASES:
            orig = load(orig_name)
            rebuild = load(rebuild_name)
            self.assertNotEqual(orig, rebuild, f"{rebuild_name}: not independent")

    def test_var_value_is_unchanged_across_the_independent_rebuild(self):
        for orig_name, rebuild_name, expected in self.CASES:
            orig_var = find_short_form_var(load(orig_name), A_SCALE_K36_40)
            rebuild_var = find_short_form_var(load(rebuild_name), A_SCALE_REBUILD)
            self.assertEqual(orig_var, [expected] * 4, orig_name)
            self.assertEqual(rebuild_var, [expected] * 4, rebuild_name)


if __name__ == "__main__":
    unittest.main()
