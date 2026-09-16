"""Batched MatMul `var`-byte `M` mod-4 rule: it continues to hold past
`M=32`, with zero exceptions across 7 new points plus 2 independent
rebuilds under a different calibration draw.

`tests/test_axera_matmul_var_byte_m_period.py` (PR #1533) found `M`'s
`[16,32]` window follows a clean, exception-free rule: `var = 0x30`
iff `M mod 4 == 0`, else `0x34` -- qualitatively simpler than `K`'s own
`[16,32]` window, which turned out (`tests/test_axera_matmul_var_byte_k_period32.py`,
PR #1532) to hide a messier period-32 structure once tested past its
own initial `[16,32]` boundary. PR #1533 explicitly left open "whether
this simple rule continues past `M=32`."

## Result: the mod-4 rule holds robustly past M=32 -- no K-style complexity appears

Building `M = 33, 34, 35, 36, 40, 44, 48` (holding `batch=2, K=8, N=8`
fixed, the established baseline) gives `var = 0x34, 0x34, 0x34, 0x30,
0x30, 0x30, 0x30` -- every non-multiple-of-4 (`33,34,35`) gives `0x34`,
every multiple of 4 (`36,40,44,48`) gives `0x30`. Zero exceptions.
Unlike `K`, whose `[16,32]` rule broke down into a more complex
period-32 alternation once `K=33..40` was tested, `M`'s mod-4 rule
shows no sign of new complexity in the immediately-adjacent range
tested here (`M` up to 48, i.e. 1.5x past the original `[16,32]`
window `test_axera_matmul_var_byte_m_period.py` covered).

**Confirmed above the noise floor and independent of calibration.**
`M=36` and `M=40` were each rebuilt with a genuinely different RNG
seed and a different calibration range (`a_scale =
0.011763946153223515` vs. the main sweep's `0.007842687889933586` --
a real, different draw, not a byte-identical replay) and reproduce the
identical `var` value both times (`M=36 -> 0x30`, `M=40 -> 0x30`).

## What remains open

Whether the mod-4 rule holds indefinitely, or eventually breaks down
the way `K`'s own rule did (which only showed its true period-32
complexity once tested a full 8 values past its first `[16,32]`
window), is not established here -- this file only extends the tested
range to `M=48`, not to the equivalent `K=64` that would fully mirror
`K`'s own period-32 discovery distance. `M`'s `5..15` gap remains its
own, differently-shaped, non-modular pattern
(`tests/test_axera_matmul_var_byte_m_plateaus.py`); whether it
connects to this clean `mod 4` rule at any deeper level is still
unknown. *Why* `M mod 4` selects between these two values is not
decoded.
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


A_SCALE = 0.007842687889933586
A_SCALE_REBUILD = 0.011763946153223515


class TestMod4RuleContinuesPastM32(unittest.TestCase):
    CASES = {
        "matmul_var_m33.mcode.gz": 0x34,
        "matmul_var_m34.mcode.gz": 0x34,
        "matmul_var_m35.mcode.gz": 0x34,
        "matmul_var_m36.mcode.gz": 0x30,
        "matmul_var_m40.mcode.gz": 0x30,
        "matmul_var_m44.mcode.gz": 0x30,
        "matmul_var_m48.mcode.gz": 0x30,
    }

    def test_each_m_matches_its_recorded_value(self):
        for fname, expected in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, A_SCALE)
            self.assertEqual(var, [expected] * 4, fname)

    def test_mod4_rule_predicts_every_new_point(self):
        m_values = {
            33: 0x34,
            34: 0x34,
            35: 0x34,
            36: 0x30,
            40: 0x30,
            44: 0x30,
            48: 0x30,
        }
        for m, expected in m_values.items():
            predicted = 0x30 if m % 4 == 0 else 0x34
            self.assertEqual(
                expected, predicted, f"M={m}: mod-4 rule mismatch in test data itself"
            )

    def test_only_two_distinct_values_appear(self):
        seen = set()
        for fname in self.CASES:
            seen.update(find_short_form_var(load(fname), A_SCALE))
        self.assertEqual(seen, {0x34, 0x30})


class TestRuleSurvivesAnIndependentRebuildWithDifferentCalibration(unittest.TestCase):
    """M=36 and M=40 each rebuilt with a different RNG seed and a
    different calibration range -- a genuinely different draw, not a
    byte-identical replay -- reproduce var=0x30 both times."""

    CASES = [
        ("matmul_var_m36.mcode.gz", "matmul_var_m36_rebuild.mcode.gz"),
        ("matmul_var_m40.mcode.gz", "matmul_var_m40_rebuild.mcode.gz"),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        for orig_name, rebuild_name in self.CASES:
            orig = load(orig_name)
            rebuild = load(rebuild_name)
            self.assertNotEqual(orig, rebuild, f"{rebuild_name}: not independent")

    def test_var_value_is_030_across_the_independent_rebuild(self):
        for orig_name, rebuild_name in self.CASES:
            orig_var = find_short_form_var(load(orig_name), A_SCALE)
            rebuild_var = find_short_form_var(load(rebuild_name), A_SCALE_REBUILD)
            self.assertEqual(orig_var, [0x30] * 4, orig_name)
            self.assertEqual(rebuild_var, [0x30] * 4, rebuild_name)


if __name__ == "__main__":
    unittest.main()
