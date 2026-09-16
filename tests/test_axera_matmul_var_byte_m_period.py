"""Batched MatMul's `var` byte's `M` gap, `M=17..31`: unlike `K`'s
analogous `[16,32]` window (period-16, with a break), `M`'s window is
a clean period-4 -- `var` is `0x30` exactly when `M mod 4 == 0`, and
`0x34` everywhere else.

`tests/test_axera_matmul_var_byte_k_extend.py` (PR #1529, merged) found
`K`'s own "settled-looking" `[16,32]` window actually hides a period-16
structure (with a 4-wide break at `K=21..24`), after the original
coarse sweep (`tests/test_axera_matmul_batched_var_byte.py`, PR #1510)
saw only `K=8` and `K=32` and assumed the value had settled at `0x30`.
`tests/test_axera_matmul_var_byte_thresholds.py` (PR #1524, merged)
was in the exact same "only two coarse points checked" situation for
`M`: `M=16` and `M=32` both give `0x30`, and the gap between them was
never densely sampled -- even though `tests/test_axera_matmul_var_byte_m_plateaus.py`
(PR #1528, merged) already densely mapped `M`'s *other* gap (`5..15`)
and found it richer and differently-shaped than `K`'s.

## Result: `M`'s `[17,31]` window is periodic too -- but far more simply than `K`'s

Densely sampling `M=17..31` (holding `batch=2, K=8, N=8` fixed, the
established baseline) gives:

| M | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `var` | `0x34` | `0x34` | `0x34` | **`0x30`** | `0x34` | `0x34` | `0x34` | **`0x30`** | `0x34` | `0x34` | `0x34` | **`0x30`** | `0x34` | `0x34` | `0x34` |

Every single `M` where `var == 0x30` (`20, 24, 28`) is a multiple of 4;
every other `M` in the range gives `0x34`. Combined with the two
already-known endpoints -- `M=16` (a multiple of 4) and `M=32` (also a
multiple of 4) both already confirmed `0x30` in
`test_axera_matmul_var_byte_thresholds.py` -- this is a clean,
falsifiable rule across the *entire* `[16,32]` range with zero
exceptions among the 17 points now checked: **`var = 0x30` iff
`M mod 4 == 0`, else `0x34`.**

This is qualitatively different from `K`'s equivalent window: `K`'s
`[16,32]` hides a period-16 structure with three distinct values and a
4-wide anomalous break; `M`'s hides a period-4 structure with exactly
two values and (across the range tested) no exceptions at all. The two
dimensions are not symmetric in how their `var`-selection behaves --
another confirmation of this thread's running finding (`M`'s `5..15`
gap already showed a differently-shaped pattern from `K`'s `9..16` gap
in PR #1528) that `var` depends on *which* dimension carries a given
value, not just the value itself.

## Confirmed above the noise floor and independent of calibration

`M=20`, `M=21`, and `M=24` were each rebuilt with a genuinely different
RNG seed (`11` vs. the sweep's own `3`, giving different `A_scale`
values at every rebuilt point -- e.g. `0.00783697608858347` vs.
`0.007840401493012905` for `M=20`) and reproduce the identical `var`
value every time: `M=20 -> 0x30`, `M=21 -> 0x34`, `M=24 -> 0x30`. The
`mod 4` rule holds under a different calibration draw, not just as a
property of one specific build.

## What remains open

This is the cleanest `var`-byte pattern found in this whole thread so
far -- unlike `K`'s messy multi-value periodicity, or `M`'s own equally
messy `5..15` gap, `M`'s `[16,32]` behavior reduces to a single,
two-valued, exception-free modular rule. It is NOT decoded *why*
`M mod 4` specifically selects between these two values, and it is
untested whether this simple rule continues past `M=32` or whether
`M`'s own irregular `5..15` pattern (which has no such clean rule) and
this clean `16..32` pattern are connected in some way not yet
understood. Also untested: whether an analogous clean modular rule
exists in `K`'s `[33,48]` extension (a sibling investigation) or
elsewhere in `var`'s dependence on `batch`.
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


class TestMGapIsCleanPeriod4(unittest.TestCase):
    # fixture: (a_scale, expected var)
    CASES = {
        "matmul_var_m17.mcode.gz": (0.007840047590434551, 0x34),
        "matmul_var_m18.mcode.gz": (0.007840401493012905, 0x34),
        "matmul_var_m19.mcode.gz": (0.007840401493012905, 0x34),
        "matmul_var_m20.mcode.gz": (0.007840401493012905, 0x30),
        "matmul_var_m21.mcode.gz": (0.007840401493012905, 0x34),
        "matmul_var_m22.mcode.gz": (0.007840401493012905, 0x34),
        "matmul_var_m23.mcode.gz": (0.007840401493012905, 0x34),
        "matmul_var_m24.mcode.gz": (0.007840401493012905, 0x30),
        "matmul_var_m25.mcode.gz": (0.007842687889933586, 0x34),
        "matmul_var_m26.mcode.gz": (0.007842687889933586, 0x34),
        "matmul_var_m27.mcode.gz": (0.007842687889933586, 0x34),
        "matmul_var_m28.mcode.gz": (0.007842687889933586, 0x30),
        "matmul_var_m29.mcode.gz": (0.007842687889933586, 0x34),
        "matmul_var_m30.mcode.gz": (0.007842687889933586, 0x34),
        "matmul_var_m31.mcode.gz": (0.007842687889933586, 0x34),
    }

    def test_each_m_matches_its_recorded_value(self):
        for fname, (a_scale, expected) in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, fname)

    def test_only_two_distinct_values_appear(self):
        seen = set()
        for fname, (a_scale, _expected) in self.CASES.items():
            data = load(fname)
            seen.update(find_short_form_var(data, a_scale))
        self.assertEqual(seen, {0x34, 0x30})

    def test_030_appears_exactly_at_multiples_of_four(self):
        m_values = {
            17: 0x34,
            18: 0x34,
            19: 0x34,
            20: 0x30,
            21: 0x34,
            22: 0x34,
            23: 0x34,
            24: 0x30,
            25: 0x34,
            26: 0x34,
            27: 0x34,
            28: 0x30,
            29: 0x34,
            30: 0x34,
            31: 0x34,
        }
        for m, expected in m_values.items():
            predicted = 0x30 if m % 4 == 0 else 0x34
            self.assertEqual(
                expected, predicted, f"M={m}: mod-4 rule mismatch in test data itself"
            )


class TestPeriod4RuleSurvivesAnIndependentRebuildWithDifferentCalibration(
    unittest.TestCase
):
    """M=20, M=21, and M=24 each rebuilt with a different RNG seed -- a
    genuinely different calibration draw -- reproduce the identical var
    value every time, confirming the mod-4 rule is a real property of
    M at this shape, not sampling noise or calibration-dependence."""

    CASES = [
        (
            "matmul_var_m20.mcode.gz",
            0.007840401493012905,
            "matmul_var_m20_rebuild.mcode.gz",
            0.00783697608858347,
            0x30,
        ),
        (
            "matmul_var_m21.mcode.gz",
            0.007840401493012905,
            "matmul_var_m21_rebuild.mcode.gz",
            0.00783697608858347,
            0x34,
        ),
        (
            "matmul_var_m24.mcode.gz",
            0.007840401493012905,
            "matmul_var_m24_rebuild.mcode.gz",
            0.00783844105899334,
            0x30,
        ),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        for (
            orig_name,
            _orig_scale,
            rebuild_name,
            _rebuild_scale,
            _expected,
        ) in self.CASES:
            orig = load(orig_name)
            rebuild = load(rebuild_name)
            self.assertNotEqual(orig, rebuild, f"{rebuild_name}: not independent")

    def test_var_value_is_unchanged_across_the_independent_rebuild(self):
        for (
            orig_name,
            orig_scale,
            rebuild_name,
            rebuild_scale,
            expected,
        ) in self.CASES:
            orig_var = find_short_form_var(load(orig_name), orig_scale)
            rebuild_var = find_short_form_var(load(rebuild_name), rebuild_scale)
            self.assertEqual(orig_var, [expected] * 4, orig_name)
            self.assertEqual(rebuild_var, [expected] * 4, rebuild_name)


if __name__ == "__main__":
    unittest.main()
