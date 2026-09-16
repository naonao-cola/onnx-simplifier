"""Filling in batched MatMul's `M` gaps in `[5,7]` and `[9,15]`: a
richer, *differently-shaped* plateau structure than `K`'s own gap --
not a mirror of it.

`tests/test_axera_matmul_var_byte_thresholds.py` (PR #1524) mapped the
short-form `A` quad's "var" byte across `M ∈ {1,2,4,8,16,32}` (holding
`batch=2, K=8, N=8` fixed) and found `M<=4 -> 0x62`, a unique singleton
`M=8 -> 0x2c`, and `M>=16 -> 0x30`, leaving `M=5,6,7` and `M=9..15`
untested. `tests/test_axera_matmul_var_byte_k_plateaus.py` (PR #1526)
densely sampled the analogous `K` gap and found three previously-unseen
plateaus there (not a single threshold) -- raising the question of
whether `M`'s own gap would show the same qualitative shape.

## Result: richer than K's gap, and NOT the same shape

Densely sampling `M = 5, 6, 7, 9, 10, 11, 12, 13, 14, 15` (holding
`batch=2, K=8, N=8` fixed, the same baseline) gives:

| M | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `var` | `0x66` | `0x66` | `0x34` | `0x2c` | `0x34` | `0x34` | `0x34` | **`0x30`** | `0x34` | `0x34` | `0x34` | `0x30` |

Two new plateaus appear (`0x66` at `M=5,6`; `0x34` at `M=7,9-11,13-15`),
plus the already-known `M=8` singleton (`0x2c`). Unlike `K`'s gap --
where the plateaus filled in cleanly between the two known endpoint
values and the final value (`0x30`) only appeared once, at the very
end (`K=16`) -- `M=12` produces `0x30` *early*, sandwiched between two
`0x34` stretches (`M=9-11` and `M=13-15`), before the `0x34` plateau
resumes and only permanently switches to `0x30` at `M=16`. This is a
genuinely different shape from `K`'s monotone-looking sequence of
distinct plateaus: `M`'s map has the *same* value (`0x30`) appearing
once as an isolated "preview" and then again, permanently, four values
later -- not simply a wider or narrower version of the same pattern.

**Confirmed above the noise floor.** `M=6` (start of the new `0x66`
plateau) and `M=12` (the surprising early-`0x30` point, the single most
informative new data point) were each rebuilt with a *different* RNG
seed (`seed=7` vs. the first pass's `seed=3`, giving genuinely
different `A_scale` values: `0.007828372530639172` and
`0.007834210991859436` respectively, vs. the first pass's
`0.007840047590434551`) and reproduce the identical `var` value both
times -- ruling out ordinary build noise and calibration-dependence,
the same standard `tests/test_axera_matmul_var_byte_k_plateaus.py`
(PR #1526) applied to `K`.

## What remains open

No formula is fitted to `M -> var` across the full range now mapped
(`1,2,4 -> 0x62`; `5,6 -> 0x66`; `7,9,10,11,13,14,15 -> 0x34`;
`8 -> 0x2c`; `12 -> 0x30` (early, isolated); `16,32 -> 0x30`
(permanent)). This is at least as irregular as `K`'s own gap, and
differently shaped rather than symmetric to it -- consistent with
`var` encoding some compiler-internal tiling/allocation-strategy
selection whose boundaries do not follow a simple arithmetic rule in
either dimension. Untested here: `N`'s own gap was never even
checked for irregularity (PR #1510 found `N` never moves `var` at the
6 points it tried, but that was a coarse check by the standard this
file and PR #1526 now apply); and whether `M`'s plateau boundaries
shift if `batch`/`K`/`N` are varied simultaneously rather than one at
a time.
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
A_SCALE_M6_REBUILD = 0.007828372530639172
A_SCALE_M12_REBUILD = 0.007834210991859436


class TestMGapHasTwoNewPlateausPlusTheKnownSingleton(unittest.TestCase):
    # fixture: (a_scale, expected var)
    CASES = {
        "matmul_var_m5.mcode.gz": (A_SCALE_PASS1, 0x66),
        "matmul_var_m6.mcode.gz": (A_SCALE_PASS1, 0x66),
        "matmul_var_m7.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m9.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m10.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m11.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m12.mcode.gz": (A_SCALE_PASS1, 0x30),
        "matmul_var_m13.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m14.mcode.gz": (A_SCALE_PASS1, 0x34),
        "matmul_var_m15.mcode.gz": (A_SCALE_PASS1, 0x34),
    }

    def test_each_m_matches_its_recorded_plateau_value(self):
        for fname, (a_scale, expected) in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, fname)

    def test_two_new_values_appear_in_this_gap(self):
        seen = set()
        for fname, (a_scale, _expected) in self.CASES.items():
            data = load(fname)
            seen.update(find_short_form_var(data, a_scale))
        # 0x62 (M<=4) and 0x30 (M>=16) were already known; 0x2c is the
        # already-known M=8 singleton (not in this file's own CASES,
        # since it's already committed/tested elsewhere). 0x66 and
        # 0x34 are new to this file.
        self.assertEqual(seen, {0x66, 0x34, 0x30})

    def test_m12_is_an_early_isolated_return_to_the_final_value(self):
        """Unlike K's gap (where the endpoint value 0x30 only ever
        appeared once, at the very end), M=12 produces 0x30 -- the
        SAME value M>=16 permanently settles on -- as an isolated
        one-off, sandwiched between two 0x34 stretches that resume
        immediately after it."""
        m11 = find_short_form_var(load("matmul_var_m11.mcode.gz"), A_SCALE_PASS1)
        m12 = find_short_form_var(load("matmul_var_m12.mcode.gz"), A_SCALE_PASS1)
        m13 = find_short_form_var(load("matmul_var_m13.mcode.gz"), A_SCALE_PASS1)
        self.assertEqual(m11, [0x34] * 4)
        self.assertEqual(m12, [0x30] * 4)
        self.assertEqual(m13, [0x34] * 4)
        self.assertNotEqual(m12, m11)
        self.assertNotEqual(m12, m13)


class TestMPlateausSurviveAnIndependentRebuildWithDifferentCalibration(
    unittest.TestCase
):
    """M=6 (start of the new 0x66 plateau) and M=12 (the surprising
    early-0x30 point) were each rebuilt with a different RNG seed -- a
    genuinely different calibration draw -- and reproduce the
    identical var value, ruling out both ordinary noise and
    calibration-dependence."""

    CASES = [
        (
            "matmul_var_m6.mcode.gz",
            "matmul_var_m6_rebuild.mcode.gz",
            A_SCALE_M6_REBUILD,
            0x66,
        ),
        (
            "matmul_var_m12.mcode.gz",
            "matmul_var_m12_rebuild.mcode.gz",
            A_SCALE_M12_REBUILD,
            0x30,
        ),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        for orig_name, rebuild_name, _a_scale, _expected in self.CASES:
            orig = load(orig_name)
            rebuild = load(rebuild_name)
            self.assertNotEqual(orig, rebuild, f"{rebuild_name}: not independent")

    def test_var_value_is_unchanged_across_the_independent_rebuild(self):
        for orig_name, rebuild_name, rebuild_scale, expected in self.CASES:
            orig_var = find_short_form_var(load(orig_name), A_SCALE_PASS1)
            rebuild_var = find_short_form_var(load(rebuild_name), rebuild_scale)
            self.assertEqual(orig_var, [expected] * 4, orig_name)
            self.assertEqual(rebuild_var, [expected] * 4, rebuild_name)


class TestMGapDiffersFromKGapShape(unittest.TestCase):
    """M's gap is not a simple mirror of K's: K's endpoint value (0x30)
    only ever appeared at the very end of its gap (K=16), while M's
    endpoint value (0x30, per PR #1524 -- not re-loaded here since
    that PR's M=16 fixture isn't committed to this branch) appears
    once early at M=12 and then again, permanently, at M=16 -- a
    qualitatively different pattern, not just a different set of byte
    values. This test only checks the part directly observable from
    this file's own fixtures: that M=12's value differs from both its
    immediate neighbors, confirming it's a true isolated singleton
    within the M=9..15 range, not a plateau edge."""

    def test_m12_is_an_isolated_singleton_within_the_m9_15_range(self):
        m11 = find_short_form_var(load("matmul_var_m11.mcode.gz"), A_SCALE_PASS1)
        m12 = find_short_form_var(load("matmul_var_m12.mcode.gz"), A_SCALE_PASS1)
        m13 = find_short_form_var(load("matmul_var_m13.mcode.gz"), A_SCALE_PASS1)
        self.assertEqual(m12, [0x30] * 4)
        self.assertNotEqual(m12, m11)
        self.assertNotEqual(m12, m13)


if __name__ == "__main__":
    unittest.main()
