"""Filling in batched MatMul's `K` gap in (8,16]: not a single binary
threshold as PR #1524 speculated, but (at least) three additional
plateau values before landing on `0x30` at `K=16`.

`tests/test_axera_matmul_var_byte_thresholds.py` (PR #1524, not yet
merged at the time this file was written -- see that PR's diff) mapped
the short-form `A` quad's "var" byte to `K<=8 -> 0x62`, `K=32 -> 0x30`,
and described the gap as "`K`'s crossing is somewhere in `(8,16]`, not
pinned more precisely here" -- implying a single binary crossing point,
by analogy with `batch`'s own clean 3-vs-4 threshold found in the same
file.

**That analogy does not hold. Densely sampling `K=9..16` (holding
`batch=2, M=4, N=8` fixed, PR #1524's own baseline) finds three
plateaus, not one crossing:**

| K | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `var` | `0x9a` | `0x9a` | `0x9a` | **`0x3a`** | `0x98` | `0x98` | `0x98` | `0x30` |

Three previously-unseen values appear (`0x9a`, `0x3a`, `0x98`), on top
of the two already known (`0x62` for `K<=8`, `0x30` for `K>=16`) --
`K=12` is a lone singleton value between two 3-wide plateaus (`9-11`
and `13-15`), the same "isolated one-off at a specific size" shape
this project already found for `M=8`'s own `0x2c` value in
`test_axera_matmul_var_byte_thresholds.py`. `var` is not moving through
a small fixed set gated by one threshold; it is a genuinely richer,
multi-valued function of `K` in this range.

**Confirmed above the noise floor.** Three of these points (`K=9`,
`K=12`, `K=13` -- one from each plateau, including the `K=12`
singleton) were independently rebuilt with a *different* RNG seed
(giving a visibly different `A_scale`, `0.007828372530639172` vs. the
first pass's `0.007840047590434551` -- a genuinely different
calibration draw, not a byte-identical rebuild) and reproduce the
identical `var` value every time: `K=9` -> `0x9a` again, `K=12` ->
`0x3a` again, `K=13` -> `0x98` again. This rules out both ordinary
build noise and any dependence on the specific calibration draw --
`var` tracks `K` itself at this shape, not something incidental to one
build.

**What remains open.** No formula is fitted to `K -> var` across the
full `1..32` range (values now known: `1,2,4,8 -> 0x62`; `9,10,11 ->
0x9a`; `12 -> 0x3a`; `13,14,15 -> 0x98`; `16,32 -> 0x30`) -- this looks
at least as irregular as `M`'s own already-documented pattern, not
simpler. Untested here: whether `K=17..31` holds more plateaus before
settling at `0x30`, and whether the exact plateau boundaries (`8|9`,
`11|12`, `12|13`, `15|16`) shift if `batch`/`M`/`N` are varied
simultaneously rather than one at a time. Reported as a precise,
falsified assumption (PR #1524's implied single-threshold framing) plus
a denser, honestly-partial map -- not a decode of what `var` means.
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
A_SCALE_REBUILD = 0.007828372530639172


class TestKGapHasThreePlateausNotOneThreshold(unittest.TestCase):
    # fixture: (a_scale, expected var)
    CASES = {
        "matmul_var_k9.mcode.gz": (A_SCALE_PASS1, 0x9A),
        "matmul_var_k10.mcode.gz": (A_SCALE_PASS1, 0x9A),
        "matmul_var_k11.mcode.gz": (A_SCALE_PASS1, 0x9A),
        "matmul_var_k12.mcode.gz": (A_SCALE_PASS1, 0x3A),
        "matmul_var_k13.mcode.gz": (A_SCALE_PASS1, 0x98),
        "matmul_var_k14.mcode.gz": (A_SCALE_PASS1, 0x98),
        "matmul_var_k15.mcode.gz": (A_SCALE_PASS1, 0x98),
        "matmul_var_k16.mcode.gz": (A_SCALE_PASS1, 0x30),
    }

    def test_each_k_matches_its_recorded_plateau_value(self):
        for fname, (a_scale, expected) in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, fname)

    def test_three_new_values_appear_between_k8_and_k16(self):
        seen = set()
        for fname, (a_scale, _expected) in self.CASES.items():
            data = load(fname)
            seen.update(find_short_form_var(data, a_scale))
        # 0x62 (K<=8) and 0x30 (K>=16) were already known; these three
        # are new.
        self.assertEqual(seen, {0x9A, 0x3A, 0x98, 0x30})

    def test_k12_is_a_singleton_between_two_wider_plateaus(self):
        k11 = find_short_form_var(load("matmul_var_k11.mcode.gz"), A_SCALE_PASS1)
        k12 = find_short_form_var(load("matmul_var_k12.mcode.gz"), A_SCALE_PASS1)
        k13 = find_short_form_var(load("matmul_var_k13.mcode.gz"), A_SCALE_PASS1)
        self.assertNotEqual(k12, k11)
        self.assertNotEqual(k12, k13)
        self.assertEqual(k11, [0x9A] * 4)
        self.assertEqual(k13, [0x98] * 4)


class TestPlateausSurviveAnIndependentRebuildWithDifferentCalibration(
    unittest.TestCase
):
    """K=9 (start of the 0x9a plateau), K=12 (the singleton), and K=13
    (start of the 0x98 plateau) were each rebuilt with a different RNG
    seed -- a genuinely different calibration draw (different A_scale)
    -- and reproduce the identical var value, ruling out both ordinary
    noise and calibration-dependence."""

    CASES = [
        ("matmul_var_k9.mcode.gz", "matmul_var_k9_rebuild.mcode.gz", 0x9A),
        ("matmul_var_k12.mcode.gz", "matmul_var_k12_rebuild.mcode.gz", 0x3A),
        ("matmul_var_k13.mcode.gz", "matmul_var_k13_rebuild.mcode.gz", 0x98),
    ]

    def test_rebuild_uses_a_different_calibration_draw(self):
        # Sanity check the rebuild really is independent, not a
        # byte-identical replay.
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
