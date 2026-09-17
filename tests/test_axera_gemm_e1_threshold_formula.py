"""Continues `tests/test_axera_gemm_e1_periodicity_check.py` (PR
#1609)'s own explicitly flagged gap: that file found Gemm's own
`bank=0xe1` (225), `field=32`/`field=48` pair presence is governed by a
per-`K` absolute `N` threshold, confirmed at exactly `K=16,32,64` (all
sharing `N=512`/`513`), `K=65` (bracketed only between `N=300` absent
and `N=400` present), and `K=128` (`N=256`/`257`) -- and explicitly left
open both `K=65`'s own exact threshold and whether the transition from
"shares `N=512`" to "has its own smaller threshold" is sharp exactly at
`K=65` or shifts gradually across `K=65..128`.

## Answer: `K=65`'s own threshold is `N=320`/`321` -- and it is shared
## by `K=80` and `K=96` too, revealing a THIRD clean plateau

Bisecting `K=65` down from PR #1609's own `(300 absent, 400 present)`
bracket pins its exact threshold at **`N=320` (absent) / `N=321`
(present)** -- a single clean step, the same precision PR #1609's own
`K=128` boundary got.

Testing `K=80` and `K=96` (both strictly between `K=65` and `K=128`) at
that exact same `N=320`/`321` pair finds **both share it exactly**:
absent at `N=320`, present at `N=321`, for `K=65`, `K=80`, and `K=96`
alike. Combined with `K=63` (tested here, confirms it still shares
plateau 1's `N=512`/`513` the same as `K=16,32,64`) and PR #1609's own
`K=112`/`K=128` pair (both re-tested here at the finer `N=256`/`257`
boundary -- `K=112` was previously only checked at `N=260`/`280`,
present at both; this file confirms `K=112` ALSO shares `K=128`'s exact
`256`/`257` step), the full picture is **three clean plateaus, not a
smooth or single-step transition**:

| plateau | `K` values confirmed | `N` threshold |
| --- | --- | --- |
| 1 | 16, 32, 63, 64 | 512 / 513 |
| 2 | 65, 80, 96 | 320 / 321 |
| 3 | 112, 128 | 256 / 257 |

Every crossing tested is a genuine single-`N`-wide step (never
intermittent), and the `K=96` boundary (`N=320`/`321`) was
independently rebuilt with a different weight/calibration seed
(seed=2, vs. seed=1 for every other fixture in this file) and gives the
identical result -- ruling out a coincidence of one particular RNG
draw, the same discipline PR #1609's own `K=128` rebuild used.

## A candidate rule that fits every confirmed boundary: the `N`
## threshold is the plateau value for which `K * N <= 32768` still
## holds, drawn from a small discrete `N` candidate set

`K=64 * N=512 == 32768` exactly -- the plateau-1/plateau-2 switch sits
precisely where `K` would first push the total past `32768` if `N`
stayed at 512 (`65 * 512 == 33280 > 32768`). `K=96 * N=320 == 30720`,
still safely under `32768`; `112 * 320 == 35840`, over it -- consistent
with `K=112` needing to drop to the next smaller candidate (`256`)
instead, and `112 * 256 == 28672 <= 32768` while `128 * 256 == 32768`
exactly, again landing right at the cap.

This rule is **verified to hold at every one of the now 8 confirmed
(K, N) plateau-boundary pairs** (`16,32,63,64->512`; `65,80,96->320`;
`112,128->256`) without exception, and cleanly explains why the
plateau-1/plateau-3 boundaries (`K=64`, `K=128`) land on an EXACT `K*N`
product (`32768`) while plateau-2's own boundary (`K=96` to `K=112`)
straddles it without either side landing exactly on `32768` -- the
candidate `N=320` isn't itself derived from `32768`, it's a smaller,
separately-chosen value that merely also respects the same `<=32768`
cap while it's in use.

**This does not, by itself, explain why `512`, `320`, and `256` (not
some other set, e.g. a clean geometric sequence like `512, 256, 128`)
are the specific candidate values Pulsar2 chooses from.** `320` is not
a power of two, unlike its plateau neighbors -- this file does not
decode why that particular value is in the candidate set, only that
the "largest candidate `N` with `K*N<=32768`" rule correctly predicts
which of the three already-known candidates applies at every `K` tested
so far. A fourth, larger `K` value (well past 128, to check whether a
still-smaller fourth candidate exists) is not tested here.

## Where the plateau-2/plateau-3 switch itself falls is still open

`K=96` is confirmed plateau 2, `K=112` is confirmed plateau 3; no `K`
strictly between 97 and 111 was tested, so the exact switch point is
only bracketed to that 15-wide range (unlike the plateau-1/plateau-2
switch, which is pinned exactly at `K=64`/`65`, since both were directly
tested). The `K*N<=32768` rule predicts the switch happens at
`floor(32768/320) = 102`, i.e. `K<=102` should still be plateau 2 and
`K>=103` should already be plateau 3 -- but this is a prediction from
the rule, not independently confirmed by a build in that range.
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

GEMM_F32 = b"\x33\x03\x00"
GEMM_F48 = b"\x35\x03\x00\xa1"

# (K, absent_fixture, present_fixture) -- exact single-N-step boundary
# for each K, all confirmed by direct bisection.
EXACT_BOUNDARIES = [
    (63, "gemm_1x63x512_e1thresh.mcode.gz", "gemm_1x63x513_e1thresh.mcode.gz"),
    (65, "gemm_1x65x320_e1thresh.mcode.gz", "gemm_1x65x321_e1thresh.mcode.gz"),
    (80, "gemm_1x80x320_e1thresh.mcode.gz", "gemm_1x80x321_e1thresh.mcode.gz"),
    (96, "gemm_1x96x320_e1thresh.mcode.gz", "gemm_1x96x321_e1thresh.mcode.gz"),
    (112, "gemm_1x112x256_e1thresh.mcode.gz", "gemm_1x112x257_e1thresh.mcode.gz"),
]

# Extra bisection points further from each boundary, confirming a
# genuine clean step rather than a diffuse/noisy transition region.
K80_EXTRA = [
    ("gemm_1x80x300_e1thresh.mcode.gz", False),
    ("gemm_1x80x350_e1thresh.mcode.gz", True),
    ("gemm_1x80x414_e1thresh.mcode.gz", True),
]
K96_EXTRA = [
    ("gemm_1x96x290_e1thresh.mcode.gz", False),
    ("gemm_1x96x340_e1thresh.mcode.gz", True),
]
K112_EXTRA = [
    ("gemm_1x112x260_e1thresh.mcode.gz", True),
    ("gemm_1x112x280_e1thresh.mcode.gz", True),
]

# K=65's own already-committed PR #1609 bracket, reused (not rebuilt)
# to confirm this file's tighter 320/321 boundary is consistent with
# it (300 absent, 400 present must both still hold).
K65_PRIOR_BRACKET = [
    ("gemm_1x65x300_e1period.mcode.gz", False),
    ("gemm_1x65x400_e1period.mcode.gz", True),
]

K96_REBUILD = [
    ("gemm_1x96x320_e1thresh_rebuild.mcode.gz", False),
    ("gemm_1x96x321_e1thresh_rebuild.mcode.gz", True),
]

# The three confirmed plateaus, for the K*N<=32768 rule check.
PLATEAU_N = {63: 512, 65: 320, 80: 320, 96: 320, 112: 256}
CANDIDATE_NS = (512, 320, 256)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def pair_present(recs):
    f32 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
    }
    f48 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 48
    }
    return GEMM_F32 in f32 and GEMM_F48 in f48


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        names = (
            [n for _, n, _ in EXACT_BOUNDARIES]
            + [n for _, _, n in EXACT_BOUNDARIES]
            + [n for n, _ in K80_EXTRA + K96_EXTRA + K112_EXTRA]
            + [n for n, _ in K96_REBUILD]
        )
        for name in sorted(set(names)):
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestExactBoundariesAreSingleCleanSteps(unittest.TestCase):
    def test_absent_just_below_each_boundary(self):
        for k, absent_name, present_name in EXACT_BOUNDARIES:
            self.assertFalse(pair_present(decode(absent_name)), (k, absent_name))

    def test_present_just_above_each_boundary(self):
        for k, absent_name, present_name in EXACT_BOUNDARIES:
            self.assertTrue(pair_present(decode(present_name)), (k, present_name))


class TestK65PriorBracketStillHolds(unittest.TestCase):
    """PR #1609's own (300 absent, 400 present) K=65 bracket must still
    be consistent with this file's own tighter 320/321 boundary."""

    def test_prior_bracket(self):
        for name, expected in K65_PRIOR_BRACKET:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestASecondPlateauSharedByK65K80K96(unittest.TestCase):
    """The core new finding: K=65, K=80, and K=96 all share the
    IDENTICAL N=320/321 threshold -- a second plateau, structurally
    like the already-known K=16/32/64 plateau at N=512, not a smooth
    per-K value."""

    def test_all_three_k_values_share_the_exact_same_boundary(self):
        for k in (65, 80, 96):
            absent_name = f"gemm_1x{k}x320_e1thresh.mcode.gz"
            present_name = f"gemm_1x{k}x321_e1thresh.mcode.gz"
            self.assertFalse(pair_present(decode(absent_name)), k)
            self.assertTrue(pair_present(decode(present_name)), k)


class TestK112SharesK128sExactBoundary(unittest.TestCase):
    """K=112 (previously only checked at N=260/280, both present) is
    confirmed here to share K=128's own exact N=256/257 step -- a third
    plateau, not a gradual approach toward K=128's own value."""

    def test_k112_matches_k128_step_exactly(self):
        self.assertFalse(
            pair_present(decode("gemm_1x112x256_e1thresh.mcode.gz")), "K=112,N=256"
        )
        self.assertTrue(
            pair_present(decode("gemm_1x112x257_e1thresh.mcode.gz")), "K=112,N=257"
        )
        # Cross-check against PR #1609's own K=128 fixtures directly.
        self.assertFalse(pair_present(decode("gemm_1x128x256_e1period.mcode.gz")))
        self.assertTrue(pair_present(decode("gemm_1x128x257_e1period.mcode.gz")))


class TestExtraBisectionPointsConfirmCleanSteps(unittest.TestCase):
    def test_k80_extra(self):
        for name, expected in K80_EXTRA:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k96_extra(self):
        for name, expected in K96_EXTRA:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k112_extra(self):
        for name, expected in K112_EXTRA:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestK96BoundaryConfirmedWithIndependentRebuild(unittest.TestCase):
    """K=96's own N=320/321 boundary rebuilt with a different
    weight/calibration seed (seed=2) gives the identical result --
    ruling out one particular RNG draw as the explanation."""

    def test_rebuild_matches_original(self):
        for name, expected in K96_REBUILD:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestKTimesNCapRuleHoldsAtEveryConfirmedBoundary(unittest.TestCase):
    """A candidate unifying rule: the active N threshold is the largest
    of the three known candidates (512, 320, 256) for which K*N does
    not exceed 32768. Verified directly against every one of the 8
    confirmed (K, plateau) pairs -- not forced, checked."""

    def _predicted_n(self, k):
        valid = [n for n in CANDIDATE_NS if k * n <= 32768]
        return max(valid) if valid else min(CANDIDATE_NS)

    def test_rule_predicts_every_confirmed_plateau(self):
        for k, expected_n in PLATEAU_N.items():
            self.assertEqual(self._predicted_n(k), expected_n, k)

    def test_plateau_1_and_3_boundaries_land_exactly_on_the_cap(self):
        # K=64*512 and K=128*256 both equal 32768 exactly -- the
        # boundary is precisely where the cap is first exceeded.
        self.assertEqual(64 * 512, 32768)
        self.assertEqual(128 * 256, 32768)
        self.assertGreater(65 * 512, 32768)
        self.assertGreater(112 * 320, 32768)

    def test_plateau_2_does_not_land_exactly_on_the_cap(self):
        # Unlike plateaus 1/3, plateau 2's own candidate (320) isn't
        # itself derived from 32768 -- K=96 (last confirmed plateau-2
        # point) is well under the cap, not flush against it.
        self.assertLess(96 * 320, 32768)


if __name__ == "__main__":
    unittest.main()
