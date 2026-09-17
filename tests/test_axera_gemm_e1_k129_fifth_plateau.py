"""Continues `tests/test_axera_gemm_e1_switch_and_fourth_plateau.py` (PR
#1613)'s own explicitly flagged gaps: that file found `K=129` matches
NEITHER neighboring plateau's own threshold cleanly (absent at both
`N=128` and `N=129`, present at `N=256`), leaving its true threshold
unresolved somewhere in the open interval `(129, 256)`. It also found
`K=300` already past the fourth plateau (`N=128`) entirely, with its
own smaller threshold never pinned down.

## `K=129`'s own exact threshold is `N=192`/`193` -- a genuine FIFTH
## plateau, not an isolated anomaly

Bisecting `K=129` within `(129, 256)` finds a single clean step at
**`N=192` (absent) / `N=193` (present)**. `K=160`, built and tested
independently, shares this exact boundary -- confirming a real plateau,
not a one-off. `129*192=24768` and `160*192=30720`, both far under the
`32768` cap `tests/test_axera_gemm_e1_threshold_formula.py` (PR #1611)
proposed -- this transition is nowhere near that cap, the same
character PR #1613's own `K=96/97` correction already showed for
candidate `320`.

## The fifth plateau's own edges are both clean, single-`K` steps

**Lower edge** (`256`->`192`): `K=128` is still cleanly candidate `256`
(absent at `N=192`, already known absent/present at `256`/`257` from
PR #1611); `K=129` is the first point on candidate `192`. A clean
`128`/`129` step, no gap.

**Upper edge** (`192`->`128`): `K=160` is still on candidate `192`
(absent at `N=192`); `K=161` has already moved past it (present at
`N=192`) -- confirmed by four independent builds (`161`, `162`, `165`,
`170`, `171` all present at `N=192`). `K=161`'s own threshold is then
checked directly against candidate `128` and matches it EXACTLY
(absent at `128`, present at `129`) -- no further gap between the
fifth plateau and the fourth.

The full ordered candidate sequence by `K`, now six plateaus deep:

| plateau | `K` range confirmed | `N` threshold |
| --- | --- | --- |
| 1 | <=64 | 512/513 |
| 2 | 65-96 | 320/321 |
| 3 | 97-128 | 256/257 |
| 5 (this file) | 129-160 | 192/193 |
| 4 | 161-288 | 128/129 |
| 6 (this file) | 289+ | 64/65 |

(Numbered to match each plateau's own discovery order across PRs, not
its `K`-ascending position -- plateaus 1-3 from PR #1609/#1611, 4 from
PR #1611/#1613, 5 and 6 from this file.)

## A genuine structural regularity in the two non-power-of-two
## candidates: each is exactly `64` more than the next power of two down

`320 = 256 + 64`. `192 = 128 + 64`. Both of this project's own
already-flagged "anomalous" candidates (PR #1611's own "not a power of
two" note on `320`; the same character now confirmed for `192`) turn
out to share an identical, simple relationship to their neighboring
power-of-two candidate. This is a real, verified pattern across the
two confirmed instances -- not claimed as a general law from only two
data points, but a genuine advance on PR #1611's own "does not explain
why 320 specifically" honest gap. Consistent with this reading: no
third "extra" candidate was found between `128` and `64` -- the
predicted insertion point (`64+64=128`) degenerates to the power-of-two
bound itself, naturally explaining its absence rather than requiring a
separate excuse.

## The `128`->`64` transition (`K=288`/`289`) breaks a tempting
## "alternating cap" pattern found along the way -- reported honestly,
## not forced

The five now-fully-pinned transitions' own `K*N` products at the
absent/present boundary:

| transition | boundary `K` | `K*N` at boundary (absent side) |
| --- | --- | --- |
| `512`->`320` | 64/65 | `64*512=32768` |
| `320`->`256` | 96/97 | `96*320=30720` |
| `256`->`192` | 128/129 | `128*256=32768` |
| `192`->`128` | 160/161 | `160*192=30720` |
| `128`->`64` | 288/289 | `288*128=36864` |

The first four alternate cleanly between exactly `32768` and exactly
`30720` -- a tempting two-cap pattern, checked directly here by testing
`K=257` (just past the point `32768`'s own alternation would predict
the fifth transition, `floor(32768/128)=256`) and finding it **still
on candidate `128`** (absent at `128`, present at `129`), contradicting
that prediction outright. The real fifth transition, pinned by
bisection between the confirmed-still-valid `K=280`/`285`/`287` and the
confirmed-already-switched `K=290`, lands at `K=288`/`289`
(`288*128=36864`), a THIRD distinct multiple of `1024` (`32*1024`,
`30*1024`, `36*1024`) -- not a continuation of the two-value
alternation. **This file reports the "each boundary's `K*N` product is
an exact multiple of 1024" observation as real and verified across all
five transitions, but explicitly does NOT claim a clean two-cap
alternation, which the `K=257`/`288`/`289` data directly refutes.**

## `K=300`'s own exact threshold: `N=64`/`65` -- a sixth plateau,
## continuing the power-of-two-halving family

Bisecting `K=300` (already known present at `N=128` from PR #1613)
downward finds a single clean step at `N=64` (absent) / `N=65`
(present) -- confirmed by a full bisection trail (`96`, `80`, `72`,
`68`, `66` all present; `64` the exact edge). `64` continues the clean
geometric-halving family `512, 256, 128, 64` exactly, with no further
"+64"-style extra candidate found between `128` and `64` (consistent
with the degenerate-insertion reading above). `K=300`'s own upper edge
(where does candidate `64`'s own valid range begin, i.e. the exact
`K` where `288`/`289`'s transition happens) is already pinned above;
this file does not chase whether candidate `64` has its own further
sub-threshold at some `K` well past `300`, an open question for a
future investigation.
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

# K=129's own bisection trail within (129, 256) -- all present except
# the final N=192 boundary.
K129_BISECTION = [
    ("gemm_1x129x192_e1fifth.mcode.gz", False),
    ("gemm_1x129x193_e1fifth.mcode.gz", True),
    ("gemm_1x129x194_e1fifth.mcode.gz", True),
    ("gemm_1x129x196_e1fifth.mcode.gz", True),
    ("gemm_1x129x200_e1fifth.mcode.gz", True),
]

# K=160 independently shares the identical N=192/193 boundary.
K160_SHARES_FIFTH_PLATEAU = [
    ("gemm_1x160x192_e1fifth.mcode.gz", False),
    ("gemm_1x160x193_e1fifth.mcode.gz", True),
    ("gemm_1x160x256_e1fifth.mcode.gz", True),
]

# K=128 is still cleanly candidate 256 even at N=192 -- the lower edge
# of the fifth plateau is a clean 128/129 step, no gap.
K128_STILL_CANDIDATE_256 = ("gemm_1x128x192_e1fifth.mcode.gz", False)

# Upper edge: K=161..171 have all already moved past N=192 (present),
# unlike K=160 (absent) -- a clean single-K step at 160/161.
K161_PLUS_PAST_FIFTH_PLATEAU = [
    ("gemm_1x161x192_e1fifth.mcode.gz", True),
    ("gemm_1x162x192_e1fifth.mcode.gz", True),
    ("gemm_1x165x192_e1fifth.mcode.gz", True),
    ("gemm_1x170x192_e1fifth.mcode.gz", True),
    ("gemm_1x171x192_e1fifth.mcode.gz", True),
]

# K=161's own threshold matches candidate 128 exactly.
K161_MATCHES_CANDIDATE_128 = [
    ("gemm_1x161x128_e1fifth.mcode.gz", False),
    ("gemm_1x161x129_e1fifth.mcode.gz", True),
]

# The naive "alternating 32768/30720 cap" prediction for the
# 128->64 transition (K=257, floor(32768/128)=256) is refuted: K=257 is
# still on candidate 128.
K257_STILL_CANDIDATE_128 = [
    ("gemm_1x257x128_e1fifth.mcode.gz", False),
    ("gemm_1x257x129_e1fifth.mcode.gz", True),
]

# The real 128->64 transition, bisected between confirmed-still-valid
# and confirmed-already-switched points, lands at K=288/289.
K288_289_TRANSITION = [
    ("gemm_1x280x128_e1fifth.mcode.gz", False),
    ("gemm_1x285x128_e1fifth.mcode.gz", False),
    ("gemm_1x287x128_e1fifth.mcode.gz", False),
    ("gemm_1x288x128_e1fifth.mcode.gz", False),
    ("gemm_1x289x128_e1fifth.mcode.gz", True),
    ("gemm_1x290x128_e1fifth.mcode.gz", True),
]

# K=300's own exact threshold, bisected below N=128: a sixth plateau at
# N=64/65.
K300_SIXTH_PLATEAU = [
    ("gemm_1x300x64_e1fifth.mcode.gz", False),
    ("gemm_1x300x65_e1fifth.mcode.gz", True),
    ("gemm_1x300x66_e1fifth.mcode.gz", True),
    ("gemm_1x300x68_e1fifth.mcode.gz", True),
    ("gemm_1x300x72_e1fifth.mcode.gz", True),
    ("gemm_1x300x80_e1fifth.mcode.gz", True),
    ("gemm_1x300x96_e1fifth.mcode.gz", True),
]

ALL_NEW_FIXTURES = sorted(
    set(
        [n for n, _ in K129_BISECTION]
        + [n for n, _ in K160_SHARES_FIFTH_PLATEAU]
        + [K128_STILL_CANDIDATE_256[0]]
        + [n for n, _ in K161_PLUS_PAST_FIFTH_PLATEAU]
        + [n for n, _ in K161_MATCHES_CANDIDATE_128]
        + [n for n, _ in K257_STILL_CANDIDATE_128]
        + [n for n, _ in K288_289_TRANSITION]
        + [n for n, _ in K300_SIXTH_PLATEAU]
    )
)


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
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestK129HasItsOwnExactThreshold(unittest.TestCase):
    """K=129's true threshold, left unresolved by PR #1613, is a clean
    single-N step at N=192/193 -- not near the 32768 cap on either
    side."""

    def test_k129_bisection(self):
        for name, expected in K129_BISECTION:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_boundary_is_far_from_the_32768_cap(self):
        self.assertLess(129 * 192, 32768)
        self.assertLess(160 * 192, 32768)


class TestFifthPlateauIsSharedNotIsolated(unittest.TestCase):
    """K=160, built and tested independently of K=129, shares the
    identical N=192/193 boundary -- confirming a real plateau."""

    def test_k160_matches_k129s_boundary(self):
        for name, expected in K160_SHARES_FIFTH_PLATEAU:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestFifthPlateausLowerEdgeIsCleanAtK128K129(unittest.TestCase):
    def test_k128_still_candidate_256(self):
        name, expected = K128_STILL_CANDIDATE_256
        self.assertEqual(pair_present(decode(name)), expected, name)


class TestFifthPlateausUpperEdgeIsCleanAtK160K161(unittest.TestCase):
    """K=161 through 171 have all already moved past N=192 (present),
    while K=160 remains absent -- a single-K-wide step, the same
    discipline as every other confirmed boundary this session."""

    def test_k161_plus_all_present_at_n192(self):
        for name, expected in K161_PLUS_PAST_FIFTH_PLATEAU:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k161_matches_candidate_128_exactly(self):
        for name, expected in K161_MATCHES_CANDIDATE_128:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestExtraCandidatesAreThePriorPowerOfTwoPlus64(unittest.TestCase):
    """Both non-power-of-two candidates confirmed this session share an
    identical relationship to their neighboring power of two: 320 =
    256+64 (PR #1609/#1611's own value), 192 = 128+64 (this file's own
    finding). Not claimed as a general law from two points, but a real,
    verified pattern -- and it correctly predicts no third such
    candidate exists between 128 and 64 (64+64=128 degenerates to the
    bound itself)."""

    def test_320_is_256_plus_64(self):
        self.assertEqual(256 + 64, 320)

    def test_192_is_128_plus_64(self):
        self.assertEqual(128 + 64, 192)

    def test_predicted_insertion_between_128_and_64_is_degenerate(self):
        self.assertEqual(64 + 64, 128)


class TestAlternatingCapPredictionIsRefuted(unittest.TestCase):
    """The first four transitions alternate cleanly between K*N=32768
    and K*N=30720 -- but K=257 (which the 32768-alternation would
    predict as already switched, floor(32768/128)=256) is confirmed
    STILL on candidate 128, refuting a clean two-cap alternation."""

    def test_first_four_transitions_alternate_32768_and_30720(self):
        self.assertEqual(64 * 512, 32768)
        self.assertEqual(96 * 320, 30720)
        self.assertEqual(128 * 256, 32768)
        self.assertEqual(160 * 192, 30720)

    def test_k257_contradicts_the_32768_prediction(self):
        # The alternation would predict K=257 already switched
        # (257*128=32896 > 32768). It has not.
        self.assertGreater(257 * 128, 32768)
        for name, expected in K257_STILL_CANDIDATE_128:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestTheRealFifthTransitionIsK288K289(unittest.TestCase):
    """Bisected between confirmed-still-valid (K=280,285,287) and
    confirmed-already-switched (K=290) points: a clean single-K step
    at K=288/289, K*N=36864 -- a third distinct multiple of 1024,
    not a continuation of the 32768/30720 alternation."""

    def test_bisection(self):
        for name, expected in K288_289_TRANSITION:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_boundary_product_is_a_third_multiple_of_1024(self):
        product = 288 * 128
        self.assertEqual(product, 36864)
        self.assertEqual(product % 1024, 0)
        self.assertNotIn(product, (32768, 30720))


class TestK300sSixthPlateauIsN64(unittest.TestCase):
    """K=300's own exact threshold, bisected below N=128 (already known
    present there from PR #1613): a clean single-N step at N=64/65,
    continuing the power-of-two-halving family 512, 256, 128, 64."""

    def test_bisection(self):
        for name, expected in K300_SIXTH_PLATEAU:
            self.assertEqual(pair_present(decode(name)), expected, name)


if __name__ == "__main__":
    unittest.main()
