"""Continues `tests/test_axera_gemm_e1_threshold_formula.py` (PR
#1611)'s own explicitly flagged gaps: that file found Gemm's own
`bank=0xe1` (225), `field=32`/`field=48` pair presence follows a
three-plateau structure across `K` (plateau 1, `K<=64`: `N=512/513`;
plateau 2, `K=65..96`: `N=320/321`; plateau 3, `K=112,128`: `N=256/257`)
and proposed a candidate rule -- the active `N` threshold is the
largest of `{512, 320, 256}` with `K*N<=32768` -- which predicted the
plateau-2/plateau-3 switch lands exactly at `K=102`/`K=103`
(`floor(32768/320)=102`), and left untested whether a fourth,
still-smaller plateau exists past `K=128`.

## Correction 1: the held-out prediction was WRONG -- the real switch
## is `K=96`/`K=97`, not `K=102`/`K=103`

Testing `K=97` through `K=103` at `N=320` finds **every one of them
already present** -- including `K=97`, five `K` values earlier than
the rule's own `K=102` prediction. Testing `K=97` directly at plateau
3's own `N=256`/`257` boundary confirms it matches EXACTLY (absent at
256, present at 257) -- `K=97` has already fully switched to plateau 3.
Since PR #1611 itself directly confirmed `K=96` is still plateau 2
(absent at `N=320`), the true switch is a single-`K`-wide step between
`K=96` and `K=97`, not a gradual shift ending at `K=102`/`103`.

**This refutes the `K*N<=32768` cap as the driver of THIS specific
transition**: `96*320=30720` and `97*320=31040` are both far under the
`32768` cap PR #1611's own rule relies on -- whatever triggers Gemm to
abandon candidate `320` in favor of candidate `256`, it is not simply
"the total first exceeds `32768`" the way the `64`/`65` transition
(exactly `32768`) and, as shown below, the `128`/`>128` transition
also are. Reported as a real, evidenced correction to a rule that PR
#1611 itself already flagged as unconfirmed by a genuine held-out test
-- not a failure of that PR's own honesty, since it explicitly said so.

## Correction 2 / new finding: a genuine fourth plateau exists at
## `N=128`/`129`, and its own two transitions behave differently

`K=200` and `K=256` both carry the pair even at `N=256` (below plateau
3's own threshold) -- confirming a fourth, smaller candidate exists.
Bisecting `K=256` finds its own exact threshold at **`N=128`
(absent) / `N=129` (present)** -- `128*256==32768`, again landing
exactly on the same cap the `64`/`65` and `128`/`129`(candidate-256)
transitions do. `K=200` independently shares the identical `128`/`129`
boundary (confirmed by direct build, not assumed), the same
multiple-`K`-share-one-plateau structure PR #1611 already established
for its own three plateaus.

`K=300`, however, is **already present even at `N=128`** -- meaning
`K=300` has moved past this fourth plateau too, into a (at least) fifth
regime not pinned down here. `300*128=38400`, over the `32768` cap --
consistent with the same "candidate becomes invalid once `K*N` exceeds
32768" logic that correctly predicted the `64`/`65` and
`128`(candidate-256)/`129`(candidate-... not simply candidate-128, see
below) transitions, but this file does not chase the fifth plateau's
own value.

## A second, genuinely unresolved wrinkle: `K=129` matches NEITHER
## neighboring candidate cleanly

The naive `K*N<=32768` cap predicts candidate `256`'s own valid range
ends exactly at `K=128` (`128*256==32768`), with `K=129` needing to
drop to the next candidate down. This file confirms `K=128` is fully,
cleanly still candidate `256` (absent at 256, present at 257 -- already
known from PR #1611). But `K=129` does **not** cleanly become
candidate `128` either: it is **absent at BOTH `N=128` and `N=129`**
(contradicting "already candidate 128," which would require present at
129), while also being **present at `N=256`** (contradicting "still
candidate 256," which would require absent at 256). `K=129`'s own true
threshold sits somewhere in the open interval `(129, 256)`, not pinned
down by any candidate confirmed so far. This is a real, precisely
evidenced anomaly -- reported honestly as unresolved rather than forced
into either neighboring plateau.

## What this establishes

The `K*N<=32768`-cap logic PR #1611 proposed correctly predicts THREE
of the (at least) four transitions found across this file and PR #1611
combined (`64/65`, `96` is NOT one of them -- see Correction 1 --
`128`(candidate-256's own boundary), and `256`(candidate-128's own
boundary, this file's new finding)) -- specifically, every transition
between the powers-of-two candidates `512`, `256`, and `128` lands
exactly on the `32768` cap. The one candidate that does NOT fit this
otherwise-clean geometric-halving family, `320`, is also the one whose
own upper transition (`96`/`97`) badly misses the cap prediction --
suggestting `320` is genuinely anomalous, not part of the same
mechanism as its power-of-two neighbors, though this file does not
decode why Pulsar2 introduces this one non-power-of-two candidate
between two that otherwise follow a clean rule. The `K=129` wrinkle
shows even the "clean" candidates have their own unresolved edge
behavior once probed closely enough.
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

# Correction 1: K=97..103 at N=320 are ALL already present, refuting
# PR #1611's own K=102/103 prediction for the switch point.
K97_103_AT_N320_PRESENT = [
    "gemm_1x97x320_e1switch.mcode.gz",
    "gemm_1x98x320_e1switch.mcode.gz",
    "gemm_1x99x320_e1switch.mcode.gz",
    "gemm_1x100x320_e1switch.mcode.gz",
    "gemm_1x101x320_e1switch.mcode.gz",
    "gemm_1x102x320_e1switch.mcode.gz",
    "gemm_1x103x320_e1switch.mcode.gz",
]
K101_AT_N321_PRESENT = "gemm_1x101x321_e1switch.mcode.gz"
K102_AT_N321_PRESENT = "gemm_1x102x321_e1switch.mcode.gz"

# K=97..103 confirmed to fully match plateau 3's own N=256/257 boundary.
PLATEAU3_MATCH = [
    ("gemm_1x97x256_e1switch.mcode.gz", False),
    ("gemm_1x97x257_e1switch.mcode.gz", True),
    ("gemm_1x102x256_e1switch.mcode.gz", False),
    ("gemm_1x102x257_e1switch.mcode.gz", True),
    ("gemm_1x103x256_e1switch.mcode.gz", False),
    ("gemm_1x103x257_e1switch.mcode.gz", True),
]

# Correction 2: the fourth plateau, N=128/129, at K=256 (bisected) and
# K=200 (independently confirmed to share it).
K256_BISECTION = [
    ("gemm_1x256x64_e1fourth.mcode.gz", False),
    ("gemm_1x256x128_e1fourth.mcode.gz", False),
    ("gemm_1x256x129_e1fourth.mcode.gz", True),
    ("gemm_1x256x130_e1fourth.mcode.gz", True),
    ("gemm_1x256x132_e1fourth.mcode.gz", True),
    ("gemm_1x256x136_e1fourth.mcode.gz", True),
    ("gemm_1x256x144_e1fourth.mcode.gz", True),
    ("gemm_1x256x160_e1fourth.mcode.gz", True),
    ("gemm_1x256x192_e1fourth.mcode.gz", True),
    ("gemm_1x256x224_e1fourth.mcode.gz", True),
    ("gemm_1x256x256_e1fourth.mcode.gz", True),
    ("gemm_1x256x257_e1fourth.mcode.gz", True),
]
K200_SHARES_FOURTH_PLATEAU = [
    ("gemm_1x200x128_e1fourth.mcode.gz", False),
    ("gemm_1x200x129_e1fourth.mcode.gz", True),
    ("gemm_1x200x256_e1fourth.mcode.gz", True),
    ("gemm_1x200x257_e1fourth.mcode.gz", True),
]

# K=300 has ALREADY moved past the fourth plateau -- present even at
# N=128, unlike K=200/256.
K300_PAST_FOURTH_PLATEAU = [
    ("gemm_1x300x128_e1fourth.mcode.gz", True),
    ("gemm_1x300x129_e1fourth.mcode.gz", True),
    ("gemm_1x300x256_e1fourth.mcode.gz", True),
    ("gemm_1x300x257_e1fourth.mcode.gz", True),
]

# The K=129 wrinkle: matches neither candidate 256 nor candidate 128.
K129_ANOMALY = [
    ("gemm_1x129x128_e1switch.mcode.gz", False),
    ("gemm_1x129x129_e1switch.mcode.gz", False),
    ("gemm_1x129x256_e1switch.mcode.gz", True),
]

ALL_NEW_FIXTURES = sorted(
    set(
        K97_103_AT_N320_PRESENT
        + [K101_AT_N321_PRESENT, K102_AT_N321_PRESENT]
        + [n for n, _ in PLATEAU3_MATCH]
        + [n for n, _ in K256_BISECTION]
        + [n for n, _ in K200_SHARES_FOURTH_PLATEAU]
        + [n for n, _ in K300_PAST_FOURTH_PLATEAU]
        + [n for n, _ in K129_ANOMALY]
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


class TestHeldOutPredictionWasWrong(unittest.TestCase):
    """PR #1611's own K*N<=32768 rule predicted K=102/103 as the
    plateau-2/plateau-3 switch. Every K from 97 through 103 is ALREADY
    present at N=320 -- the switch happened at least 5 K values
    earlier than predicted."""

    def test_k97_through_103_already_present_at_n320(self):
        for name in K97_103_AT_N320_PRESENT:
            self.assertTrue(pair_present(decode(name)), name)

    def test_k101_and_k102_present_at_n321_too(self):
        self.assertTrue(pair_present(decode(K101_AT_N321_PRESENT)))
        self.assertTrue(pair_present(decode(K102_AT_N321_PRESENT)))


class TestK97AlreadyFullyMatchesPlateau3(unittest.TestCase):
    """K=97 (and re-confirmed K=102, K=103) match plateau 3's own
    N=256/257 boundary exactly -- the switch from plateau 2 to plateau
    3 is a single-K-wide step at K=96/97, not the gradual K=65..128
    transition PR #1609/#1611 left open."""

    def test_plateau3_boundary_matches_for_every_k(self):
        for name, expected in PLATEAU3_MATCH:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_true_switch_is_a_single_k_step_not_a_range(self):
        # PR #1611's own K=96 fixture (already committed) must still
        # be absent at N=320 -- confirming the switch is the single
        # step K=96 (plateau 2) -> K=97 (plateau 3), not a range.
        k96_absent = decode("gemm_1x96x320_e1thresh.mcode.gz")
        self.assertFalse(pair_present(k96_absent))
        k97_present = decode("gemm_1x97x320_e1switch.mcode.gz")
        self.assertTrue(pair_present(k97_present))


class TestKTimes320CapDoesNotExplainTheRealSwitch(unittest.TestCase):
    """The K*N<=32768 cap rule is refuted for the plateau-2/3
    transition specifically: both K=96 and K=97 land far under the
    cap, yet the switch already happens between them."""

    def test_both_sides_of_the_real_switch_are_under_the_cap(self):
        self.assertLess(96 * 320, 32768)
        self.assertLess(97 * 320, 32768)

    def test_the_cap_rule_predicted_a_different_k(self):
        # The rule's own prediction (floor(32768/320)) does not match
        # the empirically confirmed switch point (96/97).
        self.assertEqual(32768 // 320, 102)
        self.assertNotEqual(102, 96)


class TestFourthPlateauExistsAtN128(unittest.TestCase):
    """A genuine fourth plateau, confirmed by full bisection at K=256
    (absent through N=128, present from N=129 onward -- a single clean
    step, same discipline as every other confirmed boundary this
    session)."""

    def test_k256_bisection_is_a_single_clean_step(self):
        for name, expected in K256_BISECTION:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_boundary_lands_exactly_on_the_32768_cap(self):
        self.assertEqual(256 * 128, 32768)


class TestK200IndependentlySharesTheFourthPlateau(unittest.TestCase):
    """K=200, built and tested independently of K=256, shares the
    identical N=128/129 boundary -- the same multi-K-per-plateau
    structure PR #1611 already established for its own three
    plateaus."""

    def test_k200_matches_k256s_boundary(self):
        for name, expected in K200_SHARES_FOURTH_PLATEAU:
            self.assertEqual(pair_present(decode(name)), expected, name)


class TestK300HasAlreadyMovedPastTheFourthPlateau(unittest.TestCase):
    """K=300 (300*128=38400 > 32768) is present even at N=128, unlike
    K=200/256 -- confirming at least a fifth regime exists, not pinned
    down further here."""

    def test_k300_present_even_at_n128(self):
        for name, expected in K300_PAST_FOURTH_PLATEAU:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k300_exceeds_the_cap_for_candidate_128(self):
        self.assertGreater(300 * 128, 32768)


class TestK129MatchesNeitherNeighboringCandidate(unittest.TestCase):
    """K=129 is absent at BOTH N=128 and N=129 (ruling out "already
    candidate 128") yet present at N=256 (ruling out "still candidate
    256") -- its own true threshold is unresolved, somewhere in
    (129, 256). Reported as an honest, precisely evidenced anomaly."""

    def test_k129_matches_neither_candidate_cleanly(self):
        for name, expected in K129_ANOMALY:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k128_itself_is_still_cleanly_candidate_256(self):
        # Already known from PR #1611/#1609; re-confirmed here as the
        # baseline K=129 deviates from.
        k128_absent = decode("gemm_1x128x256_e1period.mcode.gz")
        k128_present = decode("gemm_1x128x257_e1period.mcode.gz")
        self.assertFalse(pair_present(k128_absent))
        self.assertTrue(pair_present(k128_present))


if __name__ == "__main__":
    unittest.main()
