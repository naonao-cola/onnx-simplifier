"""Pinning the exact `kx1` regime-transition point PR #1527 left open:
`k=8` still belongs to the "old" regime -- it's the last point where
segments 0/1 keep growing and segments 3/4 stay fixed -- and the sharp
jump to the "new" regime (segments 0/1 shrink and converge, segments
3/4 grow) happens in exactly one step, between `k=8` and `k=9`.

`tests/test_axera_conv_orientation_k9_regime_shift.py` (PR #1527) found
Conv's `kx1` kernel-orientation mcode undergoes a structural regime
shift somewhere in `(7,9]`: at `k=3,5,7` two op-program segments (0/1)
grow (segment 0 exactly linearly, 96 bytes/tap; segment 1 the same way
from `k=3` to `k=5` but falling 32 bytes short of that rule at `k=7`)
while two other segments (3/4) stay byte-length-fixed. At `k=9`,
segments 0/1 both *shrink* and converge to an identical 864-byte
length, while segments 3/4 grow substantially for the first time (one
exactly doubling, one growing 9x) with real decoded instruction
content. PR #1527 explicitly flagged "a `k=8` build (untested here)
would pin the exact transition point" as the natural next step.

## Result: k=8 belongs to the old regime, and reveals it was still
growing linearly -- from k=7's *actual* value, not the idealized one

Building `8x1` at the same shape (`cin=cout=4`, `hw=8`, same
weight-init/calibration convention) gives:

| segment | k=7 | k=8 | k=9 |
| --- | --- | --- | --- |
| 0 (op-program) | 992 | **1088** | 864 |
| 1 (op-program) | 960 | **1056** | 864 |
| 2 | 1184 | **1216** | 1184 |
| 3 | 32 | **32** | 288 |
| 4 | 352 | **352** | 704 |

**Segments 0 and 1 both keep growing at `k=8`, by exactly +96 bytes
each (the same 96-bytes/tap rate PR #1523 established for segment 0)
-- but computed from `k=7`'s own *actual* values, not the idealized
linear-from-`k=5` prediction.** Segment 0: `992 + 96 = 1088`, exact.
Segment 1: `960 + 96 = 1056`, exact -- meaning segment 1's own `k=7`
shortfall (960, not the idealized 992) was never "corrected"; growth
simply continued at the same per-tap rate from wherever segment 1
actually was. **Segments 3 and 4 are still exactly fixed at `k=8`**
(32 and 352 bytes, matching every one of `k=3,5,7`), the same
"boilerplate through k<=7" status PR #1527 established -- they have
not yet started growing.

**This pins the transition precisely: it happens in exactly one step,
between `k=8` and `k=9`, not gradually.** `k=8` is unambiguously still
the old regime (by every measure PR #1523/#1527 used: segments 0/1
still growing per-tap, segments 3/4 still fixed) and `k=9` is
unambiguously the new one -- there is no intermediate state observed at
`k=8` itself, i.e. no partial shrinkage of 0/1 or partial growth of
3/4. The regime shift, whatever causes it, is a single sharp jump
rather than a smooth transition PR #1527's "somewhere in `(7,9]`"
framing might have suggested.

## A second finding: segment 2 bumps once, and k=8 alone shows a real
## decode-coverage gap

Segment 2 -- constant at 1184 bytes across every `k` tested in PR
#1523/#1527, including `k=9` -- is briefly **1216 bytes at `k=8` only**,
a one-off +32-byte bump that reverts back to 1184 at `k=9`. Not chased
further here (this file's scope is pinning the transition point, not
decoding segment 2), but recorded so it isn't mistaken for noise by a
future search: it survives the same independent rebuild check below
that confirms the rest of `k=8`'s data.

**`mcode.check()` also reports a real, reproducible coverage gap at
`k=8` that no other `k` in this thread shows** (`k=3,5,7,9` all
`check()` clean, `[]`): "only 93.8% of non-zero bytes explained" at
four small byte ranges, `(364,367)`, `(369,370)`, `(391,392)`,
`(685,691)`. This is not noise -- the independent rebuild reports the
identical four ranges, byte-for-byte. Whether this connects causally
to the segment-2 bump or the upcoming regime shift is not established
here; flagged as another concrete, reproducible `k=8`-specific
anomaly for whoever chases the transition's mechanism next.

**Confirmed above the noise floor.** An independent rebuild of `8x1`
reproduces the identical 5-segment length breakdown exactly
(1088/1056/1216/32/352), with only 6 ordinary noise-floor bytes
differing elsewhere in the stream -- well within this project's
established `~6-20` byte scale.

## What remains open

*Why* the transition happens between exactly `k=8` and `k=9` -- what
hardware or compiler-internal boundary that specific tap count crosses
-- is not decoded here, nor is segment 2's one-off `k=8` bump, nor (per
PR #1527) what the new segment-3/4 content at `k=9` computes. This
file only narrows "somewhere in `(7,9]`" to "exactly between `k=8` and
`k=9`, in one sharp step."
"""

import gzip
import os
import sys
import unittest
from collections import Counter

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def all_segment_lengths(data):
    _, segs = mcode.segments(data)
    return [length for _, length, _ in segs]


class TestK8SegmentLengths(unittest.TestCase):
    def test_k8_segment_lengths(self):
        data = load("conv_4c4c_8x8_k8x1.mcode.gz")
        # Unlike every other k tested in this thread (mcode.check()==[]
        # at k=3,5,7,9), k=8 alone reports a small coverage gap -- see
        # module docstring's "A third, previously-unremarked finding".
        # Not asserted as well-formed here; the gap is real and checked
        # separately below, not hidden by a stricter assertion.
        issues = mcode.check(data)
        self.assertEqual(
            len(issues), 1, f"expected exactly one check() issue: {issues}"
        )
        self.assertIn("coverage", issues[0])
        self.assertEqual(all_segment_lengths(data), [1088, 1056, 1216, 32, 352])


class TestSegments0And1StillGrowLinearlyAtK8(unittest.TestCase):
    """Segment 0 and segment 1 both grow by exactly +96 bytes from
    k=7's own actual values -- k=8 continues the per-tap rate rather
    than correcting segment 1's earlier shortfall."""

    def test_segment0_continues_the_96_bytes_per_tap_rate(self):
        k7_len0 = all_segment_lengths(load("conv_4c4c_8x8_k7x1.mcode.gz"))[0]
        k8_len0 = all_segment_lengths(load("conv_4c4c_8x8_k8x1.mcode.gz"))[0]
        self.assertEqual(k7_len0, 992)
        self.assertEqual(k8_len0, 1088)
        self.assertEqual(k8_len0 - k7_len0, 96)

    def test_segment1_continues_from_its_own_actual_k7_value(self):
        """Segment 1 grows +96 from k=7's real value (960), not from
        the idealized linear prediction (992) -- its k=7 shortfall was
        never corrected, just carried forward."""
        k7_len1 = all_segment_lengths(load("conv_4c4c_8x8_k7x1.mcode.gz"))[1]
        k8_len1 = all_segment_lengths(load("conv_4c4c_8x8_k8x1.mcode.gz"))[1]
        self.assertEqual(k7_len1, 960)
        self.assertEqual(k8_len1, 1056)
        self.assertEqual(k8_len1 - k7_len1, 96)


class TestSegments3And4StillFixedAtK8(unittest.TestCase):
    """Unlike k=9, segments 3/4 have not started growing yet at k=8 --
    still exactly the boilerplate lengths seen at k=3,5,7."""

    def test_segments_3_and_4_match_the_k3_5_7_boilerplate_lengths(self):
        for fname in (
            "conv_4c4c_8x8_k3x1.mcode.gz",
            "conv_4c4c_8x8_k5x1.mcode.gz",
            "conv_4c4c_8x8_k7x1.mcode.gz",
            "conv_4c4c_8x8_k8x1.mcode.gz",
        ):
            lens = all_segment_lengths(load(fname))[3:5]
            self.assertEqual(lens, [32, 352], fname)


class TestTheJumpToK9IsSharpNotGradual(unittest.TestCase):
    """k=8 shows no partial shrinkage of segments 0/1 and no partial
    growth of segments 3/4 -- the regime shift is a single sharp step
    between k=8 and k=9, not a smooth transition through k=8."""

    def test_k9_segments_0_1_are_smaller_than_both_k7_and_k8(self):
        len0_1_k7 = all_segment_lengths(load("conv_4c4c_8x8_k7x1.mcode.gz"))[:2]
        len0_1_k8 = all_segment_lengths(load("conv_4c4c_8x8_k8x1.mcode.gz"))[:2]
        len0_1_k9 = all_segment_lengths(load("conv_4c4c_8x8_k9x1.mcode.gz"))[:2]
        self.assertEqual(len0_1_k7, [992, 960])
        self.assertEqual(len0_1_k8, [1088, 1056])
        self.assertEqual(len0_1_k9, [864, 864])
        for a, b in zip(len0_1_k8, len0_1_k9):
            self.assertGreater(a, b, "k=9 should be strictly smaller than k=8")

    def test_k9_segments_3_4_are_larger_than_both_k7_and_k8(self):
        len3_4_k8 = all_segment_lengths(load("conv_4c4c_8x8_k8x1.mcode.gz"))[3:5]
        len3_4_k9 = all_segment_lengths(load("conv_4c4c_8x8_k9x1.mcode.gz"))[3:5]
        self.assertEqual(len3_4_k8, [32, 352])
        self.assertEqual(len3_4_k9, [288, 704])
        for a, b in zip(len3_4_k8, len3_4_k9):
            self.assertLess(a, b, "k=9 should be strictly larger than k=8")


class TestSegment2BumpsOnceAtK8(unittest.TestCase):
    """Segment 2, constant at 1184 bytes everywhere else tested
    (k=3,5,7,9), is briefly 1216 bytes at k=8 only -- a real, rebuild-
    confirmed one-off, not chased further here."""

    def test_segment2_is_1184_except_at_k8(self):
        for fname in (
            "conv_4c4c_8x8_k3x1.mcode.gz",
            "conv_4c4c_8x8_k5x1.mcode.gz",
            "conv_4c4c_8x8_k7x1.mcode.gz",
            "conv_4c4c_8x8_k9x1.mcode.gz",
        ):
            self.assertEqual(all_segment_lengths(load(fname))[2], 1184, fname)

    def test_segment2_is_1216_at_k8_and_survives_rebuild(self):
        original = load("conv_4c4c_8x8_k8x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k8x1_rebuild.mcode.gz")
        self.assertEqual(all_segment_lengths(original)[2], 1216)
        self.assertEqual(all_segment_lengths(rebuild)[2], 1216)


class TestK8HasAReproducibleCoverageGap(unittest.TestCase):
    """k=8 alone (not k=3,5,7,9) reports a check() coverage gap, and
    the exact same four byte ranges reproduce identically in the
    independent rebuild -- real, not noise."""

    EXPECTED_RANGES = "(364, 367), (369, 370), (391, 392), (685, 691)"

    def test_other_k_values_check_clean(self):
        for fname in (
            "conv_4c4c_8x8_k3x1.mcode.gz",
            "conv_4c4c_8x8_k5x1.mcode.gz",
            "conv_4c4c_8x8_k7x1.mcode.gz",
            "conv_4c4c_8x8_k9x1.mcode.gz",
        ):
            self.assertEqual(mcode.check(load(fname)), [], fname)

    def test_k8_coverage_gap_reproduces_at_the_identical_ranges(self):
        original_issues = mcode.check(load("conv_4c4c_8x8_k8x1.mcode.gz"))
        rebuild_issues = mcode.check(load("conv_4c4c_8x8_k8x1_rebuild.mcode.gz"))
        self.assertEqual(len(original_issues), 1)
        self.assertEqual(len(rebuild_issues), 1)
        self.assertIn(self.EXPECTED_RANGES, original_issues[0])
        self.assertIn(self.EXPECTED_RANGES, rebuild_issues[0])


class TestK8SurvivesAnIndependentRebuild(unittest.TestCase):
    """This project's hard-learned rule: verify any new data point
    against an independent rebuild before trusting it."""

    def test_k8_segment_lengths_reproduce_exactly(self):
        original = load("conv_4c4c_8x8_k8x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k8x1_rebuild.mcode.gz")
        self.assertEqual(all_segment_lengths(original), all_segment_lengths(rebuild))

    def test_only_ordinary_noise_floor_bytes_differ(self):
        original = load("conv_4c4c_8x8_k8x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k8x1_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLess(len(diffs), 20, "only ordinary noise-floor bytes should differ")

    def test_record_kinds_reproduce_across_all_segments(self):
        original = load("conv_4c4c_8x8_k8x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k8x1_rebuild.mcode.gz")
        _, segs1 = mcode.segments(original)
        _, segs2 = mcode.segments(rebuild)
        for i in range(len(segs1)):
            pos1, len1, _ = segs1[i]
            pos2, len2, _ = segs2[i]
            end1 = pos1 + len1
            while end1 > pos1 and original[end1 - 1] == 0:
                end1 -= 1
            end2 = pos2 + len2
            while end2 > pos2 and rebuild[end2 - 1] == 0:
                end2 -= 1
            recs1 = mcode.decode(original, start=pos1, end=end1, **mcode.FULL_RULE)
            recs2 = mcode.decode(rebuild, start=pos2, end=end2, **mcode.FULL_RULE)
            self.assertEqual(
                Counter(r["kind"] for r in recs1),
                Counter(r["kind"] for r in recs2),
                f"segment {i}: record kinds",
            )


if __name__ == "__main__":
    unittest.main()
