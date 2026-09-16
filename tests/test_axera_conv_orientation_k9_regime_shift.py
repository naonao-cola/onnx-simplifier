"""Investigating PR #1523's `kx1` segment-1 deviation at `k=7` (960
bytes instead of the linearly-predicted 992) by extending the sweep to
`k=9`: the deviation is not a one-time correction to an otherwise-
linear rule. By `k=9` the whole framing breaks down -- growth stops
happening in the two op-program segments at all, and jumps instead
into segments this project had previously confirmed fixed/boilerplate.

`tests/test_axera_conv_orientation_scaling.py` (PR #1523, still open at
the time of this file, not yet on `origin/master` -- its test file and
fixtures were pulled directly from its branch,
`take-cheeze/mcode-conv-orientation-scaling`, rather than waiting for
merge) found `kx1` Conv's first op-program segment grows exactly
linearly (96 bytes/tap: `3x1`->608, `5x1`->800, `7x1`->992) while its
second op-program segment follows the same rule from `k=3`->`k=5`
(608->800) but falls short at `k=7` (960, not 992) -- flagged as a
real, rebuild-confirmed, but unexplained deviation, with a `k=9`/`k=11`
sweep suggested as the natural next step.

## Segment-level record diff between k=5 and k=7: genuinely scattered, not one missing block

Before extending the sweep, this file first tried to localize the
32-byte segment-1 shortfall directly by diffing `k=5` and `k=7`'s
segment-1 record sequences with `difflib.SequenceMatcher` (the same
technique `tests/test_axera_conv_kernel_orientation.py`, PR #1499,
used for its own 35%-subsequence-match finding). The match ratio is
0.52 -- close to PR #1499's own 35% figure for a different orientation
comparison -- and the differences are scattered throughout the segment
(dozens of small replace/insert/delete opcodes, register numbers
shifting, payload bytes changing) rather than one identifiable missing
record or block. This rules out a simple "one record got dropped"
explanation and is consistent with PR #1499/#1523's own finding that
orientation/kernel-length changes cause genuine widespread content
differences, not simple insertions/deletions of a fixed sub-block.

## The real finding: k=9 moves the growth out of segments 0/1 entirely

Building `9x1` at the same shape (`cin=cout=4`, `hw=8`, same
weight-init/calibration convention as PR #1523's own probe script)
gives all 5 of this shape's mcode segments, compared against the
already-established `k=3,5,7` values:

| segment | k=3 | k=5 | k=7 | k=9 |
| --- | --- | --- | --- | --- |
| 0 (op-program) | 608 | 800 | 992 | **864** |
| 1 (op-program) | 608 | 800 | 960 | **864** |
| 2 | 1184 | 1184 | 1184 | 1184 |
| 3 | 32 | 32 | 32 | **288** |
| 4 | 352 | 352 | 352 | **704** |

Segment 2 stays exactly constant at every `k` tested (as it does for
the `1xk` orientation too, per PR #1523). **Segments 0 and 1 -- the
only two segments PR #1499 found carry any of the `3x1`-vs-`1x3` delta,
and the only two segments PR #1523's own linear-growth rule was framed
around -- both *shrink* from `k=7` to `k=9`** (992->864, 960->864;
notably converging on the identical 864 value, unlike every earlier
`k` where they differed). **Segments 3 and 4 -- both byte-length-fixed
at every `k` up to and including 7, the same "boilerplate" status PR
#1499 established for the *analogous* segments 3/4/5 in the `3x1`-vs-
`1x3` comparison -- jump for the first time at `k=9`**: segment 3 goes
32 -> 288 (9x), segment 4 goes 352 -> 704 (exactly 2x).

**This is real content, not padding.** `mcode.decode()` on `k=9`'s
segment 3 finds a genuine mix of record kinds (31 `S`, 17 `raw`, 9
`B`, 8 `V`, 1 `A` -- not a single repeated filler value); segment 4 is
88 `V` (verb) records, a real, fully-decoded instruction sequence, not
zero-padding.

**Confirmed above the noise floor.** An independent rebuild of `9x1`
reproduces the identical 5-segment length breakdown exactly
(864/864/1184/288/704), with only 14 ordinary noise-floor bytes
differing elsewhere in the stream -- well within this project's
established `~6-20` byte scale, not a sign this data point is
unreliable.

## What this means for PR #1523's framing

The "segment 0 grows linearly at 96 bytes/tap, segment 1 mostly
follows but deviates once at k=7" picture does not survive to `k=9`:
by then, growth has moved entirely out of both op-program segments and
into segments 3/4 instead, and segments 0/1 are no longer growing at
all (they shrank). The `k=7` deviation this file set out to explain is
better read, in light of this, as an early symptom of segments 3/4
starting to take on load that segments 0/1 previously carried alone --
not a one-off correction to an otherwise-clean linear rule, and not
(as far as tested here) a simple periodic or asymptotic pattern either.
**Not decoded here**: the exact kernel length where segments 3/4 first
grow (somewhere in `(7,9]`, not pinned more precisely), why growth
relocates there, or what the new segment-3/4 records compute. A `k=8`
build (untested here) would pin the exact transition point; `k=11`+
would show whether segments 0/1 keep shrinking, stabilize, or resume
growing, none of which is established by this file.
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


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def all_segment_lengths(data):
    _, segs = mcode.segments(data)
    return [length for _, length, _ in segs]


class TestSegments0And1ShrinkAtK9(unittest.TestCase):
    """The two op-program segments, which grew at every step from k=3
    to k=7, both shrink from k=7 to k=9 -- and converge on the
    identical length, unlike every earlier k where they differed."""

    def test_k9_segment_lengths(self):
        data = load("conv_4c4c_8x8_k9x1.mcode.gz")
        self.assertEqual(mcode.check(data), [], "well-formed")
        lens = all_segment_lengths(data)
        self.assertEqual(lens, [864, 864, 1184, 288, 704])

    def test_segments_0_and_1_shrink_relative_to_k7(self):
        k7 = load("conv_4c4c_8x8_k7x1.mcode.gz")
        k9 = load("conv_4c4c_8x8_k9x1.mcode.gz")
        len0_k7, len1_k7 = all_segment_lengths(k7)[:2]
        len0_k9, len1_k9 = all_segment_lengths(k9)[:2]
        self.assertEqual((len0_k7, len1_k7), (992, 960))
        self.assertEqual((len0_k9, len1_k9), (864, 864))
        self.assertLess(len0_k9, len0_k7)
        self.assertLess(len1_k9, len1_k7)


class TestSegments3And4GrowForTheFirstTimeAtK9(unittest.TestCase):
    """Segments 3 and 4 are byte-length-fixed across k=3,5,7 (32 and
    352 bytes respectively) but grow substantially at k=9 -- the
    growth PR #1523 found confined to segments 0/1 has relocated."""

    FIXED_CASES = [
        ("conv_4c4c_8x8_k3x1.mcode.gz", [32, 352]),
        ("conv_4c4c_8x8_k5x1.mcode.gz", [32, 352]),
        ("conv_4c4c_8x8_k7x1.mcode.gz", [32, 352]),
    ]

    def test_segments_3_and_4_fixed_through_k7(self):
        for fname, expected in self.FIXED_CASES:
            data = load(fname)
            lens = all_segment_lengths(data)[3:5]
            self.assertEqual(lens, expected, fname)

    def test_segments_3_and_4_grow_at_k9(self):
        data = load("conv_4c4c_8x8_k9x1.mcode.gz")
        lens = all_segment_lengths(data)[3:5]
        self.assertEqual(lens, [288, 704])
        # segment 4 exactly doubles; segment 3 grows 9x.
        self.assertEqual(704 / 352, 2.0)
        self.assertEqual(288 / 32, 9.0)

    def test_segment_2_stays_constant_at_every_k(self):
        for fname in (
            "conv_4c4c_8x8_k3x1.mcode.gz",
            "conv_4c4c_8x8_k5x1.mcode.gz",
            "conv_4c4c_8x8_k7x1.mcode.gz",
            "conv_4c4c_8x8_k9x1.mcode.gz",
        ):
            data = load(fname)
            self.assertEqual(all_segment_lengths(data)[2], 1184, fname)

    def test_k9_new_segments_carry_real_record_content_not_padding(self):
        data = load("conv_4c4c_8x8_k9x1.mcode.gz")
        _, segs = mcode.segments(data)
        pos3, len3, _ = segs[3]
        pos4, len4, _ = segs[4]
        recs3 = decode_segment(data, pos3, len3)
        recs4 = decode_segment(data, pos4, len4)
        kinds3 = Counter(r["kind"] for r in recs3)
        kinds4 = Counter(r["kind"] for r in recs4)
        self.assertGreater(len(kinds3), 1, "segment 3: mix of record kinds, not filler")
        self.assertEqual(kinds4, Counter({"V": 88}), "segment 4: 88 verb records")


class TestK9SurvivesAnIndependentRebuild(unittest.TestCase):
    """This project's hard-learned rule: verify any new data point
    against an independent rebuild before trusting it."""

    def test_k9_segment_lengths_reproduce_exactly(self):
        original = load("conv_4c4c_8x8_k9x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k9x1_rebuild.mcode.gz")
        self.assertEqual(
            all_segment_lengths(original),
            all_segment_lengths(rebuild),
        )

    def test_only_ordinary_noise_floor_bytes_differ(self):
        original = load("conv_4c4c_8x8_k9x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k9x1_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLess(len(diffs), 20, "only ordinary noise-floor bytes should differ")


class TestK5VsK7Segment1DiffIsScatteredNotLocalized(unittest.TestCase):
    """Record-level diff between k=5 and k=7's segment 1 (via
    difflib.SequenceMatcher, the same technique PR #1499 used) shows a
    match ratio close to PR #1499's own ~35% figure, with dozens of
    scattered small differences -- not one identifiable missing
    record accounting for the 32-byte shortfall."""

    def test_match_ratio_is_low_and_diffs_are_scattered(self):
        import difflib

        k5 = load("conv_4c4c_8x8_k5x1.mcode.gz")
        k7 = load("conv_4c4c_8x8_k7x1.mcode.gz")
        _, segs5 = mcode.segments(k5)
        _, segs7 = mcode.segments(k7)
        pos5, len5, _ = segs5[1]
        pos7, len7, _ = segs7[1]
        recs5 = decode_segment(k5, pos5, len5)
        recs7 = decode_segment(k7, pos7, len7)

        def sig(r):
            return tuple((k, v) for k, v in sorted(r.items()) if k != "at")

        sm = difflib.SequenceMatcher(
            a=[sig(r) for r in recs5], b=[sig(r) for r in recs7], autojunk=False
        )
        self.assertLess(sm.ratio(), 0.6, "should not be a near-identical sequence")
        opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
        self.assertGreater(
            len(opcodes), 15, "differences should be scattered across many small edits"
        )


if __name__ == "__main__":
    unittest.main()
