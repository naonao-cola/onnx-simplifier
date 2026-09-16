"""Scaling `3x1`-vs-`1x3` past `k=3`: the "wide" (`1xk`) orientation's
mcode is frozen at a fixed size for every kernel length tested, while
the "tall" (`kx1`) orientation grows -- and its growth is exactly
linear in one of its two op-program segments.

`tests/test_axera_conv_kernel_orientation.py` (PR #1499, merged)
localized the `3x1`-vs-`1x3` mcode-length delta (3,528 vs 3,208 bytes,
`cin=cout=4`, `8x8` spatial) to two op-program segments, found `1x3`
has fewer records of every kind in both, and showed the two segments'
record content is only a 35% in-order subsequence match -- real content
difference, not merely fewer repeated copies of one sub-block. It left
"the specific scan/tiling rule itself... undecoded."

This scales the same shape to `k=5` and `k=7` (`5x1`/`1x5`,
`7x1`/`1x7`, same `cin=cout=4`, `8x8`, same weight-init/calibration
convention) to see whether the delta scales predictably with kernel
length.

## The central finding: the two orientations do NOT scale the same way

**`1xk` (kernel width `k`, height `1`) is length-invariant**: `1x3`,
`1x5`, and `1x7` all produce **exactly 3,208 bytes** of mcode -- the
same total length as `k=3`'s already-known `1x3` case, unchanged as
`k` grows to `5` and `7`. Both of its two op-program segments are
individually **exactly 448 bytes at every `k` tested**, byte-length-
identical across all three builds even though their record-kind
composition drifts slightly (segment 0's `V` count grows `11 -> 12 ->
13`, its `S` count shrinks `57 -> 56 -> 54`, and `raw` grows `24 -> 26
-> 33`, netting to the same 448-byte total every time; `B` stays
exactly constant, `15` in segment 0 and `13` in segment 1, at every
`k`). Landing on the identical byte total three times running, via
different record compositions each time, is not plausible as
coincidence -- this reads as a fixed per-segment instruction-size
budget for this scan direction, not a per-tap-unrolled encoding.

**`kx1` (kernel height `k`, width `1`) grows, and its first op-program
segment grows exactly linearly**: `3x1`/`5x1`/`7x1`'s segment 0 is
608/800/992 bytes -- **+192 bytes for every +2 in `k`, exactly, at
both measured steps** (96 bytes per additional kernel tap). Segment 1
grows the same way from `k=3` to `k=5` (`608 -> 800`, `+192`) but
**deviates at `k=7`** (`800 -> 960`, only `+160`) -- confirmed via an
independent rebuild of `7x1` (byte-identical segment-0/segment-1
record-kind composition and length, only 6 ordinary noise-floor bytes
differing elsewhere in the stream) to rule out this asymmetry being a
non-determinism artifact rather than a real, reproducible deviation
from the linear rule that holds for segment 0.

## What this does and does not establish

This narrows "the scan/tiling rule" considerably: the delta is not a
symmetric function of "how different are the two orientations" in any
simple sense -- one orientation (`1xk`) has a fixed instruction-size
budget regardless of kernel length (up to at least `k=7`), while the
other (`kx1`) genuinely unrolls additional per-tap structure, cleanly
so in its first op-program segment (a precise, decoded 96-bytes/tap
rate) but with a real, confirmed exception in its second segment at
`k=7`. The exact instruction-level content responsible for that
96-bytes/tap growth, and the cause of segment 1's `k=7` deviation, are
not decoded here -- consistent with, and a further precision-upgrade
of, `test_axera_conv_kernel_orientation.py`'s own finding that this is
genuine instruction-content difference, not a simple repeat count.

Verified via `mcode.check()` (both streams well-formed at every `k`)
and independent rebuilds at `k=5` and `k=7` (record-kind counts exactly
reproduced; only a handful of ordinary noise-floor bytes move,
consistent with this project's established `~6-20` byte noise scale at
this class of shape).
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


def op_program_segment_lengths(data):
    _, segs = mcode.segments(data)
    return [length for _, length, _ in segs[:2]]


class TestWideOrientationIsLengthInvariant(unittest.TestCase):
    """1xk's total mcode length, and each of its two op-program
    segments individually, stay exactly fixed at k=3,5,7."""

    CASES = [
        ("conv_4c4c_8x8_k1x3.mcode.gz", 3208),
        ("conv_4c4c_8x8_k1x5.mcode.gz", 3208),
        ("conv_4c4c_8x8_k1x7.mcode.gz", 3208),
    ]

    def test_total_length_is_invariant_to_k(self):
        for fname, expected in self.CASES:
            data = load(fname)
            self.assertEqual(mcode.check(data), [], f"{fname}: well-formed")
            self.assertEqual(len(data), expected, fname)

    def test_both_op_program_segments_stay_exactly_448_bytes(self):
        for fname, _ in self.CASES:
            data = load(fname)
            lens = op_program_segment_lengths(data)
            self.assertEqual(lens, [448, 448], f"{fname}: segment lengths")


class TestTallOrientationSegment0GrowsLinearly(unittest.TestCase):
    """kx1's first op-program segment grows by exactly 192 bytes for
    every +2 in k (96 bytes/tap), confirmed at both measured steps."""

    CASES = [
        ("conv_4c4c_8x8_k3x1.mcode.gz", 3, 608),
        ("conv_4c4c_8x8_k5x1.mcode.gz", 5, 800),
        ("conv_4c4c_8x8_k7x1.mcode.gz", 7, 992),
    ]

    def test_segment0_length_per_k(self):
        for fname, k, expected_len0 in self.CASES:
            data = load(fname)
            self.assertEqual(mcode.check(data), [], f"{fname}: well-formed")
            len0, _ = op_program_segment_lengths(data)
            self.assertEqual(len0, expected_len0, f"k={k}")

    def test_growth_rate_is_exactly_96_bytes_per_tap_both_steps(self):
        lens = {k: op_program_segment_lengths(load(f))[0] for f, k, _ in self.CASES}
        self.assertEqual(lens[5] - lens[3], 192, "k=3->5 growth")
        self.assertEqual(lens[7] - lens[5], 192, "k=5->7 growth")


class TestTallOrientationSegment1DeviatesAtK7(unittest.TestCase):
    """Segment 1 grows the same way as segment 0 from k=3 to k=5, but
    deviates at k=7 -- a real, rebuild-confirmed exception to the
    linear rule that holds cleanly for segment 0."""

    def test_segment1_grows_linearly_from_k3_to_k5(self):
        len1_k3 = op_program_segment_lengths(load("conv_4c4c_8x8_k3x1.mcode.gz"))[1]
        len1_k5 = op_program_segment_lengths(load("conv_4c4c_8x8_k5x1.mcode.gz"))[1]
        self.assertEqual(len1_k3, 608)
        self.assertEqual(len1_k5, 800)
        self.assertEqual(len1_k5 - len1_k3, 192)

    def test_segment1_deviates_from_linear_at_k7(self):
        len1_k5 = op_program_segment_lengths(load("conv_4c4c_8x8_k5x1.mcode.gz"))[1]
        len1_k7 = op_program_segment_lengths(load("conv_4c4c_8x8_k7x1.mcode.gz"))[1]
        self.assertEqual(len1_k5, 800)
        self.assertEqual(len1_k7, 960, "actual value, not the linear-predicted 992")
        self.assertEqual(
            len1_k7 - len1_k5, 160, "not the +192 the linear rule predicts"
        )

    def test_k7_deviation_survives_an_independent_rebuild(self):
        """Confirms the 960 (not 992) segment-1 length at k=7 is real
        structure, not a non-determinism artifact: an independent
        rebuild reproduces the identical segment lengths and record-kind
        composition, with only ordinary noise-floor bytes differing
        elsewhere in the stream."""
        original = load("conv_4c4c_8x8_k7x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k7x1_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        self.assertEqual(
            op_program_segment_lengths(original), op_program_segment_lengths(rebuild)
        )

        ok1, segs1 = mcode.segments(original)
        ok2, segs2 = mcode.segments(rebuild)
        for i in range(2):
            pos1, len1, _ = segs1[i]
            pos2, len2, _ = segs2[i]
            recs1 = decode_segment(original, pos1, len1)
            recs2 = decode_segment(rebuild, pos2, len2)
            self.assertEqual(
                Counter(r["kind"] for r in recs1),
                Counter(r["kind"] for r in recs2),
                f"segment {i}: record kinds",
            )

        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLess(len(diffs), 20, "only ordinary noise-floor bytes should differ")


class TestK5AlsoSurvivesAnIndependentRebuild(unittest.TestCase):
    """Determinism check for the k=5 data point too, matching this
    project's rule of never trusting a single build."""

    def test_k5_5x1_record_kinds_reproduce_exactly(self):
        original = load("conv_4c4c_8x8_k5x1.mcode.gz")
        rebuild = load("conv_4c4c_8x8_k5x1_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        recs1 = mcode.decode(original, start=0, end=len(original), **mcode.FULL_RULE)
        recs2 = mcode.decode(rebuild, start=0, end=len(rebuild), **mcode.FULL_RULE)
        self.assertEqual(
            Counter(r["kind"] for r in recs1), Counter(r["kind"] for r in recs2)
        )


if __name__ == "__main__":
    unittest.main()
