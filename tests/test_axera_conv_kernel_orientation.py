"""Localizing (not fully decoding) Conv's `3x1` vs `1x3` kernel-orientation
mcode-length difference.

`scripts/axera/README.md`'s "Two more Conv attributes tried" section
found that a `Conv` with a `3x1` kernel and the same `Conv` with a `1x3`
kernel (same `cin`/`cout`, same total weight count -- `Wbt` came out
byte-identical *in length*, 1,320 bytes both ways) produce different
mcode lengths: 3,528 bytes for `3x1` vs 3,208 bytes for `1x3`, a real
320-byte difference. It was flagged there as "not further decoded (no
same-length pair here to diff cleanly), but a real, motivated target
for whoever chases the scanning-order encoding next" -- found and
measured, but never actually localized or diffed, since the two builds
aren't the same length. (Checked before starting: no later section of
that 7490-line README revisits "3x1"/"1x3"/"scanning-order" again, and
this is a different question from the per-channel requant-multiplier
work `scripts/axera/emitter.py` already closes -- that's about weight
scale/bias, not kernel-orientation scan order.)

This reproduces the exact same lengths (`cin=cout=4`, `hw=8`, same-size
weights, MinMax calibration) and, since a raw byte diff is impossible
across different lengths, uses the real structural codec
(`mcode.segments()` / `mcode.decode(..., **mcode.FULL_RULE)`) to compare
at the *record* level instead.

## What this localizes

**The entire 320-byte delta lives in the two op-program segments, and
nowhere else.** Both streams parse cleanly (`mcode.check()` returns `[]`
for both) into 5 segments each. Segments 3/4/5 (the large non-op-program
segment, the tiny 32-byte segment, and the 352-byte epilogue-adjacent
segment) are **byte-length-identical and record-kind-count-identical**
between the two orientations -- completely unaffected by kernel shape.
Only segments 1 and 2 (each containing exactly one op-program verb,
confirmed via `mcode.op_programs()`) differ: 608 bytes each for `3x1`,
448 bytes each for `1x3` -- a 160-byte shrink per segment, 320 bytes
total, exactly accounting for the whole-stream delta.

**Within those two segments, `1x3` has systematically fewer records of
every recognized kind than `3x1`** (verbs, width-rule `S` units, bare
`B` pairs, and unrecognized `raw` bytes all decrease), not merely a
shorter *unrecognized* region -- this is a real difference in how many
structured instructions the compiler emits per op-program segment
according to kernel orientation, not just less padding.

**Confirmed above the noise floor.** An independent second rebuild of
the `3x1` config alone (this project's hard-learned rule for trusting
any diff) shows the familiar few-byte determinism noise (15 raw bytes
differ, within the op-program segment's own span) but **zero** change
in record kind counts at any granularity -- `decode()`'s record
structure is exactly reproducible even when a handful of underlying
byte values are not. The orientation delta (49 fewer `S` records, 10
fewer `raw`, 7 fewer `V`, 6 fewer `B` -- summed across both segments) is
far above that zero noise floor.

**Not a simple "fewer repeated iterations" pattern.** Comparing content
signatures (each record's fields, ignoring its absolute offset) between
the two op-program segments shows `1x3`'s records are only found as an
in-order subsequence of `3x1`'s **35% of the time** (38 of 108) -- if
`3x1` were simply `1x3` plus N extra copies of one repeating sub-block
(an unrolled per-tap loop with a fixed body per extra kernel tap), that
figure would be near 100%. It isn't: most records on each side are
unique-valued (their own `reg`/`payload`/`tag` combination, not shared
with the other build), meaning the instruction *content*, not just the
instruction *count*, genuinely differs by orientation -- consistent
with the README's own hypothesis ("the compiler treats a 'tall' and a
'wide' kernel of otherwise identical size differently, plausibly
because how it scans/tiles the input differs by row vs. column
direction"), now with a precise location, but the specific
scan/tiling rule itself remains undecoded -- an honest partial result,
not a full decode.
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


def sig(r):
    return tuple(sorted((k, v) for k, v in r.items() if k != "at"))


class TestConvOrientationLengthDelta(unittest.TestCase):
    K31 = "conv_4c4c_8x8_k3x1.mcode.gz"
    K13 = "conv_4c4c_8x8_k1x3.mcode.gz"

    def test_reproduces_the_readmes_lengths(self):
        d31, d13 = load(self.K31), load(self.K13)
        self.assertEqual(len(d31), 3528)
        self.assertEqual(len(d13), 3208)
        self.assertEqual(len(d31) - len(d13), 320)

    def test_both_streams_are_well_formed(self):
        self.assertEqual(mcode.check(load(self.K31)), [])
        self.assertEqual(mcode.check(load(self.K13)), [])

    def test_delta_is_confined_to_the_two_op_program_segments(self):
        d31, d13 = load(self.K31), load(self.K13)
        _, segs31 = mcode.segments(d31)
        _, segs13 = mcode.segments(d13)
        self.assertEqual(len(segs31), 5)
        self.assertEqual(len(segs13), 5)

        # Segments 3/4/5 (0-indexed 2/3/4): identical length and identical
        # record-kind composition, regardless of kernel orientation.
        for i in (2, 3, 4):
            pos31, len31, _ = segs31[i]
            pos13, len13, _ = segs13[i]
            self.assertEqual(len31, len13, f"segment index {i}: length")
            r31 = self._decode_segment(d31, pos31, len31)
            r13 = self._decode_segment(d13, pos13, len13)
            self.assertEqual(
                Counter(r["kind"] for r in r31),
                Counter(r["kind"] for r in r13),
                f"segment index {i}: record kinds",
            )

        # Segments 1/2 (0-indexed 0/1): each shrinks by exactly 160 bytes.
        for i in (0, 1):
            self.assertEqual(segs31[i][1] - segs13[i][1], 160, f"segment index {i}")

    def _decode_segment(self, data, pos, length):
        end = pos + length
        while end > pos and data[end - 1] == 0:
            end -= 1
        return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)

    def test_op_program_segments_each_contain_exactly_one_program(self):
        d31, d13 = load(self.K31), load(self.K13)
        op31, op13 = mcode.op_programs(d31), mcode.op_programs(d13)
        counts31 = [len(v) for v in op31.values()]
        counts13 = [len(v) for v in op13.values()]
        self.assertEqual(sorted(counts31), [0, 0, 1, 1, 1])
        self.assertEqual(sorted(counts13), [0, 0, 1, 1, 1])

    def test_1x3_has_fewer_records_of_every_kind_in_the_op_program_segments(self):
        d31, d13 = load(self.K31), load(self.K13)
        _, segs31 = mcode.segments(d31)
        _, segs13 = mcode.segments(d13)
        c31 = Counter()
        c13 = Counter()
        for i in (0, 1):
            pos31, len31, _ = segs31[i]
            pos13, len13, _ = segs13[i]
            c31 += Counter(r["kind"] for r in self._decode_segment(d31, pos31, len31))
            c13 += Counter(r["kind"] for r in self._decode_segment(d13, pos13, len13))
        for kind in ("V", "S", "B", "raw"):
            self.assertGreater(c31[kind], c13[kind], f"kind {kind}: expected 3x1 > 1x3")

    def test_content_is_mostly_not_a_shared_repeated_subblock(self):
        """If 3x1 were just 1x3 plus extra copies of one repeating
        sub-block, 1x3's record sequence would be an in-order subsequence
        of 3x1's almost entirely. It isn't -- confirming the difference is
        in instruction content, not just a repeat count."""
        d31, d13 = load(self.K31), load(self.K13)
        _, segs31 = mcode.segments(d31)
        _, segs13 = mcode.segments(d13)
        pos31, len31, _ = segs31[0]
        pos13, len13, _ = segs13[0]
        s31 = [sig(r) for r in self._decode_segment(d31, pos31, len31)]
        s13 = [sig(r) for r in self._decode_segment(d13, pos13, len13)]

        i = 0
        for r in s31:
            if i < len(s13) and r == s13[i]:
                i += 1
        match_fraction = i / len(s13)
        self.assertLess(
            match_fraction,
            0.6,
            "expected well under full subsequence match, confirming genuinely"
            " different instruction content rather than a longer repeat count",
        )


if __name__ == "__main__":
    unittest.main()
