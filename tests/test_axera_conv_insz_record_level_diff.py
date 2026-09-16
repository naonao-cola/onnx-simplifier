"""Record-level structural diff of Conv's spatial-size dependence:
mostly diffuse, as PR #1537 found -- but two specific records turn out
to hold a literal, decoded `insz - 1` value, confirmed across three
data points and stable under an independent rebuild.

`tests/test_axera_conv_shape_selector_search.py` (PR #1537) found a
raw byte diff between `Conv(insz=9)` and `Conv(insz=10)` (same weight
tensor, same 3888-byte mcode length, `cin=cout=4, k=3, group=1` all
held fixed -- the cleanest possible isolation) shows 1308 bytes
differing (34%), concentrated in `mcode.segments()`'s op-program
segments 0-2, and flagged record-level decoding as future work.

## What a record-level (not byte-level) diff actually shows

`mcode.decode()` on segment 1 (576 bytes at this shape) gives the
identical record-KIND composition for `insz=9` and `insz=10` -- 76 `S`,
39 `raw`, 17 `B`, 15 `V`, 1 `A`, 148 records each. Comparing records
position-by-position (valid here since the counts match exactly) shows
only **16 of 148 records actually differ** between the two builds --
the 194-byte segment-1 raw diff PR #1537 reported is dominated by a
handful of records changing length (different `p` value -> different
payload byte count) and shifting every later record's absolute offset
by a few bytes, not by most of the segment's *content* changing. Most
of segment 1 is untouched.

## Two of those 16 differing records hold a literal `insz - 1` value

Building a third data point, `insz=11` (via `tests/test_axera_mcode_structure.py`'s
`_dilation_conv_model(1, 1, insz=11)`, the same "no dilation, pad=1"
convention PR #1537 used), two specific records show a value that
tracks `insz` exactly, at all three points:

- **A `V`-kind (verb) record**, `verb=161, bank=14, field=96` (bytes
  `a1 00 60 0e <byte> 00 00` in every build -- only one byte moves):
  operand's first byte is `0x08` at `insz=9`, `0x09` at `insz=10`,
  `0x0a` at `insz=11` -- exactly `insz - 1` at every point tested.
- **An `S`-kind short unit**, `tag=130, reg=140, p=1`, payload `0x0e
  <byte>` (raw bytes `01 0e <byte> 82`): the same byte, in the same
  place, tracks the identical `insz - 1` sequence (`0x08, 0x09, 0x0a`).

Both are confirmed stable under an independent rebuild of `insz=9`
(`conv_4c4c_insz9_rebuild.mcode.gz`, already committed by PR #1537):
both records are byte-for-byte identical between the original `insz=9`
build and its rebuild, ruling out noise.

## What this does NOT establish -- an honest complication

A third candidate that looked equally promising from the `insz=9`-vs-
`10` pair alone -- an `S`-kind record, `tag=131, reg=64, p=2`, whose
payload's last byte was `0x07` at `insz=9` and `0x08` at `insz=10`
(also `insz - 1`) -- does **not** continue the pattern at `insz=11`:
that exact record is simply absent from `insz=11`'s segment 1 (only
one `tag=131,reg=64` record remains, not two). Segment 1 itself grows
from 576 to 608 bytes at `insz=11` (148 records at `insz=9,10` to 153
at `insz=11`) with a real record-kind-count shift (`S: 76->75, raw:
39->45, V: 15->16, B: 17->16`) -- `insz=11` crosses into a genuinely
different structural regime for this segment, the same kind of sharp
regime shift this project's kernel-orientation work already found
(`tests/test_axera_conv_orientation_k8_transition.py`'s `k=8`-to-`k=9`
jump). So the two confirmed `insz-1` fields are real, decoded literal
values -- not the whole story, and not proof that every record in
segment 1 that "looks like" it tracks `insz` actually does so robustly
across a wider range. Untested here: `insz=12+`, whether the two
confirmed fields keep tracking `insz-1` past the `insz=11` regime
shift, and what the two confirmed fields (verb 161/bank 14/field 96,
and tag 130/reg 140) actually mean semantically (an intermediate
coordinate, a loop bound, a tile count) -- only that their value is
literally `insz - 1`.
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


class TestSegment1RecordCountsMatchDespiteBroadByteDiff(unittest.TestCase):
    """insz=9 and insz=10's segment 1 have identical record-kind
    composition and count -- the 194-byte raw diff PR #1537 reported
    is not evenly spread across all 148 records."""

    def test_record_kind_counts_are_identical(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        b = load("conv_4c4c_insz10.mcode.gz")
        _, segsA = mcode.segments(a)
        _, segsB = mcode.segments(b)
        posA, lenA, _ = segsA[1]
        posB, lenB, _ = segsB[1]
        recsA = decode_segment(a, posA, lenA)
        recsB = decode_segment(b, posB, lenB)
        self.assertEqual(len(recsA), len(recsB))
        self.assertEqual(
            Counter(r["kind"] for r in recsA), Counter(r["kind"] for r in recsB)
        )

    def test_only_a_small_minority_of_records_differ_position_for_position(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        b = load("conv_4c4c_insz10.mcode.gz")
        _, segsA = mcode.segments(a)
        _, segsB = mcode.segments(b)
        posA, lenA, _ = segsA[1]
        posB, lenB, _ = segsB[1]
        recsA = decode_segment(a, posA, lenA)
        recsB = decode_segment(b, posB, lenB)
        n_diff = 0
        for ra, rb in zip(recsA, recsB):
            keys = (set(ra.keys()) | set(rb.keys())) - {"at"}
            if any(ra.get(k) != rb.get(k) for k in keys):
                n_diff += 1
        self.assertLess(
            n_diff,
            20,
            "expected the vast majority of same-position records to be"
            " untouched, contrary to what the raw 194-byte diff alone suggests",
        )


class TestTwoRecordsEncodeInszMinusOneLiterally(unittest.TestCase):
    """A verb record (verb=161, bank=14, field=96) and a short unit
    (tag=130, reg=140) each carry a single byte equal to exactly
    `insz - 1`, confirmed at insz=9, 10, and 11."""

    CASES = [
        ("conv_4c4c_insz9.mcode.gz", 9, 0x08),
        ("conv_4c4c_insz10.mcode.gz", 10, 0x09),
        ("conv_4c4c_insz11.mcode.gz", 11, 0x0A),
    ]

    def test_verb_record_operand_byte_is_insz_minus_one(self):
        for fname, insz, expected in self.CASES:
            data = load(fname)
            _, segs = mcode.segments(data)
            pos, length, _ = segs[1]
            recs = decode_segment(data, pos, length)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 14
                and r.get("field") == 96
            ]
            self.assertEqual(len(hits), 1, f"{fname}: expected exactly one match")
            self.assertEqual(
                hits[0]["operand"][0],
                expected,
                f"{fname}: insz={insz}, expected operand[0] == insz-1 == {expected:#04x}",
            )

    def test_short_unit_payload_byte_is_insz_minus_one(self):
        for fname, insz, expected in self.CASES:
            data = load(fname)
            _, segs = mcode.segments(data)
            pos, length, _ = segs[1]
            recs = decode_segment(data, pos, length)
            hits = [r for r in recs if r.get("tag") == 130 and r.get("reg") == 140]
            self.assertEqual(len(hits), 1, f"{fname}: expected exactly one match")
            self.assertEqual(hits[0]["payload"][0], 0x0E, f"{fname}: fixed first byte")
            self.assertEqual(
                hits[0]["payload"][1],
                expected,
                f"{fname}: insz={insz}, expected payload[1] == insz-1 == {expected:#04x}",
            )

    def test_both_fields_survive_an_independent_rebuild(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        a_rebuild = load("conv_4c4c_insz9_rebuild.mcode.gz")
        for data in (a, a_rebuild):
            _, segs = mcode.segments(data)
            pos, length, _ = segs[1]
            recs = decode_segment(data, pos, length)
            v = next(
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 14
                and r.get("field") == 96
            )
            s = next(r for r in recs if r.get("tag") == 130 and r.get("reg") == 140)
            self.assertEqual(v["operand"][0], 0x08)
            self.assertEqual(s["payload"][1], 0x08)


class TestAThirdCandidateFieldDoesNotContinueThePattern(unittest.TestCase):
    """A third record (tag=131, reg=64, p=2) also looked like it tracked
    insz-1 from the insz=9-vs-10 pair alone, but is simply absent at
    insz=11 -- an honest complication, not a third confirmed field.
    Segment 1 itself grows and its record-kind counts shift at insz=11,
    a real regime change."""

    def test_candidate_record_present_at_9_and_10_but_absent_at_11(self):
        counts = {}
        for fname in (
            "conv_4c4c_insz9.mcode.gz",
            "conv_4c4c_insz10.mcode.gz",
            "conv_4c4c_insz11.mcode.gz",
        ):
            data = load(fname)
            _, segs = mcode.segments(data)
            pos, length, _ = segs[1]
            recs = decode_segment(data, pos, length)
            hits = [r for r in recs if r.get("tag") == 131 and r.get("reg") == 64]
            counts[fname] = len(hits)
        self.assertEqual(counts["conv_4c4c_insz9.mcode.gz"], 2)
        self.assertEqual(counts["conv_4c4c_insz10.mcode.gz"], 2)
        self.assertEqual(
            counts["conv_4c4c_insz11.mcode.gz"],
            1,
            "expected the candidate record to have vanished by insz=11",
        )

    def test_segment_1_length_and_record_count_shift_at_insz11(self):
        b = load("conv_4c4c_insz10.mcode.gz")
        c = load("conv_4c4c_insz11.mcode.gz")
        _, segsB = mcode.segments(b)
        _, segsC = mcode.segments(c)
        posB, lenB, _ = segsB[1]
        posC, lenC, _ = segsC[1]
        self.assertEqual(lenB, 576)
        self.assertEqual(lenC, 608, "expected segment 1 to grow at insz=11")
        recsB = decode_segment(b, posB, lenB)
        recsC = decode_segment(c, posC, lenC)
        self.assertEqual(len(recsB), 148)
        self.assertEqual(len(recsC), 153)
        self.assertNotEqual(
            Counter(r["kind"] for r in recsB), Counter(r["kind"] for r in recsC)
        )


if __name__ == "__main__":
    unittest.main()
