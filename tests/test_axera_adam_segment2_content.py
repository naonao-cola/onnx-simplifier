"""First characterization of `adam_update_fp32.mcode.gz` segment 2 (388,
832) -- the densest, largest segment in this training-graph fixture,
left fully open by `tests/test_axera_training_graph_setup_sequence.py`
(PR #1552, this session's first semantic training-graph mcode decode).

## The V-kind records are the already-decoded general-corpus field-write mechanism

`scripts/axera/README.md`'s "`a1 00 xx yy` is a field write" section
(line 2206, from this project's much older, non-training-graph corpus
work) established: `a1 00 <field> <bank> <value>` writes a 32-bit value
to field `xx` of bank `yy`, where `xx` is always a multiple of `0x10`
(16-byte-granular), confirmed 1,255/1,255 headers across two real
models with zero exceptions.

Segment 2 has 15 `V`-kind records. Every one uses verb `0xa1` (161,
matching `a1 00`'s own first byte) except four boundary/terminator
verbs (`0xa2`, `0xa7`, `0xa8` x2, already named elsewhere in this
project's Conv work as setup/terminator markers, not field writes).
**Every `field` value across all 15 records, without exception, is a
multiple of 16** -- exactly the already-established pattern, now
confirmed for the first time in a training-graph mcode stream. Not a
new finding in itself, but a real generalization: this general-corpus
mechanism was never previously checked against training/FP32-override
content.

## A related but distinct echo: short-unit records also carry 16-byte-granular values

Beyond the `V`-kind field writes, two classes of `S`-kind (short-unit)
records in segment 2 independently carry values with the same 16-byte
granularity, in clean incrementing runs -- a pattern the README's own
field-write section only documented for `V`-kind records, never for
short units:

- **`tag=131, p=1` (2-byte little-endian payload):** 13 records total;
  six of them (offsets 424-449) form a clean arithmetic run --
  `992, 1008, 1024, 1040, 1056, 1072` -- each step exactly `+16`, with
  `reg` incrementing `10, 12, 14, 16` for the first four then holding
  at `8` for the last two. A second, shorter 2-record run at offsets
  597-602 (`8368, 8384`, `reg` `8, 10`) repeats the same shape at a
  different absolute value. The other 5 of the 13 `tag=131,p=1`
  records (offsets 500, 535, 560, 708, 1069) do not fit into either
  run and are not characterized further here.
- **`tag=132, p=0` (1-byte payload):** 60 records total; 41 of them
  (68%) have a payload value that is itself a multiple of 16 (range
  0-240), frequently in incrementing runs of 3-5 consecutive records
  (e.g. offsets 949-969: values `0, 16, 32, 48, 64, 80`, `reg`
  `224, 226, 228, 230, 8, 8`; offsets 1101-1117: values
  `0, 16, 32, 48, 64`, `reg` `40, 42, 44, 46, 8`). The remaining 19 of
  60 have small non-multiple-of-16 values in `[5, 30]` -- a visibly
  different population (plausibly counts, bit-widths, or small
  indices rather than field offsets), not decoded here.

## What this establishes, and what remains open

This is a real, concrete first pass at segment 2, not a full decode:
it confirms the general-corpus field-write mechanism generalizes to
training-graph mcode (the `V`-kind result), and it surfaces a new,
previously-undocumented echo of that same 16-byte addressing
granularity inside two specific short-unit record shapes -- but it
does not explain *why* these particular addresses are being
written/read (no correlation attempted here to which live tensor --
weight, gradient, or this op's `Sub` intermediate -- each write
targets), nor characterize the remaining ~185 of 254 segment-2 records
(the `raw` bytes, the `B`-kind records, and the `S`-kind records with
other tag/p shapes not covered above). No literal float32 constant
(a learning rate, epsilon, or similar) was found or searched for
successfully -- no build script reproducing this exact fixture's
source graph was located in `scripts/axera/`, so the real numeric
values this op uses could not be cross-checked against the mcode
bytes the way this project's INT8-quantized forward-op work has
routinely done with real scale/zero-point values.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def segment2_records():
    data = load("adam_update_fp32.mcode.gz")
    pos, length = 388, 832
    return mcode.decode(data, start=pos, end=pos + length, **mcode.FULL_RULE)


class TestSegment2RecordBreakdown(unittest.TestCase):
    def test_record_kind_counts(self):
        recs = segment2_records()
        kinds = {}
        for r in recs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        self.assertEqual(kinds, {"S": 120, "raw": 79, "B": 40, "V": 15})


class TestVKindRecordsAreTheKnownFieldWriteMechanism(unittest.TestCase):
    """Every V-kind record's field offset is a multiple of 16, matching
    scripts/axera/README.md's already-established `a1 00 xx yy` field-
    write mechanism (line 2206) -- confirmed here for a training-graph
    fixture for the first time."""

    def test_every_field_is_16_byte_granular(self):
        recs = [r for r in segment2_records() if r["kind"] == "V"]
        self.assertEqual(len(recs), 15)
        for r in recs:
            self.assertEqual(
                r["field"] % 16,
                0,
                f"at={r['at']} verb={r['verb']:#x} field={r['field']}",
            )

    def test_most_verbs_are_the_field_write_verb_0xa1(self):
        recs = [r for r in segment2_records() if r["kind"] == "V"]
        verb_a1_count = sum(1 for r in recs if r["verb"] == 0xA1)
        self.assertEqual(verb_a1_count, 11)


class TestShortUnitTag131P1HasTwoStep16Runs(unittest.TestCase):
    """tag=131,p=1 short units (2-byte LE payload): two runs of
    consecutive records whose values step by exactly +16, an echo of
    the same 16-byte granularity the V-kind field writes use, never
    previously documented for short-unit records."""

    def _tag131_p1(self):
        return [
            r
            for r in segment2_records()
            if r["kind"] == "S" and r.get("tag") == 131 and r.get("p") == 1
        ]

    def test_total_count(self):
        self.assertEqual(len(self._tag131_p1()), 13)

    def test_first_run_steps_by_16(self):
        recs = self._tag131_p1()
        run = [r for r in recs if 424 <= r["at"] <= 449]
        self.assertEqual(len(run), 6)
        values = [int.from_bytes(r["payload"], "little") for r in run]
        self.assertEqual(values, [992, 1008, 1024, 1040, 1056, 1072])
        self.assertTrue(all(b - a == 16 for a, b in zip(values, values[1:])))

    def test_second_run_steps_by_16(self):
        recs = self._tag131_p1()
        run = [r for r in recs if 597 <= r["at"] <= 602]
        self.assertEqual(len(run), 2)
        values = [int.from_bytes(r["payload"], "little") for r in run]
        self.assertEqual(values, [8368, 8384])


class TestShortUnitTag132P0SplitsIntoTwoPopulations(unittest.TestCase):
    """tag=132,p=0 short units (1-byte payload): 63% have a
    multiple-of-16 value (the same granularity as the field-write
    mechanism, often in incrementing runs); the rest are small values
    in [5,15] -- a visibly different, uncharacterized population."""

    def _tag132_p0(self):
        return [
            r
            for r in segment2_records()
            if r["kind"] == "S" and r.get("tag") == 132 and r.get("p") == 0
        ]

    def test_total_count(self):
        self.assertEqual(len(self._tag132_p0()), 60)

    def test_multiple_of_16_count(self):
        recs = self._tag132_p0()
        mult16 = [r for r in recs if r["payload"][0] % 16 == 0]
        self.assertEqual(len(mult16), 41)

    def test_a_sample_run_steps_by_16(self):
        recs = self._tag132_p0()
        run = [r for r in recs if 949 <= r["at"] <= 969]
        self.assertEqual(len(run), 6)
        values = [r["payload"][0] for r in run]
        self.assertEqual(values, [0, 16, 32, 48, 64, 80])

    def test_non_multiple_values_are_small(self):
        recs = self._tag132_p0()
        non_mult16 = [r["payload"][0] for r in recs if r["payload"][0] % 16 != 0]
        self.assertEqual(len(non_mult16), 19)
        self.assertTrue(all(5 <= v <= 30 for v in non_mult16))


if __name__ == "__main__":
    unittest.main()
