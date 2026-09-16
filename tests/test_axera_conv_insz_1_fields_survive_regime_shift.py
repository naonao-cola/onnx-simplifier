"""Testing whether the two `insz-1` fields PR #1539 decoded survive
past the `insz=11` regime shift: they do, at every one of four new
points (`insz=12,13,15,20`) -- but a new structural wrinkle appears
starting at `insz=13`: both fields begin showing up in a SECOND
segment as well, not just the one segment PR #1539 found them in.

`tests/test_axera_conv_insz_record_level_diff.py` (PR #1539) decoded
two Conv records, in segment 1 of a `cin=cout=4, k=3, dilation=1, pad=1`
model, whose value tracks `insz - 1` exactly at `insz=9,10,11`:

- A `V`-kind (verb) record, `verb=161, bank=14, field=96`, operand
  byte 0.
- An `S`-kind short unit, `tag=130, reg=140, p=1`, payload byte 1.

It found a THIRD, superficially similar candidate did not survive to
`insz=11` (the record vanished as segment 1 itself entered a new
structural regime -- record-kind counts shifted, segment length grew
576->608 bytes), and left open whether the two *confirmed* fields keep
tracking `insz-1` past that same `insz=11` regime shift.

## Both confirmed fields survive robustly at insz=12, 13, 15, 20

Building `insz = 12, 13, 15, 20` (same `cin=cout=4, k=3, dilation=1,
pad=1` convention) and searching every one of `mcode.segments()`'s
segments (not assuming segment 1 specifically, since the regime could
shift again) for both record signatures:

| insz | insz-1 | segment(s) with both fields | value found |
| --- | --- | --- | --- |
| 12 | 11 (`0x0b`) | segment 1 only | `0x0b` -- exact |
| 13 | 12 (`0x0c`) | segments 0 **and** 1 | `0x0c` -- exact, both copies |
| 15 | 14 (`0x0e`) | segments 0 **and** 1 | `0x0e` -- exact, both copies |
| 20 | 19 (`0x13`) | segments 0 **and** 1 | `0x13` -- exact, both copies |

Zero exceptions across all four new points (eight total individual
field instances, since two segments each carry both fields at three of
the four points) -- the `insz-1` relationship PR #1539 found is real
and holds well beyond the narrow `9..11` range it was originally
confirmed at.

## The new wrinkle: segment 0 starts carrying the same fields from insz=13 onward

At `insz=12` (matching `insz=9,10,11`'s own layout), only segment 1
carries either field -- segment 0 has neither. **Starting at
`insz=13`, segment 0 ALSO carries both fields**, with the identical
`insz-1` value as segment 1's own copy, and this continues at `insz=15`
and `insz=20`. This is a distinct structural change from the `insz=11`
regime shift PR #1539 already found (that one only affected segment
1's own record-kind composition and made the third candidate vanish;
it did not duplicate either confirmed field into segment 0). Not
decoded further here: why segment 0 begins duplicating this content at
exactly `insz=13`, or whether it's connected to any of this project's
other `insz=13`-adjacent findings (`tests/test_axera_conv_dilation_insz_threshold.py`'s
unrelated dilation-3/4 trigger work, at a *different* dilation
configuration, found its own `insz=13` transition point -- the two are
not asserted to share a cause, only noted as a coincidence worth a
future look).

## Confirmed above the noise floor

An independent rebuild of `insz=13` reproduces both fields' values
identically in both segments (`0x0c` throughout) and the same
record-kind composition; only 4 bytes differ between the original and
the rebuild anywhere in the whole 3920-byte stream, well within this
project's established noise-floor scale and nowhere near either
field's own record. Both builds pass `mcode.check()` cleanly (`[]`).
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


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def find_fields_by_segment(data):
    """Returns {segment_index: (verb_operand_byte0, short_unit_payload_byte1)}
    for every segment that carries either the verb=161/bank=14/field=96
    record or the tag=130/reg=140 short unit."""
    _, segs = mcode.segments(data)
    out = {}
    for idx, (pos, length, _) in enumerate(segs):
        recs = decode_segment(data, pos, length)
        verb = next(
            (
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 14
                and r.get("field") == 96
            ),
            None,
        )
        su = next(
            (r for r in recs if r.get("tag") == 130 and r.get("reg") == 140), None
        )
        if verb is not None or su is not None:
            out[idx] = (
                verb["operand"][0] if verb is not None else None,
                su["payload"][1] if su is not None else None,
            )
    return out


class TestFieldsSurviveAtInsz12(unittest.TestCase):
    """insz=12 matches the insz=9,10,11 layout: only segment 1 carries
    both fields, value 11 (0x0b) = insz-1."""

    def test_only_segment_1_carries_both_fields(self):
        data = load("conv_4c4c_insz12.mcode.gz")
        by_seg = find_fields_by_segment(data)
        self.assertEqual(set(by_seg.keys()), {1})
        self.assertEqual(by_seg[1], (0x0B, 0x0B))


class TestFieldsSurviveAndDuplicateFromInsz13Onward(unittest.TestCase):
    """insz=13, 15, and 20 all carry both fields in BOTH segment 0 and
    segment 1, each holding the identical insz-1 value."""

    CASES = [
        ("conv_4c4c_insz13.mcode.gz", 0x0C),
        ("conv_4c4c_insz15.mcode.gz", 0x0E),
        ("conv_4c4c_insz20.mcode.gz", 0x13),
    ]

    def test_both_segments_carry_both_fields_with_the_correct_value(self):
        for fname, expected in self.CASES:
            data = load(fname)
            by_seg = find_fields_by_segment(data)
            self.assertEqual(set(by_seg.keys()), {0, 1}, fname)
            self.assertEqual(by_seg[0], (expected, expected), fname)
            self.assertEqual(by_seg[1], (expected, expected), fname)


class TestInsz13FieldsSurviveAnIndependentRebuild(unittest.TestCase):
    def test_rebuild_reproduces_identical_field_values(self):
        original = load("conv_4c4c_insz13.mcode.gz")
        rebuild = load("conv_4c4c_insz13_rebuild.mcode.gz")
        self.assertEqual(
            find_fields_by_segment(original), find_fields_by_segment(rebuild)
        )

    def test_rebuild_diff_is_small_and_well_formed(self):
        original = load("conv_4c4c_insz13.mcode.gz")
        rebuild = load("conv_4c4c_insz13_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLess(len(diffs), 20, "expected only ordinary noise-floor drift")
        self.assertEqual(mcode.check(original), [])
        self.assertEqual(mcode.check(rebuild), [])


if __name__ == "__main__":
    unittest.main()
