"""Resolving the `insz=13` coincidence PR #1541 flagged: NOT the same
mechanism as PR #1519's dilation-3/4 trigger threshold -- but chasing
it down generalizes PR #1539/#1541's own `insz-1` field into a cleaner
`insz - dilation` formula.

`tests/test_axera_conv_insz_1_fields_survive_regime_shift.py` (PR
#1541) found that at `dilation=1`, a decoded verb record
(`verb=161, bank=14, field=96`) whose value tracks `insz - 1` starts
appearing in a *second* mcode segment (segment 0, alongside its
original home in segment 1) exactly at `insz=13`.
`tests/test_axera_conv_dilation_insz_threshold.py` (PR #1519, merged)
separately found Conv's *unrelated* "extra `B`-run" trigger -- present
only at raw dilation `3`/`4`, never at other dilation values including
`1` -- first appears at exactly the same `insz=13`, for a differently-
configured model (`pad=dilation` instead of `pad=1`). PR #1541 flagged
this shared number as "not asserted to share a cause, only noted as a
coincidence worth a future look."

## Checked directly, using PR #1519's own already-committed fixtures --
## no new builds needed

`scripts/axera/fixtures/conv_dilation{3,4}_insz{12,13,14}.mcode.gz`
and `conv_dilation{3,4}.mcode.gz` (the original `insz=16` builds) were
already committed by PR #1519/#1509's own work. Searching them for PR
#1541's own verb-record signature (`verb=161, bank=14, field=96`)
gives:

| dilation | insz | segment 0 | segment 1 | value | `insz - dilation` |
| --- | --- | --- | --- | --- | --- |
| 3 | 12 | 9 | 9 | 9 | 9 |
| 3 | 13 | 10 | 10 | 10 | 10 |
| 3 | 14 | 11 | 11 | 11 | 11 |
| 3 | 16 | 13 | 13 | 13 | 13 |
| 4 | 12 | 8 | 8 | 8 | 8 |
| 4 | 13 | 9 | 9 | 9 | 9 |
| 4 | 14 | 10 | 10 | 10 | 10 |
| 4 | 16 | 12 | 12 | 12 | 12 |

Two things fall out of this table:

1. **The verb field's value is `insz - dilation`, not just `insz - 1`**
   -- PR #1539/#1541's `insz - 1` was the `dilation=1` special case of a
   more general rule. Exact match at all 8 points across two dilation
   values and four `insz` values, none of which were part of the
   original `dilation=1` discovery.
2. **Both segments already carry this field at `insz=12` for
   `dilation=3` and `dilation=4` -- there is no `insz=13` onset for the
   segment-0 duplication at these dilation values**, unlike `dilation=1`
   where PR #1541 found segment 0 picks the field up only starting at
   `insz=13` (segment 0 has neither field at `insz<=12` in that
   file's own data). If the `insz=13` "duplicate into segment 0"
   transition were a shared, dilation-independent hardware/compiler
   threshold, `dilation=3`/`4` would be expected to show the same
   before/after-13 split; they don't -- both segments already carry the
   field at `insz=12`, the earliest point checked here for these
   dilation values.

## Conclusion: the shared `insz=13` number is coincidental, not causal

PR #1519's dilation-3/4 trigger and PR #1541's segment-0 duplication
are governed by different mechanisms: the trigger is dilation-gated
(present only at raw dilation 3/4, confirmed absent at every other
dilation tested across this project's earlier work) and reaches its
own first-appearance point at `insz=13` for reasons PR #1519 left
undecoded; the verb field's segment-0 duplication is a `dilation=1`-
specific regime transition (dilation=3/4 never show a "before" state
at any `insz` tested here) that also happens to land on `insz=13`.
Two independently-timed transitions sharing one number, in a codec
this project has already found to be dense with distinct shape-driven
transition points (`insz=8/9`, `11`, `12/13`, and others), is exactly
the kind of coincidence that should NOT be assumed connected without
evidence -- and the evidence here (dilation=3/4's field is already
duplicated well before `insz=13`) actively argues against a shared
cause. This closes PR #1541's flagged question negatively, while
incidentally generalizing its own decoded field from `insz-1` to the
more general `insz-dilation`.

Not established here: why `dilation=1`'s segment-0 duplication itself
begins at `insz=13` (a real, still-undecoded threshold in its own
right, just not the one PR #1519 found for a different mechanism); or
what the short-unit field (`tag=130, reg=140`) PR #1539/#1541 also
decoded looks like at `dilation=3`/`4` -- it was not found under that
same `reg`/`tag` in any of these builds, and locating its dilation-3/4
analog (if one exists) is left for future work.
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


def find_verb_field_by_segment(data):
    """Returns {segment_index: operand_byte0} for every segment
    carrying the verb=161/bank=14/field=96 record PR #1539/#1541
    decoded as insz-1 (here generalized to insz-dilation)."""
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
        if verb is not None:
            out[idx] = verb["operand"][0]
    return out


class TestVerbFieldGeneralizesToInszMinusDilation(unittest.TestCase):
    # fixture: (dilation, insz, expected value = insz - dilation)
    CASES = [
        ("conv_dilation3_insz12.mcode.gz", 3, 12),
        ("conv_dilation3_insz13.mcode.gz", 3, 13),
        ("conv_dilation3_insz14.mcode.gz", 3, 14),
        ("conv_dilation3.mcode.gz", 3, 16),
        ("conv_dilation4_insz12.mcode.gz", 4, 12),
        ("conv_dilation4_insz13.mcode.gz", 4, 13),
        ("conv_dilation4_insz14.mcode.gz", 4, 14),
        ("conv_dilation4.mcode.gz", 4, 16),
    ]

    def test_value_equals_insz_minus_dilation_in_both_segments(self):
        for fname, dilation, insz in self.CASES:
            data = load(fname)
            by_seg = find_verb_field_by_segment(data)
            expected = insz - dilation
            self.assertEqual(set(by_seg.keys()), {0, 1}, fname)
            self.assertEqual(by_seg[0], expected, fname)
            self.assertEqual(by_seg[1], expected, fname)


class TestNoInsz13OnsetAtDilation3Or4(unittest.TestCase):
    """Unlike dilation=1 (PR #1541: segment 0 has neither field until
    insz=13), dilation=3 and dilation=4 already carry the field in
    BOTH segments at insz=12 -- the earliest point checked here. This
    is the direct evidence that the insz=13 coincidence is not a
    shared, dilation-independent threshold."""

    def test_both_segments_already_populated_at_insz12(self):
        for fname in (
            "conv_dilation3_insz12.mcode.gz",
            "conv_dilation4_insz12.mcode.gz",
        ):
            data = load(fname)
            by_seg = find_verb_field_by_segment(data)
            self.assertEqual(
                set(by_seg.keys()),
                {0, 1},
                f"{fname}: expected both segments already populated,"
                " unlike dilation=1's insz=13 onset",
            )


if __name__ == "__main__":
    unittest.main()
