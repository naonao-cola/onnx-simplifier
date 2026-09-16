"""Refuting the receptive-field-size hypothesis for Conv dilation
"field 2" -- the signal tracks something closer to raw dilation, not
`dilation*(kernel_size-1)+1`.

`tests/test_axera_conv_dilation_fields.py` (PR #1503, merged) found two
Conv mcode fields sensitive to dilation at a `k=3` shape and explicitly
left untested: "A `k=5` kernel variant at dilation `{1,2,3}` (matching
receptive-field sizes 5/9/13 to the `k=3` sweep's `d=2/d=4/d=6`) was
built to test whether the fields track receptive-field size rather than
raw dilation, but serializes to a different total length ... with the
two fields not yet relocated inside that different layout -- left for
future work."

This finishes that thread using `mcode.decode()`/record-level comparison
(not raw offsets, which don't survive the length change) to relocate
the relevant structure in the `k=5` layout.

## What "field 2" actually is, restated at the record level

Decoding around `test_axera_conv_dilation_fields.py`'s FIELD2_SLOTS
(552-558 / 1160-1166, the `k=3` shape) shows those raw-byte windows sit
inside a short-unit record cluster right after a fixed anchor record
(`S`, reg=22, tag=0x83, payload=`00 0b 0f`). What actually changes
between dilation values, precisely, is which record *kind* follows that
anchor:

- `d=2, d=3, d=5, d=6, d=7`: the cluster stays as ordinary short `S`
  units (matching the previously-merged file's byte-level "3 distinct
  values" finding -- those are different S-unit payload bytes, not a
  different record shape).
- `d=3` *also* inserts two adjacent extra runs -- 4 then 3 bare
  `B`-kind pair records (7 records total, at offset 603/617, mirrored
  at 1211/1225) -- not present at `d=2`.
- `d=4` inserts a single bigger extra run of 8 `B`-kind pairs at the
  same location (offset 552, mirrored at 1160, the same fixed
  608-byte engine-copy separation this README already documents
  elsewhere).
- `d=5, d=6, d=7`: no such extra `B`-run at all -- back to `d=2`'s
  shape, at these offsets. So "extra `B`-run record count" for `k=3`
  reads `d=2:0, d=3:7, d=4:8, d=5:0, d=6:0, d=7:0` (per engine copy)
  -- a real, non-monotone, narrow spike at `d=3`/`d=4` specifically,
  not a threshold.

This confirms and sharpens (rather than contradicts) the merged PR's
own "not simple binary, non-monotone" finding -- the earlier work found
*that* the byte values were non-monotone; this shows *why*, at the
record-structure level: dilation `3`/`4` uniquely trigger extra
instructions at this location, and no other dilation value does.

## The `k=5` sweep, and the refutation

Built `Conv(k=5)` at `dilation={1,2,3,4}` (receptive fields `5,9,13,17`
-- `d=1,d=2,d=3` chosen to match `k=3`'s `d=2,d=4,d=6` receptive fields
`5,9,13` exactly; `d=4` added afterward to check whether the `k=3`
sweep's raw-dilation-`3`/`4` spike also shows up here). `d=1` and `d=3`
serialize to the same 3,912-byte length; `d=2` (3,944 bytes) and `d=4`
(4,176 bytes) each serialize to their own different length -- expected,
this project has repeatedly found mcode length is shape/attribute-
sensitive in non-uniform ways, which is exactly why record-level
comparison rather than raw-offset comparison is used throughout this
file.

Searching each `k=5` build for the same signature (an extra run of >=3
consecutive bare `B`-kind records at an offset below 1600 -- excluding
one specific 3-record run, `((13,225),(126,129),(8,132))` at offset
~984-993, confirmed present in *every* `k=5` build regardless of
dilation and so treated as boilerplate, not signal; every other
boilerplate `B`-run common to all `k=5` builds starts at 2010+, well
past the cutoff):

| build | receptive field | extra early B-run |
| --- | --- | --- |
| `k=5, d=1` | 5 | **none** |
| `k=5, d=2` | **9** | **none** |
| `k=5, d=3` | 13 | **yes** -- 7 records (4+3 split), offset 713/727, mirrored 1513/1527 |
| `k=5, d=4` | 17 | yes -- 3 records, offset 594, mirrored 1424 |

**`k=5, d=2` has receptive field 9 -- identical to `k=3, d=4`'s
receptive field -- and shows *no* extra run, where `k=3, d=4` shows an
8-record one.** Same receptive field, opposite outcome: this directly
refutes the hypothesis that field 2 (or this structural marker of it)
is driven by effective receptive-field span
(`dilation*(kernel_size-1)+1`).

**What the data is consistent with instead: raw dilation value, around
`3`-`4`, in both kernel-size sweeps.** `k=3`'s spike is at raw dilation
`3`,`4`; `k=5`'s spike (of the four dilation values tested) is also at
raw dilation `3`,`4` -- despite those two sweeps' receptive fields at
that point being completely different (`7`,`9` for `k=3` vs `13`,`17`
for `k=5`). This is offered as an observation the data supports, not a
decoded rule -- only two kernel sizes and a handful of dilation values
were tested, not enough to fit a real formula, and `k=5, d=4`'s extra
run has a different shape (3 records, different reg values) from
`k=3`'s, so "the same phenomenon" is inferred from position and
qualitative kind (bare `B`-pair records inserted at this specific
anchor-adjacent slot), not byte-identical content.

**Confirmed above the noise floor.** `k=5, d=1`'s absence and `k=5,
d=3`'s presence (4-record run, exact reg/tag values, exact offsets)
both survive an independent rebuild unchanged; the only bytes that move
between a build and its rebuild are 3 (`d=3`) or 15 (`d=1`) ordinary
noise bytes elsewhere in the stream, consistent with this project's
established noise scale.
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


_UNIVERSAL_EARLY_BOILERPLATE = ((13, 225), (126, 129), (8, 132))


def early_b_runs(data, min_len=3, cutoff=1600):
    """Runs of >=min_len consecutive bare `B`-kind records starting
    before `cutoff`, excluding one specific 3-record run
    (`_UNIVERSAL_EARLY_BOILERPLATE`, offset ~984-993 depending on shape)
    confirmed present in every `k=5` build regardless of dilation --
    real boilerplate, not signal. The remaining boilerplate `B`-runs
    common to every build regardless of dilation start at 1626+ (`k=3`
    shape) / 2010+ (`k=5` shape), well past `cutoff`, so no further
    content-based exclusion is needed there."""
    recs = mcode.decode(data, start=0, end=len(data), **mcode.FULL_RULE)
    runs = []
    i = 0
    while i < len(recs):
        if recs[i].get("kind") == "B":
            j = i
            while j < len(recs) and recs[j].get("kind") == "B":
                j += 1
            if j - i >= min_len and recs[i]["at"] < cutoff:
                sig = tuple((r["reg"], r["tag"]) for r in recs[i:j])
                if sig != _UNIVERSAL_EARLY_BOILERPLATE:
                    runs.append((recs[i]["at"], j - i, sig))
            i = j
        else:
            i += 1
    return runs


class TestK3SweepExtraBRunIsNonMonotone(unittest.TestCase):
    """Restates test_axera_conv_dilation_fields.py's byte-level "field 2
    is not simple binary" finding at the record level: a real, narrow
    spike at d=3/d=4, absent everywhere else in the d=2..7 sweep."""

    def test_d2_has_no_extra_early_b_run(self):
        self.assertEqual(early_b_runs(load("conv_dilation2.mcode.gz")), [])

    def test_d3_has_extra_runs_at_both_engine_copies(self):
        runs = early_b_runs(load("conv_dilation3.mcode.gz"))
        offsets_and_counts = [(at, count) for at, count, _ in runs]
        # Two engine copies (separated by the same fixed 608-byte gap
        # this README already documents elsewhere), each split into a
        # 4-record run immediately followed by a 3-record run.
        self.assertEqual(offsets_and_counts, [(603, 4), (617, 3), (1211, 4), (1225, 3)])

    def test_d4_has_an_8_record_extra_run_at_both_engine_copies(self):
        runs = early_b_runs(load("conv_dilation4.mcode.gz"))
        offsets_and_counts = [(at, count) for at, count, _ in runs]
        self.assertEqual(offsets_and_counts, [(552, 8), (1160, 8)])

    def test_d5_d6_d7_have_no_extra_early_b_run(self):
        for name in (
            "conv_dilation5.mcode.gz",
            "conv_dilation6.mcode.gz",
            "conv_dilation7.mcode.gz",
        ):
            self.assertEqual(early_b_runs(load(name)), [], name)


class TestK5SweepRefutesReceptiveFieldHypothesis(unittest.TestCase):
    """The key comparison: k=3,d=4 and k=5,d=2 share receptive field 9
    but show opposite outcomes for this structural marker."""

    def test_k5_d1_rf5_has_no_extra_run(self):
        self.assertEqual(early_b_runs(load("conv_k5_dilation1.mcode.gz")), [])

    def test_k5_d2_rf9_has_no_extra_run_unlike_k3_d4_at_the_same_rf(self):
        # k=3,d=4 (receptive field 9) has an 8-record extra run at each
        # engine copy -- test_d4_has_an_8_record_extra_run_at_both_
        # engine_copies above. k=5,d=2 is also receptive field 9
        # (dilation*(k-1)+1 = 2*4+1 = 9) but shows NONE: same rf,
        # opposite outcome.
        self.assertEqual(early_b_runs(load("conv_k5_dilation2.mcode.gz")), [])

    def test_k5_d3_rf13_has_extra_runs_at_both_engine_copies(self):
        runs = early_b_runs(load("conv_k5_dilation3.mcode.gz"))
        offsets_and_counts = [(at, count) for at, count, _ in runs]
        self.assertEqual(offsets_and_counts, [(713, 4), (727, 3), (1513, 4), (1527, 3)])

    def test_k5_d4_rf17_also_has_extra_runs_at_both_engine_copies(self):
        # Different length (4176) from the d1/d3 pair (3912), so not
        # directly byte-comparable, but record-level decode still
        # locates real extra runs here too -- consistent with "raw
        # dilation ~3-4" rather than any function of receptive field
        # (17 here vs 9/13 for the other k=5 cases, and 7/9 for k=3's
        # own d=3/d=4).
        runs = early_b_runs(load("conv_k5_dilation4.mcode.gz"))
        offsets_and_counts = [(at, count) for at, count, _ in runs]
        self.assertEqual(offsets_and_counts, [(594, 3), (1424, 3)])


class TestK5FindingsSurviveAnIndependentRebuild(unittest.TestCase):
    """This project's hard-learned determinism rule, applied to both
    the 'absent' state (d=1) and the 'present' state (d=3)."""

    def test_d1_absence_is_reproducible(self):
        a = load("conv_k5_dilation1.mcode.gz")
        b = load("conv_k5_dilation1_rebuild.mcode.gz")
        self.assertEqual(len(a), len(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 0, "sanity: some noise should exist")
        self.assertEqual(early_b_runs(a), [])
        self.assertEqual(early_b_runs(b), [])

    def test_d3_presence_is_reproducible(self):
        a = load("conv_k5_dilation3.mcode.gz")
        b = load("conv_k5_dilation3_rebuild.mcode.gz")
        self.assertEqual(len(a), len(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 0, "sanity: some noise should exist")
        self.assertEqual(early_b_runs(a), early_b_runs(b))


if __name__ == "__main__":
    unittest.main()
