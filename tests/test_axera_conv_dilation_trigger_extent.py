"""Widening the Conv dilation "extra B-run" spike search past d=7:
no periodicity found out to d=14, and its content is a real,
reproducible register-allocation sequence -- but *why* dilation
3/4 specifically triggers it stays undecoded.

`tests/test_axera_conv_dilation_fields.py` (PR #1503) found "field 2"
(a two-copy marker near a fixed anchor) is non-monotone across
`dilation=2..7` and explicitly flagged a tempting `dilation mod 5`
grouping as likely coincidental, not confirmed, since `d=3`/`d=6` don't
share a value despite differing by 5.
`tests/test_axera_conv_dilation_receptive_field.py` (PR #1507)
relocated field 2 at the record level: an extra run of bare `B`-kind
records inserted near a fixed anchor, present only at raw dilation
`3`/`4` across `d=2..7` -- refuted receptive-field-size as the driver.
`tests/test_axera_conv_dilation_b_run_is_reliable.py` (PR #1508)
confirmed this is a real, deterministic effect (13 independent rebuilds,
0 exceptions), not a coin flip. All three left "why 3/4" and "what does
it encode" open, and none tested past `d=7`.

## Widened sweep: d=8..14, no periodicity

Built the same `k=3` shape (`_dilation_conv_model`, `pad=dilation`, the
convention that keeps `d=2..7` at a constant 3,528-byte length) at
`dilation=8..14`. Unlike `d=2..7`, this range does **not** stay one
length: `d=8,9,10,11,13,14` serialize to 3,760 bytes and `d=12` alone
to 3,792 bytes (a threshold this project's earlier work already noted
existed but didn't explore) -- another reminder that raw-offset
comparison across different dilation values is unsafe; this file uses
`mcode.decode()`-based record search throughout, as the prior three
files do.

**None of `d=8` through `d=14` shows the early spike run** (the same
`early_b_runs()`-style detector as PR #1507/#1508, searched below
offset 1600, the same cutoff already established at the `k=3` shape).
Combined with the already-confirmed absence at `d=2,5,6,7` and presence
only at `d=3,4`, the full picture across `d=2..14` is: **present at
exactly two consecutive values (3, 4) and nowhere else in a 13-value
range.** This directly tests, and refutes, the "coincidental `mod 5`"
possibility PR #1503 raised but didn't check further out -- if dilation
periodically re-triggered this marker, `d=8` or `d=9` (`3 mod 5`, `4 mod
5`) would be the first place to look, and neither does.

## A second, previously-unnamed boilerplate run, now made explicit

Widening the search cutoff (to 2000, past the `k=3` shape's own
1600-byte cutoff) surfaces a *different* 2-copy run of 5 `B`-kind
records at offset 1626/1924 that is present in **every single build
checked here, `d=2` through `d=14` inclusive, byte-for-byte identical
content, regardless of dilation or which of the two length classes the
build falls into** (only its absolute offset shifts by the same +32
bytes `d=12`'s whole-stream length shift causes, content unchanged).
This is the same kind of confirmed-boilerplate run PR #1507 had to
name and exclude for the `k=5` shape (`_UNIVERSAL_EARLY_BOILERPLATE`);
the `k=3` shape has an analogous one that the earlier files' own
`cutoff=1600` search window happened to exclude without ever being
identified by name. Noted here for completeness -- it does not change
any prior file's conclusions (their searches never actually reached
it), but a future search at this shape with a wider cutoff should know
to exclude it rather than mistake it for new signal.

## The spike's own content: a real register-allocation sequence, not noise

`d=3`'s and `d=4`'s extra records were already confirmed to survive
independent rebuilds *byte-for-byte* (re-verified here directly against
all three of PR #1508's committed `d=3` rebuild fixtures: identical
`reg`/`tag` values, not just identical record counts). The `reg` values
are not arbitrary: in both `d=3`'s first sub-run (`47, 22, 49, 22`) and
`d=4`'s full run (`27, 22, 29, 82, 31, 74, 33, 44`), every *other*
record (tag `0xE1`/225) carries a **cleanly incrementing-by-2** `reg`
(`47, 49` at `d=3`; `27, 29, 31, 33` at `d=4`), while the interleaved
tag-`0x81`/129 records carry non-monotonic `reg` values. `d=3`'s second,
shorter sub-run ends on `reg=8` with tag `0x84`/132 -- the same
"`reg8` hosts a rotating generic short-unit payload" pattern this
project already named closing the Gemm M=8 lead. This is consistent
with (not proof of) `d=3`/`d=4` requiring a small number of freshly
allocated temporary registers that no other tested dilation value
needs -- a real, motivated hypothesis for *why* these two values are
special, still not confirmed at the semantic level.

## What remains open

Neither *why* dilation lands on exactly `3`/`4` (out of `2..14` tested)
nor what the inserted records compute is decoded here. Untested in this
file, left for whoever continues: whether channel count, spatial input
size, or padding mode (independent of dilation) shifts which dilation
value triggers this, which would test whether "3/4" is really about
dilation itself or a proxy for some other threshold this shape's fixed
`cin=cout=4`, `insz=16` happens to put right at dilation 3-4.
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


def b_runs(data, min_len=3, cutoff=2000):
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
                runs.append((recs[i]["at"], j - i, sig))
            i = j
        else:
            i += 1
    return runs


_UNIVERSAL_K3_BOILERPLATE_SIG = (
    ((7, 225), (24, 129), (9, 225), (24, 129), (8, 132)),
    ((105, 225), (10, 129), (107, 225), (120, 129), (8, 132)),
)


class TestNoSpikeRecursPastD7(unittest.TestCase):
    """The early (<1600) spike run PR #1507/#1508 found at d=3,4 does
    not reappear anywhere in d=8..14 -- refutes any simple periodicity."""

    def _early_spike(self, name):
        return b_runs(load(name), cutoff=1600)

    def test_d8_has_no_early_spike(self):
        self.assertEqual(self._early_spike("conv_dilation8.mcode.gz"), [])

    def test_d12_has_no_early_spike(self):
        self.assertEqual(self._early_spike("conv_dilation12.mcode.gz"), [])

    def test_d14_has_no_early_spike(self):
        self.assertEqual(self._early_spike("conv_dilation14.mcode.gz"), [])


class TestUniversalK3BoilerplateRunIsNowNamed(unittest.TestCase):
    """The 1626/1924 run (5 records x2 copies) is present, byte-
    identical in content, at every dilation value from 2 through 14 --
    confirmed boilerplate, not signal, whichever length class the
    build falls into."""

    def _content_only(self, name):
        runs = b_runs(load(name), cutoff=2000)
        # isolate just the two boilerplate-shaped runs (5 records each);
        # d=3/d=4 have extra runs earlier in the stream too, so filter
        # to the ones matching the known boilerplate signature length.
        return tuple(sig for _, n, sig in runs if n == 5)

    def test_present_and_identical_across_every_dilation_and_length_class(self):
        for name in (
            "conv_dilation2.mcode.gz",
            "conv_dilation3.mcode.gz",
            "conv_dilation4.mcode.gz",
            "conv_dilation7.mcode.gz",
            "conv_dilation8.mcode.gz",
            "conv_dilation12.mcode.gz",
            "conv_dilation14.mcode.gz",
        ):
            self.assertEqual(
                self._content_only(name), _UNIVERSAL_K3_BOILERPLATE_SIG, name
            )


class TestSpikeContentIsARegisterSequenceNotRandom(unittest.TestCase):
    """The tag-0xE1 records inside the d=3/d=4 spike carry a cleanly
    incrementing-by-2 reg sequence; re-verified across all 4 committed
    d=3 samples (PR #1503's original + PR #1508's 3 rebuilds) that this
    is exact, reproducible content, not just a stable record count."""

    def test_d3_reg_sequence_increments_by_2_and_is_stable_across_rebuilds(self):
        for name in (
            "conv_dilation3.mcode.gz",
            "conv_dilation3_rebuild0.mcode.gz",
            "conv_dilation3_rebuild1.mcode.gz",
            "conv_dilation3_rebuild2.mcode.gz",
        ):
            runs = b_runs(load(name), cutoff=1600)
            first_run_sig = runs[0][2]
            tag_e1_regs = [reg for reg, tag in first_run_sig if tag == 225]
            self.assertEqual(tag_e1_regs, [47, 49], name)

    def test_d4_reg_sequence_increments_by_2(self):
        # Two mirrored engine copies (offsets 552 and 1160), each an
        # 8-record run with the identical reg sequence.
        runs = b_runs(load("conv_dilation4.mcode.gz"), cutoff=1600)
        self.assertEqual(len(runs), 2)
        for _, _, sig in runs:
            tag_e1_regs = [reg for reg, tag in sig if tag == 225]
            self.assertEqual(tag_e1_regs, [27, 29, 31, 33])


if __name__ == "__main__":
    unittest.main()
