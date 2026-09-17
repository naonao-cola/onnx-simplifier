"""Continues `tests/test_axera_conv_binary_cluster_synthesis.py` (PR
#1603)'s own partial characterization: that file spot-checked
`scripts/axera/README.md`'s "second matching pair" observation (a
3-byte value repeated identically at offsets 552 and 1160) at
`dilation=2` and `dilation=4`, confirmed the paired-matching STRUCTURE
exists at both, and confirmed the two dilations resolve to different
values -- but explicitly did not decode what either instance actually
represents, or how (if at all) it relates to this project's two other
already-known dilation-triggered Conv mechanisms: the 28-offset
"binary path switch" (`tests/test_axera_conv_reg60_mechanism.py`, PR
#1586) and the "extra B-run" trigger
(`tests/test_axera_conv_dilation_b_run_is_reliable.py`).

## The README's own full passage, read in context

`scripts/axera/README.md`'s "Extending the periodic field..." section
(~line 1406) describes TWO separate things at this location, easy to
conflate:

1. The already-decoded, dilation-dependent "offset 2561" field
   (`reg=224` -- see PR #1603's own `TestReg224IsTheReadmesAlreadyKnownOffset2561Field`,
   unrelated to this file).
2. **"A new, real, threshold-like effect"**: comparing `d=4`/`d=6`
   against the `d=2` baseline (but explicitly *not* `d=3` vs `d=2`)
   surfaces "a second pair of matching 3-byte runs at entirely
   different offsets (552/1160 for `d=4`; 754/1362 for `d=6`) that
   don't exist at all in the `d=3`-vs-`d=2` comparison" -- flagged in
   the README as "not chased further here."

## Answer: this "second pair" is not a new field at all -- it is the
## ALREADY-KNOWN "extra B-run" trigger, just relocated to a different
## absolute byte offset at `dilation=4` than at `dilation=3`

`tests/test_axera_conv_dilation_b_run_is_reliable.py`'s own
`early_b_runs()` detector finds `dilation=3`'s extra-B-run block at
offsets `(603, 4), (617, 3), (1211, 4), (1225, 3)` -- NOT at 552/1160.
Running the identical detector against `dilation=4` finds its own
extra-B-run block at exactly `(552, 8), (1160, 8)` -- the README's own
"552/1160" offsets, byte for byte. **This is why the README's own
fixed-offset diff (comparing raw bytes at 552/1160 specifically) found
"nothing" for `d=3` vs `d=2` but "something new" for `d=4` vs `d=2`: it
wasn't that `d=3` lacks this mechanism, it's that `d=3`'s own B-run
sits at a different absolute offset (603/1211) that the README's
552/1160-anchored comparison never looked at.** The mechanism itself
(an extra run of bare `B`-kind mcode records, `tests/test_axera_conv_dilation_b_run_is_reliable.py`'s
own already-established finding) is present at both `d=3` and `d=4` --
only its exact byte position moves.

Decoded directly (`mcode.decode()`, not raw bytes): `dilation=4`'s
block at offset 552 is 8 consecutive `B`-kind records --
`(tag=225,reg=27), (tag=129,reg=22), (tag=225,reg=29), (tag=129,reg=82),
(tag=225,reg=31), (tag=129,reg=74), (tag=225,reg=33), (tag=129,reg=44)`
-- and the block at offset 1160 is byte-for-byte identical to it (the
"matching pair" the README describes is this whole 8-record block
repeated twice, not a single 3-byte value). **Content-stable across
all 9 available `dilation=4` samples** (`conv_dilation4.mcode.gz` +
`tests/test_axera_conv_binary_cluster_higher_dilation.py`'s own 8
independent `_r{0..7}` rebuilds) -- zero exceptions, matching this
project's established rebuild-stability discipline.

**This does NOT match PR #1586's own 28-offset binary-cluster switch
list** (`KNOWN_CORRELATED_OFFSETS`, all in the 687-2584 range at
`dilation=3` -- 552/1160 is not among them, confirmed directly below)
-- these are three genuinely separate dilation-triggered mechanisms
co-existing in the same small model, not one mechanism described three
ways.

**A note on the leading byte, to avoid a false cross-op link**: the
`tag=225` B-kind records above happen to share the numeric value
`0xE1` with the `bank=0xE1` (225) V-kind records
`tests/test_axera_conv_dilation8_length_growth.py` (PR #1602) tied to
Gemm's own PR #1570 constant. This is very likely coincidental, not
the same field: PR #1602's finding is about a `V`-kind record's
*bank* number; this file's finding is about a `B`-kind record's *tag*
number -- different record kinds, different byte-stream role (`mcode.py`'s
own `V`/`B`/`S` kind distinction), and this B-kind `tag=225` record
carries no `field=32`/`field=48` operand pattern at all (`B`-kind
records don't have that structure). Flagged here explicitly so a
future reader does not conflate the two just because `225` appears in
both.

## What this closes, and what remains genuinely open

This resolves the README's own "not chased further" flag: the
"second matching pair" is the extra-B-run mechanism at a shifted
offset, not a third undecoded field. What remains open (unchanged by
this file, explicitly not attempted here): *why* dilation=3's own
B-run sits at 603/1211 while dilation=4's sits at 552/1160 (a
~51-byte shift, presumably from some other dilation-dependent content
earlier in the stream growing/shrinking by that amount between `d=3`
and `d=4` -- not traced here), and *what the inserted B-run records
themselves compute* (unchanged from `tests/test_axera_conv_dilation_b_run_is_reliable.py`'s
own honest "not decoded" conclusion).
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_axera_conv_dilation_b_run_is_reliable import early_b_runs  # noqa: E402
from test_axera_conv_reg60_mechanism import (  # noqa: E402
    KNOWN_CORRELATED_OFFSETS as D3_CLUSTER_OFFSETS,
)

FIX = os.path.join(_AXERA_DIR, "fixtures")

D2_NAMES = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
    "conv_dilation2_rebuild0.mcode.gz",
    "conv_dilation2_rebuild1.mcode.gz",
] + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]

D3_NAMES = ["conv_dilation3.mcode.gz"]

D4_NAMES = ["conv_dilation4.mcode.gz"] + [
    f"conv_dilation4_r{i}.mcode.gz" for i in range(8)
]

EXPECTED_D3_RUN = [(603, 4), (617, 3), (1211, 4), (1225, 3)]
EXPECTED_D4_RUN = [(552, 8), (1160, 8)]

EXPECTED_D4_BLOCK = [
    (225, 27),
    (129, 22),
    (225, 29),
    (129, 82),
    (225, 31),
    (129, 74),
    (225, 33),
    (129, 44),
]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def block_at(name, start, end):
    recs = decode(name)
    return [
        (r["tag"], r["reg"])
        for r in recs
        if r.get("at") is not None and start <= r["at"] <= end
    ]


class TestDilation3sOwnBRunIsAtADifferentOffsetThan552(unittest.TestCase):
    """The already-established dilation=3 extra-B-run
    (`tests/test_axera_conv_dilation_b_run_is_reliable.py`) sits at
    603/617/1211/1225 -- nowhere near 552/1160. This is exactly why the
    README's own fixed-offset 552/1160 comparison found "nothing" for
    d=3 vs d=2: it was looking in the wrong place for d=3's own
    (real, already-known) instance of the same mechanism."""

    def test_d3_run_matches_established_offsets(self):
        for n in D3_NAMES:
            self.assertEqual(early_b_runs(load(n)), EXPECTED_D3_RUN, n)

    def test_d3_has_nothing_at_552_1160(self):
        for n in D3_NAMES:
            recs = decode(n)
            hits = [r for r in recs if r.get("at") in (552, 1160) and r["kind"] == "B"]
            self.assertEqual(hits, [], n)


class TestDilation4sBRunIsExactlyAtTheReadmesOwn552And1160(unittest.TestCase):
    """The README's own "second matching pair" offsets (552, 1160) are
    precisely where dilation=4's own extra-B-run block sits --
    confirmed with the same `early_b_runs()` detector the already-known
    mechanism uses, not a new detection method."""

    def test_d4_run_is_exactly_552_1160(self):
        for n in D4_NAMES:
            self.assertEqual(early_b_runs(load(n)), EXPECTED_D4_RUN, n)


class TestDilation4sBlockIsContentStableAcrossAllNineSamples(unittest.TestCase):
    """The 8-record block at offset 552 (and its byte-identical copy at
    1160) is exactly the same (tag, reg) sequence in all 9 available
    dilation=4 samples (the original pre-existing fixture plus 8
    independent rebuilds from `tests/test_axera_conv_binary_cluster_higher_dilation.py`,
    PR #1601) -- zero exceptions."""

    def test_block_matches_expected_everywhere(self):
        for n in D4_NAMES:
            self.assertEqual(block_at(n, 552, 567), EXPECTED_D4_BLOCK, n)

    def test_both_copies_are_byte_identical_within_each_sample(self):
        for n in D4_NAMES:
            self.assertEqual(block_at(n, 552, 567), block_at(n, 1160, 1175), n)


class TestNotTheBinaryClusterSwitch(unittest.TestCase):
    """PR #1586's own 28-offset binary-cluster switch list is entirely
    in the 687-2584 range at dilation=3 -- 552 and 1160 are not among
    them. Confirmed directly against the actual frozenset, not assumed
    from its documented range."""

    def test_552_and_1160_not_in_known_cluster_offsets(self):
        self.assertNotIn(552, D3_CLUSTER_OFFSETS)
        self.assertNotIn(1160, D3_CLUSTER_OFFSETS)
        self.assertNotIn(687, (552, 1160))  # sanity: cluster's own lowest offset


class TestLeadingByteE1IsCoincidentalNotTheSameFieldAsGemmsConstant(unittest.TestCase):
    """The block's tag=225 (0xE1) is a B-kind record's TAG, not a
    V-kind record's BANK -- structurally different from
    `tests/test_axera_conv_dilation8_length_growth.py` (PR #1602)'s
    own bank=0xE1 field=32/48 V-records, despite sharing the numeric
    value 225. No field=32/48-style operand exists on these records at
    all (B-kind records carry no field/operand)."""

    def test_no_v_kind_bank_0xe1_records_in_this_block(self):
        for n in D4_NAMES:
            recs = decode(n)
            in_block = [
                r
                for r in recs
                if r.get("at") is not None
                and (552 <= r["at"] <= 567 or 1160 <= r["at"] <= 1175)
            ]
            self.assertTrue(all(r["kind"] == "B" for r in in_block), (n, in_block))


if __name__ == "__main__":
    unittest.main()
