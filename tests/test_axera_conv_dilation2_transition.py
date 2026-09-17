"""Continues `tests/test_axera_conv_binary_cluster_other_shape.py` (PR
#1597)'s own explicitly flagged gap: that file found Conv's binary-
cluster switch (`tests/test_axera_conv_reg60_mechanism.py`, PR #1586,
originally decoded at `Conv(k=3, dilation=3, pad=3, cin=4, cout=4,
insz=16)`) is completely ABSENT at `dilation=1, pad=1` (same output
shape), and separately noticed -- but did not chase -- that the
`reg=8` pool group's own slot/register-label structure ALSO differs
between those two shapes (dilation=1: clean 4-of-4 permutation, like
Mul/MatMul; dilation=3: 3-of-4, variable-register-label, per
`tests/test_axera_conv_reg8_reg60_noise_source.py` PR #1580). PR #1597
explicitly named `dilation=2` as the missing data point needed to tell
"a clean `dilation>=some threshold`" apart from "`dilation=3`-specific."

## Method

`Conv(k=3, dilation=2, pad=2, cin=4, cout=4, insz=16)` already has
**four independent fixtures committed from earlier in this session's
own work** (`conv_dilation2.mcode.gz`, `conv_dilation2_rebuild.mcode.gz`,
`conv_dilation2_rebuild0.mcode.gz`, `conv_dilation2_rebuild1.mcode.gz`
-- built via `tests/test_axera_mcode_structure.py`'s own
`_dilation_conv_model(2, 2)` + `_build_and_get_mcode_bytes` helpers,
the same `RandomState(0)`-seeded convention every other dilation
fixture in this project uses, confirmed by cross-referencing
`tests/test_axera_conv_dilation_b_run_is_reliable.py`'s own docstring).
This file adds **8 more fresh, independent rebuilds**
(`conv_dilation2_r{0..7}.mcode.gz`) via the identical
`_dilation_conv_model`/`_build_and_get_mcode_bytes` pattern (built with
`pulsar2:7.0-lite`, the only image loaded in this worktree's Docker
daemon; matching `pulsar2:6.0-lite`-built fixtures within the same
noise floor other PRs in this session already established for this
specific shape family), for **12 total samples** -- the largest sample
size any single dilation value has gotten in this project's own
binary-cluster investigation.

All 12 decode to the identical 3,528-byte length -- the same length PR
#1586's own `dilation=3` fixtures and PR #1597's own `dilation=1`
fixtures share, confirming the pad/dilation pairing keeps output shape
(and total serialized length) constant across all three dilation
values, isolating dilation's own effect.

## Finding 1: the binary-cluster switch IS present at `dilation=2` --
## the boundary is between 1 and 2, not `dilation=3`-specific

`reg=60` splits **2-vs-10** across the 12 samples:
`conv_dilation2_rebuild0.mcode.gz` and `conv_dilation2_rebuild1.mcode.gz`
read `0x7f`; the other 10 (including both pre-existing
`conv_dilation2`/`conv_dilation2_rebuild` and all 8 fresh `_r{i}`
samples) read `0x7e`. Critically, **this variation is confirmed within
the 4 pre-existing fixtures ALONE** (`conv_dilation2`/`_rebuild` read
`0x7e`, `_rebuild0`/`_rebuild1` read `0x7f`) -- ruling out any
provenance confound between the pre-existing batch and this file's own
8 fresh builds, the same discipline this project's whole rebuild-
stability line of work is built on.

Running PR #1586's exact brute-force method (the union of every
pairwise byte-diff between the two `reg=60`-defined groups) finds
**exactly 28 differing offsets -- the same cluster SIZE as `dilation=3`**,
and the records they belong to are the *same* ones: `reg=60` (2
copies), `reg=54`, `reg=232`, three `verb=161,bank=15` float-operand
records, and `reg=224` (4 copies). 26 of the 28 offsets sit at the
*exact same absolute byte position* PR #1586 found at `dilation=3`
(`2161`, `2190`-`2216`, `2550`, `2561`-`2584`); only `reg=60`'s own two
record offsets shift by exactly `-2` bytes (`685`/`1293` here vs.
`687`/`1295` there, using PR #1586's own payload-byte-relative
numbering) -- an ordinary, unrelated small reflow upstream of `reg=60`'s
own records between the two shapes, not a second phenomenon.

**One of the cluster's two computed float32 values is byte-identical
across `dilation=2` and `dilation=3`**: the `verb=161,bank=15` record's
operand reads `37 3f 07 42` (`33.811733`) in the "common" group and
`43 07 ff 42` (`127.514183`) in the "rare" group at `dilation=2` --
literally the same two float32 values PR #1586 found at `dilation=3`,
not just the same magnitude class. (`reg=224`'s own float pair differs
in its exact value between the two dilations -- `0.01565`/`0.01085`
here vs. PR #1586's `0.01459`/`0.00980` -- consistent with a real,
shape-dependent per-channel statistic, not a fixed marker; this file
does not chase why only one of the two float positions is dilation-
invariant.)

## Finding 2: the `reg=8` group's own form also matches `dilation=3`,
## not `dilation=1`

At `dilation=2`, every one of the 12 samples' `reg=8` group is anchored
at the `reg=170`-payload-`\\x12` record (PR #1580's own `dilation=3`
anchor), not the `verb=162,bank=0,field=0` anchor PR #1597 found at
`dilation=1`. The group is a **3-of-4, variable-register-label**
structure in every sample (never a clean 4-of-4 permutation): register
labels on the 2nd/3rd slots draw from `{8, 242, 176}`, exactly PR
#1580's own decoded pool of alternate labels. **`reg=172`'s tag is
`134` if and only if the first slot's payload is the short form
(class `P4`) -- PR #1580's own exact biconditional -- holds with zero
exceptions across all 12 samples.**

## What this pins down

Both of PR #1597's flagged questions resolve the same way, together:
the transition for BOTH the binary-cluster switch and the `reg=8`
group's own slot structure sits precisely between `dilation=1`
(absent/clean-4-of-4) and `dilation=2` (present/3-of-4-variable-label,
identical in kind to `dilation=3`) -- not specific to `dilation=3`
alone. This is consistent with a real dilation-triggered code path
change at `dilation>=2` (this project's own established dilation-
trigger precedent -- see `scripts/axera/README.md`'s Conv dilation
material for other fields that activate past a threshold, though this
file's own boundary, `1` vs `2`, is a different exact value from those
already-known triggers, not a re-derivation of them) rather than one
specific shape's own idiosyncratic tiling decision. The root TRIGGER
for which of the two states a given `dilation>=2` build lands on
remains exactly as open as PR #1586 left it -- this file only pins
down *which shapes* the phenomenon appears at, not *why* a given build
picks one state or the other.
"""

import gzip
import itertools
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

OLD_BATCH = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
    "conv_dilation2_rebuild0.mcode.gz",
    "conv_dilation2_rebuild1.mcode.gz",
]
FRESH_BATCH = [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]
ALL_NAMES = OLD_BATCH + FRESH_BATCH

GROUP_COMMON = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
] + FRESH_BATCH
GROUP_RARE = ["conv_dilation2_rebuild0.mcode.gz", "conv_dilation2_rebuild1.mcode.gz"]

CLASS_OF = {b"#\x00 ": "P1", b"#\x00\x10": "P2", b"#\x00@": "P3", b"0": "P4"}

# Recomputed from scratch in TestBinaryClusterMatchesDilation3sCluster
# below -- not trusted from the module docstring's own claim.
KNOWN_CLUSTER_OFFSETS = frozenset(
    [
        685,
        1293,
        2161,
        2190,
        2191,
        2192,
        2198,
        2199,
        2200,
        2206,
        2207,
        2208,
        2214,
        2215,
        2216,
        2550,
        2561,
        2562,
        2563,
        2568,
        2569,
        2570,
        2575,
        2576,
        2577,
        2582,
        2583,
        2584,
    ]
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg60_value(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 60 and r.get("tag") == 131
    ]
    assert len(hits) >= 1, hits
    return hits[0]["payload"][-1]


def reg8_group(recs):
    """Returns (reg172_tag, [(reg, class), ...]) for the reg=170-anchored
    3-of-4 pool group -- PR #1580's own Conv dilation=3 mechanism."""
    r170 = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 170
        and r.get("payload") == b"\x12"
        and 800 <= r.get("at", 0) <= 900
    ]
    assert len(r170) == 1, r170
    anchor = r170[0]["at"]
    r172 = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 172 and r.get("at") == anchor + 4
    ]
    assert len(r172) == 1, r172
    group = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("payload") in CLASS_OF
        and r.get("at") is not None
        and anchor + 8 <= r["at"] <= anchor + 30
    ]
    slots = [(r["reg"], CLASS_OF[r["payload"]]) for r in group]
    return r172[0]["tag"], slots


class TestAllTwelveSamplesDecodeCleanly(unittest.TestCase):
    def test_3528_bytes_and_zero_check_errors(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 3528, n)
            self.assertEqual(mcode.check(load(n)), [], n)


class TestReg60VariesAtDilation2(unittest.TestCase):
    """2-vs-10 split, confirmed a real variation (not a batch-provenance
    artifact): the pre-existing 4-fixture batch alone already shows
    both values."""

    def test_within_old_batch_alone_both_values_appear(self):
        vals = {reg60_value(decode(n)) for n in OLD_BATCH}
        self.assertEqual(vals, {0x7E, 0x7F})

    def test_group_common_is_0x7e_group_rare_is_0x7f(self):
        for n in GROUP_COMMON:
            self.assertEqual(reg60_value(decode(n)), 0x7E, n)
        for n in GROUP_RARE:
            self.assertEqual(reg60_value(decode(n)), 0x7F, n)


class TestBinaryClusterMatchesDilation3sCluster(unittest.TestCase):
    """PR #1586's exact brute-force method, applied fresh here: the
    union of every pairwise byte-diff between the two reg=60-defined
    groups. Recomputes the cluster from scratch (not trusted from the
    module docstring) and confirms it is exactly 28 offsets, matching
    KNOWN_CLUSTER_OFFSETS above -- the same cluster SIZE PR #1586 found
    at dilation=3, and (except for reg=60's own 2-byte-shifted copies)
    the same absolute offsets."""

    def test_exactly_28_cluster_offsets_matching_known_set(self):
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        for n in ALL_NAMES:
            self.assertEqual(len(datas[n]), length, n)

        found = set()
        for i in range(length):
            vals_common = {datas[n][i] for n in GROUP_COMMON}
            vals_rare = {datas[n][i] for n in GROUP_RARE}
            if (
                len(vals_common) == 1
                and len(vals_rare) == 1
                and vals_common != vals_rare
            ):
                found.add(i)

        self.assertEqual(found, KNOWN_CLUSTER_OFFSETS)
        self.assertEqual(len(found), 28)

    def test_no_diffs_within_group_common_alone(self):
        """A sanity control: pairwise diffs *within* the 10-sample
        common group (excluding the 2-sample rare group) should be
        confined to the already-known reg=8 pool window, never the
        28-offset cluster -- confirming the cluster genuinely tracks
        the reg=60 split, not generic build-to-build noise."""
        datas = {n: load(n) for n in GROUP_COMMON}
        cluster_diffs = set()
        for a, b in itertools.combinations(GROUP_COMMON, 2):
            da, db = datas[a], datas[b]
            for i in KNOWN_CLUSTER_OFFSETS:
                if da[i] != db[i]:
                    cluster_diffs.add(i)
        self.assertEqual(cluster_diffs, set())


class TestOneFloatValueIsByteIdenticalToDilation3(unittest.TestCase):
    """The verb=161,bank=15 record's float32 operand is literally the
    same two values (33.811733 / 127.514183) PR #1586 found at
    dilation=3 -- not just the same magnitude class."""

    def test_common_group_float_matches_dilation3(self):
        recs = decode("conv_dilation2.mcode.gz")
        hits = [
            r
            for r in recs
            if r["kind"] == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("field") == 96
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["operand"], b"7?\x07B")

    def test_rare_group_float_matches_dilation3(self):
        recs = decode("conv_dilation2_rebuild0.mcode.gz")
        hits = [
            r
            for r in recs
            if r["kind"] == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("field") == 96
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["operand"], b"C\x07\xffB")


class TestReg8GroupFormMatchesDilation3NotDilation1(unittest.TestCase):
    """At dilation=2, every one of the 12 samples' reg=8 group is the
    reg=170-anchored, 3-of-4, variable-register-label form PR #1580
    decoded at dilation=3 -- never dilation=1's clean 4-of-4
    verb=162-anchored permutation (PR #1597)."""

    def test_all_twelve_samples_show_the_three_of_four_form(self):
        for n in ALL_NAMES:
            recs = decode(n)
            _, slots = reg8_group(recs)
            self.assertEqual(len(slots), 3, n)
            classes = {c for _, c in slots}
            self.assertEqual(len(classes), 3, n)

    def test_register_labels_vary_matching_pr1580s_known_pool(self):
        seen_labels = set()
        for n in ALL_NAMES:
            recs = decode(n)
            _, slots = reg8_group(recs)
            seen_labels |= {r for r, _ in slots[1:]}
        self.assertEqual(seen_labels, {8, 242, 176})


class TestReg172BiconditionalHoldsAtDilation2Too(unittest.TestCase):
    """PR #1580's own exact biconditional (reg=172's tag is 134 iff
    slot 1's payload is the short form P4) holds with zero exceptions
    across all 12 dilation=2 samples."""

    def test_biconditional_holds_for_every_sample(self):
        for n in ALL_NAMES:
            recs = decode(n)
            tag, slots = reg8_group(recs)
            slot1_is_short = slots[0][1] == "P4"
            self.assertEqual(
                tag == 134,
                slot1_is_short,
                f"{n}: slot1={slots[0]} but reg172_tag={tag}",
            )


if __name__ == "__main__":
    unittest.main()
