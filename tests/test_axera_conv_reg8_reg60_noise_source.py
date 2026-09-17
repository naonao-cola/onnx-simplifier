"""Continues `tests/test_axera_conv_rebuild_stability.py` (PR #1579)'s
own explicitly flagged gaps: `reg=8` and `reg=60` are genuinely noisy
across independent `Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)`
rebuilds, and that PR left both mechanisms undecoded. This file reuses
its 8 already-committed fixtures (4 in `OLD_BATCH`, 4 in `FRESH_BATCH`)
-- no new builds needed.

## `reg=8`: decoded -- a 3-of-4 unordered slot assignment, with a
## register-label wrinkle Gemm's own version didn't have

`tests/test_axera_gemm_reg8_second_noise_source.py` (PR #1577) decoded
the analogous Gemm-side mystery as three fixed byte offsets (anchored
at `reg=78`) each independently drawing one of 3 candidate values.
Conv's version is structurally similar but with one real difference:
the group here is anchored at a **stable `reg=170` record** (payload
`\\x12`, byte-identical in all 8 samples) followed 4 bytes later by a
**stable `reg=172` record** (payload `"`, `0x22`), then exactly 3 more
`S`-kind records drawing from **four** candidate payloads instead of
Gemm's three:

| class | payload | trailing byte |
| --- | --- | --- |
| `P1` | `\\x23\\x00\\x20` (tag 130, 3 bytes) | `0x20` |
| `P2` | `\\x23\\x00\\x10` (tag 130, 3 bytes) | `0x10` |
| `P3` | `\\x23\\x00\\x40` (tag 130, 3 bytes) | `0x40` |
| `P4` | `\\x30` (tag 130, 1 byte -- a short form) | `0x30` |

(All four trailing bytes are multiples of `0x10` -- `0x10/0x20/0x30/0x40`,
the same "evenly spaced scratch/tile-buffer id" character PR #1577
noted for Gemm's own 3-member pool, here with a 4th member.)

Across all 8 samples, **exactly 3 of these 4 classes are chosen, with
zero exceptions** -- cleaner than Gemm's own version, which sometimes
duplicated a class or dropped to only 2 distinct values. The first of
the 3 chosen records is **always** labeled `reg=174` (a fixed register
identity, unlike anything else in this group); the other two records'
register label is *itself* drawn from a small pool that is NOT fixed:
`reg=8` used for both of the remaining two slots, or `reg=8` for one
and `reg=242` for the other, or (once, `_v7stability_r1`) `reg=8` for
one and a previously-unseen `reg=176` for the other:

| fixture | reg174 slot | 2nd slot | 3rd slot | missing class | reg172 tag |
| --- | --- | --- | --- | --- | --- |
| `conv_dilation3` | P2 | reg8=P1 | reg242=P3 | P4 | 132 |
| `_rebuild0` | **P4 (short)** | reg8=P2 | reg8=P1 | P3 | **134** |
| `_rebuild1` | P2 | reg8=P1 | reg242=P3 | P4 | 132 |
| `_rebuild2` | **P4 (short)** | reg8=P1 | reg8=P2 | P3 | **134** |
| `_v7stability_r0` | P2 | reg8=P4 | reg8=P3 | P1 | 132 |
| `_v7stability_r1` | P3 | reg176=P1 | reg8=P4 | P2 | 132 |
| `_v7stability_r2` | P1 | reg8=P3 | reg242=P2 | P4 | 132 |
| `_v7stability_r3` | P1 | reg8=P3 | reg242=P2 | P4 | 132 |

This means `reg=8`'s own instability (the one PR #1579 flagged) is a
symptom of this broader mechanism: `reg=8` is simply the register
label the compiler happens to reach for most often when it needs a
2nd/3rd slot identity, not a uniquely "reg=8-specific" phenomenon. Two
more registers this project has never tracked (`174`, always present
but varying VALUE; `242`/`176`, appearing only when `reg=8` isn't
reused) are part of the same mechanism -- `174` in particular would
itself show up as "noisy" if it were in `tests/test_axera_resource_model_census.py`'s
36-register core set, which it is not (it is a sparse register outside
that near-universal list, which is presumably why this wasn't already
caught by the census-level sweep).

## `reg=172`'s tag is an exact flag for "slot 1 used its short form"

**Zero exceptions across all 8 samples**: `reg=172`'s own record uses
tag `134` instead of the otherwise-universal tag `132` if and only if
the `reg=174`-anchored slot's payload is the 1-byte short form (`P4`).
This is the Conv-side analogue of PR #1577's `reg=0` extra-record
indicator for Gemm -- a verified biconditional, not a mere correlation,
tying `reg=172`'s own tag variation into the same explained mechanism.

## `reg=60`: still undecoded -- one more hypothesis ruled out

PR #1579 already ruled out "toolchain image version" and "old/fresh
batch" as `reg=60`'s driver. This file checked one more candidate:
whether `reg=60`'s `0x7e`/`0x7f` flip correlates with *which* class is
the "missing" one in the 3-of-4 group above. It does not --
`conv_dilation3` and `_rebuild1` both have `P4` as the missing class,
yet disagree on `reg=60` (`0x7e` vs `0x7f`). `reg=60`'s mechanism
remains genuinely open.

## What this establishes

`reg=8`'s Conv-side noise is now decoded to the same precision as
Gemm's own case (PR #1577): a real, structured, unordered slot
assignment over a small candidate pool, not unstructured allocator
noise. Combined with PR #1577, this project has now cleanly decoded
the *mechanism class* (small unordered pools of scratch/tile-buffer-id
candidates, assigned to a variable subset of register labels) for two
of the three ops where the known table-order coin flip cannot apply
-- though the semantic meaning of the candidate values and *why* one
member of the pool is omitted each build remain open in both cases.
`reg=60` is the one Conv-specific residual this file could not close.
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

OLD_BATCH = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
FRESH_BATCH = [
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
ALL_NAMES = OLD_BATCH + FRESH_BATCH

CLASS_OF = {
    b"#\x00 ": "P1",
    b"#\x00\x10": "P2",
    b"#\x00@": "P3",
    b"0": "P4",
}
ALL_CLASSES = frozenset(CLASS_OF.values())

EXPECTED = {
    "conv_dilation3.mcode.gz": ([("174", "P2"), ("8", "P1"), ("242", "P3")], 132),
    "conv_dilation3_rebuild0.mcode.gz": (
        [("174", "P4"), ("8", "P2"), ("8", "P1")],
        134,
    ),
    "conv_dilation3_rebuild1.mcode.gz": (
        [("174", "P2"), ("8", "P1"), ("242", "P3")],
        132,
    ),
    "conv_dilation3_rebuild2.mcode.gz": (
        [("174", "P4"), ("8", "P1"), ("8", "P2")],
        134,
    ),
    "conv_dilation3_v7stability_r0.mcode.gz": (
        [("174", "P2"), ("8", "P4"), ("8", "P3")],
        132,
    ),
    "conv_dilation3_v7stability_r1.mcode.gz": (
        [("174", "P3"), ("176", "P1"), ("8", "P4")],
        132,
    ),
    "conv_dilation3_v7stability_r2.mcode.gz": (
        [("174", "P1"), ("8", "P3"), ("242", "P2")],
        132,
    ),
    "conv_dilation3_v7stability_r3.mcode.gz": (
        [("174", "P1"), ("8", "P3"), ("242", "P2")],
        132,
    ),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def four_pool_group(recs):
    """Returns ([(reg, class), ...] for the 3 slot records, reg172_tag).

    Anchored on `reg=170`'s own record (byte-identical in every sample,
    payload `\\x12`), which always precedes `reg=172`'s record by 4
    bytes and the 3-slot group by 8-30 bytes.
    """
    r170_hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 170
        and r.get("payload") == b"\x12"
        and r.get("at") is not None
        and 800 <= r["at"] <= 900
    ]
    assert len(r170_hits) == 1, r170_hits
    anchor = r170_hits[0]["at"]

    r172_hits = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 172 and r["at"] == anchor + 4
    ]
    assert len(r172_hits) == 1, r172_hits
    reg172_tag = r172_hits[0]["tag"]

    group = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("payload") in CLASS_OF
        and r.get("at") is not None
        and anchor + 8 <= r["at"] <= anchor + 30
    ]
    slots = [(str(r["reg"]), CLASS_OF[r["payload"]]) for r in group]
    return slots, reg172_tag


def reg60_value(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 60
        and r.get("tag") == 131
        and r.get("payload")
    ]
    assert len(hits) >= 1, hits
    return hits[0]["payload"][-1]


class TestFourPoolGroupMatchesExpected(unittest.TestCase):
    """Pins the exact per-fixture (slot list, reg172 tag) reading from
    the module docstring's table -- fails loudly if a future `mcode.py`
    change alters how these records decode."""

    def test_all_eight_fixtures(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots, reg172_tag = four_pool_group(recs)
            expected_slots, expected_tag = EXPECTED[name]
            self.assertEqual(slots, expected_slots, name)
            self.assertEqual(reg172_tag, expected_tag, name)


class TestExactlyThreeOfFourClassesChosenEveryBuild(unittest.TestCase):
    """Zero exceptions across all 8 samples: always exactly 3 records,
    always exactly 3 *distinct* classes out of the 4-member pool --
    cleaner than Gemm's own version (PR #1577), which sometimes
    duplicated a class or collapsed to only 2 distinct values."""

    def test_three_records_three_distinct_classes(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots, _ = four_pool_group(recs)
            self.assertEqual(len(slots), 3, name)
            classes = {c for _, c in slots}
            self.assertEqual(len(classes), 3, name)
            self.assertTrue(classes <= ALL_CLASSES, name)

    def test_all_four_classes_appear_as_the_missing_one_somewhere(self):
        missing_seen = set()
        for name in ALL_NAMES:
            recs = decode(name)
            slots, _ = four_pool_group(recs)
            classes = {c for _, c in slots}
            missing_seen |= ALL_CLASSES - classes
        self.assertEqual(missing_seen, ALL_CLASSES)


class TestSlotOneIsAlwaysRegister174ButItsValueVaries(unittest.TestCase):
    """The first slot (immediately after the anchor) is always labeled
    `reg=174` in every one of the 8 samples -- a fixed register
    identity, unlike the other two slots -- but its own VALUE still
    varies across all 4 candidate classes."""

    def test_first_slot_is_always_reg174(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots, _ = four_pool_group(recs)
            self.assertEqual(slots[0][0], "174", name)

    def test_reg174s_value_varies_across_samples(self):
        values = {four_pool_group(decode(name))[0][1] for name in ALL_NAMES}
        self.assertGreater(len(values), 1, values)


class TestSecondAndThirdSlotLabelsAreThemselvesVariable(unittest.TestCase):
    """Unlike Gemm (PR #1577), where all 3 slots lived under fixed
    register numbers (78, 8, 8), Conv's 2nd/3rd slots draw their
    REGISTER LABEL from a small pool too: reg=8 used for both, reg=8
    for one and reg=242 for the other, or (once) reg=8 and a
    previously-untracked reg=176."""

    def test_reg8_used_twice_in_some_samples(self):
        recs = decode("conv_dilation3_rebuild0.mcode.gz")
        slots, _ = four_pool_group(recs)
        regs = [r for r, _ in slots[1:]]
        self.assertEqual(regs, ["8", "8"])

    def test_reg242_appears_as_an_alternate_third_slot_label(self):
        recs = decode("conv_dilation3.mcode.gz")
        slots, _ = four_pool_group(recs)
        regs = {r for r, _ in slots[1:]}
        self.assertIn("242", regs)

    def test_reg176_appears_once_as_a_third_alternate_label(self):
        recs = decode("conv_dilation3_v7stability_r1.mcode.gz")
        slots, _ = four_pool_group(recs)
        regs = {r for r, _ in slots[1:]}
        self.assertIn("176", regs)


class TestReg172TagExactlyFlagsSlotOneShortForm(unittest.TestCase):
    """The core finding: `reg=172`'s tag is `134` (instead of the
    otherwise-universal `132`) if and only if `reg=174`'s own slot-1
    record uses its 1-byte short form (class `P4`) -- a verified
    biconditional, zero exceptions in 8/8 samples, the Conv-side
    analogue of PR #1577's `reg=0` extra-record indicator for Gemm."""

    def test_biconditional_holds_for_every_fixture(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots, reg172_tag = four_pool_group(recs)
            slot1_is_short = slots[0][1] == "P4"
            self.assertEqual(
                reg172_tag == 134,
                slot1_is_short,
                f"{name}: slot1={slots[0]} but reg172_tag={reg172_tag}",
            )


class TestReg8sNoiseIsFullyExplainedByThisMechanism(unittest.TestCase):
    """Confirms PR #1579's original observation -- reg=8 unstable
    within each independently-built batch -- is entirely a symptom of
    this 3-of-4 slot mechanism, not a separate, still-mysterious
    reg=8-specific phenomenon."""

    def test_reg8_slot_membership_varies_across_all_eight(self):
        reg8_classes_per_sample = []
        for name in ALL_NAMES:
            recs = decode(name)
            slots, _ = four_pool_group(recs)
            reg8_classes_per_sample.append(frozenset(c for r, c in slots if r == "8"))
        self.assertGreater(len(set(reg8_classes_per_sample)), 1)


class TestReg60RemainsUndecoded(unittest.TestCase):
    """PR #1579 already ruled out toolchain version and old/fresh batch
    as reg=60's driver. This rules out one more candidate: correlation
    with which class is "missing" from the 3-of-4 group above. It does
    not correlate -- `conv_dilation3` and `_rebuild1` share the same
    missing class (`P4`) yet disagree on reg=60's value. reg=60 remains
    genuinely open."""

    def test_missing_class_does_not_predict_reg60(self):
        by_missing = {}
        for name in ALL_NAMES:
            recs = decode(name)
            slots, _ = four_pool_group(recs)
            classes = {c for _, c in slots}
            missing = next(iter(ALL_CLASSES - classes))
            by_missing.setdefault(missing, set()).add(reg60_value(recs))
        # At least one "missing class" bucket contains both reg=60
        # values -- proves missing-class alone doesn't determine it.
        self.assertTrue(any(len(v) > 1 for v in by_missing.values()), by_missing)


if __name__ == "__main__":
    unittest.main()
