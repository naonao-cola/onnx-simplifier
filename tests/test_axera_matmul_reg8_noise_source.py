"""Continues `tests/test_axera_matmul_rebuild_stability.py` (PR #1581)'s
own explicitly flagged gap: `reg=8` is unstable across independent
`MatMul(4,8,8)` rebuilds even though the known `A_offset`/`B_offset`
"table-order coin flip" cleanly explains 6 of its 7 near-universal-
register neighbors in that same build. This file decodes `reg=8`'s own
mechanism using the 10 fixtures PR #1581 already committed (8 in
`ALL_NAMES`, plus 2 pre-existing `matmul_4x8x8_rebuild_mode{A,B}.mcode.gz`
fixtures from `tests/test_axera_matmul_offset_table_coinflip.py`,
PR #1506, from a different build session/toolchain) -- no new builds
needed.

## The mechanism: a fixed 4-slot group, always a full permutation

Unlike Gemm's 3-slot pool (`tests/test_axera_gemm_reg8_second_noise_source.py`,
PR #1577, trailing bytes `0x20/0x30/0x40`) and Conv's 3-of-4-slot pool
(`tests/test_axera_conv_reg8_reg60_noise_source.py`, PR #1580, trailing
bytes `0x10/0x20/0x30/0x40`), MatMul's version is a clean **4-of-4**
permutation of the same 4-member class pool Conv uses:

| class | payload (tag 130, 3-byte) | trailing byte |
| --- | --- | --- |
| `P1` | `\\x33\\x00\\x20` | `0x20` |
| `P2` | `\\x23\\x00\\x40` | `0x40` |
| `P3` | `\\x23\\x00\\x30` | `0x30` |
| `P4` | `\\x23\\x00\\x10` | `0x10` |

The group is anchored at a **stable `verb=162, bank=0, field=0` `V`-kind
record** (operand `\\x12\\x00\\x00\\x00`, byte-identical -- same absolute
offset `345` -- in all 10 samples, regardless of `A_offset`/`B_offset`
table order). Immediately after it, at three FIXED offsets (`353`,
`361`, `369`), three more `verb=162, bank=0, field=0` `V`-kind records
each carry one of the 4 candidate classes in their operand's first 3
bytes. A fourth, fixed-offset (`378`) record -- always decoded as
`S`-kind, `reg=8`, `tag=130` -- carries the 4th class. **Zero exceptions
across all 10 samples**: the four classes assigned to these four fixed
positions are always a genuine permutation of all 4 pool members --
never a duplicate, never an omission -- unlike Gemm (which sometimes
collapsed to 2 distinct classes via a duplicate) and Conv (which always
omits exactly one class). This makes MatMul's version the cleanest of
the three op-specific pool mechanisms decoded so far.

**The register-label assignment is also fixed, unlike Conv's**: the
first three physical slots are always `V`-kind (never a register-
labeled `S` record at all), and the fourth slot is always the one and
only `S`-kind, `reg=8` record. Conv's own version (PR #1580) found the
slot->register-label mapping itself varied (`reg=8` reused for 2 slots,
or paired with `reg=242`/`reg=176`); MatMul shows no such variation --
`reg=8`'s own instability here is entirely a symptom of "this fixed
physical slot's class assignment varies," with nothing else moving.

No paired biconditional indicator (the way `reg=0`'s extra record
worked for Gemm, or `reg=172`'s tag worked for Conv) is needed or found
here, since there is no "how many distinct classes" ambiguity to flag
-- it is always exactly 4 of 4, every time.

## Cross-toolchain confirmation

The two pre-existing `matmul_4x8x8_rebuild_mode{A,B}.mcode.gz` fixtures
(a different build session/toolchain than this file's own 8 fresh
`v7stability` samples, per PR #1581's own cross-version compatibility
check) show the identical group structure at the identical byte
offsets, and also each resolve to a full 4-of-4 permutation -- the
mechanism is not an artifact of one particular build script or
toolchain version.

## What this establishes

Combined with PR #1577 (Gemm) and PR #1580 (Conv), this project has now
decoded the *mechanism class* (a small unordered pool of scratch/tile-
buffer-id candidates, assigned across a handful of fixed or near-fixed
physical slots) for `reg=8`'s residual noise in three of this project's
four main ops -- MatMul's own version being the simplest and cleanest:
fixed slot count, fixed slot labels, always a complete permutation with
no omissions or duplicates. The semantic meaning of the four candidate
values (still evenly spaced by `0x10`, still consistent with a
scratch/tile-buffer-id reading) remains open in all three cases, as
does why Mul's own `reg=8` noise (PR #1566) was fully explained by the
*different*, already-known table-order coin flip instead of this
mechanism -- whether Mul simply doesn't have this second mechanism at
all, or has it but it's masked/absorbed by the coin flip's own larger
table, is not decoded here.
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

V7_BATCH = [
    "matmul_4x8x8_v7stability_diag0.mcode.gz",
    "matmul_4x8x8_v7stability_r1.mcode.gz",
    "matmul_4x8x8_v7stability_r2.mcode.gz",
    "matmul_4x8x8_v7stability_r3.mcode.gz",
    "matmul_4x8x8_v7stability_r4.mcode.gz",
    "matmul_4x8x8_v7stability_r5.mcode.gz",
    "matmul_4x8x8_v7stability_r6.mcode.gz",
    "matmul_4x8x8_v7stability_r7.mcode.gz",
]
LEGACY_BATCH = [
    "matmul_4x8x8_rebuild_modeA.mcode.gz",
    "matmul_4x8x8_rebuild_modeB.mcode.gz",
]
ALL_NAMES = V7_BATCH + LEGACY_BATCH

CANDS = {b"\x33\x00\x20", b"\x23\x00\x40", b"\x23\x00\x30", b"\x23\x00\x10"}

ANCHOR_AT = 345
SLOT_OFFSETS = (353, 361, 369, 378)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def find_group(recs):
    """Returns {offset: (kind, class)} for the anchor-relative 4-slot
    group. Raises via assertion if the fixed structure (anchor at 345,
    exactly 3 V-kind + 1 S-kind reg=8 slot at the fixed offsets) is not
    found exactly as expected."""
    anchor_hits = [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("verb") == 162
        and r.get("field") == 0
        and r.get("bank") == 0
        and r.get("operand", b"")[:4] == b"\x12\x00\x00\x00"
        and r.get("at") == ANCHOR_AT
    ]
    assert len(anchor_hits) == 1, anchor_hits

    slots = {}
    for r in recs:
        at = r.get("at")
        if at not in SLOT_OFFSETS[:3]:
            continue
        if (
            r["kind"] == "V"
            and r.get("verb") == 162
            and r.get("field") == 0
            and r.get("bank") == 0
            and r.get("operand", b"")[:3] in CANDS
        ):
            slots[at] = ("V", r["operand"][:3])

    for r in recs:
        if (
            r["kind"] == "S"
            and r.get("reg") == 8
            and r.get("tag") == 130
            and r.get("payload") in CANDS
            and r.get("at") == SLOT_OFFSETS[3]
        ):
            slots[SLOT_OFFSETS[3]] = ("S,reg8", r["payload"])

    return slots


class TestAnchorIsStableAtAFixedOffset(unittest.TestCase):
    """The `verb=162,bank=0,field=0` operand=`12 00 00 00` anchor
    record sits at the identical absolute offset (345) in all 10
    samples, regardless of A_offset/B_offset table order -- the whole
    group's position is independent of the known coin flip."""

    def test_anchor_present_at_345_in_every_sample(self):
        for name in ALL_NAMES:
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 162
                and r.get("operand", b"")[:4] == b"\x12\x00\x00\x00"
            ]
            self.assertEqual(len(hits), 1, name)
            self.assertEqual(hits[0]["at"], ANCHOR_AT, name)


class TestFourFixedSlotsAlwaysFormACompletePermutation(unittest.TestCase):
    """The core finding, zero exceptions across all 10 samples (8 fresh
    + 2 cross-toolchain): the four fixed-offset slots always carry all
    4 distinct classes -- never a duplicate, never an omission. Cleaner
    than both Gemm (sometimes collapses to 2 distinct via a duplicate)
    and Conv (always omits exactly one class)."""

    def test_all_four_slots_present_with_all_four_distinct_classes(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots = find_group(recs)
            self.assertEqual(set(slots.keys()), set(SLOT_OFFSETS), name)
            classes = [c for _, c in slots.values()]
            self.assertEqual(len(classes), 4, name)
            self.assertEqual(set(classes), CANDS, name)
            self.assertEqual(len(set(classes)), 4, f"{name}: expected no duplicates")


class TestSlotLabelsAreFixedUnlikeConv(unittest.TestCase):
    """Unlike Conv (PR #1580), where the register LABEL on 2 of 3
    slots was itself variable (reg=8, reg=242, or reg=176), MatMul's
    slot->label mapping never varies: the first three physical
    positions are always V-kind (never any S-kind register), and the
    fourth position is always the one and only S-kind reg=8 record."""

    def test_first_three_slots_always_v_kind(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots = find_group(recs)
            for off in SLOT_OFFSETS[:3]:
                self.assertEqual(slots[off][0], "V", f"{name} offset={off}")

    def test_fourth_slot_always_s_kind_reg8(self):
        for name in ALL_NAMES:
            recs = decode(name)
            slots = find_group(recs)
            self.assertEqual(slots[SLOT_OFFSETS[3]][0], "S,reg8", name)


class TestClassAssignmentVariesFreelyAcrossBuilds(unittest.TestCase):
    """Confirms this is genuine per-build non-determinism, not a fixed
    assignment that happens to look variable: at least 3 distinct
    permutations are observed among the 8 fresh v7stability samples
    alone."""

    def test_at_least_three_distinct_permutations_observed(self):
        perms = set()
        for name in V7_BATCH:
            recs = decode(name)
            slots = find_group(recs)
            perms.add(tuple(slots[off][1] for off in SLOT_OFFSETS))
        self.assertGreaterEqual(len(perms), 3, perms)

    def test_reg8_slot_alone_takes_all_four_values_across_the_ten_samples(self):
        values = set()
        for name in ALL_NAMES:
            recs = decode(name)
            slots = find_group(recs)
            values.add(slots[SLOT_OFFSETS[3]][1])
        self.assertEqual(values, CANDS)


class TestCrossToolchainFixturesConfirmTheSameStructure(unittest.TestCase):
    """The two pre-existing matmul_4x8x8_rebuild_mode{A,B} fixtures --
    a different build session/toolchain per PR #1581's own compatibility
    check -- show the identical group structure at the identical
    offsets, and also resolve to full 4-of-4 permutations. Not an
    artifact of one particular build script."""

    def test_legacy_fixtures_match_the_structure(self):
        for name in LEGACY_BATCH:
            recs = decode(name)
            slots = find_group(recs)
            self.assertEqual(set(slots.keys()), set(SLOT_OFFSETS), name)
            classes = [c for _, c in slots.values()]
            self.assertEqual(set(classes), CANDS, name)


if __name__ == "__main__":
    unittest.main()
