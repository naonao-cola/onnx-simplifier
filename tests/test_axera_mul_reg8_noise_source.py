"""Continues `tests/test_axera_reg8_cross_op_synthesis.py` (PR #1582)'s
own explicitly flagged gap: "whether Mul simply doesn't have this
second mechanism at all, or has it but it's masked/absorbed by the
[table-order] coin flip's own larger table, is not decoded here."
This file decodes it, using the same 8 already-committed
`mul_1x8_universal_stability_r{0..7}.mcode.gz` fixtures
`tests/test_axera_universal_bank_reg_stability.py` (PR #1566) built --
no new builds needed.

## The mechanism: byte-for-byte the SAME physical structure as MatMul's

Grouping the 8 samples by `x_offset`/`y_offset` table order first
(5 samples `(y,x)`-ordered, 3 `(x,y)`-ordered, matching PR #1566's own
split exactly) and then looking for a fixed-offset slot group the way
`tests/test_axera_matmul_reg8_noise_source.py` (PR #1583) did for
MatMul finds one immediately -- and it is not merely structurally
*similar* to MatMul's, it is **identical down to the absolute byte
offsets**:

- A stable **`verb=162, bank=0, field=0` `V`-kind anchor record**
  (operand `\\x12\\x00\\x00\\x00`) at **offset 345** -- the exact same
  verb/bank/field/operand and the exact same absolute offset PR #1583
  found for MatMul's own anchor, in every one of the 8 samples,
  regardless of table order.
- Three more `verb=162, bank=0, field=0` `V`-kind records at the exact
  same fixed offsets PR #1583 found (**353, 361, 369**), each carrying
  a 4-byte operand `\\x13\\x00 YY ZZ` where `YY` is one of the 4 pool
  candidates and `ZZ` is a fixed per-position byte (`\\x00` at 353/361,
  `\\x82` at 369 -- constant across all 8 samples, part of the record's
  own fixed framing, not part of the class assignment).
- A fourth slot at **offset 378**, always the one and only `S`-kind,
  `reg=8`, `tag=130` record with a 3-byte payload `\\x13\\x00 YY`.

**Zero exceptions across all 8 samples, and entirely independent of
table order** (the group sits at the identical offsets in both the
5-sample `(y,x)` group and the 3-sample `(x,y)` group): the four
classes assigned to these four fixed positions are always a genuine,
duplicate-free permutation of all 4 pool members -- the same "clean
4-of-4 permutation" MatMul showed (PR #1583), and unlike Gemm's
duplicate-tolerant 3-of-3 (PR #1577) or Conv's always-omits-one 3-of-4
(PR #1580). All 8 samples land on 8 *distinct* permutations.

Mul's own class payloads use a single leading byte (`\\x13`) on every
one of the 4 slots -- even more uniform than MatMul's own version,
which split `0x33` (class P1) from `0x23` (classes P2/P3/P4). This
matches `tests/test_axera_reg8_cross_op_synthesis.py`'s (PR #1582)
own finding that Mul's leading byte (`0x13`) is disjoint from every
other op's.

## This cleanly separates from the table-order coin flip -- it doesn't
## just coexist with it, it fully explains the "residual" PR #1566 left
## open

PR #1566's own docstring flagged an honest residual: "within the
`(y,x)` group (5 samples, all count-50), the first record's payload
still varies -- `0x20` (rebuilds 0, 1, 7), `0x40` (rebuild 2), `0x10`
(rebuild 5)." That "first record" is precisely this file's own
offset-378 slot -- directly confirmed below
(`TestMatchesPR1566sOwnResidual`). PR #1566 observed the values without
having the slot-group structure to explain *why* they varied; this file
supplies that structure.

Removing this group's own one `reg=8` record from each sample's total
`reg=8` count (`kind in ("S","B"), reg==8`) still leaves the *exact*
same 2-record gap PR #1566 found between table-order groups (49 vs 47,
not 50 vs 48) -- proving this group is a **separate** source of
`reg=8` variation, cleanly additive to (not a restatement of) the
table-order coin flip's own effect elsewhere in the record stream. Mul
is the one op where BOTH mechanisms are visibly present and separable
in the same build: the coin flip explains a 2-record COUNT difference
elsewhere in the stream (unchanged from PR #1566's own finding, not
re-decoded here), and this newly-found slot group explains the
specific VALUE residual PR #1566 could observe but not explain.

## What this completes

Combined with PR #1577 (Gemm), PR #1580 (Conv), and PR #1583 (MatMul),
this project has now decoded `reg=8`'s "second noise mechanism" in
**all four** of its main tracked ops. Mul's own version turns out to be
the *simplest and most tightly matched to MatMul's*: identical anchor
record, identical fixed offsets, identical always-a-full-permutation
behavior, and (unlike MatMul) a single shared leading byte across all
four slots. The semantic meaning of the four candidate values (still
consistent with, not proven to be, a scratch/tile-buffer-id reading --
see PR #1577/#1580/#1582's own repeated caveat) remains open here too.
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

ALL_NAMES = [f"mul_1x8_universal_stability_r{i}.mcode.gz" for i in range(8)]

CANDS = {b"\x13\x00 ", b"\x13\x00@", b"\x13\x000", b"\x13\x00\x10"}
POOL_TRAILING_BYTES = {0x10, 0x20, 0x30, 0x40}

ANCHOR_AT = 345
SLOT_OFFSETS = (353, 361, 369, 378)

# Expected (order, [(offset, class), ...]) per fixture, recomputed
# directly against the fixtures in TestFourFixedSlotsAlwaysFormACompletePermutation
# rather than trusted from this table alone -- kept here so the exact
# per-fixture reading is visible without re-running the decode.
EXPECTED_ORDER = {
    "mul_1x8_universal_stability_r0.mcode.gz": ("y", "x"),
    "mul_1x8_universal_stability_r1.mcode.gz": ("y", "x"),
    "mul_1x8_universal_stability_r2.mcode.gz": ("y", "x"),
    "mul_1x8_universal_stability_r3.mcode.gz": ("x", "y"),
    "mul_1x8_universal_stability_r4.mcode.gz": ("x", "y"),
    "mul_1x8_universal_stability_r5.mcode.gz": ("y", "x"),
    "mul_1x8_universal_stability_r6.mcode.gz": ("x", "y"),
    "mul_1x8_universal_stability_r7.mcode.gz": ("y", "x"),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def table_order(data):
    window = data[180:320]
    ix = window.find(b"x_offset")
    iy = window.find(b"y_offset")
    assert ix != -1 and iy != -1, "both x_offset and y_offset must be present"
    return ("x", "y") if ix < iy else ("y", "x")


def find_group(recs):
    """Returns {offset: (kind, trailing_byte)} for the anchor-relative
    4-slot group. Mirrors tests/test_axera_matmul_reg8_noise_source.py's
    own find_group() almost exactly -- same anchor shape, same fixed
    offsets."""
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
            slots[at] = ("V", r["operand"][2])

    for r in recs:
        if (
            r["kind"] == "S"
            and r.get("reg") == 8
            and r.get("tag") == 130
            and r.get("payload") in CANDS
            and r.get("at") == SLOT_OFFSETS[3]
        ):
            slots[SLOT_OFFSETS[3]] = ("S,reg8", r["payload"][2])

    return slots


class TestAllEightSamplesAreTheSameLength(unittest.TestCase):
    def test_2232_bytes_every_time(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 2232, n)


class TestTableOrderMatchesPR1566(unittest.TestCase):
    """Reconfirms PR #1566's own 5-vs-3 table-order split before
    building on it."""

    def test_order_matches_expected_table(self):
        for n in ALL_NAMES:
            self.assertEqual(table_order(load(n)), EXPECTED_ORDER[n], n)

    def test_five_yx_three_xy(self):
        orders = [table_order(load(n)) for n in ALL_NAMES]
        self.assertEqual(orders.count(("y", "x")), 5)
        self.assertEqual(orders.count(("x", "y")), 3)


class TestAnchorIsStableAtAFixedOffsetRegardlessOfTableOrder(unittest.TestCase):
    """The verb=162,bank=0,field=0 operand=12 00 00 00 anchor record
    sits at the identical absolute offset (345) in all 8 samples,
    regardless of x_offset/y_offset table order -- the same
    independence PR #1583 found for MatMul's own anchor, at the exact
    same offset."""

    def test_anchor_present_at_345_in_every_sample(self):
        for n in ALL_NAMES:
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 162
                and r.get("operand", b"") == b"\x12\x00\x00\x00"
            ]
            self.assertEqual(len(hits), 1, n)
            self.assertEqual(hits[0]["at"], ANCHOR_AT, n)


class TestFourFixedSlotsAlwaysFormACompletePermutation(unittest.TestCase):
    """The core finding, zero exceptions across all 8 samples: the four
    fixed-offset slots always carry all 4 distinct pool trailing bytes
    -- never a duplicate, never an omission, matching MatMul's own
    clean 4-of-4 result (PR #1583) rather than Gemm's or Conv's."""

    def test_all_four_slots_present_with_all_four_distinct_classes(self):
        for n in ALL_NAMES:
            recs = decode(n)
            slots = find_group(recs)
            self.assertEqual(set(slots.keys()), set(SLOT_OFFSETS), n)
            classes = [c for _, c in slots.values()]
            self.assertEqual(len(classes), 4, n)
            self.assertEqual(set(classes), POOL_TRAILING_BYTES, n)
            self.assertEqual(len(set(classes)), 4, f"{n}: expected no duplicates")

    def test_all_eight_samples_land_on_distinct_permutations(self):
        perms = set()
        for n in ALL_NAMES:
            recs = decode(n)
            slots = find_group(recs)
            perms.add(tuple(slots[off][1] for off in SLOT_OFFSETS))
        self.assertEqual(len(perms), 8, perms)

    def test_slot_assignment_is_independent_of_table_order(self):
        """The group's presence, offsets, and permutation behavior are
        identical whether a sample is (y,x)- or (x,y)-ordered -- both
        table-order groups show the same 4-of-4 structure at the same
        offsets."""
        for order in (("y", "x"), ("x", "y")):
            names = [n for n in ALL_NAMES if EXPECTED_ORDER[n] == order]
            self.assertGreater(len(names), 0)
            for n in names:
                slots = find_group(decode(n))
                self.assertEqual(set(slots.keys()), set(SLOT_OFFSETS), n)
                self.assertEqual(len(set(c for _, c in slots.values())), 4, n)


class TestSlotLabelsMatchMatMulExactly(unittest.TestCase):
    """Unlike Conv (PR #1580), where the register LABEL itself varied
    across slots, Mul's slot->label mapping is fixed and identical to
    MatMul's own (PR #1583): the first three physical positions are
    always V-kind, the fourth is always the one and only S-kind reg=8
    record."""

    def test_first_three_slots_always_v_kind(self):
        for n in ALL_NAMES:
            slots = find_group(decode(n))
            for off in SLOT_OFFSETS[:3]:
                self.assertEqual(slots[off][0], "V", f"{n} offset={off}")

    def test_fourth_slot_always_s_kind_reg8(self):
        for n in ALL_NAMES:
            slots = find_group(decode(n))
            self.assertEqual(slots[SLOT_OFFSETS[3]][0], "S,reg8", n)


class TestMatchesPR1566sOwnResidual(unittest.TestCase):
    """PR #1566 observed (without decoding the mechanism) that within
    the 5-sample (y,x) group, the first reg=8 record's payload varied:
    0x20 (rebuilds 0,1,7), 0x40 (rebuild 2), 0x10 (rebuild 5). That
    "first record" is precisely this file's own offset-378 slot --
    confirmed directly."""

    def test_yx_group_residual_matches_pr1566_exactly(self):
        expected = {
            "mul_1x8_universal_stability_r0.mcode.gz": 0x20,
            "mul_1x8_universal_stability_r1.mcode.gz": 0x20,
            "mul_1x8_universal_stability_r2.mcode.gz": 0x40,
            "mul_1x8_universal_stability_r5.mcode.gz": 0x10,
            "mul_1x8_universal_stability_r7.mcode.gz": 0x20,
        }
        for n, exp in expected.items():
            self.assertEqual(EXPECTED_ORDER[n], ("y", "x"), n)
            slots = find_group(decode(n))
            self.assertEqual(slots[SLOT_OFFSETS[3]][1], exp, n)


class TestGroupIsSeparateFromTheTableOrderCoinFlipsOwnCountEffect(unittest.TestCase):
    """Removing this group's own single reg=8 record (offset 378) from
    each sample's total reg=8 count still leaves the exact same 2-record
    gap PR #1566 found between table-order groups (49 vs 47, not the
    raw 50 vs 48) -- proving this slot group is a distinct, additively
    separate source of reg=8 variation from the coin flip's own effect
    elsewhere in the record stream, not a restatement of it."""

    def _reg8_all(self, recs):
        return [r for r in recs if r["kind"] in ("S", "B") and r.get("reg") == 8]

    def test_total_counts_match_pr1566(self):
        for n in ALL_NAMES:
            recs = decode(n)
            count = len(self._reg8_all(recs))
            if EXPECTED_ORDER[n] == ("y", "x"):
                self.assertEqual(count, 50, n)
            else:
                self.assertEqual(count, 48, n)

    def test_non_group_count_still_shows_a_two_record_gap(self):
        non_group_counts_by_order = {("y", "x"): set(), ("x", "y"): set()}
        for n in ALL_NAMES:
            recs = decode(n)
            non_group = [r for r in self._reg8_all(recs) if r.get("at") != 378]
            non_group_counts_by_order[EXPECTED_ORDER[n]].add(len(non_group))
        self.assertEqual(non_group_counts_by_order[("y", "x")], {49})
        self.assertEqual(non_group_counts_by_order[("x", "y")], {47})


if __name__ == "__main__":
    unittest.main()
