"""Characterizes the "second, non-coin-flip noise source" that
`tests/test_axera_gemm_rebuild_stability.py` (PR #1576) found and
explicitly left open: `reg=8` (also `reg=0`, `reg=78`) is unstable
across independent `Gemm(1,512,1000)` rebuilds even though that PR
directly confirmed Gemm cannot use the "table-order coin flip"
mechanism (`Y_offset`/`A_offset` order is fixed at offsets `(148, 196)`
in every one of its 8 samples).

This file reuses PR #1576's own 8 already-committed fixtures (4 in
`OLD_BATCH`, `tests/test_axera_gemm_transb_rotation_stability.py`
PR #1525; 4 in `FRESH_BATCH`, PR #1576 itself) -- no new builds were
needed, since the existing 8 independent samples (two separately
provisioned batches) already produce a clean, zero-exception rule.

## The mechanism: a second, 3-slot unordered assignment

Three byte offsets, always at the same *relative* spacing from each
other in this fixed-shape (4,368-byte) build -- `reg=78`'s own record,
then two `reg=8` records at `+4` and `+10` bytes from it -- each
independently carry one of exactly three candidate byte values. (The
group's absolute position shifts by 2 bytes, `887`/`891`/`897` vs.
`887`/`893`/`899`, depending on whether `reg=78`'s own record happens
to use its 1-byte or 3-byte payload form that build -- an unrelated
reflow, not a fourth source of variation; the code below locates the
group by its `reg=78` anchor, not a hardcoded absolute offset.)

| class | long form (`tag=130`, 3-byte payload) | short form (`tag=132`, 1-byte) |
| --- | --- | --- |
| `A` | `\\x93\\x00\\x40` | -- |
| `B` | `\\x93\\x00\\x30` | -- |
| `C` | `\\x23\\x00\\x20` | `\\x23` |

(`A`/`B`/`C`'s trailing byte -- `0x40`, `0x30`, `0x20` -- is evenly
spaced by `0x10`; `A`/`B` also share the same leading two bytes. This
is *suggestive* of three page/bank-granular scratch-buffer addresses,
consistent with the "16-byte field-offset granularity" convention
already established elsewhere in this codec (README ~line 2206) if the
trailing byte is itself a coarser unit -- but this file does not claim
that interpretation is proven, only that the byte pattern is
consistent with it.)

Across all 8 samples, offset `887` (`reg=78`) always carries exactly
one class. Offsets `891`/`897` (`reg=8`) carry the *other* one or two
classes -- **not always disjoint from `887`'s class, and not always
both present**:

| fixture | 887 (`reg=78`) | 891 (`reg=8`) | 897 (`reg=8`) | distinct classes used | `reg=0` extra `B,147`? |
| --- | --- | --- | --- | --- | --- |
| `gemm_1x512x1000_tb0` | C | A | B | 3 (A,B,C) | no |
| `..._rebuild0` | C | B | A | 3 (A,B,C) | no |
| `..._rebuild1` | C | **C** | A | **2 (A,C)** | **yes** |
| `..._rebuild2` | C | *(absent)* | A | **2 (A,C)** | **yes** |
| `..._stability_r0` | B | A | C | 3 (A,B,C) | no |
| `..._stability_r1` | A | C | B | 3 (A,B,C) | no |
| `..._stability_r2` | C | *(absent)* | B | **2 (B,C)** | **yes** |
| `..._stability_r3` | B | C | A | 3 (A,B,C) | no |

Two things vary independently per build: **which class lands at each
of the three offsets** (genuinely unordered -- `891`/`897`'s relative
order is not a fixed sort, e.g. `[A,B]` then `[B,A]` then `[A,C]`, the
same "don't assume a canonical order" character as the already-known
coin flip, just over a different, 3-member internal collection instead
of the 2-member `Y_offset`/`A_offset` table PR #1576 ruled out), and
**whether all three classes actually get used, or only two** (via
either a genuine duplicate -- `rebuild1` uses `C` twice and never uses
`B` at all -- or a whole record going missing -- `rebuild2`/
`stability_r2` drop offset `891` entirely).

## `reg=0`'s extra `B,147` record is an exact indicator, not a coincidence

**Zero exceptions across all 8 samples**: `reg=0` carries its extra
`(B, tag=147)` record if and only if fewer than 3 distinct classes
appear across the three offsets above. This is verified as an exact
biconditional below, not just a correlation -- it ties `reg=0`,
`reg=8`, and `reg=78`'s previously-separate "genuine noise" findings
(PR #1576's own three-way split) into **one** explained mechanism: a
3-slot, unordered, duplicate-tolerant address assignment, with
`reg=0`'s extra record acting as a "fewer than 3 distinct addresses
this build" flag.

## What remains open

- The exact *semantic* meaning of classes `A`/`B`/`C` (candidate
  scratch/tile-buffer addresses is the natural reading given the
  evenly-spaced trailing byte, but not proven here).
- *Why* a build sometimes collapses to 2 distinct classes instead of
  3 -- whether that's a real scheduling/tiling decision (e.g. the
  compiler decided it only needs 2 scratch buffers for this
  particular tiling of a 512x1000 FC layer) or itself an artifact of
  the same non-determinism, is not decoded here.
- Whether this exact 3-slot mechanism generalizes to other Gemm shapes,
  or to Conv/MatMul's own analogous noise -- only checked at this one
  `Gemm(1,512,1000)` config, the same one PR #1576 used.
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
    "gemm_1x512x1000_tb0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
]
FRESH_BATCH = [
    "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
]
ALL_NAMES = OLD_BATCH + FRESH_BATCH

CLASS_OF = {
    b"\x93\x00@": "A",
    b"\x93\x000": "B",
    b"#\x00 ": "C",
    b"#": "C",
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def three_slot_group(recs):
    """Returns (reg78_class, [reg8_classes]) for the one reg=78 record
    and the 1-2 reg=8 records adjacent to it, classified by payload.

    `reg=78` carries several unrelated records elsewhere in the stream
    (see the module docstring's raw dump) -- only the one whose payload
    matches a known class (`A`/`B`/`C`) is the slot this file is about.
    """
    reg78_hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 78
        and r.get("tag") in (130, 132)
        and r["payload"] in CLASS_OF
    ]
    assert len(reg78_hits) == 1, reg78_hits
    anchor = reg78_hits[0]["at"]
    reg8 = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 8 and r.get("tag") == 130
    ]
    near_reg8 = [r for r in reg8 if 0 < r["at"] - anchor <= 12]
    reg78_class = CLASS_OF[reg78_hits[0]["payload"]]
    reg8_classes = [
        CLASS_OF[r["payload"]] for r in near_reg8 if r["payload"] in CLASS_OF
    ]
    return reg78_class, reg8_classes


def reg0_has_extra_b147(recs):
    return any(
        r["kind"] == "B" and r.get("tag") == 147 and r.get("reg") == 0 for r in recs
    )


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


EXPECTED = {
    "gemm_1x512x1000_tb0.mcode.gz": ("C", {"A", "B"}, False),
    "gemm_1x512x1000_tb0_rebuild0.mcode.gz": ("C", {"A", "B"}, False),
    "gemm_1x512x1000_tb0_rebuild1.mcode.gz": ("C", {"A", "C"}, True),
    "gemm_1x512x1000_tb0_rebuild2.mcode.gz": ("C", {"A"}, True),
    "gemm_1x512x1000_tb0_stability_r0.mcode.gz": ("B", {"A", "C"}, False),
    "gemm_1x512x1000_tb0_stability_r1.mcode.gz": ("A", {"B", "C"}, False),
    "gemm_1x512x1000_tb0_stability_r2.mcode.gz": ("C", {"B"}, True),
    "gemm_1x512x1000_tb0_stability_r3.mcode.gz": ("B", {"A", "C"}, False),
}


class TestThreeSlotGroupMatchesExpectedClasses(unittest.TestCase):
    """Pins the exact per-fixture (reg78 class, reg8 classes) reading
    from the module docstring's table -- fails loudly if a future
    `mcode.py` change alters how these records decode."""

    def test_all_eight_fixtures(self):
        for name in ALL_NAMES:
            recs = decode(name)
            reg78_class, reg8_classes = three_slot_group(recs)
            expected_78, expected_8, _ = EXPECTED[name]
            self.assertEqual(reg78_class, expected_78, name)
            self.assertEqual(set(reg8_classes), expected_8, name)


class TestReg0ExtraRecordExactlyIndicatesFewerThanThreeDistinctClasses(
    unittest.TestCase
):
    """The core finding: `reg=0`'s extra `(B, tag=147)` record is present
    if and only if fewer than 3 distinct classes appear across the
    3-slot group -- a verified biconditional, zero exceptions in 8/8
    samples, not a mere correlation."""

    def test_biconditional_holds_for_every_fixture(self):
        for name in ALL_NAMES:
            recs = decode(name)
            reg78_class, reg8_classes = three_slot_group(recs)
            distinct = {reg78_class} | set(reg8_classes)
            fewer_than_three = len(distinct) < 3
            has_extra = reg0_has_extra_b147(recs)
            self.assertEqual(
                fewer_than_three,
                has_extra,
                f"{name}: distinct classes={distinct} (len={len(distinct)}) "
                f"but reg=0 extra B,147 present={has_extra}",
            )

    def test_matches_expected_table(self):
        for name, (_, _, expected_extra) in EXPECTED.items():
            recs = decode(name)
            self.assertEqual(reg0_has_extra_b147(recs), expected_extra, name)


class TestReg78AndReg8AreNotAlwaysDisjoint(unittest.TestCase):
    """Refutes a simpler "clean 3-way permutation, one class always
    dropped whole" hypothesis: `rebuild1` proves a class (`C`) can be
    duplicated across `reg=78` and `reg=8` while a different class
    (`B`) is dropped entirely -- the assignment is duplicate-tolerant,
    not a strict permutation."""

    def test_rebuild1_has_a_genuine_duplicate(self):
        recs = decode("gemm_1x512x1000_tb0_rebuild1.mcode.gz")
        reg78_class, reg8_classes = three_slot_group(recs)
        self.assertEqual(reg78_class, "C")
        self.assertIn("C", reg8_classes)
        self.assertNotIn("B", {reg78_class} | set(reg8_classes))


class TestReg8sTwoSlotsHaveNoFixedOrder(unittest.TestCase):
    """When both reg=8 slots are present, their relative order is not a
    fixed sort by class or by value -- consistent with an unordered
    (coin-flip-like) assignment, the same character as the already-known
    Y_offset/A_offset mechanism but over a different internal
    collection."""

    def test_order_varies_across_three_distinct_class_samples(self):
        orders = set()
        for name in ALL_NAMES:
            recs = decode(name)
            _, reg8_classes = three_slot_group(recs)
            if len(reg8_classes) == 2:
                orders.add(tuple(reg8_classes))
        # At least 3 distinct 2-tuples among the 5 three-distinct-class
        # samples confirms no single canonical order.
        self.assertGreaterEqual(len(orders), 3, orders)


if __name__ == "__main__":
    unittest.main()
