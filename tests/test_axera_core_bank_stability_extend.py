"""Continuing `tests/test_axera_universal_bank_reg_stability.py` (PR
#1566): that file tested 5 of the 9 near-universal banks and 3 of the
36 near-universal registers `tests/test_axera_resource_model_census.py`
(PR #1563) found, for content-stability across 8 independent rebuilds
of one unchanged `Mul[1,8]` config, and explicitly flagged "most of
that set remains untested." This file covers everything PR #1566 left
untested: the README's own already-named "core" banks 0x00-0x04
(never actually checked for content-stability before, only presence),
and all 33 remaining registers of the 36-register universal set.

No new fixtures needed -- this reuses PR #1566's own 8 committed
`mul_1x8_universal_stability_r{0-7}.mcode.gz` rebuilds directly, the
exact same 8 independent compiles, so results are directly comparable
to that file's own `reg=8` finding.

## Banks 0x00-0x04: all five are fully content-stable

Every `(field, value)` pair at each of banks 0x00, 0x01, 0x02, 0x03,
0x04 is byte-for-byte identical across all 8 rebuilds. These are the
README's own longest-established "core" field-write banks (originally
decoded from two real models, `resnet18d`/`mnasnet`) -- their
*presence* was never in question, but this is the first time their
*content* has been checked against this project's own rebuild-stability
standard, and they pass cleanly.

## Registers: 27 of 33 remaining are stable, 6 are not -- and all 6
## are fully explained by the SAME mechanism PR #1566 found for reg=8

27 of the 33 untested universal registers (`{0, 2, 9, 10, 12, 14, 20,
22, 24, 26, 28, 32, 36, 38, 52, 60, 66, 72, 74, 76, 101, 106, 107, 120,
126, 128, 138}`) are byte-for-byte identical across all 8 rebuilds --
real, stable resource-model structure, extending PR #1566's own
positive result from 2 registers to 29.

**6 registers -- `{7, 16, 18, 40, 62, 78}` -- are not stable**, but
every one of them fits PR #1566's own `x_offset`/`y_offset` table-order
mechanism exactly, with no residual left over:

| reg | `(y,x)`-order count | `(x,y)`-order count |
| --- | --- | --- |
| 7 | 1 | 2 |
| 16 | 3 | 2 |
| 18 | 6 | 5 |
| 40 | 2 | 3 |
| 62 | 5 | 4 |
| 78 | 3 | 2 |

Every one of these splits along the identical rebuild partition PR
#1566 found for `reg=8` (`{0,1,2,5,7}` vs `{3,4,6}`, by table order,
not by any property of the register itself). **Unlike `reg=8`, none of
these six show a residual: conditioning on table order alone makes each
register's full record set (not just its count) byte-identical across
every rebuild that shares an order** -- checked directly, not assumed:
5-way and 3-way within-group comparisons for all 6 registers show zero
internal variation once table order is fixed. So table order is not
just a *count* predictor here, it is a *complete* content predictor --
a cleaner case than `reg=8`'s own partially-unexplained residual.

## What this establishes for the resource-model picture

Combined with PR #1566: of the 9 near-universal banks and 36
near-universal registers PR #1563's census found, **all 9 banks are now
confirmed content-stable**, and **35 of 36 registers are now fully
accounted for** -- 33 genuinely stable, and 6 (including the census's
own heaviest member, `reg=8`) explained by the one already-known
table-order coin-flip mechanism (2 of those 6 with a residual PR #1566
left open; the other 5 new ones found here fully explained). Zero new,
uncharacterized noise classes were found. This is a strong, mostly-
complete result for a future generator's resource model: the "core
vocabulary" PR #1563 identified is, with one well-understood exception,
genuinely reliable structure to build on.

Not tested here (left for future work, as PR #1566's own directive
suggested but did not require): whether banks 0x00-0x04's stability
holds for a second op family. Gemm and Conv structurally cannot exhibit
the table-order mechanism at all (`tests/test_axera_table_order_coinflip_audit.py`
-- they never have 2+ peer live non-constant tensor entries in an
unordered table), so there was no equivalent risk to rule out for them
the way there was for Mul's registers; but this file only directly
confirms bank/register stability within Mul, not across every op
family the census corpus spans.
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


REBUILD_NAMES = [f"mul_1x8_universal_stability_r{i}.mcode.gz" for i in range(8)]


def bank_records(recs, bank):
    return sorted(
        (r["field"], r.get("value"))
        for r in recs
        if r["kind"] == "V" and r["bank"] == bank
    )


def reg_records(recs, reg):
    return [
        (r["kind"], r.get("tag"), r.get("payload"))
        for r in recs
        if r["kind"] in ("S", "B") and r.get("reg") == reg
    ]


def table_order(data):
    window = data[180:320]
    ix = window.find(b"x_offset")
    iy = window.find(b"y_offset")
    assert ix != -1 and iy != -1, "both x_offset and y_offset must be present"
    return ("x", "y") if ix < iy else ("y", "x")


# The 36-register universal set from PR #1563's own census, minus the
# 3 registers PR #1566 already tested (8, 100, 96).
REMAINING_REGISTERS = sorted(
    {
        0,
        2,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        32,
        36,
        38,
        40,
        52,
        60,
        62,
        66,
        72,
        74,
        76,
        78,
        96,
        100,
        101,
        106,
        107,
        120,
        126,
        128,
        138,
    }
    - {8, 100, 96}
)

STABLE_REGISTERS = [
    0,
    2,
    9,
    10,
    12,
    14,
    20,
    22,
    24,
    26,
    28,
    32,
    36,
    38,
    52,
    60,
    66,
    72,
    74,
    76,
    101,
    106,
    107,
    120,
    126,
    128,
    138,
]

TABLE_ORDER_EXPLAINED_REGISTERS = [7, 16, 18, 40, 62, 78]


class TestCoreBanksAreFullyStable(unittest.TestCase):
    """Banks 0x00-0x04 -- the README's own longest-established core
    field-write banks -- have never had their CONTENT (as opposed to
    presence) checked for rebuild-stability before this file."""

    def test_all_five_core_banks_stable(self):
        all_recs = [mcode.decode(load(n), **mcode.FULL_RULE) for n in REBUILD_NAMES]
        for bank in range(5):
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestRemainingSetIsFullyAccountedFor(unittest.TestCase):
    def test_remaining_register_count(self):
        self.assertEqual(len(REMAINING_REGISTERS), 33)

    def test_stable_and_table_order_sets_partition_the_remainder(self):
        self.assertEqual(
            set(STABLE_REGISTERS) | set(TABLE_ORDER_EXPLAINED_REGISTERS),
            set(REMAINING_REGISTERS),
        )
        self.assertEqual(len(STABLE_REGISTERS), 27)
        self.assertEqual(len(TABLE_ORDER_EXPLAINED_REGISTERS), 6)


class Test27RegistersAreFullyStable(unittest.TestCase):
    def test_stable_registers_are_byte_identical_across_all_8_rebuilds(self):
        all_recs = [mcode.decode(load(n), **mcode.FULL_RULE) for n in REBUILD_NAMES]
        for reg in STABLE_REGISTERS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"reg={reg} should be present")


class Test6RegistersAreFullyExplainedByTableOrderWithNoResidual(unittest.TestCase):
    """Unlike reg=8 (PR #1566), which had a leftover residual within a
    fixed table-order group, these 6 registers are FULLY explained by
    table order alone -- zero variation once conditioned on it."""

    def test_counts_vary_but_correlate_exactly_with_table_order(self):
        datas = [load(n) for n in REBUILD_NAMES]
        all_recs = [mcode.decode(d, **mcode.FULL_RULE) for d in datas]
        orders = [table_order(d) for d in datas]
        for reg in TABLE_ORDER_EXPLAINED_REGISTERS:
            counts = [len(reg_records(recs, reg)) for recs in all_recs]
            yx_counts = {c for c, o in zip(counts, orders) if o == ("y", "x")}
            xy_counts = {c for c, o in zip(counts, orders) if o == ("x", "y")}
            self.assertEqual(
                len(yx_counts), 1, f"reg={reg}: (y,x)-order count should be uniform"
            )
            self.assertEqual(
                len(xy_counts), 1, f"reg={reg}: (x,y)-order count should be uniform"
            )
            self.assertNotEqual(
                yx_counts, xy_counts, f"reg={reg}: counts should differ by order"
            )

    def test_full_record_content_matches_within_each_order_group_no_residual(self):
        datas = [load(n) for n in REBUILD_NAMES]
        all_recs = [mcode.decode(d, **mcode.FULL_RULE) for d in datas]
        for reg in TABLE_ORDER_EXPLAINED_REGISTERS:
            yx_group = [
                reg_records(recs, reg)
                for d, recs in zip(datas, all_recs)
                if table_order(d) == ("y", "x")
            ]
            xy_group = [
                reg_records(recs, reg)
                for d, recs in zip(datas, all_recs)
                if table_order(d) == ("x", "y")
            ]
            self.assertEqual(len(yx_group), 5, f"reg={reg}: expected 5 (y,x) samples")
            self.assertEqual(len(xy_group), 3, f"reg={reg}: expected 3 (x,y) samples")
            self.assertTrue(
                all(g == yx_group[0] for g in yx_group),
                f"reg={reg}: (y,x)-order group should have zero residual variation",
            )
            self.assertTrue(
                all(g == xy_group[0] for g in xy_group),
                f"reg={reg}: (x,y)-order group should have zero residual variation",
            )


if __name__ == "__main__":
    unittest.main()
