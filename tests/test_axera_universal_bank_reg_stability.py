"""Closing a gap `tests/test_axera_resource_model_census.py` (PR #1563)
explicitly flagged: presence in every one of 306 fixtures doesn't tell
you whether a bank/register's *content* is stable across independent
rebuilds of the SAME config -- this project has repeatedly found
register/bank assignment can be allocator noise (Gemm's M=8 lead,
`tests/test_axera_gemm_m8_diff_is_allocation_noise.py`; Gemm's `transB`
rotation, `tests/test_axera_gemm_transb_rotation_stability.py`;
MatMul's `A_offset`/`B_offset` table order,
`tests/test_axera_matmul_offset_table_coinflip.py`).

This tests five of PR #1563's "universal" (present in all 306 fixtures)
banks/registers against 8 independent rebuilds of one unchanged
`Mul(x[1,8], y[1,8])` config, comparing exact record content (not just
presence) the same way those noise-determination PRs did.

## Result: split, not uniform -- the newly-found banks are stable,
## reg=8 is not

- **Banks `0x0e`, `0x0f`, `0x1c`, `0x1e`** (PR #1563's own newly-found
  near-universal banks, not previously singled out by the README's
  older 2-model sample): every `(field, value)` pair at each bank is
  byte-for-byte identical across all 8 rebuilds. Real, stable
  structure -- these are load-bearing in the sense that mattered for
  this test (reproducible), even though what they compute is still
  undecoded.
- **`reg=100` and `reg=96`** (two arbitrarily-chosen members of PR
  #1563's 36-register universal set, picked to include a large-numbered
  one per that PR's own "magnitude doesn't predict universality"
  finding): also byte-for-byte identical across all 8 rebuilds.
- **`reg=8`** (the census's single heaviest universal register, 27,369
  corpus-wide uses): genuinely NOT stable. Record count varies (48 or
  50 records across the 8 rebuilds) and specific payload bytes differ
  even among same-count builds.

## reg=8's variation is not unexplained noise -- it is the SAME
## table-order mechanism PR #1561 already decoded for MatMul

Mul has two peer live, non-constant tensor inputs (`x`, `y`) -- the
exact structural precondition PR #1561's synthesis identified for the
`A_offset`/`B_offset`-style name-table coin flip
(`tests/test_axera_table_order_coinflip_audit.py`: this freedom
requires 2+ peer entries in an unordered table). Checking each
rebuild's `x_offset`/`y_offset` relative order against its `reg=8`
record count:

| table order | rebuilds | reg=8 record count |
| --- | --- | --- |
| `(y, x)` | 0, 1, 2, 5, 7 | **50**, every time |
| `(x, y)` | 3, 4, 6 | **48**, every time |

The correlation is exact and two-valued across all 8 samples -- the
same kind of clean bimodal split PR #1506 (`tests/test_axera_matmul_offset_table_coinflip.py`)
found for MatMul's own coin flip, now confirmed for Mul's `reg=8` usage
too. This is a genuine, useful extension: `reg=8`'s apparent
"universality" in the census is compatible with (likely *is*) the same
generic table-order-coin-flip mechanism, not a new, uncharacterized
noise class.

**One honest residual, not explained by table order alone**: within
the `(y, x)` group (5 samples, all count-50), the first record's
payload byte still varies -- `0x20` (rebuilds 0, 1, 7), `0x40`
(rebuild 2), `0x10` (rebuild 5). Table order predicts the *count*
exactly but not every byte within a fixed count -- there is a second,
smaller source of variation this file does not decode, on top of the
table-order effect.

## What this establishes for the resource-model picture

Presence-in-every-fixture (PR #1563's own "universal" criterion) does
NOT imply stable content -- confirmed directly, not just flagged as a
theoretical gap. At least one of PR #1563's universal registers
(`reg=8`) is a live instance of the exact noise class PR #1561 already
named, extending that mechanism's known scope from MatMul to Mul. The
other four probes here (two banks beyond the already-checked core set,
two registers) came back stable, which is a real, useful positive
result for a future generator's resource model -- but this is 5 probes
out of 9 near-universal banks and 36 near-universal registers PR #1563
found; most of that set remains untested for content stability.
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


class TestNewlyFoundBanksAreStableAcrossRebuilds(unittest.TestCase):
    """Banks 0x0e/0x0f/0x1c/0x1e (PR #1563's newly-found near-universal
    banks) carry byte-identical (field, value) content across all 8
    independent rebuilds of one unchanged Mul[1,8] config."""

    def test_four_banks_stable(self):
        all_recs = [mcode.decode(load(n), **mcode.FULL_RULE) for n in REBUILD_NAMES]
        for bank in (0x0E, 0x0F, 0x1C, 0x1E):
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestSomeUniversalRegistersAreStable(unittest.TestCase):
    """reg=100 and reg=96 (arbitrary members of PR #1563's 36-register
    universal set, including a >0x40 one) carry byte-identical content
    across all 8 rebuilds."""

    def test_two_registers_stable(self):
        all_recs = [mcode.decode(load(n), **mcode.FULL_RULE) for n in REBUILD_NAMES]
        for reg in (100, 96):
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"reg={reg} should be present")


class TestReg8IsNotStableAndMatchesTheTableOrderCoinFlip(unittest.TestCase):
    """reg=8 -- the census's heaviest universal register -- varies
    across rebuilds, and its record count correlates exactly with
    Mul's own x_offset/y_offset table order (the same mechanism PR
    #1506 decoded for MatMul's A_offset/B_offset)."""

    def test_reg8_record_count_varies(self):
        all_recs = [mcode.decode(load(n), **mcode.FULL_RULE) for n in REBUILD_NAMES]
        counts = [len(reg_records(recs, 8)) for recs in all_recs]
        self.assertEqual(
            len(set(counts)), 2, f"expected exactly 2 distinct counts, got {counts}"
        )
        self.assertEqual(set(counts), {48, 50})

    def test_reg8_count_correlates_exactly_with_table_order(self):
        datas = [load(n) for n in REBUILD_NAMES]
        all_recs = [mcode.decode(d, **mcode.FULL_RULE) for d in datas]
        for d, recs in zip(datas, all_recs):
            order = table_order(d)
            count = len(reg_records(recs, 8))
            if order == ("y", "x"):
                self.assertEqual(count, 50, f"order={order} should give count 50")
            else:
                self.assertEqual(order, ("x", "y"))
                self.assertEqual(count, 48, f"order={order} should give count 48")

    def test_within_one_table_order_group_a_residual_variation_remains(self):
        """Table order predicts reg=8's record COUNT exactly, but not
        every payload byte within a fixed count -- an honest residual,
        not explained here."""
        datas = [load(n) for n in REBUILD_NAMES]
        all_recs = [mcode.decode(d, **mcode.FULL_RULE) for d in datas]
        yx_group = [
            reg_records(recs, 8)
            for d, recs in zip(datas, all_recs)
            if table_order(d) == ("y", "x")
        ]
        self.assertEqual(len(yx_group), 5, "expected 5 (y,x)-ordered rebuilds")
        first_records = [g[0] for g in yx_group]
        self.assertGreater(
            len(set(first_records)),
            1,
            "expected residual variation within the same table-order group",
        )


if __name__ == "__main__":
    unittest.main()
