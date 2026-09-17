"""Extends the census-based rebuild-stability check
(`tests/test_axera_universal_bank_reg_stability.py` PR #1566, Mul;
`tests/test_axera_gemm_rebuild_stability.py` PR #1576, Gemm;
`tests/test_axera_conv_rebuild_stability.py` PR #1579, Conv) to
**MatMul**, completing the sweep across all four of this project's main
tracked ops for the first time.

MatMul is the one op with the MOST prior per-field noise investigation
-- `tests/test_axera_matmul_offset_table_coinflip.py` is the ORIGINAL
discovery of the "table-order coin flip" mechanism (an unordered
per-compile allocator non-determinism affecting the `A_offset`/
`B_offset` parameter name-table order), later found to also explain
some of Mul's own noise. But that prior work characterized ONE
specific field's swing (the ~900-byte cross-order diff), never ran the
systematic census-sweep the other three ops just got across every
near-universal bank/register.

## Method

Built `MatMul(A[4,8], B[8,8])` -- the exact shape
`tests/test_axera_matmul_offset_table_coinflip.py` already used for its
own 8-rebuild discovery -- **8 fresh, independent samples**
(`matmul_4x8x8_v7stability_{diag0,r1..r7}.mcode.gz`, all 3,112 bytes),
via a standalone build script using an identical `RandomState(0)`
calibration seed for every sample (both `A` and `B` inputs), so
calibration provenance is not a confound here (matching PR #1579's
Conv approach, not PR #1576's Gemm approach, which had to leave that
ambiguity open for its two differently-provisioned batches). Unlike
Gemm/Conv, this file does not need an "old batch vs. fresh batch"
split for provenance reasons -- all 8 samples come from one script, one
config, one calibration seed -- but still splits them into two groups
of 4 in build order (`BATCH1`/`BATCH2`) as an extra within-group
stability check, the same discipline the other three files used.

**A methodological wrinkle, surfaced and handled explicitly**: like PR
#1579's Conv work, this worktree's Docker daemon only has
`pulsar2:7.0-lite` loaded, not the project's `pulsar2:6.0-lite`
default. Before trusting any comparison against the two pre-existing
`matmul_4x8x8_rebuild_mode{A,B}.mcode.gz` fixtures (built by
`tests/test_axera_matmul_offset_table_coinflip.py`'s own original
work, presumably with `6.0-lite`), this file confirmed compatibility: a
diagnostic 7.0-lite build showed (a) identical length (3,112 bytes),
(b) a same-table-order diff of 46 bytes against `modeB` -- somewhat
above this project's tightest same-version noise floor but still
clearly ordinary-noise scale, not structural -- and (c) an
opposite-table-order diff of 915 bytes against `modeA`, landing exactly
in the coin flip's own established 913-917-byte "big" bucket and
spanning up to the same `3084` upper bound. The lower bound starts at
`156` here rather than the original file's `204`, purely because this
file's own model names its output tensor `y` (lowercase) where the
original build used `Y` -- the extra 3 differing bytes are a
`y_offset`/`Y_offset` ASCII-case artifact of this file's own model
construction, not a new finding; confirmed by inspecting the raw bytes
at that offset directly. Treated as compatible; the two pre-existing
fixtures are used below only as an independent cross-check on the
reg=8 finding, not pooled into the main 8-sample stability analysis.

## Result: all 9 banks stable, and MatMul is the clearest case yet for
## the coin flip explaining most (but not all) of its own noise

**All 9 near-universal banks (`0x00`-`0x04`, `0x0e`, `0x0f`, `0x1c`,
`0x1e`) are byte-for-byte content-stable** across all 8 samples --
Mul, Gemm, Conv, and now MatMul all show the identical result. Four
ops, four confirmations.

**29 of 36 registers are stable everywhere.** The remaining 7
(`7, 8, 10, 16, 20, 40, 62`) vary -- but unlike Gemm and Conv, **6 of
those 7 (`7, 10, 16, 20, 40, 62`) are *fully* explained by the
`A_offset`/`B_offset` table order**: grouping the 8 samples by which
order they landed in, each of these 6 registers is perfectly stable
*within* either order-group and differs *between* them, with zero
exceptions. This is the first op in this project's four-op sweep where
the known coin-flip mechanism cleanly accounts for the bulk of a
near-universal register's observed noise -- fitting, since this is the
exact shape where that mechanism was originally discovered.

**`reg=8` is the one exception, and it is NOT explained by table order
even here.** Grouping the same 8 samples by `A_offset`/`B_offset`
order, `reg=8`'s content is unstable *within* both order-groups
independently (verified directly, not inferred): the `(A,B)`-order
group alone shows two distinct payloads across its 4 samples, and so
does the `(B,A)`-order group. **This is the fourth op in a row
(Mul, Gemm, Conv, MatMul) where `reg=8` shows rebuild instability, and
the third op in a row (Gemm, Conv, MatMul) where the known coin-flip
mechanism -- even when it demonstrably explains 6 of `reg=8`'s own
near-universal-register neighbors in this exact build -- does not
explain `reg=8` itself.** Two pre-existing fixtures from a different
build session/toolchain version (`matmul_4x8x8_rebuild_mode{A,B}.mcode.gz`)
cross-check this: `modeA` (`A,B`-order) carries the same `reg=8` value
as one of this file's own `(A,B)`-order samples, and a *different*
value from another `(A,B)`-order sample in this file -- confirming the
same within-order-group variation independently, across two separate
build sessions.

`reg=8`'s exact payload across all 8 samples takes one of 4 distinct
values (`\\x33\\x00\\x20`, `\\x23\\x00\\x40`, `\\x23\\x00\\x30`,
`\\x23\\x00\\x10`) -- trailing bytes `0x20/0x30/0x40/0x10`, evenly
spaced by `0x10`, structurally reminiscent of Gemm's own already-decoded
3-slot mechanism (`tests/test_axera_gemm_reg8_second_noise_source.py`
PR #1577, trailing bytes `0x20/0x30/0x40`) but with a 4th value here
and no `reg=0`/`reg=78`-style paired indicator checked in this file --
not decoded further here, left as a concrete, evidenced lead for
whoever chases MatMul's own version of PR #1577's work next.

`reg=0`, `reg=60`, and `reg=78` -- genuinely noisy for Gemm and/or Conv
-- are all fully stable for MatMul, further confirming (a third time)
that the second noise mechanism's affected-register set is not a
fixed, op-independent set.

## What this completes for the resource-model picture

The census-sweep rebuild-stability check now covers all four of this
project's main tracked ops. All four share the identical 9-bank /
~36-register near-universal structure, confirmed content-stable for
the overwhelming majority of it in every case. `reg=8` is now confirmed
unstable in all four ops, and in three of four (Gemm, Conv, and now
MatMul -- the one op where the coin flip was actually discovered) that
instability is NOT fully explained by the known mechanism, even though
that same mechanism cleanly explains most of the OTHER noisy registers
in this exact MatMul build. Whatever the true source of `reg=8`'s
residual noise is, it is general across every op this project has
checked and independent of the table-order coin flip -- the clearest
and most consistently reproduced open thread in this project's
noise-vs-load-bearing resource-model picture.
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

BATCH1 = [
    "matmul_4x8x8_v7stability_diag0.mcode.gz",
    "matmul_4x8x8_v7stability_r1.mcode.gz",
    "matmul_4x8x8_v7stability_r2.mcode.gz",
    "matmul_4x8x8_v7stability_r3.mcode.gz",
]
BATCH2 = [
    "matmul_4x8x8_v7stability_r4.mcode.gz",
    "matmul_4x8x8_v7stability_r5.mcode.gz",
    "matmul_4x8x8_v7stability_r6.mcode.gz",
    "matmul_4x8x8_v7stability_r7.mcode.gz",
]
ALL_NAMES = BATCH1 + BATCH2

CORE_BANKS = (0x00, 0x01, 0x02, 0x03, 0x04, 0x0E, 0x0F, 0x1C, 0x1E)
CORE_REGS = (
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
)

COINFLIP_EXPLAINED_REGS = (7, 10, 16, 20, 40, 62)
RESIDUAL_NOISE_REGS = (8,)
GENUINE_NOISE_REGS = COINFLIP_EXPLAINED_REGS + RESIDUAL_NOISE_REGS
STABLE_EVERYWHERE_REGS = tuple(sorted(set(CORE_REGS) - set(GENUINE_NOISE_REGS)))

REG8_VALUES = {
    b"3\x00 ",
    b"#\x00@",
    b"#\x000",
    b"#\x00\x10",
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def offset_table_order(data):
    """Returns ("A", "B") if A_offset's record precedes B_offset's,
    else ("B", "A") -- mirrors
    tests/test_axera_matmul_offset_table_coinflip.py's own inspection,
    widened to a 200-byte window since this file's own model naming
    shifts the exact byte offset slightly from that file's 195:270."""
    window = data[100:400]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    assert ia != -1 and ib != -1, "both A_offset and B_offset must be present"
    return ("A", "B") if ia < ib else ("B", "A")


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


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def decode_all(names):
    return [decode(n) for n in names]


class TestAllEightSamplesAreTheSameLength(unittest.TestCase):
    def test_3112_bytes_every_time(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 3112, n)


class TestFreshBatchImageVersionIsCompatible(unittest.TestCase):
    """This worktree's Docker daemon only has `pulsar2:7.0-lite`
    loaded, not the `pulsar2:6.0-lite` the pre-existing
    `matmul_4x8x8_rebuild_mode{A,B}.mcode.gz` fixtures (PR #1506) used.
    Before treating any comparison against them as meaningful, confirm
    the two toolchain versions are compatible for this exact
    shape/config, the same three-check discipline PR #1579 used for
    Conv."""

    def test_same_length_as_pre_existing_fixtures(self):
        self.assertEqual(
            len(load("matmul_4x8x8_v7stability_diag0.mcode.gz")),
            len(load("matmul_4x8x8_rebuild_modeA.mcode.gz")),
        )

    def test_same_order_diff_is_ordinary_noise_scale(self):
        diag = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        mode_b = load("matmul_4x8x8_rebuild_modeB.mcode.gz")
        self.assertEqual(offset_table_order(diag), offset_table_order(mode_b))
        diffs = sum(1 for i in range(len(diag)) if diag[i] != mode_b[i])
        self.assertGreater(diffs, 0)
        self.assertLess(
            diffs, 100, "same-order cross-version diff should be ordinary noise scale"
        )

    def test_opposite_order_diff_lands_in_the_known_big_bucket(self):
        diag = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        mode_a = load("matmul_4x8x8_rebuild_modeA.mcode.gz")
        self.assertNotEqual(offset_table_order(diag), offset_table_order(mode_a))
        diffs = [i for i in range(len(diag)) if diag[i] != mode_a[i]]
        self.assertGreaterEqual(len(diffs), 900)
        self.assertLessEqual(len(diffs), 1000)
        # 156, not the original file's 204: this file's own model names
        # its output tensor "y" where the original build used "Y" -- a
        # y_offset/Y_offset ASCII-case artifact of this file's own model
        # construction (see module docstring), not a new finding.
        self.assertEqual(min(diffs), 156)
        self.assertEqual(max(diffs), 3084)


class TestCoreBanksAreContentStable(unittest.TestCase):
    """All 9 near-universal banks are byte-identical across all 8
    independent MatMul(4,8,8) rebuilds, matching Mul's, Gemm's, and
    Conv's own results."""

    def test_nine_banks_stable_across_all_eight(self):
        all_recs = decode_all(ALL_NAMES)
        for bank in CORE_BANKS:
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestCoinFlipFullyExplainsSixOfSevenNoisyRegisters(unittest.TestCase):
    """Grouping the 8 samples by A_offset/B_offset table order, 6 of
    the 7 nominally "noisy" registers are perfectly stable *within*
    either order-group and differ *between* them -- the coin flip
    fully accounts for their variation, with zero exceptions."""

    def test_registers_vary_when_pooled(self):
        all_recs = decode_all(ALL_NAMES)
        for reg in COINFLIP_EXPLAINED_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertFalse(
                all(s == sets[0] for s in sets), f"reg={reg} should vary when pooled"
            )

    def test_registers_are_stable_within_each_order_group(self):
        orders = [offset_table_order(load(n)) for n in ALL_NAMES]
        all_recs = decode_all(ALL_NAMES)
        for reg in COINFLIP_EXPLAINED_REGS:
            vals = [reg_records(recs, reg) for recs in all_recs]
            ab = [v for o, v in zip(orders, vals) if o == ("A", "B")]
            ba = [v for o, v in zip(orders, vals) if o == ("B", "A")]
            self.assertTrue(len(ab) > 0 and len(ba) > 0, f"reg={reg}: need both groups")
            self.assertTrue(
                all(v == ab[0] for v in ab),
                f"reg={reg} should be stable within the (A,B)-order group",
            )
            self.assertTrue(
                all(v == ba[0] for v in ba),
                f"reg={reg} should be stable within the (B,A)-order group",
            )
            self.assertNotEqual(
                ab[0],
                ba[0],
                f"reg={reg}: the two order-groups should actually differ",
            )


class TestReg8IsNotExplainedByTheCoinFlipEvenHere(unittest.TestCase):
    """reg=8 is unstable WITHIN both order-groups independently -- the
    known coin-flip mechanism, which fully explains 6 of its near-
    universal-register neighbors in this exact build, does not explain
    reg=8. The fourth op (after Mul, Gemm, Conv) where this register
    shows rebuild instability, and the third (Gemm, Conv, MatMul) where
    the known mechanism cannot be the explanation."""

    def test_reg8_unstable_within_ab_order_group(self):
        orders = [offset_table_order(load(n)) for n in ALL_NAMES]
        all_recs = decode_all(ALL_NAMES)
        vals = [reg_records(recs, 8) for recs in all_recs]
        ab = [v for o, v in zip(orders, vals) if o == ("A", "B")]
        self.assertGreaterEqual(len(ab), 2)
        self.assertFalse(all(v == ab[0] for v in ab))

    def test_reg8_unstable_within_ba_order_group(self):
        orders = [offset_table_order(load(n)) for n in ALL_NAMES]
        all_recs = decode_all(ALL_NAMES)
        vals = [reg_records(recs, 8) for recs in all_recs]
        ba = [v for o, v in zip(orders, vals) if o == ("B", "A")]
        self.assertGreaterEqual(len(ba), 2)
        self.assertFalse(all(v == ba[0] for v in ba))

    def test_reg8_takes_four_distinct_payload_values(self):
        all_recs = decode_all(ALL_NAMES)
        payloads = set()
        for recs in all_recs:
            hits = reg_records(recs, 8)
            first_tag130 = next(p for _, tag, p in hits if tag == 130)
            payloads.add(first_tag130)
        self.assertEqual(payloads, REG8_VALUES)

    def test_pre_existing_fixtures_cross_check_the_within_order_variation(self):
        """matmul_4x8x8_rebuild_modeA (a different build session/
        toolchain) shares its (A,B)-order group with two of this
        file's own samples, which do not all agree with each other --
        confirming the same within-order variation independently."""
        mode_a = load("matmul_4x8x8_rebuild_modeA.mcode.gz")
        self.assertEqual(offset_table_order(mode_a), ("A", "B"))
        mode_a_reg8 = reg_records(decode("matmul_4x8x8_rebuild_modeA.mcode.gz"), 8)
        mode_a_val = next(p for _, tag, p in mode_a_reg8 if tag == 130)

        orders = [offset_table_order(load(n)) for n in ALL_NAMES]
        all_recs = decode_all(ALL_NAMES)
        ab_vals = {
            next(p for _, tag, p in reg_records(recs, 8) if tag == 130)
            for n, o, recs in zip(ALL_NAMES, orders, all_recs)
            if o == ("A", "B")
        }
        self.assertIn(mode_a_val, ab_vals)
        self.assertGreater(len(ab_vals), 1)


class TestRegZeroSixtySeventyEightAreStableForMatMul(unittest.TestCase):
    """reg=0/reg=78 (noisy for Gemm) and reg=60 (noisy for Conv) are
    all fully stable for MatMul -- a third confirmation that the
    second noise mechanism's affected-register set is op-specific, not
    a fixed set."""

    def test_reg0_stable(self):
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 0) for recs in all_recs]
        self.assertTrue(all(s == sets[0] for s in sets))

    def test_reg60_stable(self):
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 60) for recs in all_recs]
        self.assertTrue(all(s == sets[0] for s in sets))

    def test_reg78_stable(self):
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 78) for recs in all_recs]
        self.assertTrue(all(s == sets[0] for s in sets))


class TestRemainingRegistersAreStableEverywhere(unittest.TestCase):
    def test_stable_in_every_sample(self):
        self.assertEqual(len(STABLE_EVERYWHERE_REGS) + len(GENUINE_NOISE_REGS), 36)
        all_recs = decode_all(ALL_NAMES)
        for reg in STABLE_EVERYWHERE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} should be content-stable across all 8 samples",
            )


if __name__ == "__main__":
    unittest.main()
