"""Extends the census-based rebuild-stability check
(`tests/test_axera_universal_bank_reg_stability.py` PR #1566,
`tests/test_axera_core_bank_stability_extend.py` PR #1567) from Mul --
the only op it has ever been run against -- to **Gemm**, for the first
time.

Gemm structurally cannot exhibit the specific "table-order coin flip"
mechanism those PRs found for Mul's `reg=8` (Pulsar2 forces Gemm's `B`/
`C` to compile-time constants, leaving only one live, non-constant
tensor -- `A` -- so there is no 2+-peer unordered name table for that
exact mechanism's precondition to bite). But "cannot have coin-flip
noise" is not the same claim as "is stable" -- Gemm's own near-universal
banks/registers had never actually been checked for rebuild-to-rebuild
content stability before this file, only specific individual fields
(the `transB` "3-way rotation",
`tests/test_axera_gemm_transb_rotation_stability.py`; the M=8 lead,
`test_axera_gemm_m8_diff_is_allocation_noise.py`).

## Method

Reused `Gemm(A[1,512], B[512,1000], C[1000])`, `transB=0` -- the same
FC-layer-shaped config `tests/test_axera_gemm_transb_rotation_stability.py`
(PR #1525) already built 4 independent samples of (the committed
`gemm_1x512x1000_tb0.mcode.gz` plus `_rebuild{0,1,2}.mcode.gz`, all
4,368 bytes). This file adds **4 more fresh, independent rebuilds**
(`_stability_r{0,1,2,3}.mcode.gz`, also 4,368 bytes each) via
`pulsar2_docker.build()`, for 8 total samples, and checks every one of
`tests/test_axera_resource_model_census.py`'s 9 near-universal banks
and 36 near-universal registers for exact content stability, the same
protocol PR #1566/#1567 ran for Mul.

**A methodological wrinkle, surfaced and handled explicitly**: the
original 4 fixtures' exact calibration data is not recoverable (only
the RNG seed convention -- `np.random.default_rng(0)` -- was documented
in the prior PR's docstring, not the exact sample count/shape/build
path). Comparing "old 4" vs "new 4" as one pooled batch therefore risks
conflating genuine allocator noise with an artifact of not reproducing
the *exact* original calibration tensors. This file avoids that trap by
checking stability **within each batch independently first**, and only
treating a register as confirmed noise if it is unstable within BOTH
independently-built batches on their own -- the same "don't trust a
single comparison, check independent rebuilds" discipline this
project's whole rebuild-stability line of work is built on, applied one
level deeper here because of the batch-provenance gap.

## Result

**All 9 near-universal banks (`0x00`-`0x04`, `0x0e`, `0x0f`, `0x1c`,
`0x1e`) are byte-for-byte content-stable** across all 8 samples, both
within each batch and pooled -- the same clean result Mul got.

**Registers split three ways**, not two:

1. **29 of 36 registers are stable everywhere** -- within each batch
   AND across both batches pooled (`2, 7, 9, 10, 12, 14, 16, 20, 22, 24,
   26, 28, 36, 40, 52, 60, 62, 66, 72, 76, 96, 100, 101, 106, 107, 120,
   126, 128, 138`). Genuinely constant, reusable content for a future
   generator.
2. **3 registers (`0, 8, 78`) are unstable within EACH batch
   independently** -- confirmed genuine allocator noise, not a
   calibration-provenance artifact, since the same instability
   reproduces inside the "old" batch alone and inside the "new" batch
   alone. `reg=8` being among these matches Mul's own finding (PR
   #1566) that `reg=8` is this project's single most noise-prone
   register -- now confirmed noisy for a second, structurally different
   op that cannot even use the mechanism PR #1566 attributed Mul's
   `reg=8` noise to. **This means reg=8's instability is NOT fully
   explained by the table-order coin flip** -- that mechanism cannot
   apply to Gemm at all here (only one live tensor, no unordered peer
   table), yet `reg=8` is still unstable, so whatever this second
   mechanism is, it is not Mul-specific and not the coin flip. Left
   honestly undecoded.
3. **4 registers (`18, 32, 38, 74`) are stable within each batch alone,
   but differ between the two batches** -- consistent with either (a) a
   real dependency on the (unreproduced) calibration data, or (b)
   coincidental batch-correlated noise. This file does not have the
   original calibration tensors and cannot distinguish these two
   explanations; reported as an open, three-way-honest finding rather
   than forced into either.

## What this establishes for the resource-model picture

The "9 core banks, ~36 core registers" structure Mul's census suggested
generalizes to Gemm's own resource usage (same near-universal set,
independently confirmed content-stable for a structurally different
op). But Gemm's noise profile is NOT simply "Mul's coin-flip minus the
registers that needed 2+ peer tensors" -- `reg=8`'s instability survives
even where the known mechanism's precondition is absent, meaning there
is at least one more, currently uncharacterized, source of rebuild-to-
rebuild non-determinism in this codec beyond the table-order coin flip.
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

GENUINE_NOISE_REGS = (0, 8, 78)
BATCH_CONFOUNDED_REGS = (18, 32, 38, 74)
STABLE_EVERYWHERE_REGS = tuple(
    sorted(set(CORE_REGS) - set(GENUINE_NOISE_REGS) - set(BATCH_CONFOUNDED_REGS))
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


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


def decode_all(names):
    return [mcode.decode(load(n), **mcode.FULL_RULE) for n in names]


class TestAllEightSamplesAreTheSameLength(unittest.TestCase):
    def test_4368_bytes_every_time(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 4368, n)


class TestCoreBanksAreContentStable(unittest.TestCase):
    """All 9 near-universal banks are byte-identical across all 8
    independent Gemm(1,512,1000) rebuilds, matching Mul's own result."""

    def test_nine_banks_stable_across_all_eight(self):
        all_recs = decode_all(ALL_NAMES)
        for bank in CORE_BANKS:
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestNoTableOrderCoinFlipMechanismApplies(unittest.TestCase):
    """Gemm has only one live, non-constant tensor (`A`) in its name
    table -- `Y_offset`/`A_offset` order is fixed at the same two
    absolute offsets in every one of the 8 samples, confirming the
    2+-peer-entry precondition PR #1561's coin-flip mechanism requires
    genuinely does not hold here."""

    def test_offset_table_order_never_varies(self):
        for n in ALL_NAMES:
            d = load(n)
            iy = d.find(b"Y_offset")
            ia = d.find(b"A_offset")
            self.assertNotEqual(iy, -1, n)
            self.assertNotEqual(ia, -1, n)
            self.assertEqual((iy, ia), (148, 196), n)


class TestGenuineAllocatorNoiseRegisters(unittest.TestCase):
    """reg=0, reg=8, reg=78 are unstable WITHIN each independently-built
    4-sample batch on its own -- ruling out the batch-provenance
    confound and confirming real, structural allocator noise."""

    def test_unstable_within_old_batch_alone(self):
        all_recs = decode_all(OLD_BATCH)
        for reg in GENUINE_NOISE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertFalse(
                all(s == sets[0] for s in sets),
                f"reg={reg} expected to vary within the old batch alone",
            )

    def test_unstable_within_fresh_batch_alone(self):
        all_recs = decode_all(FRESH_BATCH)
        for reg in GENUINE_NOISE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertFalse(
                all(s == sets[0] for s in sets),
                f"reg={reg} expected to vary within the fresh batch alone",
            )

    def test_reg8_is_noisy_even_without_the_coinflip_precondition(self):
        """The one register this project has now seen unstable in both
        Mul (via the table-order coin flip) and Gemm (structurally
        unable to use that mechanism) -- an open, unexplained residual
        instability source, not fully accounted for by PR #1561's
        finding."""
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 8) for recs in all_recs]
        self.assertFalse(all(s == sets[0] for s in sets))


class TestBatchConfoundedRegistersAreHonestlyAmbiguous(unittest.TestCase):
    """reg=18, 32, 38, 74: stable within EACH batch alone, but differ
    between the two batches -- consistent with a real calibration-data
    dependency or with coincidental batch-correlated noise; this file
    cannot distinguish the two without the original calibration
    tensors, and does not claim either."""

    def test_stable_within_old_batch_alone(self):
        all_recs = decode_all(OLD_BATCH)
        for reg in BATCH_CONFOUNDED_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} expected stable within the old batch alone",
            )

    def test_stable_within_fresh_batch_alone(self):
        all_recs = decode_all(FRESH_BATCH)
        for reg in BATCH_CONFOUNDED_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} expected stable within the fresh batch alone",
            )

    def test_differs_between_the_two_batches(self):
        old_recs = decode_all(OLD_BATCH)
        fresh_recs = decode_all(FRESH_BATCH)
        for reg in BATCH_CONFOUNDED_REGS:
            old_val = reg_records(old_recs[0], reg)
            fresh_val = reg_records(fresh_recs[0], reg)
            self.assertNotEqual(
                old_val,
                fresh_val,
                f"reg={reg} expected to differ between the two batches",
            )


class TestRemainingRegistersAreStableEverywhere(unittest.TestCase):
    """29 of the 36 near-universal registers are content-stable across
    all 8 samples -- both within either batch and pooled together."""

    def test_stable_in_every_sample(self):
        self.assertEqual(
            len(STABLE_EVERYWHERE_REGS)
            + len(GENUINE_NOISE_REGS)
            + len(BATCH_CONFOUNDED_REGS),
            36,
        )
        all_recs = decode_all(ALL_NAMES)
        for reg in STABLE_EVERYWHERE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} should be content-stable across all 8 samples",
            )


if __name__ == "__main__":
    unittest.main()
