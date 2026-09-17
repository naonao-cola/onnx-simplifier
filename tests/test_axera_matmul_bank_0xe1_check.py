"""Continues `tests/test_axera_conv_dilation8_length_growth.py` (PR
#1602)'s own explicitly flagged gap: that file found Conv's
`dilation=8` mcode carries `bank=0xe1` (225), `field=32`/`field=48`
V-records byte-identical to the two "unexplained" `0xe1` field=32/48
records `tests/test_axera_gemm_bank_81_e1_decode.py` (PR #1570)
originally found at large-N Gemm shapes -- the first time that exact
constant pair had been seen outside Gemm. PR #1602 explicitly left
open: "Whether the same `0xe1` field=32/48 pair also appears in MatMul
at its own analogous 'large' threshold is not checked here."

## Answer: yes -- and it is common, not a "large shape" edge case the
## way Gemm/Conv show it

Scanning every one of this corpus's 135 already-committed MatMul
fixtures (no new builds needed) for `bank=225` V-records finds the
Gemm/Conv-matching pair -- `field=32` operand `33 03 00`, `field=48`
operand `35 03 00 a1`, byte-identical to PR #1570's/#1602's own -- in
**58 of 135** fixtures, spanning shapes from `MatMul(A[4,8],B[8,8])`
(the project's own smallest, most-used MatMul probe shape) up through
`K=40`+ sweeps. This is a real generalization of the constant pair
across all three ops that carry sparse bank `0xe1` -- but unlike Gemm's
own clean "N crosses ~32-33" threshold and Conv's own clean "dilation
crosses from 4 to 8" threshold, MatMul's own trigger is NOT a single
shape-size threshold, as the sections below establish precisely.

## A second, MatMul-specific pair exists at the same two field slots,
## never seen in Gemm or Conv

Alongside (sometimes instead of) the Gemm-matching pair, many MatMul
fixtures carry a DIFFERENT `field=32`/`field=48` pair at the identical
two field numbers: operand `09 03 00` (field=32) / `0b 03 00 a1`
(field=48). This exact pair does not appear in any Gemm or Conv fixture
checked in this corpus (confirmed below) -- it is a MatMul-specific
analogue occupying the same two structural slots, not a mistaken
re-read of the Gemm/Conv pair.

## The MatMul-specific pair's presence is fully explained by the known
## `A_offset`/`B_offset` table-order coin flip -- zero exceptions across
## 42 same-family samples

Grouping 42 fixtures from two shape families that share a fixed
`(batch=2, M=4, N=8)` base -- the `K`-sweep
(`tests/test_axera_matmul_var_byte_k_plateaus.py`'s own `matmul_var_k9`
through `matmul_var_k40` fixtures, 31 samples) and the
`MatMul(A[4,8],B[8,8])` rebuild-stability family
(`tests/test_axera_matmul_rebuild_stability.py` PR #1581's own 11
samples, both `modeA`/`modeB` and `v7stability_*`) -- by their own
`A_offset`/`B_offset` table order finds: **the MatMul-specific pair is
present if and only if the order is `(B, A)`, with zero exceptions
across all 42 samples.** This is the same mechanism class already
established for this project's other rebuild-noise findings (the
coin-flip explaining a record-count or record-content difference
directly), now extended to a THIRD kind of coin-flip-driven effect
(bank `0xe1`'s own presence/absence, not a byte value or record count).

## Honest complication: that exact order-mapping is NOT universal --
## it inverts in the `M`-sweep family at `M>=24`

Checking the same rule against `tests/test_axera_matmul_var_byte_thresholds.py`'s
own `M`-sweep fixtures (`matmul_var_m24` through `matmul_var_m48`,
`matmul_var_shape_check_m64`) finds **19 mismatches**: at `M>=24`, the
MatMul-specific pair's presence tracks the *opposite* table order --
present at `(A,B)`, absent at `(B,A)` -- the exact reverse of the
`K`-sweep/small-shape mapping. This means the pair's presence is not a
single, universal function of table order alone; something about `M`
crossing its own threshold (in the same M=24-ish range PR #1524/PR
#1545's own `var`-byte threshold work already found other MatMul
quantities transitioning) flips which table order the pair prefers.
This file does not chase that second boundary further -- it is recorded
honestly as a real, verified complication rather than forced into one
formula.

## The Gemm-matching pair's own trigger is even less clean -- reported
## as genuinely open, not forced

An initial hypothesis (order `(B,A)`, OR order `(A,B)` with `K` not a
multiple of 4) fit all 31 points of the `K`-sweep family perfectly --
but broke immediately against `matmul_var_bitcost_k32.mcode.gz`
(`tests/test_axera_matmul_var_byte_thresholds.py`'s own `K=32` point,
built against the *identical* `(batch=2,M=4,N=8)` base shape as the
`K`-sweep family, order `(B,A)`) which shows the Gemm-matching pair
**absent** -- contradicting the "order=(B,A) implies present" half of
the hypothesis even within what should be the same shape family. This
project has repeatedly preferred a precise, evidenced negative over a
forced pattern (see `tests/test_axera_gemm_field144_q_rule.py` PR
#1575, `tests/test_axera_conv_reg60_mechanism.py` PR #1586's own root-
trigger residual); the Gemm-matching pair's own presence rule is left
here as genuinely unresolved, not claimed to be `K mod 4`.

## What this establishes

The cross-op connection PR #1602 found (Conv's dilation=8 growth
sharing Gemm's own `0xe1` field=32/48 constant) generalizes to MatMul
too, confirming this is a real, shared codec-level constant pair used
by at least three ops -- not a Gemm-specific artifact. But MatMul's own
version is structurally richer than Gemm's/Conv's clean single-
threshold pictures: it coexists with a second, MatMul-specific pair at
the same field slots, that second pair's presence is precisely
explained by the known table-order coin flip in one shape family and
precisely inverted in another, and the Gemm-matching pair's own trigger
resists a clean formula even within one shape family. All three of
these findings are real and verified; none is forced into a false
unification.
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

GEMM_F32 = b"\x33\x03\x00"
GEMM_F48 = b"\x35\x03\x00\xa1"
OTHER_F32 = b"\x09\x03\x00"
OTHER_F48 = b"\x0b\x03\x00\xa1"

K_SWEEP_NAMES = [
    f"matmul_var_k{k}.mcode.gz" for k in list(range(9, 32)) + list(range(33, 41))
]
DIAG_FAMILY_NAMES = [
    "matmul_4x8x8.mcode.gz",
    "matmul_4x8x8_rebuild_modeA.mcode.gz",
    "matmul_4x8x8_rebuild_modeB.mcode.gz",
    "matmul_4x8x8_v7stability_diag0.mcode.gz",
    "matmul_4x8x8_v7stability_r1.mcode.gz",
    "matmul_4x8x8_v7stability_r2.mcode.gz",
    "matmul_4x8x8_v7stability_r3.mcode.gz",
    "matmul_4x8x8_v7stability_r4.mcode.gz",
    "matmul_4x8x8_v7stability_r5.mcode.gz",
    "matmul_4x8x8_v7stability_r6.mcode.gz",
    "matmul_4x8x8_v7stability_r7.mcode.gz",
]
M_SWEEP_MISMATCH_NAMES = [
    "matmul_var_m24.mcode.gz",
    "matmul_var_m24_rebuild.mcode.gz",
    "matmul_var_m25.mcode.gz",
    "matmul_var_m26.mcode.gz",
    "matmul_var_m27.mcode.gz",
    "matmul_var_m28.mcode.gz",
    "matmul_var_m29.mcode.gz",
    "matmul_var_m30.mcode.gz",
    "matmul_var_m31.mcode.gz",
    "matmul_var_m33.mcode.gz",
    "matmul_var_m34.mcode.gz",
    "matmul_var_m35.mcode.gz",
    "matmul_var_m36.mcode.gz",
    "matmul_var_m36_rebuild.mcode.gz",
    "matmul_var_m40.mcode.gz",
    "matmul_var_m40_rebuild.mcode.gz",
    "matmul_var_m44.mcode.gz",
    "matmul_var_m48.mcode.gz",
    "matmul_var_shape_check_m64.mcode.gz",
]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def table_order(data):
    window = data[100:400]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    assert ia != -1 and ib != -1, "both A_offset and B_offset must be present"
    return ("A", "B") if ia < ib else ("B", "A")


def field32_operands(recs):
    return {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
    }


def field48_operands(recs):
    return {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 48
    }


class TestMatMulCarriesTheExactGemmConvConstantPair(unittest.TestCase):
    """The core cross-op finding: at least one already-committed MatMul
    fixture carries bank=0xe1's field=32/48 pair byte-identical to
    Gemm's (PR #1570) and Conv's (PR #1602) own."""

    def test_small_baseline_shape_carries_it(self):
        # matmul_var_k18 (batch=2,M=4,K=18,N=8) is a small, ordinary
        # shape -- not a "large N" edge case -- yet carries the pair.
        recs = decode("matmul_var_k18.mcode.gz")
        self.assertIn(GEMM_F32, field32_operands(recs))
        self.assertIn(GEMM_F48, field48_operands(recs))

    def test_at_least_half_of_the_k_sweep_carries_it(self):
        count = sum(1 for n in K_SWEEP_NAMES if GEMM_F32 in field32_operands(decode(n)))
        self.assertGreaterEqual(count, len(K_SWEEP_NAMES) // 2)


class TestMatMulSpecificSecondPairIsNeverInGemmOrConv(unittest.TestCase):
    """The 09 03 00 / 0b 03 00 a1 pair occupies the same two field
    numbers but is MatMul-specific -- not a misread of Gemm's/Conv's
    own constant."""

    def test_other_pair_appears_in_at_least_one_matmul_fixture(self):
        recs = decode("matmul_4x8x8.mcode.gz")
        self.assertIn(OTHER_F32, field32_operands(recs))
        self.assertIn(OTHER_F48, field48_operands(recs))

    def test_other_pair_absent_from_gemms_own_decode_fixtures(self):
        for n in (
            "gemm_1x128x1000.mcode.gz",
            "gemm_1x256x1000.mcode.gz",
            "gemm_1x512x1000_tb0.mcode.gz",
        ):
            recs = decode(n)
            self.assertNotIn(OTHER_F32, field32_operands(recs), n)

    def test_other_pair_absent_from_convs_own_decode_fixtures(self):
        for n in (
            "conv_dilation8.mcode.gz",
            "conv_dilation8_r0.mcode.gz",
        ):
            recs = decode(n)
            self.assertNotIn(OTHER_F32, field32_operands(recs), n)


class TestOtherPairMatchesTableOrderInTheKAndDiagFamilies(unittest.TestCase):
    """Zero exceptions across 42 samples from two shape families sharing
    a (batch=2, M=4, N=8) base: the MatMul-specific pair is present iff
    the A_offset/B_offset table order is (B, A)."""

    def test_forty_two_samples_zero_exceptions(self):
        names = K_SWEEP_NAMES + DIAG_FAMILY_NAMES
        for n in names:
            data = load(n)
            order = table_order(data)
            has_other = OTHER_F32 in field32_operands(decode(n))
            self.assertEqual(
                has_other,
                order == ("B", "A"),
                f"{n}: order={order} has_other={has_other}",
            )


class TestOtherPairsOrderMappingInvertsInTheMSweepFamily(unittest.TestCase):
    """Honest complication: the exact same rule, applied to the M-sweep
    family's own fixtures (M>=24), is wrong on all 19 points checked --
    the mapping is inverted there, not merely noisy. Reported precisely
    rather than treated as scatter around the K-family's own rule."""

    def test_m_sweep_mismatches_the_k_family_rule_on_every_point(self):
        mismatches = 0
        for n in M_SWEEP_MISMATCH_NAMES:
            data = load(n)
            order = table_order(data)
            has_other = OTHER_F32 in field32_operands(decode(n))
            if has_other != (order == ("B", "A")):
                mismatches += 1
        self.assertEqual(mismatches, len(M_SWEEP_MISMATCH_NAMES))

    def test_m_sweep_follows_the_inverted_rule_instead(self):
        for n in M_SWEEP_MISMATCH_NAMES:
            data = load(n)
            order = table_order(data)
            has_other = OTHER_F32 in field32_operands(decode(n))
            self.assertEqual(
                has_other,
                order == ("A", "B"),
                f"{n}: order={order} has_other={has_other}",
            )


class TestGemmMatchingPairsTriggerIsNotACleanFormula(unittest.TestCase):
    """A K-mod-4 + table-order hypothesis fit all 31 K-sweep points but
    breaks on bitcost_k32 -- the same (batch=2,M=4,N=8) base shape,
    K=32, order (B,A) -- which the hypothesis predicts present but is
    actually absent. Reported as a genuine, unresolved complication."""

    def test_k_sweep_hypothesis_fits_all_31_points(self):
        for k in list(range(9, 32)) + list(range(33, 41)):
            name = f"matmul_var_k{k}.mcode.gz"
            data = load(name)
            order = table_order(data)
            has_gemm = GEMM_F32 in field32_operands(decode(name))
            predicted = order == ("B", "A") or (k % 4 != 0)
            self.assertEqual(has_gemm, predicted, name)

    def test_bitcost_k32_breaks_the_same_hypothesis(self):
        name = "matmul_var_bitcost_k32.mcode.gz"
        data = load(name)
        order = table_order(data)
        self.assertEqual(order, ("B", "A"), name)
        has_gemm = GEMM_F32 in field32_operands(decode(name))
        # The K-sweep hypothesis predicts True here (order == (B,A));
        # the actual value is False, refuting the hypothesis as a
        # universal rule even within this one shape family.
        self.assertFalse(has_gemm, name)


if __name__ == "__main__":
    unittest.main()
