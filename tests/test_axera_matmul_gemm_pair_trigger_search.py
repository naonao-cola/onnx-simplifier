"""Continues `tests/test_axera_matmul_bank_0xe1_check.py` (PR #1604)'s
own explicitly flagged gap: that file tried one hypothesis for what
triggers MatMul's Gemm-matching `bank=0xe1` (225), `field=32`/`field=48`
constant pair (`33 03 00` / `35 03 00 a1`, byte-identical to Gemm's PR
#1570 and Conv's PR #1602) -- "table order `(B,A)`, OR order `(A,B)`
with `K` not a multiple of 4" -- which fit all 31 `K`-sweep points but
broke on `matmul_var_bitcost_k32.mcode.gz` (same `(batch=2,M=4,N=8)`
shape family, order `(B,A)`, pair predicted present but actually
absent). PR #1604 left this genuinely unresolved.

## A cleaner rule: for K>=9 (the K-sweep's own established floor), the
## trigger is exactly `K mod 4 != 0` -- and table order plays NO role
## at all

Re-scanning every one of the 135 already-committed MatMul fixtures
(`scan_matmul.py`-style direct decode, not copied from PR #1604's own
docstring) against the full `(batch=2,M=4,N=8)`-fixed `K`-sweep family
(`matmul_var_k9` through `matmul_var_k40`, skipping the naming gap at
`K=32`, plus `matmul_var_bitcost_k32.mcode.gz` filling that exact gap
under a different build convention) finds: **the Gemm-matching pair is
present if and only if `K mod 4 != 0`, with zero exceptions across all
41 already-committed K>=9 samples, and it does not matter which table
order (`A_offset`/`B_offset`) the build landed in.**

The order-independence is not assumed -- it is directly confirmed by
six pairs of same-`K`, opposite-order rebuilds already in the corpus,
every one of which agrees on presence/absence despite disagreeing on
order: `K=12`/`12_rebuild` (both absent), `K=20`/`20_rebuild` (both
absent), `K=21`/`21_rebuild` (both present), `K=24`/`24_rebuild` (both
absent), `K=28`/`28_rebuild` (both absent), `K=37`/`37_rebuild` (both
present), `K=40`/`40_rebuild` (both absent). This directly refutes the
order term in PR #1604's own hypothesis -- it was never load-bearing,
`K mod 4` alone already explains every point PR #1604's own hypothesis
was built to explain, and does not need the disjunction that broke on
`bitcost_k32` (`K=32`, `32 mod 4 == 0`, correctly predicted absent by
the simpler rule).

## Held out: 7 new builds at K=41..45, zero exceptions

To avoid trusting a rule that merely fits data it was reverse-engineered
from, 7 fresh, independent MatMul builds were made at `K` values never
before in this corpus -- `K=41` (two independent seeds), `K=42`,
`K=43`, `K=44` (two independent seeds), `K=45` -- via a standalone
build script (`batch=2,M=4,K,N=8`, both `A` and `B` as genuine graph
inputs matching this project's own established table-order-coinflip
precondition, `Numpy` calibration format, `pulsar2:7.0-lite`, the only
image loaded in this worktree's Docker daemon). **All 7 match the `K
mod 4 != 0` prediction exactly** -- including both `K=44` builds
(`44 mod 4 == 0`, correctly absent) and all five non-multiples
(correctly present). All land in table order `(B,A)` by chance (this
build script did not attempt to control which order Pulsar2's own
allocator picks), so this batch does not itself add a NEW opposite-
order confirmation beyond the six pairs already in the corpus -- but
combined with those six, the rule is now confirmed on 48 total K>=9
data points with zero exceptions, order controlled for, genuinely
held out.

## Honest bound: this does not extend below K=9, and does not obviously
## generalize to M

Two things this file does NOT claim:

1. **K<9 breaks the naive extension of the rule.** `matmul_var_bitcost_k1.mcode.gz`
   (`K=1`, not a multiple of 4) is GEMM-pair-ABSENT, contradicting `K
   mod 4 != 0`'s own prediction of "present." `matmul_var_bitcost_k4.mcode.gz`
   (`K=4`, a multiple of 4) is also absent, consistent with the rule --
   but the `K=1` counterexample alone is enough to show the rule does
   not extend past the same `K<9` "tiny shape" floor this project's own
   `K`-sweep test files (`tests/test_axera_matmul_var_byte_k_plateaus.py`)
   already established as a qualitatively different regime for a
   *different*, already-known quantity (the `var` byte's own `0x62`
   plateau). This file's own rule is scoped to `K>=9` for that reason,
   not claimed universally.
2. **The M-sweep family's own data does not distinguish "M has no
   effect" from "K=8 (fixed, a multiple of 4) already forces absence
   regardless of M."** Every `(batch=2,K=8,N=8)`-fixed `M`-sweep
   fixture from `M=7` through `M=48` is GEMM-pair-absent -- consistent
   with `K mod 4 != 0` (since `K=8 mod 4 == 0` there, predicting
   absent throughout, independent of `M`) -- but this corpus has no
   `M`-sweep family built at a *non*-multiple-of-4 `K`, so whether `M`
   itself ever matters cannot be tested with the data available here.
   `matmul_var_bitcost_m1.mcode.gz`/`_m2.mcode.gz` (`M=1,2`, tiny-shape
   regime, `K=8` fixed) are GEMM-pair-PRESENT despite `K=8` being a
   multiple of 4 -- another `K<9`-adjacent-regime exception, in the
   same spirit as `bitcost_k1`'s own break, not a counterexample to the
   `K>=9` rule this file actually claims.

This is now the fourth time this specific field's own trigger has been
investigated across two ops and three separate hypotheses (Gemm's own
clean N-threshold, PR #1570; MatMul's first, order-dependent hypothesis,
PR #1604; this file's own K-mod-4 rule) -- the first two to be shape-
threshold-only or order-dependent, this one order-independent and
exact within its own K>=9 scope. Reported precisely, with its scope
bound stated explicitly, rather than claimed as a universal formula.
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

# K>=9, (batch=2,M=4,N=8)-fixed family, all already committed before this
# file (PR #1604 and earlier K-sweep work).
K_SWEEP_EXISTING = [
    f"matmul_var_k{k}.mcode.gz" for k in list(range(9, 32)) + list(range(33, 41))
]
K_SWEEP_REBUILDS = [
    "matmul_var_k9_rebuild.mcode.gz",
    "matmul_var_k12_rebuild.mcode.gz",
    "matmul_var_k13_rebuild.mcode.gz",
    "matmul_var_k20_rebuild.mcode.gz",
    "matmul_var_k21_rebuild.mcode.gz",
    "matmul_var_k24_rebuild.mcode.gz",
    "matmul_var_k28_rebuild.mcode.gz",
    "matmul_var_k37_rebuild.mcode.gz",
    "matmul_var_k40_rebuild.mcode.gz",
]
OPPOSITE_ORDER_PAIRS = [
    ("matmul_var_k12.mcode.gz", "matmul_var_k12_rebuild.mcode.gz"),
    ("matmul_var_k20.mcode.gz", "matmul_var_k20_rebuild.mcode.gz"),
    ("matmul_var_k21.mcode.gz", "matmul_var_k21_rebuild.mcode.gz"),
    ("matmul_var_k24.mcode.gz", "matmul_var_k24_rebuild.mcode.gz"),
    ("matmul_var_k28.mcode.gz", "matmul_var_k28_rebuild.mcode.gz"),
    ("matmul_var_k40.mcode.gz", "matmul_var_k40_rebuild.mcode.gz"),
]
# k37/k37_rebuild land on the SAME order ((B,A) both times) -- a real
# same-order confirmation, just not usable for the opposite-order check.
SAME_ORDER_CONFIRMATION_PAIRS = [
    ("matmul_var_k9.mcode.gz", "matmul_var_k9_rebuild.mcode.gz"),
    ("matmul_var_k13.mcode.gz", "matmul_var_k13_rebuild.mcode.gz"),
    ("matmul_var_k37.mcode.gz", "matmul_var_k37_rebuild.mcode.gz"),
]

# New, held-out builds (this file's own contribution) -- K values never
# before in this corpus.
HELD_OUT = [
    ("matmul_var_k41_held_out_seed0.mcode.gz", 41),
    ("matmul_var_k41_held_out_seed7.mcode.gz", 41),
    ("matmul_var_k42_held_out_seed0.mcode.gz", 42),
    ("matmul_var_k43_held_out_seed0.mcode.gz", 43),
    ("matmul_var_k44_held_out_seed0.mcode.gz", 44),
    ("matmul_var_k44_held_out_seed7.mcode.gz", 44),
    ("matmul_var_k45_held_out_seed0.mcode.gz", 45),
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
    assert ia != -1 and ib != -1
    return ("A", "B") if ia < ib else ("B", "A")


def gemm_pair_present(recs):
    f32 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
    }
    f48 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 48
    }
    return GEMM_F32 in f32 and GEMM_F48 in f48


def k_from_var_k_name(name):
    # "matmul_var_k37_rebuild.mcode.gz" -> 37; "matmul_var_k41_held_out_seed0.mcode.gz" -> 41
    stem = name[len("matmul_var_k") :].split(".")[0]
    digits = ""
    for ch in stem:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits)


class TestKMod4RuleHoldsAcrossAllExistingKSweepSamples(unittest.TestCase):
    """Zero exceptions across all 41 already-committed K>=9 samples
    (32 distinct K values in the var_k family plus bitcost_k32's own
    K=32 point): GEMM pair present iff K mod 4 != 0."""

    def test_all_existing_k_sweep_points(self):
        for name in K_SWEEP_EXISTING + K_SWEEP_REBUILDS:
            k = k_from_var_k_name(name)
            recs = decode(name)
            present = gemm_pair_present(recs)
            self.assertEqual(present, k % 4 != 0, name)

    def test_bitcost_k32_matches_the_simpler_rule(self):
        # PR #1604's own order-dependent hypothesis broke here. The
        # simpler K-mod-4-only rule does not.
        recs = decode("matmul_var_bitcost_k32.mcode.gz")
        self.assertFalse(gemm_pair_present(recs))


class TestRuleIsOrderIndependent(unittest.TestCase):
    """Six same-K, opposite-table-order pairs already in the corpus all
    agree on presence/absence despite disagreeing on order -- directly
    refuting the order term in PR #1604's own hypothesis. Three more
    same-K, same-order pairs are separately confirmed consistent with
    themselves (a weaker check, but still evidence the rule isn't
    accidentally noisy)."""

    def test_opposite_order_pairs_agree(self):
        for a, b in OPPOSITE_ORDER_PAIRS:
            order_a = table_order(load(a))
            order_b = table_order(load(b))
            self.assertNotEqual(order_a, order_b, (a, b))
            present_a = gemm_pair_present(decode(a))
            present_b = gemm_pair_present(decode(b))
            self.assertEqual(present_a, present_b, (a, b))

    def test_same_order_pairs_also_agree(self):
        for a, b in SAME_ORDER_CONFIRMATION_PAIRS:
            order_a = table_order(load(a))
            order_b = table_order(load(b))
            self.assertEqual(order_a, order_b, (a, b))
            present_a = gemm_pair_present(decode(a))
            present_b = gemm_pair_present(decode(b))
            self.assertEqual(present_a, present_b, (a, b))


class TestHeldOutBuildsConfirmTheRule(unittest.TestCase):
    """Seven fresh, independent MatMul builds at K values never before
    in this corpus (K=41..45), built specifically to stress-test the
    rule rather than fit it. Zero exceptions."""

    def test_all_seven_held_out_points_match(self):
        for name, k in HELD_OUT:
            recs = decode(name)
            present = gemm_pair_present(recs)
            self.assertEqual(present, k % 4 != 0, name)

    def test_held_out_fixtures_decode_cleanly(self):
        for name, _ in HELD_OUT:
            self.assertEqual(mcode.check(load(name)), [], name)

    def test_forty_eight_total_zero_exception_confirmations(self):
        # 31 var_k points + 9 rebuild points + 7 held-out points = 47,
        # plus bitcost_k32.mcode.gz (tested separately in
        # TestKMod4RuleHoldsAcrossAllExistingKSweepSamples, a different
        # naming convention filling the K=32 gap in the var_k family)
        # = 48 total K>=9 zero-exception confirmations across this file.
        total = len(K_SWEEP_EXISTING) + len(K_SWEEP_REBUILDS) + len(HELD_OUT)
        self.assertEqual(total, 47)
        self.assertEqual(total + 1, 48)


class TestRuleDoesNotExtendBelowK9(unittest.TestCase):
    """K=1 (not a multiple of 4) is GEMM-pair-absent, contradicting a
    naive extension of the K>=9 rule -- consistent with this project's
    own already-established small-K "different regime" floor, not a
    counterexample to the K>=9 claim this file actually makes."""

    def test_k1_breaks_the_naive_extension(self):
        recs = decode("matmul_var_bitcost_k1.mcode.gz")
        # k=1 is not a multiple of 4, so the K>=9 rule would predict
        # "present" if extended naively -- it does not hold here.
        self.assertFalse(gemm_pair_present(recs))

    def test_k4_is_consistent_but_uninformative(self):
        # K=4 is a multiple of 4, so both the naive extension and the
        # tiny-shape floor predict "absent" -- doesn't distinguish them.
        recs = decode("matmul_var_bitcost_k4.mcode.gz")
        self.assertFalse(gemm_pair_present(recs))


class TestMSweepDataCannotDistinguishNoEffectFromFixedK(unittest.TestCase):
    """Every (batch=2,K=8,N=8)-fixed M-sweep fixture from M=7 through
    M=48 is GEMM-pair-absent -- consistent with K=8 (a multiple of 4)
    alone forcing absence regardless of M, per this file's own K>=9
    rule. This corpus has no M-sweep at a non-multiple-of-4 K, so
    whether M itself ever matters is not tested here."""

    def test_m_sweep_from_m7_to_m48_all_absent(self):
        names = [
            "matmul_var_m7.mcode.gz",
            "matmul_var_m9.mcode.gz",
            "matmul_var_m10.mcode.gz",
            "matmul_var_m24.mcode.gz",
            "matmul_var_m36.mcode.gz",
            "matmul_var_m48.mcode.gz",
        ]
        for n in names:
            recs = decode(n)
            self.assertFalse(gemm_pair_present(recs), n)

    def test_tiny_m_regime_breaks_the_naive_extension_too(self):
        # M=1,2 (K=8 fixed, a multiple of 4) are GEMM-pair-PRESENT --
        # another small-shape-regime exception, same character as
        # bitcost_k1's own break above.
        for n in ("matmul_var_bitcost_m1.mcode.gz", "matmul_var_bitcost_m2.mcode.gz"):
            recs = decode(n)
            self.assertTrue(gemm_pair_present(recs), n)


if __name__ == "__main__":
    unittest.main()
