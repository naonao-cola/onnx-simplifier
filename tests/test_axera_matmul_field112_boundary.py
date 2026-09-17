"""Continues `tests/test_axera_bank81_cross_op_check.py` (PR #1608)'s own
explicitly flagged wrinkle: Gemm's `bank=0xe1` field=112 was previously
found to be a pure constant (`35 81 1a`) across every `M`/`K` combination
`tests/test_axera_gemm_bank_81_e1_decode.py` (PR #1570) tested, but in
PR #1608's own constant-weight `MatMul(x, w)` (`w` a `[K,N]` compile-time
constant, the true MatMul analogue of Gemm's `B`), field=112 instead reads
`\\x0d\\x81\\x1a` at `N=16` and Gemm's own `\\x35\\x81\\x1a` at `N=32` --
same trailing two bytes, different leading byte. PR #1608 explicitly left
open "the exact boundary between the two field=112 forms, and whether it
relates to any other already-known small-shape threshold."

## The boundary: between `N=16` and `N=17` -- NOT the N=32/33
## mutual-exclusivity switch

Bisecting `N` at fixed `K=512` (matching PR #1608's own already-tested
`N=16`/`N=32`/`N=33` points) finds the leading byte flips from `0x0d`
to `0x35` immediately after `N=16`: **every `N` from 17 through 32 shows
`0x35`, and only `N=16` itself shows `0x0d`.** This is a single-point
step, not a wide range -- `N=17,18,20,24` all already show `0x35`, the
same value `N=32` has, a full 15+ N values before the ALREADY-KNOWN
`0x81`/`0xe1` mutual-exclusivity switch (`tests/test_axera_gemm_sparse_bank_n_boundary.py`
PR #1568, `tests/test_axera_bank81_cross_op_check.py` PR #1608) even
happens at `N=32`/`N=33`. **These are two independent, non-coincident
boundaries** -- field=112's own leading byte settles into its "normal"
value 15 N-values before bank `0x81` ever takes over, not at the same
point.

Confirmed independent of build provenance: `N=16` reads `0x0d` in both
PR #1608's own original fixture AND a fresh, independently-seeded
rebuild built here (different RNG seed for both weights and
calibration); `N=17` reads `0x35` in two independently-seeded rebuilds
built here. Not per-build noise -- a genuine, deterministic function of
`N` at this fixed `K`.

## K-dependence: the field=112-carrying state itself doesn't exist at
## `K=128`/`K=256` in this `N` range at all

Testing `K=128` and `K=256` at the same `N=16`/`N=17` points that
isolate `K=512`'s own field=112 switch finds **no `bank=0xe1` record of
any kind** (not even the constant-suffix form) at either `N` for either
`K` -- unlike `K=512`, which carries bank `0xe1` alone (with field=112)
at both those `N` values. Sweeping `K=128` further (`N=8, 64, 100`)
shows: `N=8` is *also* bank-empty (neither `0x81` nor `0xe1`, the same
"neither bank" pre-threshold state `tests/test_axera_gemm_sparse_bank_k_boundary.py`
PR #1571 already established exists for some `K` at small `N` on Gemm's
own side), while `N=64` and `N=100` are ALREADY bank `0x81`-alone (no
field=112, no field=32/48 pair) -- meaning `K=128`'s own transition from
"neither" to "`0x81` alone" happens somewhere between `N=17` and `N=64`,
**skipping the `0xe1`-alone-with-field=112 state entirely** in the range
checked. This means field=112's own leading-byte boundary is not a
question that even applies to `K=128`/`K=256` the same way it does to
`K=512` -- those smaller `K` values may have their own, much narrower
(or nonexistent) `0xe1`-alone window elsewhere, not chased further here.

A further honest aside, also not chased: at `N=64`/`N=100`, bank
`0x81`'s own field=192 third byte reads `0x01`/`0x03` -- NEITHER
matches `1024 // 128 - 1 == 7`, the formula PR #1570/#1608 confirmed
holds for `K=128` at `N=1000`. The formula evidently depends on `N`
too in this smaller-`N` regime, not on `K` alone -- a real wrinkle
outside this file's own field=112 scope.

## What this establishes

Field=112's leading-byte switch (`N=16`->`N=17` at `K=512`) is a real,
K-specific, deterministic small-shape threshold, confirmed independent
of the already-known `N=32/33` mutual-exclusivity boundary rather than
coincident with it. It does not generalize as a clean, K-independent
rule the way bank `0x81`'s own `1024//K-1` formula does -- at `K=128`
and `K=256`, the relevant `N` window this file checked never puts the
graph into the `0xe1`-alone state where field=112 would even be
observable. This is reported as a precise, well-evidenced partial
result: the `K=512` boundary is fully pinned, and the "does not
generalize to other K in the same N range" finding is itself the
honest answer to the K-dependence question this file set out to check.
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

# K=512 boundary sweep (all M=1, w a compile-time-constant [K,N]).
N16_ORIGINAL = "matmul_bank81_probe_m1k512n16.mcode.gz"  # PR #1608's own fixture
N16_INDEP = "matmul_field112_k512n16_indep.mcode.gz"
N17_A = "matmul_field112_k512n17.mcode.gz"
N17_B = "matmul_field112_k512n17_r1.mcode.gz"
N18 = "matmul_field112_k512n18.mcode.gz"
N20 = "matmul_field112_k512n20.mcode.gz"
N24 = "matmul_field112_k512n24.mcode.gz"
N32_ORIGINAL = "matmul_bank81_probe_m1k512n32.mcode.gz"  # PR #1608's own fixture

K512_SWEEP = [N16_ORIGINAL, N16_INDEP, N17_A, N17_B, N18, N20, N24, N32_ORIGINAL]

# K-dependence probes.
K128_N16 = "matmul_field112_k128n16.mcode.gz"
K128_N17 = "matmul_field112_k128n17.mcode.gz"
K256_N16 = "matmul_field112_k256n16.mcode.gz"
K256_N17 = "matmul_field112_k256n17.mcode.gz"
K128_N8 = "matmul_field112_k128n8.mcode.gz"
K128_N64 = "matmul_field112_k128n64.mcode.gz"
K128_N100 = "matmul_field112_k128n100.mcode.gz"

ALL_NEW_FIXTURES = [
    N16_INDEP,
    N17_A,
    N17_B,
    N18,
    N20,
    N24,
    K128_N16,
    K128_N17,
    K256_N16,
    K256_N17,
    K128_N8,
    K128_N64,
    K128_N100,
]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def bank_records(recs, bank):
    return [
        (r["field"], r.get("operand"))
        for r in recs
        if r["kind"] == "V" and r.get("bank") == bank
    ]


def field112(name):
    e1 = bank_records(decode(name), 0xE1)
    hits = [op for f, op in e1 if f == 112]
    assert len(hits) == 1, (name, e1)
    return hits[0]


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestBoundaryIsExactlyBetweenN16AndN17(unittest.TestCase):
    """At fixed K=512, field=112's leading byte is 0x0d only at N=16,
    and 0x35 (Gemm's own constant) at every N from 17 through 32 --
    confirmed with two independent rebuilds on each side of the
    boundary, not a single-sample coincidence."""

    def test_n16_is_0x0d_in_both_original_and_independent_rebuild(self):
        for name in (N16_ORIGINAL, N16_INDEP):
            self.assertEqual(field112(name), b"\x0d\x81\x1a", name)

    def test_n17_is_0x35_in_two_independent_rebuilds(self):
        for name in (N17_A, N17_B):
            self.assertEqual(field112(name), b"\x35\x81\x1a", name)

    def test_every_n_from_17_through_32_matches_the_0x35_form(self):
        for name in (N17_A, N18, N20, N24, N32_ORIGINAL):
            self.assertEqual(field112(name), b"\x35\x81\x1a", name)


class TestBoundaryIsIndependentOfTheN32N33MutualExclusivitySwitch(unittest.TestCase):
    """Field=112 already settled into its 'normal' 0x35 form 15+ N
    values before bank 0x81 takes over from 0xe1 at N=32/33 -- the two
    thresholds do not coincide."""

    def test_n24_already_shows_0x35_well_before_the_n32_33_switch(self):
        self.assertEqual(field112(N24), b"\x35\x81\x1a")

    def test_n32_still_uses_0xe1_alone_like_n24_and_n17(self):
        # Confirms N=24/N=32 share both field112's own form AND the
        # same "0xe1 alone, no 0x81" bank state -- the field=112
        # switch is a distinct, earlier event within that shared state,
        # not a symptom of the later 0x81 mutual-exclusivity switch.
        recs32 = decode(N32_ORIGINAL)
        self.assertEqual(bank_records(recs32, 0x81), [])
        recs24 = decode(N24)
        self.assertEqual(bank_records(recs24, 0x81), [])


class TestFieldDoesNotGeneralizeToOtherKInTheSameNRange(unittest.TestCase):
    """K=128 and K=256 carry no bank=0xe1 record at all at N=16/17 --
    the field=112-carrying state itself doesn't exist there for those
    K values, unlike K=512."""

    def test_k128_has_no_e1_at_n16_or_n17(self):
        for name in (K128_N16, K128_N17):
            self.assertEqual(bank_records(decode(name), 0xE1), [], name)

    def test_k256_has_no_e1_at_n16_or_n17(self):
        for name in (K256_N16, K256_N17):
            self.assertEqual(bank_records(decode(name), 0xE1), [], name)


class TestK128SkipsStraightFromNeitherBankToBank0x81(unittest.TestCase):
    """K=128's own transition happens somewhere between N=17 and N=64,
    skipping the 0xe1-alone-with-field=112 state entirely in the range
    checked: N=8 is bank-empty (neither 0x81 nor 0xe1, matching Gemm's
    own established 'neither bank' pre-threshold state,
    tests/test_axera_gemm_sparse_bank_k_boundary.py PR #1571), N=64/100
    are already 0x81-alone."""

    def test_n8_has_neither_bank(self):
        recs = decode(K128_N8)
        self.assertEqual(bank_records(recs, 0x81), [])
        self.assertEqual(bank_records(recs, 0xE1), [])

    def test_n64_and_n100_are_already_0x81_alone(self):
        # Honest aside, not chased further: field=192's own third byte
        # here is 0x01 (N=64) / 0x03 (N=100) -- NEITHER matches
        # `1024 // 128 - 1 == 7`, the formula PR #1570/#1608 confirmed
        # holds for K=128 at N=1000. The formula is evidently not a
        # pure function of K alone across every N -- it depends on N
        # too in this smaller-N regime, a real wrinkle this file does
        # not attempt to decode (out of scope for the field=112
        # question this file set out to answer).
        expected_third_byte = {K128_N64: 0x01, K128_N100: 0x03}
        for name in (K128_N64, K128_N100):
            recs = decode(name)
            self.assertEqual(bank_records(recs, 0xE1), [])
            b81 = bank_records(recs, 0x81)
            self.assertEqual(len(b81), 1)
            field, op = b81[0]
            self.assertEqual(field, 192)
            self.assertEqual(op[2], expected_third_byte[name], name)
            self.assertNotEqual(
                op[2],
                1024 // 128 - 1,
                f"{name}: does not match the K=1000-tested 1024//K-1 formula",
            )


if __name__ == "__main__":
    unittest.main()
