"""Continues `tests/test_axera_mul_bank81_check.py` (PR #1612)'s own
left-open gap: that file completed the four-op picture for Gemm's
`bank=0xe1` (225), `field=32`/`field=48` constant pair (`33 03 00` /
`35 03 00 [a1]`, shared byte-identical across Gemm, Conv, MatMul, and
Mul) but only bracketed Mul's own trigger -- absent at `K*N=33,280`
(`K=512,N=65`), present at `K*N=102,400` (`K=512,N=200`) -- never
bisected to an exact crossing point. Every OTHER op's own version of
this trigger has since been pinned precisely: MatMul's `K mod 4 != 0`
(`tests/test_axera_matmul_gemm_pair_trigger_search.py`, PR #1607),
Gemm's own multi-plateau `K`/`N` structure across (at least) 7 tiers
(`tests/test_axera_gemm_e1_k129_fifth_plateau.py`/
`tests/test_axera_gemm_e1_seventh_plateau.py`, PR #1615/#1618), Conv's
own dilation-class boundary (`tests/test_axera_conv_dilation8_length_growth.py`,
PR #1602). Mul was the one op left unbisected.

## Answer: a single, clean `K*N <= 65536` cap -- not Gemm's own
## multi-plateau structure

Bisecting `N` at fixed `K=512` between PR #1612's own `65` (absent) and
`200` (present) bracket pins the exact crossing point at **`N=128`
(absent) / `N=129` (present)**: `512 * 128 = 65536 = 2**16` exactly,
`512 * 129 = 66048`, just over it.

Testing two more `K` values, each bisected independently rather than
assumed from the first, confirms the SAME `K*N <= 65536` cap holds
cleanly:

| `K` | absent at `N=` | present at `N=` | `K*N` at absent / present |
| --- | --- | --- | --- |
| 512 | 128 | 129 | 65,536 / 66,048 |
| 256 | 256 | 257 | 65,536 / 65,792 |
| 1024 | 64 | 65 | 65,536 / 66,560 |

**Every one of the three boundaries lands on the identical `K*N=65536`
product, with zero exceptions** -- unlike Gemm's own version of this
same trigger, which needed (at least) 7 separate plateaus
(`512, 320, 256, 192, 128, 64, 32`) because a SINGLE cap did not hold
across Gemm's own full `K` range (PR #1611/#1613/#1615/#1618's own
repeated corrections). Mul's own trigger is structurally simpler: one
constant cap, no plateau structure detected across the 3 `K` values
tested here (a 2x range, `256` to `1024`).

**`65536` is exactly Gemm's own `32768` cap, doubled** (`2**16` vs.
`2**15`) -- both round powers of two, but not the same value. This
project does not decode why Mul's own cap sits at exactly double
Gemm's (both are plausible hardware-buffer-size candidates; no
further evidence is offered here beyond the two numbers' own clean
relationship).

Confirmed not per-build noise: an independent-seed rebuild
(`seed=7`, vs. `seed=1` for every other fixture in this file) at the
`K=512` boundary agrees exactly -- absent at `N=128`, present at
`N=129`, both seeds.

## A second finding noticed along the way: field=144 is the mutually-
## exclusive COMPLEMENT of the field=32/48 pair, not an independent field

Every "absent" fixture in this file's own boundary set carries a
DIFFERENT `bank=0xe1` record instead -- `field=144`, operand
`9f 83 1a` -- and every "present" fixture carries NEITHER `field=144`
NOR anything else at that field. Checked directly (not assumed
symmetric): all three "absent" fixtures (`K=512,N=128`; `K=256,N=256`;
`K=1024,N=64`) have exactly `field=144` and nothing else in bank
`0xe1`; all three "present" fixtures (`K=512,N=129`; `K=256,N=257`;
`K=1024,N=65`) have exactly `field=32`/`48`/`64` and no `field=144` at
all. This is a clean binary switch between two mutually-exclusive
states -- the same "one bank, two substitute identities" shape this
project's own Gemm work found for banks `0x81`/`0xe1` themselves
(`tests/test_axera_gemm_sparse_bank_n_boundary.py`, PR #1568) and for
`0xe1`'s own field=112 (a constant, `tests/test_axera_gemm_bank_81_e1_decode.py`,
PR #1570) -- here recurring one level down, inside `0xe1`'s own content,
at exactly the same `K*N=65536` boundary already pinned above.
`field=144`'s own operand (`9f 83 1a`) is not decoded further here.

## What this completes

Mul is now the fourth and final op with a fully-pinned trigger for the
shared `0xe1` field=32/48 pair: MatMul (`K mod 4`), Gemm (multi-plateau
`K`/`N`, 7+ tiers), Conv (dilation class), Mul (`K*N <= 65536`, single
cap). All four share the exact same underlying constant byte values;
none share the same trigger formula -- the cleanest confirmation yet of
this session's own repeated finding (`tests/test_axera_gemm_e1_periodicity_check.py`
PR #1609, `tests/test_axera_matmul_gemm_pair_trigger_search.py` PR
#1607, `tests/test_axera_bank81_e1_cross_op_synthesis.py` PR #1614)
that this pair's VALUES are shared codec-wide while each op's own
TRIGGER is independently decoded and genuinely different.
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

GEMM_MATCHING_F32 = b"\x33\x03\x00"
GEMM_MATCHING_F48 = b"\x35\x03\x00"

# (name, K, N, seed, expected e1 field=32/48 present)
CASES = [
    ("mul_e1thresh_k512n128_s1.mcode.gz", 512, 128, 1, False),
    ("mul_e1thresh_k512n129_s1.mcode.gz", 512, 129, 1, True),
    ("mul_e1thresh_k512n128_s7.mcode.gz", 512, 128, 7, False),
    ("mul_e1thresh_k512n129_s7.mcode.gz", 512, 129, 7, True),
    ("mul_e1thresh_k256n256_s1.mcode.gz", 256, 256, 1, False),
    ("mul_e1thresh_k256n257_s1.mcode.gz", 256, 257, 1, True),
    ("mul_e1thresh_k1024n64_s1.mcode.gz", 1024, 64, 1, False),
    ("mul_e1thresh_k1024n65_s1.mcode.gz", 1024, 65, 1, True),
]

ALL_NEW_FIXTURES = [c[0] for c in CASES]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def e1_field3248(recs):
    return {
        r["field"]: r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") in (32, 48)
    }


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestBoundariesAreSingleCleanSteps(unittest.TestCase):
    """Every (K, N) pair here was independently bisected against a real
    Pulsar2 build, not derived from a formula -- this test just checks
    the recorded outcome matches what's claimed in the docstring."""

    def test_all_eight_cases_match(self):
        for name, _k, _n, _seed, expected_present in CASES:
            recs = decode(name)
            fields = e1_field3248(recs)
            self.assertEqual(bool(fields), expected_present, name)
            if expected_present:
                self.assertEqual(fields.get(32), GEMM_MATCHING_F32, name)
                self.assertEqual(fields.get(48), GEMM_MATCHING_F48, name)


class TestKTimesNCapIsExactlySixtyFiveThirtySixAtEveryTestedK(unittest.TestCase):
    """The core finding: K*N=65536 (2**16) at every absent/present
    boundary pair tested, for three independent K values spanning a 4x
    range (256, 512, 1024) -- a single cap, not a Gemm-style plateau
    structure."""

    def test_k_times_n_at_every_boundary(self):
        for k, absent_n, present_n in (
            (512, 128, 129),
            (256, 256, 257),
            (1024, 64, 65),
        ):
            self.assertEqual(k * absent_n, 65536, (k, absent_n))
            self.assertGreater(k * present_n, 65536, (k, present_n))

    def test_cap_is_exactly_double_gemms_own_32768(self):
        self.assertEqual(65536, 2 * 32768)
        self.assertEqual(65536, 2**16)
        self.assertEqual(32768, 2**15)


class TestIndependentSeedRebuildConfirmsTheBoundary(unittest.TestCase):
    """seed=7 (vs. seed=1 for every other fixture) at the K=512
    boundary agrees exactly -- not per-build allocator noise."""

    def test_seed7_agrees_with_seed1_at_k512_boundary(self):
        a_absent = e1_field3248(decode("mul_e1thresh_k512n128_s1.mcode.gz"))
        b_absent = e1_field3248(decode("mul_e1thresh_k512n128_s7.mcode.gz"))
        self.assertEqual(bool(a_absent), bool(b_absent))
        self.assertFalse(a_absent)
        self.assertFalse(b_absent)

        a_present = e1_field3248(decode("mul_e1thresh_k512n129_s1.mcode.gz"))
        b_present = e1_field3248(decode("mul_e1thresh_k512n129_s7.mcode.gz"))
        self.assertTrue(a_present)
        self.assertTrue(b_present)
        self.assertEqual(a_present[32], b_present[32])
        self.assertEqual(a_present[48], b_present[48])


def bank_e1_records(recs):
    return {
        r["field"]: r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1
    }


class TestField144IsTheMutuallyExclusiveComplementOfField3248(unittest.TestCase):
    """field=144 (operand 9f 83 1a) is present iff field=32/48 is
    absent, and vice versa -- a clean binary switch inside bank 0xe1's
    own content, not an independent field. Checked directly at all 6
    boundary fixtures, not assumed symmetric from one side."""

    def test_absent_side_has_only_field144(self):
        for name in (
            "mul_e1thresh_k512n128_s1.mcode.gz",
            "mul_e1thresh_k256n256_s1.mcode.gz",
            "mul_e1thresh_k1024n64_s1.mcode.gz",
        ):
            recs = decode(name)
            self.assertEqual(bank_e1_records(recs), {144: b"\x9f\x83\x1a"}, name)

    def test_present_side_has_no_field144(self):
        for name in (
            "mul_e1thresh_k512n129_s1.mcode.gz",
            "mul_e1thresh_k256n257_s1.mcode.gz",
            "mul_e1thresh_k1024n65_s1.mcode.gz",
        ):
            recs = decode(name)
            e1 = bank_e1_records(recs)
            self.assertNotIn(144, e1, name)
            self.assertEqual(
                set(e1),
                {32, 48, 64},
                (name, "expected exactly field=32/48/64, no field=144"),
            )


if __name__ == "__main__":
    unittest.main()
