"""Continues `tests/test_axera_bank81_cross_op_check.py` (PR #1608)'s own
generalization pattern: that file found Gemm's ENTIRE `bank=0x81`/`0xe1`
mutual-exclusivity mechanism (the `1024//K-1` formula, the `N=32/33`
boundary, field=144, and more) is not Gemm-specific -- it's a property of
"a matmul-like op with one live tensor and one compile-time-constant
weight matrix," reproduced byte-for-byte by a constant-weight
`MatMul(x, w)`. Every existing MatMul/Mul fixture in this corpus uses TWO
live tensors (the precondition for the `A_offset`/`B_offset`-style
table-order coin flip), so the mechanism never showed up in the routine
probe corpus for either op until PR #1608 built a constant-weight MatMul
on purpose. This file does the same for **Mul**, completing the four-op
picture (Gemm: yes; Conv: no, PR #1608; MatMul-with-constant-weight: yes,
PR #1608; Mul-with-constant-operand: this file).

## A bonus finding along the way: some ALREADY-COMMITTED, ordinary
## two-live-tensor `Mul[1,8]` fixtures already carry bank `0xe1`

Before building anything new, scanning every already-committed
`mul_*.mcode.gz` fixture (26 total) for bank `0x81`/`0xe1` finds bank
`0x81` nowhere, but bank `0xe1` in 11 of them (`mul_1x8_zp25sweep`,
`mul_1x8_universal_stability_r{3,4,6}`, `mul_1x8_w2`, four more
`zp*sweep` variants) -- and the operand is `tests/test_axera_matmul_bank_0xe1_check.py`
(PR #1604)'s own "MatMul-specific" pair (`field=32` operand `09 03 00`,
`field=48` operand `0b 03 00 a1`), NOT the Gemm-matching pair. This is a
new cross-op sighting of PR #1604's own second pair at a tiny,
completely ordinary Mul shape with two live tensors -- no constant
operand needed for THIS pair, unlike the Gemm-matching one this file
goes on to chase. Not decoded further here (which of the 26 fixtures
carry it, and why only some, is a separate question from this file's
own constant-operand focus).

## Building `Mul(x, c)`, `c` a compile-time-constant `[K,N]` initializer

Mirroring PR #1608's own constant-weight MatMul construction (a
standalone build script bypassing `pulsar2_docker.py`'s single-image-
classifier config helper with a hand-written `Numpy`-calibration config,
matching `scripts/axera/calib_search.py`'s own established pattern; a
pre-built `onnxsim_cpp2py_export.abi3.so` + `version.py` borrowed from a
sibling checkout into this worktree's `onnxsim/` package at build time,
neither committed -- confirmed gitignored), 9 shapes were built at
`(batch, K, N)` element counts spanning from well below Gemm's own
`K*N<=32768`-ish cap to nearly 8.4 million elements.

## Result: bank `0x81` NEVER appears for Mul, even at the largest shape
## tested -- but bank `0xe1`'s Gemm-matching field=32/48 pair DOES, at a
## much higher threshold than Gemm's own

| shape (`K`,`N`) | `K*N` | bank `0x81` | bank `0xe1` |
| --- | --- | --- | --- |
| 512, 16 | 8,192 | absent | absent |
| 512, 64 | 32,768 (Gemm's own cap, exactly) | absent | absent |
| 512, 65 | 33,280 | absent | absent |
| 512, 200 | 102,400 | absent | field=32/48/64 present |
| 512, 500 | 256,000 | absent | field=32/48/64 present |
| 512, 1000 | 512,000 | absent | field=32/48/64 present |
| 512, 4096 | 2,097,152 | absent | field=32/48 present (no field=64) |
| 2048, 4096 | 8,388,608 | absent | field=32/48 present, plus ~60 field=240 records |

**Bank `0x81` is a clean, confident negative across this entire range** --
zero instances at any shape, including the largest (`K=2048,N=4096`,
over 250x Gemm's own `N=33` trigger point in raw element count). This is
consistent with `0x81`'s own mechanism (the `1024//K-1` formula, keyed
on a real matrix-multiply CONTRACTION dimension `K`) being specific to
ops that actually contract over a shared dimension -- Mul is purely
elementwise, with no `K`-summation at all, so there is no dimension for
`0x81`'s own formula to apply to even in principle. This is a genuine,
informative structural boundary, not just an untested gap: bank `0x81`'s
own mechanism does not generalize to elementwise ops.

**Bank `0xe1`'s Gemm-matching field=32/48 pair (`33 03 00` / `35 03 00
[a1]`) DOES appear for Mul, confirming it generalizes across ALL FOUR
main ops now** (Gemm, Conv `tests/test_axera_conv_dilation8_length_growth.py`
PR #1602, MatMul `tests/test_axera_matmul_bank_0xe1_check.py` PR #1604,
and now Mul) -- but Mul's own threshold sits far above Gemm's own
`K*N<=32768` cap: absent at `K*N=33,280` (`K=512,N=65`), present at
`K*N=102,400` (`K=512,N=200`). The exact crossing point between those
two is not bisected further here. This matches this session's own
repeated finding (`tests/test_axera_gemm_e1_periodicity_check.py`
PR #1609, `tests/test_axera_matmul_gemm_pair_trigger_search.py` PR #1607)
that this pair's VALUES are shared codec-wide while its TRIGGER
mechanism is genuinely different per op -- Mul is simply a fourth,
distinct trigger, not reducible to any of the other three.

## Two honest wrinkles, flagged rather than forced

1. **Field=48's own operand form is not fixed.** At `K=512,N=200/500/1000`
   it's a 3-byte `35 03 00` (missing the trailing `a1` every other op's
   version has); at `K=512,N=4096` it grows to the full 4-byte
   `35 03 00 a1`, matching Gemm/Conv/MatMul exactly. The exact `N` where
   this form changes (somewhere between 1000 and 4096) is not pinned
   down here.
2. **An extra field=64 record** (`37 83 1c 01`) accompanies the pair at
   `K=512,N=200/500/1000` but disappears by `N=4096`, and a much larger
   family of ~60 field=240 records (a repeating tiling-like pattern,
   values like `b5 07 00`/`b5 01 00 a1`/`b3 07 00`) appears only at the
   largest shape (`K=2048,N=4096`) -- plausibly scheduling/tiling
   overhead specific to a shape this much larger, not decoded further.

**Content-stable across independent rebuilds**: two independently-seeded
`(K=512,N=1000)` builds show byte-identical bank `0x81`/`0xe1` content --
real compiled content, not per-build allocator noise.

## What this completes

The four-op picture for Gemm's own `0x81`/`0xe1` mutual-exclusivity
mechanism is now closed: Gemm (native), MatMul-with-constant-weight
(full reproduction, PR #1608), Conv (neither bank at any shape tested,
PR #1608), and Mul-with-constant-operand (this file: `0xe1`'s shared
sub-content only, `0x81` a clean negative). The overall shape of the
finding -- a shared vocabulary of constant VALUES with per-op-family
TRIGGER mechanisms, and one bank (`0x81`) whose mechanism needs a real
contraction dimension elementwise ops simply do not have -- is
consistent across everything this session's own cross-op `0x81`/`0xe1`
investigation has found.
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

N16 = "mul_bank81_probe_k512n16.mcode.gz"
N64 = "mul_bank81_probe_k512n64.mcode.gz"
N65 = "mul_bank81_probe_k512n65.mcode.gz"
N200 = "mul_bank81_probe_k512n200.mcode.gz"
N500 = "mul_bank81_probe_k512n500.mcode.gz"
N1000 = "mul_bank81_probe_k512n1000.mcode.gz"
N1000_R1 = "mul_bank81_probe_k512n1000_r1.mcode.gz"
N4096 = "mul_bank81_probe_k512n4096.mcode.gz"
K2048N4096 = "mul_bank81_probe_k2048n4096.mcode.gz"

ALL_NEW_FIXTURES = [
    N16,
    N64,
    N65,
    N200,
    N500,
    N1000,
    N1000_R1,
    N4096,
    K2048N4096,
]

# The MatMul-specific pair from tests/test_axera_matmul_bank_0xe1_check.py
# (PR #1604), found here at some already-committed, ordinary two-live-
# tensor Mul[1,8] fixtures -- not the Gemm-matching pair this file's own
# constant-operand builds go on to chase.
MATMUL_SPECIFIC_F32 = b"\x09\x03\x00"
MATMUL_SPECIFIC_F48 = b"\x0b\x03\x00\xa1"

GEMM_MATCHING_F32 = b"\x33\x03\x00"


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


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestExistingMulCorpusNeverShowsBank0x81ButSomeShowE1(unittest.TestCase):
    """The pre-existing, ordinary two-live-tensor Mul corpus never
    carries bank 0x81 -- but 11 of its 26 fixtures already carry bank
    0xe1's own MatMul-specific pair (PR #1604), a new cross-op sighting
    that needed no constant-operand trick at all."""

    def test_no_existing_mul_fixture_carries_bank_0x81(self):
        names = [
            n
            for n in os.listdir(FIX)
            if n.startswith("mul_")
            and n.endswith(".mcode.gz")
            and n not in ALL_NEW_FIXTURES
        ]
        self.assertGreater(len(names), 15, "expected the existing Mul corpus")
        for n in names:
            self.assertEqual(bank_records(decode(n), 0x81), [], n)

    def test_at_least_some_existing_mul_fixtures_carry_e1(self):
        names = [
            n
            for n in os.listdir(FIX)
            if n.startswith("mul_")
            and n.endswith(".mcode.gz")
            and n not in ALL_NEW_FIXTURES
        ]
        carriers = [n for n in names if bank_records(decode(n), 0xE1)]
        self.assertGreaterEqual(len(carriers), 5, carriers)
        for n in carriers:
            e1 = bank_records(decode(n), 0xE1)
            fields = {f: op for f, op in e1}
            self.assertEqual(fields.get(32), MATMUL_SPECIFIC_F32, n)
            self.assertEqual(fields.get(48), MATMUL_SPECIFIC_F48, n)


class TestBank0x81IsACleanNegativeAcrossEveryShape(unittest.TestCase):
    """Bank 0x81 never appears for a constant-operand Mul, at any shape
    from 8,192 to 8,388,608 total elements -- consistent with its own
    1024//K-1 formula needing a real contraction dimension that
    elementwise Mul simply does not have."""

    def test_zero_bank81_records_at_every_shape(self):
        for name in ALL_NEW_FIXTURES:
            self.assertEqual(bank_records(decode(name), 0x81), [], name)


class TestBank0xe1sGemmMatchingPairAppearsAboveAHigherThreshold(unittest.TestCase):
    """Unlike Gemm's own K*N<=32768-ish cap, Mul's own threshold for the
    Gemm-matching pair sits between K*N=33,280 (absent) and K*N=102,400
    (present) -- a genuinely different, higher trigger for the same
    shared constant values."""

    def test_absent_at_and_below_gemms_own_cap(self):
        for name in (N16, N64, N65):
            e1 = bank_records(decode(name), 0xE1)
            self.assertEqual(e1, [], name)

    def test_present_from_n200_upward(self):
        for name in (N200, N500, N1000, N4096, K2048N4096):
            e1 = bank_records(decode(name), 0xE1)
            fields = {f: op for f, op in e1}
            self.assertEqual(fields.get(32), GEMM_MATCHING_F32, name)
            self.assertIn(48, fields, name)


class TestField48sOperandFormChangesButField32DoesNot(unittest.TestCase):
    """Honest wrinkle: field=48's own operand grows from a 3-byte form
    (missing the trailing a1 byte every other op's version has) at
    N=200/500/1000 to the full 4-byte form, matching Gemm/Conv/MatMul
    exactly, by N=4096. field=32 is stable throughout."""

    def test_field48_is_three_bytes_at_moderate_n(self):
        for name in (N200, N500, N1000):
            e1 = dict(bank_records(decode(name), 0xE1))
            self.assertEqual(e1[48], b"\x35\x03\x00", name)

    def test_field48_grows_to_four_bytes_matching_other_ops_at_n4096(self):
        for name in (N4096, K2048N4096):
            e1 = dict(bank_records(decode(name), 0xE1))
            self.assertEqual(e1[48], b"\x35\x03\x00\xa1", name)

    def test_field32_is_stable_across_every_present_shape(self):
        for name in (N200, N500, N1000, N4096, K2048N4096):
            e1 = dict(bank_records(decode(name), 0xE1))
            self.assertEqual(e1[32], GEMM_MATCHING_F32, name)


class TestExtraField64AppearsOnlyAtModerateShapes(unittest.TestCase):
    """An extra field=64 record accompanies the pair at N=200/500/1000
    but disappears by N=4096 -- a real, honestly-flagged wrinkle, not
    decoded further."""

    def test_field64_present_at_moderate_n(self):
        for name in (N200, N500, N1000):
            e1 = dict(bank_records(decode(name), 0xE1))
            self.assertEqual(e1.get(64), b"\x37\x83\x1c\x01", name)

    def test_field64_absent_at_the_two_largest_shapes(self):
        for name in (N4096, K2048N4096):
            e1 = dict(bank_records(decode(name), 0xE1))
            self.assertNotIn(64, e1, name)


class TestLargestShapeShowsExtraField240Records(unittest.TestCase):
    """Only the largest shape (K=2048,N=4096, 8.4M elements) shows a
    large family of field=240 records -- plausibly tiling/scheduling
    overhead specific to a shape this much larger, not decoded here."""

    def test_many_field240_records_only_at_the_largest_shape(self):
        recs = decode(K2048N4096)
        f240 = bank_records(recs, 0xE1)
        count_240 = sum(1 for f, _ in f240 if f == 240)
        self.assertGreater(count_240, 30, count_240)

    def test_no_field240_records_at_smaller_shapes(self):
        for name in (N16, N64, N65, N200, N500, N1000, N4096):
            e1 = bank_records(decode(name), 0xE1)
            count_240 = sum(1 for f, _ in e1 if f == 240)
            self.assertEqual(count_240, 0, name)


class TestContentStableAcrossIndependentRebuilds(unittest.TestCase):
    """Two independently-seeded (K=512, N=1000) rebuilds show byte-
    identical bank 0x81/0xe1 content -- real compiled content, not
    per-build allocator noise."""

    def test_two_rebuilds_agree(self):
        a = (bank_records(decode(N1000), 0x81), bank_records(decode(N1000), 0xE1))
        b = (
            bank_records(decode(N1000_R1), 0x81),
            bank_records(decode(N1000_R1), 0xE1),
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
