"""Checks whether Gemm's sparse bank `0x81` -- the mutually-exclusive
substitute partner of bank `0xe1`, whose own `field=32`/`field=48`
constant pair was already found to generalize across all three main ops
(`tests/test_axera_gemm_bank_81_e1_decode.py` PR #1570, Gemm;
`tests/test_axera_conv_dilation8_length_growth.py` PR #1602, Conv;
`tests/test_axera_matmul_bank_0xe1_check.py` PR #1604 /
`tests/test_axera_matmul_gemm_pair_trigger_search.py` PR #1607, MatMul)
-- ALSO generalizes to Conv and MatMul, or is a Gemm-specific bank.

Gemm's own already-decoded content for this pair
(`tests/test_axera_gemm_sparse_bank_n_boundary.py` PR #1568,
`tests/test_axera_gemm_bank_81_e1_decode.py` PR #1570), at fixed
`M=1`:

- Bank `0x81` field=192's third operand byte is `1024 // K - 1`.
- Bank `0x81`/`0xe1` are mutually exclusive: `N<=32` uses `0xe1` alone,
  `N>=33` switches to `0x81` alone (the boundary sits exactly between
  32 and 33).
- Bank `0xe1` field=112 is a constant `35 81 1a` at every small-`N`
  shape tested.
- At large `N` (`N>=100`), `0xe1`'s own field=32/48 pair (the part that
  DOES already generalize across ops) coexists alongside `0x81`.
- Bank `0x81` field=144 (`08 03 00 a1`) appears at *some* large-`N`
  shapes and not others, unexplained (`tests/test_axera_gemm_field144_puzzle.py`
  PR #1573, `tests/test_axera_gemm_field144_q_rule.py` PR #1575).

## First pass: the existing 148 MatMul + 140 Conv fixtures in this
## corpus never carry bank `0x81` at all

Scanning every already-committed MatMul and Conv fixture in this
corpus (no new builds needed for this check) finds **zero** instances
of bank `0x81` in either op -- consistent with PR #1604's own
observation that MatMul's OTHER analogous banks (`0xe1`) needed a
specific shape trigger, not present in most of the corpus's own
routine probe shapes.

## The real reason: every existing MatMul fixture has TWO live tensors
## (`A`, `B`), not Gemm's own one-live-tensor-plus-constant-weight setup

Every MatMul fixture already in this corpus is built the way
`tests/test_axera_matmul_offset_table_coinflip.py` and its many
descendants always have: both `A` and `B` are live, non-constant graph
inputs (the precondition for the `A_offset`/`B_offset` table-order coin
flip this project has repeatedly relied on). Gemm's own bank `0x81`/
`0xe1` decode, by contrast, was always built with `B`/`C` forced to
compile-time constants (Pulsar2's own established behavior for `Gemm`)
-- a structurally different graph shape that no existing MatMul fixture
in this corpus actually has.

## Building a Gemm-equivalent MatMul (constant weight, single live
## tensor) recovers Gemm's ENTIRE `0x81`/`0xe1` mechanism, exactly

`MatMul(x, w)` with `w` a compile-time-constant `[K, N]` initializer
(the closest possible MatMul analogue of Gemm's own `B`) reproduces
every one of Gemm's own decoded findings, at the identical `N=32/33`
boundary and the identical `1024 // K - 1` formula:

| shape (`M=1`) | bank `0x81` field=192 | `1024//K-1` | bank `0xe1` |
| --- | --- | --- | --- |
| `K=512, N=16` | absent | -- | field=112 `\\x0d\\x81\\x1a` |
| `K=512, N=32` | absent | -- | field=112 `\\x35\\x81\\x1a` (**matches Gemm's own constant exactly**) |
| `K=512, N=33` | present, `01` | 1 | absent |
| `K=512, N=1000` | present, `01` (x2) + field=144 | 1 | field=32/48 pair present |
| `K=256, N=1000` | present, `03` (x2) | 3 | field=32/48 pair present |
| `K=128, N=1000` | present, `07` (x2) + field=144 | 7 | field=32/48 pair present |

**This is a real, previously-unchecked generalization**: bank `0x81`'s
own field=192 formula (`1024 // K - 1`), the exact `N=32`/`N=33`
mutual-exclusivity boundary, the large-`N` field=144 record, and the
large-`N` field=32/48 coexistence pattern are ALL identical between
Gemm and this constant-weight MatMul -- not just similar in character,
byte-identical where Gemm's own values are byte-identical (`4c 05 01`
at `K=512`, `4c 05 03` at `K=256`, `4c 05 07` at `K=128`; `08 03 00 a1`
for field=144; `33 03 00`/`35 03 00 a1` for the field=32/48 pair).

**Content-stable across independent rebuilds**: 3 independently-built
`(K=512, N=1000)` samples with different RNG seeds show byte-identical
bank `0x81`/`0xe1` content -- real compiled content, not per-build
allocator noise.

## One genuine wrinkle, honestly flagged rather than forced: field=112's
## own "constant" is not universal the way Gemm's own version was

Gemm's own field=112 was found constant (`35 81 1a`) across every
`M`/`K` combination PR #1570 tested. Here, at `N=16`, field=112 instead
reads `\\x0d\\x81\\x1a` -- same trailing two bytes, but a DIFFERENT
leading byte than both Gemm's own constant AND this file's own `N=32`
value (which matches Gemm's constant exactly). This means MatMul's
version of field=112 is not a pure constant the way Gemm's was -- it
varies with SOME shape parameter within the "small-N" regime itself
(only `N=16` and `N=32` were checked here; the exact boundary between
the two field=112 forms, and whether it relates to any other already-
known small-shape threshold, is not decoded further in this file).

## Conv: still a clean negative

Two new Conv shapes deliberately built to be "large" in ways plausibly
analogous to Gemm's own `N` trigger -- `cout=64` (a much larger output-
channel count than any prior Conv fixture in this corpus) and
`dilation=16` (twice `tests/test_axera_conv_binary_cluster_higher_dilation.py`
PR #1601's own largest-tested `dilation=8`) -- carry **no** bank `0x81`
or `0xe1` record of any kind. Conv's own only confirmed generalization
of this bank pair remains `0xe1`'s field=32/48 sub-content specifically
(PR #1602's `dilation>=8` finding), not the bank's full mechanism the
way MatMul's constant-weight form now shows.

## What this establishes

Gemm's `0x81`/`0xe1` mutual-exclusivity mechanism is not Gemm-specific
after all -- it is a property of "a matmul-like op with one live tensor
and one compile-time-constant weight matrix," which MatMul can also be
built as (just not the way this project's OWN existing MatMul corpus
happens to have built it, since that corpus's whole purpose was
studying the `A_offset`/`B_offset` coin flip, which requires the
opposite structure). Conv, which has no matching "weight-as-second-
live-matrix-operand" structure at all, only inherits the bank's own
constant SUB-fields (`0xe1`'s field=32/48 pair) without the full
mechanism (the `N`-boundary switch, the `1024//K-1` formula, field=144)
-- consistent with those sub-fields being a smaller, more portable
piece of shared codec vocabulary than the bank's overall triggering
logic.
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

SMALL_N16 = "matmul_bank81_probe_m1k512n16.mcode.gz"
SMALL_N32 = "matmul_bank81_probe_m1k512n32.mcode.gz"
BOUNDARY_N33 = "matmul_bank81_probe_m1k512n33.mcode.gz"
LARGE_K512 = "matmul_bank81_probe_m1k512n1000.mcode.gz"
LARGE_K512_R1 = "matmul_bank81_probe_m1k512n1000_r1.mcode.gz"
LARGE_K512_R2 = "matmul_bank81_probe_m1k512n1000_r2.mcode.gz"
LARGE_K256 = "matmul_bank81_probe_m1k256n1000.mcode.gz"
LARGE_K128 = "matmul_bank81_probe_m1k128n1000.mcode.gz"
CONV_COUT64 = "conv_bank81_probe_cout64.mcode.gz"
CONV_DILATION16 = "conv_bank81_probe_dilation16.mcode.gz"

ALL_NEW_FIXTURES = [
    SMALL_N16,
    SMALL_N32,
    BOUNDARY_N33,
    LARGE_K512,
    LARGE_K512_R1,
    LARGE_K512_R2,
    LARGE_K256,
    LARGE_K128,
    CONV_COUT64,
    CONV_DILATION16,
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


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestExistingCorpusNeverShowsBank0x81(unittest.TestCase):
    """Every already-committed MatMul/Conv fixture (two-live-tensor
    MatMul, ordinary Conv) has zero bank=0x81 records -- confirms the
    mechanism needs the constant-weight structure this file builds
    fresh, not something already latent in the routine probe corpus."""

    def test_no_matmul_fixture_carries_bank_0x81(self):
        fixdir = FIX
        names = [
            n
            for n in os.listdir(fixdir)
            if n.startswith("matmul_")
            and n.endswith(".mcode.gz")
            and n not in ALL_NEW_FIXTURES
        ]
        self.assertGreater(len(names), 100, "expected the large existing MatMul corpus")
        for n in names:
            recs = decode(n)
            self.assertEqual(bank_records(recs, 0x81), [], n)

    def test_no_conv_fixture_carries_bank_0x81(self):
        fixdir = FIX
        names = [
            n
            for n in os.listdir(fixdir)
            if n.startswith("conv_")
            and n.endswith(".mcode.gz")
            and n not in ALL_NEW_FIXTURES
        ]
        self.assertGreater(len(names), 100, "expected the large existing Conv corpus")
        for n in names:
            recs = decode(n)
            self.assertEqual(bank_records(recs, 0x81), [], n)


class TestBank81FormulaMatchesGemmExactly(unittest.TestCase):
    """Bank 0x81 field=192's third operand byte is 1024 // K - 1 at
    every K tested, byte-identical to Gemm's own PR #1570 values."""

    def test_k512_gives_operand_matching_1024_over_k_minus_1(self):
        recs = decode(LARGE_K512)
        hits = bank_records(recs, 0x81)
        f192 = [op for f, op in hits if f == 192]
        self.assertTrue(f192, "expected at least one field=192 record")
        for op in f192:
            self.assertEqual(op, b"\x4c\x05\x01")
            self.assertEqual(op[2], 1024 // 512 - 1)

    def test_k256_gives_operand_matching_1024_over_k_minus_1(self):
        recs = decode(LARGE_K256)
        f192 = [op for f, op in bank_records(recs, 0x81) if f == 192]
        self.assertTrue(f192)
        for op in f192:
            self.assertEqual(op, b"\x4c\x05\x03")
            self.assertEqual(op[2], 1024 // 256 - 1)

    def test_k128_gives_operand_matching_1024_over_k_minus_1(self):
        recs = decode(LARGE_K128)
        f192 = [op for f, op in bank_records(recs, 0x81) if f == 192]
        self.assertTrue(f192)
        for op in f192:
            self.assertEqual(op, b"\x4c\x05\x07")
            self.assertEqual(op[2], 1024 // 128 - 1)


class TestN32N33BoundaryMatchesGemmExactly(unittest.TestCase):
    """Reproduces Gemm's own exact N=32/N=33 mutually-exclusive
    switch (tests/test_axera_gemm_sparse_bank_n_boundary.py, PR #1568)
    at the identical N values, K=512, M=1."""

    def test_n32_uses_0xe1_alone(self):
        recs = decode(SMALL_N32)
        self.assertEqual(bank_records(recs, 0x81), [])
        e1 = bank_records(recs, 0xE1)
        self.assertEqual(len(e1), 1)
        self.assertEqual(e1[0], (112, b"\x35\x81\x1a"))

    def test_n33_uses_0x81_alone(self):
        recs = decode(BOUNDARY_N33)
        self.assertEqual(bank_records(recs, 0xE1), [])
        b81 = bank_records(recs, 0x81)
        self.assertEqual(len(b81), 1)
        self.assertEqual(b81[0], (192, b"\x4c\x05\x01"))

    def test_n32_field112_matches_gemms_own_constant_exactly(self):
        # Gemm's own field=112 constant, byte-identical.
        recs = decode(SMALL_N32)
        e1 = bank_records(recs, 0xE1)
        self.assertEqual(e1[0][1], b"\x35\x81\x1a")


class TestField112DiffersAtN16FromGemmsUniversalConstant(unittest.TestCase):
    """Honest wrinkle: N=16 shows a DIFFERENT leading byte than both
    Gemm's own constant and this file's own N=32 value -- MatMul's
    field=112 is not a pure constant across the whole small-N regime
    the way Gemm's was found to be."""

    def test_n16_field112_has_different_leading_byte(self):
        recs = decode(SMALL_N16)
        e1 = bank_records(recs, 0xE1)
        self.assertEqual(len(e1), 1)
        field, op = e1[0]
        self.assertEqual(field, 112)
        self.assertEqual(op, b"\x0d\x81\x1a")
        self.assertNotEqual(op, b"\x35\x81\x1a")

    def test_n16_and_n32_share_the_same_trailing_two_bytes(self):
        n16 = bank_records(decode(SMALL_N16), 0xE1)[0][1]
        n32 = bank_records(decode(SMALL_N32), 0xE1)[0][1]
        self.assertEqual(n16[1:], n32[1:])
        self.assertNotEqual(n16[0], n32[0])


class TestLargeNAlsoShowsField144AndTheE1Pair(unittest.TestCase):
    """At large N, bank 0x81's field=144 (present at some Gemm shapes,
    absent at others per PR #1573/#1575's own still-open puzzle) and
    bank 0xe1's own already-cross-op-confirmed field=32/48 pair both
    coexist alongside field=192 -- matching Gemm's own large-N
    structure exactly."""

    def test_k512_and_k128_show_field144_k256_does_not(self):
        # Mirrors Gemm's own PR #1573/#1575 finding that field=144's
        # presence does not reduce to a simple function of K alone --
        # not re-litigated here, just reconfirmed as present/absent in
        # the expected uneven pattern.
        self.assertIn(
            (144, b"\x08\x03\x00\xa1"), bank_records(decode(LARGE_K512), 0x81)
        )
        self.assertIn(
            (144, b"\x08\x03\x00\xa1"), bank_records(decode(LARGE_K128), 0x81)
        )
        self.assertNotIn(144, [f for f, _ in bank_records(decode(LARGE_K256), 0x81)])

    def test_e1_pair_present_at_every_large_n_shape(self):
        for name in (LARGE_K512, LARGE_K256, LARGE_K128):
            e1 = bank_records(decode(name), 0xE1)
            self.assertIn((32, b"\x33\x03\x00"), e1)
            self.assertIn((48, b"\x35\x03\x00\xa1"), e1)


class TestContentStableAcrossIndependentRebuilds(unittest.TestCase):
    """3 independently-seeded (K=512, N=1000) rebuilds show byte-
    identical bank 0x81/0xe1 content -- real compiled content, not
    per-build allocator noise."""

    def test_three_rebuilds_agree(self):
        sets = [
            (bank_records(decode(n), 0x81), bank_records(decode(n), 0xE1))
            for n in (LARGE_K512, LARGE_K512_R1, LARGE_K512_R2)
        ]
        self.assertTrue(all(s == sets[0] for s in sets))


class TestConvRemainsACleanNegative(unittest.TestCase):
    """Two deliberately large Conv shapes (cout=64; dilation=16, 2x
    the previously largest-tested dilation=8) carry no bank=0x81 or
    bank=0xe1 record at all -- Conv only inherits 0xe1's own field=32/48
    sub-content (PR #1602), not the full mutual-exclusivity mechanism."""

    def test_cout64_has_neither_bank(self):
        recs = decode(CONV_COUT64)
        self.assertEqual(bank_records(recs, 0x81), [])
        self.assertEqual(bank_records(recs, 0xE1), [])

    def test_dilation16_has_neither_bank(self):
        recs = decode(CONV_DILATION16)
        self.assertEqual(bank_records(recs, 0x81), [])
        self.assertEqual(bank_records(recs, 0xE1), [])


if __name__ == "__main__":
    unittest.main()
