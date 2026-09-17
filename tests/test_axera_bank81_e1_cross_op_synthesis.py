"""Synthesis pass over this session's own 10-PR investigation into
Gemm's `bank=0x81`/`0xe1` mutual-exclusivity mechanism and how far it
generalizes across all four of this project's main tracked ops, in the
style of `tests/test_axera_noise_vs_loadbearing_synthesis.py` (PR
#1561), `tests/test_axera_reg8_cross_op_synthesis.py` (PR #1582), and
`tests/test_axera_conv_binary_cluster_synthesis.py` (PR #1603): all
three stepped back from a scattered set of per-PR findings to build one
coherent, directly-reconfirmed picture, adding a real cross-cutting
observation no single contributing PR was positioned to make. This file
does not decode anything new from scratch by building fresh mcode --
every number below is recomputed live from the already-committed
fixtures (`TestFourOpComparisonTableMatchesFixtures` below), not copied
from any source PR's own docstring.

## The contributing PRs, in order

1. `tests/test_axera_gemm_sparse_bank_n_boundary.py` (PR #1568) +
   `tests/test_axera_gemm_bank_81_e1_decode.py` (PR #1570) -- original
   Gemm decode: `0x81`/`0xe1` mutually exclusive at `N=32`/`33` (fixed
   `M=1,K=512`); `0x81` field=192 = `1024//K-1`; `0xe1` field=112 a
   constant `35 81 1a`; a field=32/48 pair at large `N` left
   uncharacterized.
2. `tests/test_axera_conv_dilation8_length_growth.py` (PR #1602) --
   Conv's `dilation>=8` mcode carries the exact same `0xe1` field=32/48
   constant pair, the first sighting outside Gemm.
3. `tests/test_axera_matmul_bank_0xe1_check.py` (PR #1604) -- the pair
   generalizes to MatMul too (58/135 existing fixtures), plus a
   MatMul-specific second pair at the same two field slots.
4. `tests/test_axera_matmul_gemm_pair_trigger_search.py` (PR #1607) --
   decodes MatMul's own trigger as `K mod 4 != 0` (`K>=9`), refuting an
   earlier order-based hypothesis.
5. `tests/test_axera_bank81_cross_op_check.py` (PR #1608) -- the
   biggest single result: bank `0x81`'s ENTIRE mechanism (not just
   `0xe1`'s sub-content) generalizes to a constant-weight
   `MatMul(x, w)`, byte-identical to Gemm's own decoded values; Conv
   remains a clean negative for the FULL mechanism even at extreme
   shapes.
6. `tests/test_axera_matmul_field112_boundary.py` (PR #1610) -- pins
   MatMul's own field=112 boundary (`N=16`/`17`) as independent of the
   `N=32`/`33` mutual-exclusivity switch.
7. `tests/test_axera_matmul_e1_m_threshold.py` (PR #1606) -- pins
   MatMul's own `M=23`/`24` order-inversion boundary for its
   MatMul-specific second pair.
8. `tests/test_axera_gemm_e1_periodicity_check.py` (PR #1609) -- refutes
   `K mod 4` for Gemm directly; finds Gemm's own trigger is a genuine
   per-`K` absolute `N` threshold instead.
9. `tests/test_axera_gemm_e1_threshold_formula.py` (PR #1611) -- finds
   Gemm's threshold forms 3 clean plateaus (`K<=64`->`N=512`;
   `K=65-96`->`N=320`; `K=112-128`->`N=256`), proposes a
   `K*N<=32768`-cap rule.
10. `tests/test_axera_gemm_e1_switch_and_fourth_plateau.py` (PR #1613)
    -- directly tests and REFUTES that rule's own held-out prediction
    (the real plateau-2/3 switch is `K=96`/`97`, not the predicted
    `K=102`/`103`); finds a genuine 4th plateau (`N=128`/`129`); finds
    an unresolved anomaly (`K=129` matches neither neighboring
    candidate).
11. `tests/test_axera_mul_bank81_check.py` (PR #1612) -- completes the
    four-op picture: bank `0x81` is a clean negative for Mul at every
    shape tested (up to 8.4M elements); `0xe1`'s Gemm-matching pair
    DOES generalize to Mul, at yet another distinct threshold
    (`K*N` between 33,280 absent and 102,400 present).

## The four-op comparison table, directly reconfirmed against fixtures

| op | bank `0x81` full mechanism | `0xe1` field=32/48 pair | pair's own trigger |
| --- | --- | --- | --- |
| Gemm | **yes** (native) | yes | per-K absolute N threshold, 4 known plateaus (512/320/256/128), not a single formula |
| Conv | **no** (clean negative, even at `cout=64`/`dilation=16`) | yes (`dilation>=8` only) | dilation-class boundary (`dilation<=4` absent, `dilation>=8` present) |
| MatMul | **yes**, but only when built as `MatMul(x, w)` with `w` a compile-time constant (never in the ordinary two-live-tensor corpus) | yes | `K mod 4 != 0` (`K>=9`), order-independent |
| Mul | **no** (clean negative, even with a constant operand, up to 8.4M elements) | yes (constant-operand form only) | `K*N` threshold, bracketed between 33,280 (absent) and 102,400 (present) |

Every one of these headline claims is reconfirmed directly against a
real fixture below (`TestFourOpComparisonTableMatchesFixtures`), not
just restated from each source PR's own prose.

## Why "shared constant values, op-specific trigger mechanisms" is the
## right frame -- and why bank `0x81`'s own mechanism doesn't generalize
## the same way `0xe1`'s sub-content does

The `0xe1` field=32/48 pair's own byte VALUES (`33 03 00` / `35 03 00
[a1]`) are identical across all four ops -- a single, shared piece of
codec vocabulary. But the TRIGGER for when each op's own compiled
mcode includes it is different in every single case: MatMul's is
periodic in `K` alone; Gemm's is a genuine multi-plateau function of
both `K` and `N` together; Conv's is a coarse dilation-class boundary;
Mul's is bracketed but unpinned. None of the four trigger rules
transfers to another op even though they all guard the exact same
constant bytes -- this is not a single shared formula wearing four
different disguises, it is four genuinely different allocator
heuristics that all reach for the same underlying resource
representation when their own (different) conditions are met.

Bank `0x81`'s OWN mechanism (not just `0xe1`'s sub-content) is more
restrictive: it only appears where `PR #1608` found it does --
Gemm natively, and a *constant-weight* MatMul, i.e. any op built with
one live tensor contracting against one compile-time-constant weight
matrix. `1024 // K - 1` is a function of a real contraction dimension
`K`; Conv's own weight is compile-time-constant too, but convolution's
own `K` (the per-tap contraction over `cin*kh*kw`) evidently doesn't
map onto whatever internal resource this specific formula sizes, since
Conv shows a clean negative even at deliberately extreme shapes
(PR #1608's own `cout=64`/`dilation=16` probes). Mul has no contraction
dimension at all -- purely elementwise -- and PR #1612 found bank
`0x81` absolutely absent there too, which is exactly the outcome this
frame predicts rather than merely accommodates.

## What's now established

- Bank `0x81`'s full mechanism (the `1024//K-1` formula, the mutual-
  exclusivity switch, field=144's own presence pattern) requires BOTH a
  compile-time-constant second operand AND a real contraction
  dimension -- present in Gemm (native) and MatMul-as-constant-weight,
  absent in Conv and Mul regardless of shape size.
- The `0xe1` field=32/48 pair's own byte values are a single, shared
  piece of codec vocabulary used by all four ops, but each op's own
  trigger condition for including it is independently decoded and
  genuinely different (K mod 4 for MatMul; a 4-plateau K/N threshold
  structure for Gemm; a dilation-class boundary for Conv; an unpinned
  K*N bracket for Mul).
- Gemm's own K/N threshold structure is NOT a single formula even
  within one op: 4 confirmed plateaus (512, 320, 256, 128), 3 of the 4
  candidate-boundary crossings land exactly on a `K*N<=32768` cap, one
  (`320`'s own upper edge, K=96/97) badly misses that cap, and one
  point (`K=129`) matches neither of its neighboring candidates at all.

## What remains genuinely open

- WHY Pulsar2 chooses `512, 320, 256, 128` (not a clean geometric
  sequence, e.g. `512, 256, 128, 64`) as Gemm's own candidate set -- no
  semantic/hardware reading of these specific numbers is offered by any
  contributing PR.
- The `K=129` anomaly (PR #1613): its own true threshold sits somewhere
  in `(129, 256)`, not decoded.
- Gemm's own 5th-plateau value past `K=300` (PR #1613 confirmed `K=300`
  has moved past the 4th plateau but did not bisect further).
- Mul's own exact `K*N` crossing point (only bracketed between 33,280
  and 102,400, PR #1612).
- MatMul's own field=112 K-dependence (PR #1610 found the field=112-
  carrying state doesn't even exist at `K=128`/`256` in the `N=16`/`17`
  range checked, but did not find where, if anywhere, it does).
- Whether MatMul's own `K mod 4` rule and Gemm's own per-K threshold
  rule are two views of one underlying allocator policy, or genuinely
  unrelated -- this file does not resolve it, and flags it explicitly
  as an open question rather than guessing either way. The two rules
  ARE structurally different in kind (a periodic residue-class test vs.
  a monotonic-in-K threshold-with-plateaus), which is at least mild
  evidence for "genuinely different," but this is not a proof.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def bank_records(recs, bank, field=None):
    return [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("bank") == bank
        and (field is None or r.get("field") == field)
    ]


class TestFourOpComparisonTableMatchesFixtures(unittest.TestCase):
    """Every cell of the module docstring's four-op table, reconfirmed
    directly against a real fixture rather than trusted from any source
    PR's own prose."""

    def test_gemm_bank81_field192_formula(self):
        # tests/test_axera_gemm_bank_81_e1_decode.py, PR #1570.
        recs = decode("gemm_1x512x1000_tb0.mcode.gz")
        ops = {r["operand"] for r in bank_records(recs, 0x81, 192)}
        self.assertEqual(ops, {bytes([0x4C, 0x05, 1024 // 512 - 1])})

    def test_gemm_e1_field112_constant_at_small_n(self):
        recs = decode("gemm_1x512x32.mcode.gz")
        ops = {r["operand"] for r in bank_records(recs, 0xE1, 112)}
        self.assertEqual(ops, {b"\x35\x81\x1a"})

    def test_conv_shows_the_pair_only_at_dilation8_not_dilation4(self):
        # tests/test_axera_conv_dilation8_length_growth.py, PR #1602.
        d8 = decode("conv_dilation8.mcode.gz")
        d4 = decode("conv_dilation4.mcode.gz")
        self.assertEqual({r["operand"] for r in bank_records(d8, 0xE1, 32)}, {GEMM_F32})
        self.assertEqual(bank_records(d4, 0xE1), [])

    def test_conv_never_shows_bank81_full_mechanism(self):
        # tests/test_axera_bank81_cross_op_check.py, PR #1608 -- probed
        # at deliberately extreme shapes, still a clean negative.
        for name in (
            "conv_bank81_probe_cout64.mcode.gz",
            "conv_bank81_probe_dilation16.mcode.gz",
        ):
            recs = decode(name)
            self.assertEqual(bank_records(recs, 0x81), [], name)
            self.assertEqual(bank_records(recs, 0xE1), [], name)

    def test_matmul_constant_weight_reproduces_gemms_bank81_formula(self):
        # tests/test_axera_bank81_cross_op_check.py, PR #1608.
        recs = decode("matmul_bank81_probe_m1k512n1000.mcode.gz")
        ops = {r["operand"] for r in bank_records(recs, 0x81, 192)}
        self.assertEqual(ops, {bytes([0x4C, 0x05, 1024 // 512 - 1])})

    def test_matmul_constant_weight_small_n_has_neither_bank81_nor_gemms_e1_form(
        self,
    ):
        # N=16 is below the N=32/33 switch: 0x81 absent (matches Gemm's
        # own small-N behavior), and field=112 carries the *different*
        # small-N leading byte tests/test_axera_matmul_field112_boundary.py
        # (PR #1610) decoded, not Gemm's own universal constant.
        recs = decode("matmul_bank81_probe_m1k512n16.mcode.gz")
        self.assertEqual(bank_records(recs, 0x81), [])
        ops = {r["operand"] for r in bank_records(recs, 0xE1, 112)}
        self.assertEqual(ops, {b"\x0d\x81\x1a"})

    def test_matmul_two_live_tensor_corpus_follows_k_mod_4_not_gemms_threshold(self):
        # tests/test_axera_matmul_gemm_pair_trigger_search.py, PR #1607.
        k20 = decode("matmul_var_k20.mcode.gz")  # 20 % 4 == 0
        k21 = decode("matmul_var_k21.mcode.gz")  # 21 % 4 == 1
        self.assertNotIn(GEMM_F32, {r["operand"] for r in bank_records(k20, 0xE1, 32)})
        self.assertIn(GEMM_F32, {r["operand"] for r in bank_records(k21, 0xE1, 32)})

    def test_mul_bank81_is_a_clean_negative_even_at_the_largest_probed_shape(self):
        # tests/test_axera_mul_bank81_check.py, PR #1612.
        recs = decode("mul_bank81_probe_k2048n4096.mcode.gz")
        self.assertEqual(bank_records(recs, 0x81), [])

    def test_mul_e1_pair_present_above_its_own_threshold_absent_below(self):
        # K*N=33,280 absent; K*N=102,400 present (PR #1612).
        below = decode("mul_bank81_probe_k512n65.mcode.gz")
        above = decode("mul_bank81_probe_k512n200.mcode.gz")
        self.assertEqual(bank_records(below, 0xE1), [])
        self.assertIn(GEMM_F32, {r["operand"] for r in bank_records(above, 0xE1, 32)})

    def test_mul_bonus_finding_ordinary_two_live_tensor_fixture_carries_the_other_pair(
        self,
    ):
        # PR #1612's own bonus sighting: an ordinary, non-constant-operand
        # Mul[1,8] fixture already carries PR #1604's MatMul-specific
        # (not Gemm-matching) field=32/48 pair.
        recs = decode("mul_1x8_zp25sweep.mcode.gz")
        ops = {r["operand"] for r in bank_records(recs, 0xE1, 32)}
        self.assertEqual(ops, {b"\x09\x03\x00"})
        self.assertNotIn(GEMM_F32, ops)


class TestGemmsOwnFourPlateauStructureIsNotASingleFormula(unittest.TestCase):
    """Reconfirms PR #1611's/#1613's own combined plateau table directly,
    plus the specific held-out-prediction failure and the K=129 anomaly
    that make "one clean formula" the wrong description of Gemm's own
    trigger."""

    PLATEAU_POINTS = {
        # (K, N) -> expected presence, spanning all 4 confirmed plateaus.
        (64, 512): False,
        (64, 513): True,
        (96, 320): False,
        (97, 320): True,  # the REAL switch (not the rule's predicted 102/103)
        (128, 256): False,
        (128, 257): True,
        (256, 128): False,
        (256, 129): True,
    }
    FIXTURE_NAMES = {
        (64, 512): "gemm_1x64x512_e1period.mcode.gz",
        (64, 513): "gemm_1x64x513_e1period.mcode.gz",
        (96, 320): "gemm_1x96x320_e1thresh.mcode.gz",
        (97, 320): "gemm_1x97x320_e1switch.mcode.gz",
        (128, 256): "gemm_1x128x256_e1period.mcode.gz",
        (128, 257): "gemm_1x128x257_e1period.mcode.gz",
        (256, 128): "gemm_1x256x128_e1fourth.mcode.gz",
        (256, 129): "gemm_1x256x129_e1fourth.mcode.gz",
    }

    def _present(self, name):
        recs = decode(name)
        return GEMM_F32 in {r["operand"] for r in bank_records(recs, 0xE1, 32)}

    def test_all_eight_plateau_boundary_points_match(self):
        for key, expected in self.PLATEAU_POINTS.items():
            name = self.FIXTURE_NAMES[key]
            self.assertEqual(self._present(name), expected, (key, name))

    def test_k129_matches_neither_neighboring_candidate(self):
        # PR #1613's own unresolved anomaly: absent at BOTH candidate-256
        # (N=128/129) yet present at N=256, contradicting either
        # neighboring plateau's own clean rule.
        n128 = decode("gemm_1x129x128_e1switch.mcode.gz")
        n129 = decode("gemm_1x129x129_e1switch.mcode.gz")
        n256 = decode("gemm_1x129x256_e1switch.mcode.gz")
        self.assertFalse(
            GEMM_F32 in {r["operand"] for r in bank_records(n128, 0xE1, 32)}
        )
        self.assertFalse(
            GEMM_F32 in {r["operand"] for r in bank_records(n129, 0xE1, 32)}
        )
        self.assertTrue(
            GEMM_F32 in {r["operand"] for r in bank_records(n256, 0xE1, 32)}
        )

    def test_the_ktimesn_cap_rule_holds_at_three_of_four_plateau_edges_but_not_the_320_edge(
        self,
    ):
        # 64*512, 128*256, 256*128 all land exactly on 32768; 96*320
        # (30720) and 97*320 (31040) both badly miss it -- the rule that
        # explains 3 of 4 transitions does not explain the fourth.
        self.assertEqual(64 * 512, 32768)
        self.assertEqual(128 * 256, 32768)
        self.assertEqual(256 * 128, 32768)
        self.assertLess(96 * 320, 32768)
        self.assertLess(97 * 320, 32768)


class TestBank81RequiresBothAConstantOperandAndAContractionDimension(unittest.TestCase):
    """The "shared values, op-specific triggers, but bank 0x81's own full
    mechanism needs a real contraction dimension" framing, reconfirmed
    across all four ops in one place."""

    def test_gemm_has_it_natively(self):
        recs = decode("gemm_1x512x1000_tb0.mcode.gz")
        self.assertTrue(bank_records(recs, 0x81, 192))

    def test_matmul_only_has_it_as_a_constant_weight_build(self):
        # Ordinary two-live-tensor MatMul never shows bank 0x81 anywhere
        # in the existing corpus (PR #1608's own first-pass scan);
        # spot-checked here against a large two-live-tensor K-sweep
        # fixture that would be a plausible candidate if the mechanism
        # transferred without the constant-weight requirement.
        recs = decode("matmul_var_k40.mcode.gz")
        self.assertEqual(bank_records(recs, 0x81), [])
        # But the constant-weight construction reproduces it exactly.
        cw = decode("matmul_bank81_probe_m1k512n1000.mcode.gz")
        self.assertTrue(bank_records(cw, 0x81, 192))

    def test_conv_lacks_it_despite_also_having_a_constant_weight(self):
        # Conv's weight is ALSO compile-time-constant (Pulsar2's own
        # established behavior), yet bank 0x81 never appears even at
        # extreme shapes -- the constant-operand precondition alone is
        # not sufficient, consistent with this op's own contraction
        # dimension not mapping onto whatever resource this formula
        # sizes.
        for name in (
            "conv_bank81_probe_cout64.mcode.gz",
            "conv_bank81_probe_dilation16.mcode.gz",
        ):
            self.assertEqual(bank_records(decode(name), 0x81), [], name)

    def test_mul_lacks_it_with_no_contraction_dimension_at_all(self):
        recs = decode("mul_bank81_probe_k2048n4096.mcode.gz")
        self.assertEqual(bank_records(recs, 0x81), [])


if __name__ == "__main__":
    unittest.main()
