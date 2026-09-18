"""Synthesizes the whole "trivial-zero-point" thread's own framing
correction (PRs #1663/#1667/#1669/#1670/#1678, the original discovery
and cross-op/cross-arc extensions; PR #1679, the correction for the
literal-quad-style mechanism; and `tests/test_axera_x2_trivial_
zeropoint_replacement_check.py`, this session's own closing extension
to the decode-record-style `x2` mechanism) into one final, precise
record.

## The corrected picture, in one place

Every PR in this thread independently discovered a real, reproducible
fact: when a live input's own calibrated (MinMax, asymmetric uint8)
zero point lands exactly on `0`, the specific locator that PR searched
for -- Gemm's/Add's own `x1`-side literal 6-byte quad
(`02 10 1b <zp> 83 36`), Conv's own `reg=54,tag=131` decode record, or
the two-live-input cluster's own `x2`-side `reg=94,tag=132` decode
record -- has ZERO hits in the decoded stream. Every one of those PRs
characterized this as the value being "structurally absent," having
"no stored encoding at all."

That empirical observation (zero hits for the searched locator) is
correct in every one of those PRs and is NOT what this synthesis
revises. What needed correction, found by PR #1679 and closed by this
session's own follow-up work: the value is not actually unencoded. It
is replaced, at (or within a few bytes of) the identical byte offset,
by a different, fixed wire form:

| mechanism | ops covered | replacement bytes | source |
| --- | --- | --- | --- |
| literal-quad (`x1`-style) | Gemm, Add (`x1`), Conv (`reg=54`), MatMul (always) | `00 10 84` (3 bytes) | pre-existing `tests/test_axera_mcode_reciprocal.py`'s own `TestZpXImmediateRegion`, cross-referenced by PR #1679 |
| decode-record (`x2`-style, `reg=94,tag=132`) | Add, Sub, Mul, Div | `00 10 85 62 03 0f` (6 bytes) | this session's own closing check |

Both replacement forms share a `00 10` prefix but are otherwise
distinct byte sequences at a different length -- suggestive of a
related underlying codec mechanism (both trigger on the same
condition: the live input's own zero point being exactly `0`) but not
confirmed to be the literal same field or opcode.

## Why this correction does not undermine the arc's own methodology

The mistake in PRs #1663-#1678 was a narrow one: searching only for
the ABSENCE of one specific locator and inferring "not encoded"
without also checking whether a DIFFERENT, already-known wire form had
taken its place. It was found and fixed by exactly the same discipline
this whole arc has used throughout -- decode real fixtures, compare
against a control, do not assert more than the evidence shows. This is
not a reason to distrust the arc's OTHER conclusions, most of which
(the `ComputeAsymmetricUint8QuantParams` bit-exactness formulas, the
operand-order symmetry-class rule, the `m*k`-driven shape mechanisms)
were independently cross-validated multiple different ways -- multiple
ops, multiple shapes, control fixtures, specificity/near-miss checks --
not resting on the interpretation of a single "zero hits" search
result the way this one framing did.

## What remains genuinely open

- What the two replacement forms' own non-`00 10`-prefix bytes
  actually encode (`84` alone for the literal-quad case; `85 62 03 0f`
  for the decode-record case) -- both are pinned as raw, undecoded
  bytes, the same "no decode, just confirmed fixed bytes" scope
  `TestZpXImmediateRegion` itself already used.
- Whether the two forms are the same underlying codec mechanism
  (plausible, given the shared `00 10` prefix and shared trigger
  condition) or a coincidence of two independently-fixed forms.
- Why `Mul`'s own trivial-`x2` builds consistently reflow their whole
  stream by exactly 32 bytes while `Sub`'s/`Div`'s never do and `Add`'s
  does so at only one of three tested seed pairs.
- Whether `x1`'s own literal-quad replacement (`00 10 84`) and `x2`'s
  own decode-record replacement (`00 10 85 62 03 0f`) both generalize
  to shapes beyond the `(1,16)` family this whole two-live-input
  cluster has tested.
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

LITERAL_QUAD_REPLACEMENT = bytes.fromhex("001084")
X2_RECORD_REPLACEMENT = bytes.fromhex("00108562030f")
LITERAL_QUAD_PREFIX = bytes.fromhex("02101b")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestBothReplacementFormsAreDirectlyReconfirmedHere(unittest.TestCase):
    """Re-decodes real fixtures from both mechanisms, independently of
    either contributing PR's own test module, to confirm the final
    corrected picture holds in one place."""

    def test_literal_quad_replacement_present_for_conv_trivial_fixtures(self):
        # `00 10 84` is only 3 bytes, short enough to also match
        # unrelated byte runs elsewhere in a ~2KB stream by
        # coincidence -- PR #1679's own test located it at a specific,
        # control-derived offset rather than asserting a unique global
        # count; this file only asserts it appears AT LEAST once (the
        # replacement is present) and that the literal-quad prefix it
        # replaces is fully absent.
        for seed in (1, 7, 100):
            data = load(f"conv_dilation3_trivialA_seed{seed}.mcode.gz")
            self.assertGreaterEqual(len(hits(data, LITERAL_QUAD_REPLACEMENT)), 1, seed)
            self.assertEqual(hits(data, LITERAL_QUAD_PREFIX), [], seed)

    def test_literal_quad_replacement_present_for_matmuls_own_always_zero_case(self):
        data = load("matmul_4x8x8_asym_calib.mcode.gz")
        self.assertGreaterEqual(len(hits(data, LITERAL_QUAD_REPLACEMENT)), 1)
        self.assertEqual(hits(data, LITERAL_QUAD_PREFIX), [])

    def test_x2_record_replacement_present_for_every_two_live_input_op(self):
        fixture_by_op = {
            "add": "add_1x16_two_live_seed1_2_trivialx2.mcode.gz",
            "sub": "sub_1x16_two_live_seed1_2_trivialx2.mcode.gz",
            "mul": "mul_1x16_two_live_seed1_2_trivialx2.mcode.gz",
            "div": "div_1x16_two_live_seed1_2_trivialzp2.mcode.gz",
        }
        for op, name in fixture_by_op.items():
            data = load(name)
            self.assertEqual(len(hits(data, X2_RECORD_REPLACEMENT)), 1, op)

    def test_the_two_replacement_forms_are_genuinely_distinct_byte_sequences(self):
        self.assertNotEqual(LITERAL_QUAD_REPLACEMENT, X2_RECORD_REPLACEMENT[:3])
        self.assertTrue(X2_RECORD_REPLACEMENT.startswith(bytes.fromhex("0010")))
        self.assertTrue(LITERAL_QUAD_REPLACEMENT.startswith(bytes.fromhex("0010")))


class TestCorrectionDoesNotChangeTheUnderlyingSearchResults(unittest.TestCase):
    """The one claim this whole correction thread does NOT revise:
    every originally-searched-for locator genuinely has zero hits when
    the zero point is trivially 0. Re-confirmed directly here, not
    merely asserted in prose."""

    def test_reg54_still_has_zero_hits_in_conv_trivial_fixtures(self):
        for seed in (1, 7, 100):
            recs = mcode.decode(
                load(f"conv_dilation3_trivialA_seed{seed}.mcode.gz"), **mcode.FULL_RULE
            )
            reg54_hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 54 and r.get("tag") == 131
            ]
            self.assertEqual(reg54_hits, [], seed)

    def test_reg94_tag132_still_has_zero_hits_in_add_trivial_x2_fixture(self):
        recs = mcode.decode(
            load("add_1x16_two_live_seed1_2_trivialx2.mcode.gz"), **mcode.FULL_RULE
        )
        reg94_hits = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
        ]
        self.assertEqual(reg94_hits, [], reg94_hits)


if __name__ == "__main__":
    unittest.main()
