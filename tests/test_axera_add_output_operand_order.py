"""Completes the operand-order symmetry picture `tests/test_axera_sub_
operand_order.py` and `tests/test_axera_div_operand_order.py` (PR
#1665) built for the two-live-input elementwise cluster. PR #1665's
own diff explicitly left `Add` untested for this specific question:
"Add's output is also range-symmetric like Sub's own a-b/b-a case,
since a+b == b+a identically, so this is expected to be fully
invariant like Mul's, not tested directly here." This file is that
test.

## The prediction

`Add`'s output (`y = x1 + x2`) is not merely range-symmetric under an
operand swap the way `Sub`'s is (`range(a-b) == range(b-a)`, but
`a-b != b-a` pointwise) -- it is literally commutative, elementwise,
bit-for-bit: IEEE-754 float addition satisfies `a+b == b+a` exactly for
any two floats (unlike associativity, which float addition does NOT
generally satisfy). So `y`'s own calibration samples are IDENTICAL
arrays regardless of operand order, not merely equal in range. Both
`y_scale` (PR #1657's own directly-stored mechanism, using PR #1654's
round-scale-once formula) and `zp_y` (`reg=14,tag=131`, per PR #1657's
own Finding 4) should therefore be operand-order-invariant, matching
`Mul`'s own full-invariance pattern (PR #1662) rather than `Sub`'s
partial one (`y_scale` invariant, `zp_y` not -- PR #1665's own Finding
3 for `Sub`, driven by a weaker symmetry that only protects the range,
not the zero-point's own position within it).

## Fixtures: no new Docker build needed

`tests/test_axera_add_operand_order.py` (PR #1658) already built and
committed the exact two fixtures this question needs --
`add_operand_order_A_in12_op12.mcode.gz` (`Add(x1,x2)`) and
`add_operand_order_C_in12_op21.mcode.gz` (`Add(x2,x1)`), same graph-
input order, flipped node-operand order -- but PR #1658's own test
file only ever checked the INPUT-side locators (`zp_x`'s literal quad,
`reg=94,tag=132`) between them, never the OUTPUT-side fields (`y_scale`,
`zp_y`). This file decodes those same two already-committed fixtures
and checks exactly that, reusing PR #1658's own calibration convention
(`RandomState(1)` for `x1`, `RandomState(2)` for `x2`) and PR #1657's
own output-quantity formulas (`round_scale_once`, the `reg=14,tag=131`
`zp_y` locator).

## Finding: the prediction holds exactly -- BOTH `y_scale` and `zp_y`
## are bit-identical between the two operand orderings

| variant | node operands | `y_scale` (bit-exact) | `zp_y` (bit-exact) |
| --- | --- | --- | --- |
| A (`op=(x1,x2)`) | `y=x1+x2` | `0.025752649` | `122` |
| C (`op=(x2,x1)`) | `y=x2+x1` | `0.025752649` (identical) | `122` (identical) |

This is not merely "close" or "coincidentally equal" -- it is verified
directly against each variant's own real decoded bytes, and the
underlying reason is stronger than `Sub`'s own range-negation symmetry:
`x1+x2` and `x2+x1` are the exact same float32 array, element for
element, so this file's own two independent recomputations
(`YS_A`/`YS_C`) are not merely equal in some derived statistic, they
are bit-identical inputs to the same formula from the start.

This completes the four-op picture PR #1665 began: invariance level
tracks exactly the algebraic symmetry class of each op's own output --
full commutativity (`Add`, `Mul`) makes BOTH output-side fields
invariant; range-negation symmetry only (`Sub`) makes `y_scale` alone
invariant; no symmetry (`Div`) makes NEITHER field invariant.

## What this establishes, precisely, and what it does not

**Established**: `Add`'s own output-side fields (`y_scale` AND `zp_y`)
are both genuinely operand-order-invariant, bit-exact, confirmed
directly against PR #1658's own already-committed fixtures (no new
Docker build needed) -- matching `Mul`'s own full-invariance pattern,
not `Sub`'s partial one, exactly as predicted from `Add`'s own stronger
(literal commutativity, not just range symmetry) algebraic property.

**NOT established**: whether this holds at shapes other than `(1,16)`;
whether graph-input declaration order interacts with this in any way
(not tested here, and PR #1658 already showed it doesn't affect the
input-side locators' own content at this shape); whether the same
"invariance tracks output symmetry class" rule would predict correctly
for an op this arc has not yet characterized (e.g. a hypothetical
`Max`/`Min`, whose output IS commutative but whose own quantization
formula this file does not know).
"""

import gzip
import math
import os
import struct
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

SEED1, SEED2 = 1, 2

VARIANT_A = "A_in12_op12"
VARIANT_C = "C_in12_op21"


def load(variant):
    path = os.path.join(FIX, f"add_operand_order_{variant}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(variant):
    return mcode.decode(load(variant), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data PR #1657/#1658's own fixture-build scripts used."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def asymmetric_uint8_quant_params(samples):
    """`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_
    quantize_matmul.h`), the plain float32-throughout formula this
    whole arc has used since PR #1636/#1640."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zp_f = float(np.float32(-lo / scale))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), zp_f, float(scale)


def round_scale_once(samples):
    """PR #1654's own refined formula: the `hi-lo` subtraction (and the
    division by 255) done in float64, `scale` rounded to float32
    exactly ONE time -- cited directly, reused as-is, not re-derived."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


X1_SAMPLES = calib_samples(SEED1)
X2_SAMPLES = calib_samples(SEED2)

# y = x1 + x2 (variant A's own node-operand order) and y = x2 + x1
# (variant C's own order) -- kept as two independently-built lists,
# not asserted equal by construction, so the bit-identity check below
# is a real comparison of the arrays this file itself produces, not a
# tautology.
YS_A = [a + b for a, b in zip(X1_SAMPLES, X2_SAMPLES)]
YS_C = [a + b for a, b in zip(X2_SAMPLES, X1_SAMPLES)]

ZP_Y_A, _, _ = asymmetric_uint8_quant_params(YS_A)
ZP_Y_C, _, _ = asymmetric_uint8_quant_params(YS_C)
Y_SCALE_A = round_scale_once(YS_A)
Y_SCALE_C = round_scale_once(YS_C)


def bank15_floats(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("verb") == 161
        and r.get("bank") == 15
        and r.get("field") in (96, 112, 128)
    ]
    operands = {
        r["operand"] for r in hits if r.get("operand") and len(r["operand"]) == 4
    }
    return {struct.unpack("<f", op)[0] for op in operands}


def zp_y_hits(recs):
    return [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == 14 and r.get("tag") == 131
    ]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_either_variant(self):
        for variant in (VARIANT_A, VARIANT_C):
            errs = mcode.check(load(variant))
            self.assertEqual(errs, [], variant)


class TestYsArraysAreBitIdenticalRegardlessOfOperandOrder(unittest.TestCase):
    """Sanity check on this file's own recomputation, before checking
    it against real bytes: IEEE-754 float addition is commutative
    elementwise, so YS_A and YS_C should be bit-identical arrays, not
    merely equal in some derived statistic."""

    def test_every_sample_pair_is_bit_identical(self):
        for a, c in zip(YS_A, YS_C):
            np.testing.assert_array_equal(a, c)


class TestYScaleIsFullyOperandOrderInvariantLikeMulNotLikeSub(unittest.TestCase):
    """Finding: unlike Sub (range-negation symmetry only), Add's own
    y_scale is invariant for the stronger reason that y itself is
    bit-identical between orderings."""

    def test_recomputed_y_scale_is_bit_identical_between_orderings(self):
        self.assertEqual(struct.pack("<f", Y_SCALE_A), struct.pack("<f", Y_SCALE_C))

    def test_bank15_carries_the_same_y_scale_value_on_both_variants(self):
        recs_a, recs_c = decode(VARIANT_A), decode(VARIANT_C)
        floats_a, floats_c = bank15_floats(recs_a), bank15_floats(recs_c)
        self.assertIn(float(Y_SCALE_A), floats_a)
        self.assertIn(float(Y_SCALE_C), floats_c)
        # The exact float32 bit pattern recurs identically in both
        # variants' own decoded streams, not just the two independent
        # recomputations above.
        self.assertEqual(
            {struct.pack("<f", np.float32(f)) for f in floats_a if f == Y_SCALE_A},
            {struct.pack("<f", np.float32(f)) for f in floats_c if f == Y_SCALE_C},
        )


class TestZpYIsFullyOperandOrderInvariantUnlikeSubsOwnZpY(unittest.TestCase):
    """Finding: unlike Sub's own zp_y (PR #1665's Finding 3, which
    genuinely differs with operand order), Add's own zp_y is identical
    -- the clean counter-case completing the four-op contrast."""

    def test_zp_y_a_and_c_are_the_same_value(self):
        self.assertEqual(ZP_Y_A, ZP_Y_C)

    def test_variant_a_zp_y_matches_at_reg14_tag131(self):
        recs = decode(VARIANT_A)
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_A]
        self.assertEqual(len(hits), 1, ZP_Y_A)

    def test_variant_c_zp_y_matches_at_reg14_tag131(self):
        recs = decode(VARIANT_C)
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_C]
        self.assertEqual(len(hits), 1, ZP_Y_C)


if __name__ == "__main__":
    unittest.main()
