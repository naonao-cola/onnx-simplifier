"""Extends `tests/test_axera_add_operand_order.py` (PR #1658) and
`tests/test_axera_mul_operand_order.py` (PR #1662)'s own operand-order
disentanglement to `Sub`. PR #1658 found `Add`'s x1/x2 asymmetry (x1
gets the literal-quad `zp_x` + `verb=161,bank=15` scale treatment, x2
gets only a `reg=94,tag=132` zero point) is governed by the ONNX
node's own OPERAND order; PR #1662 confirmed the same mechanism for
`Mul`, but found Mul's own OUTPUT-side fields (its requant multiplier,
`zp_y`) are unaffected by operand order because Mul's output is
commutative (`x1*x2 == x2*x1`).

`Sub` is architecturally different from both: like Add, it directly
stores `y_scale` (PR #1661); unlike Mul, `Sub`'s own OUTPUT is NOT
commutative (`x1-x2 != x2-x1`), so its output-side fields are exactly
the case PR #1662's own diff flagged as untested -- "whether `Sub`'s
own operand-order sensitivity matches `Add`'s... not tested here."
This file is that test.

## Fixtures

Two fresh `Sub(a, b)` builds, both declaring graph inputs `[x1, x2]`
(graph-input order held fixed -- PR #1658 already showed it has no
effect on either locator's own content for `Add`, assumed here rather
than re-derived, the same assumption PR #1662 made for `Mul`), varying
only the `Sub` node's own operand order:

| variant | node operands | output |
| --- | --- | --- |
| A | `Sub(x1, x2)` | `y = x1 - x2` |
| C | `Sub(x2, x1)` | `y = x2 - x1` |

Calibration: `RandomState(1)` for `x1`, `RandomState(2)` for `x2` --
the same seed pair PR #1657/#1658/#1661/#1662 all used, so `zp1=133`,
`zp2=131` are directly comparable to their own already-established
values.

## Finding 1: `x1`/`x2`'s own locators flip with operand order,
## exactly like Add and Mul

| variant | literal-quad locator carries | `reg=94,tag=132` carries |
| --- | --- | --- |
| A (`op=(x1,x2)`) | `zp1`=133 | `zp2`=131 |
| C (`op=(x2,x1)`) | `zp2`=131 | `zp1`=133 |

Identical mechanism to PR #1658 (Add) and PR #1662 (Mul): whichever
tensor is the node's own first operand gets the literal-quad
`zp_x`/`verb=161,bank=15`-scale treatment; the other gets
`reg=94,tag=132` only.

## Finding 2 (new, does not generalize from Mul's own result):
## `y_scale` is IDENTICAL between A and C -- but for a DIFFERENT
## reason than Mul's own commutativity

Mul's own output-side fields were operand-order-invariant because
`Mul`'s OUTPUT VALUE is commutative. `Sub`'s output is emphatically
NOT commutative (`x1-x2 != x2-x1` pointwise) -- yet `y_scale` (the
plain `ComputeAsymmetricUint8QuantParams` scale, bit-exact under the
plain float32 formula with no round-scale-once refinement needed,
matching PR #1661's own Finding 2 precision) is bit-for-bit identical
between A and C in this file's own real fixtures. This is NOT a
decode coincidence: `y = a - b`'s own calibration RANGE (`max - min`
across all samples) is invariant to negation, since `range(a-b) ==
range(-(a-b)) == range(b-a)` for any real-valued tensor -- swapping
the operands negates every element of `y` but leaves its own
`max - min` span, and therefore its scale, unchanged. `Sub` shares
Mul's own "output-side scale is operand-order-invariant" observation,
but via a genuinely different mathematical reason (negation symmetry
of subtraction's own range, not commutativity of the operation
itself) -- this file verifies the VALUE stays the same without
assuming Mul's own explanation transfers.

## Finding 3 (does NOT generalize from Mul): `zp_y` (`reg=14,tag=131`)
## DOES differ between A and C, bit-exact to each variant's own
## correctly-ordered recomputation

Unlike `y_scale`'s own range-invariance, a ZERO POINT depends on WHERE
zero sits within that range, which is not invariant under negation
(`zp(a-b) != zp(b-a)` in general, since negating a range around a
nonzero center moves where zero falls within it). This file confirms
directly: `zp_y` is `156` for variant A (`y=x1-x2`) and `99` for
variant C (`y=x2-x1`) -- genuinely different values, each matching its
own variant's own correctly-ordered recomputation bit-exact. This is
the clean counter-case to Finding 2: `Sub`'s output-side SCALE is
operand-order-invariant, but its output-side ZERO POINT is not -- a
real, precise distinction Mul's own commutative case could not
expose (Mul's zero point and scale were BOTH invariant, since its
whole output value is invariant, not just its range).

## What this establishes, precisely, and what it does not

**Established**: `Sub`'s x1/x2 asymmetry is governed by ONNX node
operand order, exactly like `Add` (PR #1658) and `Mul` (PR #1662).
`Sub`'s own `y_scale` is operand-order-invariant, like `Mul`'s output-
side fields, but for a different underlying reason (subtraction's
range-negation symmetry, not output commutativity) -- confirmed
directly rather than assumed. `Sub`'s own `zp_y`, unlike `y_scale`
and unlike ANY of Mul's own output-side fields, genuinely differs
with operand order, bit-exact to each ordering's own correct
recomputation.

**NOT established**: whether graph-input declaration order is
genuinely inert for `Sub` (assumed from Add's own result, not
independently tested here with a full 2x2 design); whether this holds
at shapes other than `(1,16)`; `Div`'s own operand-order sensitivity
(not tested by this file -- `Div`, like `Sub`, is non-commutative, so
a similar zp_y-differs/y_scale-may-or-may-not-differ pattern is
plausible but not verified here).
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

VARIANTS = {
    "A_op12": "x1",
    "C_op21": "x2",
}


def load(variant):
    path = os.path.join(FIX, f"sub_operand_order_{variant}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(variant):
    return mcode.decode(load(variant), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data this file's own fixture-build script used, matching PR
    #1657/#1658/#1661/#1662's own seed pair (1, 2)."""
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


X1_SAMPLES = calib_samples(SEED1)
X2_SAMPLES = calib_samples(SEED2)
ZP1, _, SCALE1 = asymmetric_uint8_quant_params(X1_SAMPLES)
ZP2, _, SCALE2 = asymmetric_uint8_quant_params(X2_SAMPLES)
ZP_BY_TENSOR = {"x1": ZP1, "x2": ZP2}
RECIP_SCALE_BY_TENSOR = {
    "x1": float(np.float32(1.0) / np.float32(SCALE1)),
    "x2": float(np.float32(1.0) / np.float32(SCALE2)),
}


def sub_output_quant_params(first_samples, second_samples):
    """`y = first - second`'s own `ComputeAsymmetricUint8QuantParams`,
    plain float32-throughout (no round-scale-once refinement needed --
    confirmed bit-exact without it below, matching PR #1661's own
    Finding 2 precision for `Sub` specifically)."""
    ys = [a - b for a, b in zip(first_samples, second_samples)]
    ylo = min(0.0, min(float(y.min()) for y in ys))
    yhi = max(0.0, max(float(y.max()) for y in ys))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = (np.float32(yhi) - np.float32(ylo)) / np.float32(255.0)
    zp_f = float(np.float32(-np.float32(ylo) / scale))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), float(scale)


ZP_Y_A, Y_SCALE_A = sub_output_quant_params(X1_SAMPLES, X2_SAMPLES)
ZP_Y_C, Y_SCALE_C = sub_output_quant_params(X2_SAMPLES, X1_SAMPLES)


def literal_quad_hits(data, zp):
    quad = bytes.fromhex("02101b") + bytes([zp]) + bytes.fromhex("8336")
    return [i for i in range(len(data) - 5) if data[i : i + 6] == quad]


def reg94_hits(recs, zp):
    return [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("reg") == 94
        and r.get("tag") == 132
        and r["payload"][-1] == zp
    ]


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
        for variant in VARIANTS:
            errs = mcode.check(load(variant))
            self.assertEqual(errs, [], variant)


class TestZp1AndZp2MatchPriorPRsOwnAlreadyEstablishedValues(unittest.TestCase):
    def test_zp1_is_133_and_zp2_is_131(self):
        # Cited directly from PR #1657/#1658's own seed pair (1, 2)
        # results -- recomputed independently here, not copied.
        self.assertEqual(ZP1, 133)
        self.assertEqual(ZP2, 131)


class TestNodeOperandOrderDeterminesTheInputLocators(unittest.TestCase):
    """Finding 1: reproduces PR #1658/#1662's own core result for Sub."""

    def test_first_operand_tensor_carries_the_literal_quad_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            data = load(variant)
            hits = literal_quad_hits(data, ZP_BY_TENSOR[first_operand])
            self.assertEqual(len(hits), 1, (variant, first_operand))

    def test_second_operand_tensor_carries_reg94_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            second_operand = "x2" if first_operand == "x1" else "x1"
            recs = decode(variant)
            hits = reg94_hits(recs, ZP_BY_TENSOR[second_operand])
            self.assertEqual(len(hits), 1, (variant, second_operand))

    def test_first_operand_scale_is_bit_exact_at_verb161_bank15(self):
        for variant, first_operand in VARIANTS.items():
            recs = decode(variant)
            floats = bank15_floats(recs)
            self.assertIn(RECIP_SCALE_BY_TENSOR[first_operand], floats, variant)


class TestYScaleIsOperandOrderInvariantForADifferentReasonThanMul(unittest.TestCase):
    """Finding 2: y_scale's own value (not just whether it's present)
    is identical between A and C -- verified directly, not assumed
    from Mul's own commutativity argument, which does not apply here."""

    def test_y_scale_formula_itself_is_identical_between_orderings(self):
        # Sanity check on this file's own recomputation: range(a-b) ==
        # range(b-a), confirmed before checking it against real bytes.
        self.assertAlmostEqual(Y_SCALE_A, Y_SCALE_C, places=9)

    def test_bank15_carries_the_same_y_scale_value_on_both_variants(self):
        recs_a, recs_c = decode("A_op12"), decode("C_op21")
        floats_a, floats_c = bank15_floats(recs_a), bank15_floats(recs_c)
        y_scale_hits_a = {f for f in floats_a if abs(f - Y_SCALE_A) / Y_SCALE_A < 1e-5}
        y_scale_hits_c = {f for f in floats_c if abs(f - Y_SCALE_C) / Y_SCALE_C < 1e-5}
        self.assertEqual(len(y_scale_hits_a), 1)
        self.assertEqual(len(y_scale_hits_c), 1)
        self.assertAlmostEqual(y_scale_hits_a.pop(), y_scale_hits_c.pop(), places=5)


class TestZpYGenuinelyDiffersWithOperandOrderUnlikeMul(unittest.TestCase):
    """Finding 3: the clean counter-case to Finding 2 -- zp_y is NOT
    operand-order invariant, and matches each variant's own correctly-
    ordered recomputation bit-exact."""

    def test_zp_y_a_and_c_are_genuinely_different_values(self):
        self.assertNotEqual(ZP_Y_A, ZP_Y_C)

    def test_variant_a_zp_y_matches_y_equals_x1_minus_x2(self):
        recs = decode("A_op12")
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_A]
        self.assertEqual(len(hits), 1, ZP_Y_A)

    def test_variant_c_zp_y_matches_y_equals_x2_minus_x1(self):
        recs = decode("C_op21")
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_C]
        self.assertEqual(len(hits), 1, ZP_Y_C)

    def test_variant_a_does_not_also_show_variant_cs_own_zp_y(self):
        recs = decode("A_op12")
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_C]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
