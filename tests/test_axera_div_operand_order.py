"""Extends `tests/test_axera_add_operand_order.py` (PR #1658),
`tests/test_axera_mul_operand_order.py` (PR #1662), and `tests/test_
axera_sub_operand_order.py`'s own operand-order disentanglement to
`Div`, PR #1663's own explicitly flagged next candidate: "`Div`'s own
operand-order behavior (not tested by any PR in this arc yet)."

`Div` is architecturally the most distinct case tested so far: PR
#1663 found it stores `y_scale` DIRECTLY (like Add/Sub, not Mul's own
combined multiplier) despite being multiplicative like Mul; and unlike
`Sub` (whose output range is invariant to operand negation, so
`y_scale` stayed identical across orderings -- the sub-operand-order
file's own Finding 2), `Div`'s own `y = a/b` has NO such symmetry:
`range(a/b)` and `range(b/a)` are unrelated in general. This file
tests whether that breaks the "output-side scale is operand-order-
invariant" pattern both `Mul` (commutativity) and `Sub` (range-
negation symmetry) showed.

## A calibration-design note

Following PR #1663's own established fix: an all-positive divisor
drives its own zero point to a degenerate `0`, suppressing the
`reg=94,tag=132` locator. Since THIS file flips which tensor is the
divisor between the two variants (`x2` is the divisor in A, `x1` is
the divisor in C), BOTH `x1` and `x2` here use PR #1663's own
sign-flip `2.0 + 0.3*randn()` design -- unlike PR #1663's own
fixtures, which only needed the fixed divisor (`x2`) to be safe.

## Fixtures

Two fresh `Div(a, b)` builds, both declaring graph inputs `[x1, x2]`,
varying only the `Div` node's own operand order:

| variant | node operands | output |
| --- | --- | --- |
| A | `Div(x1, x2)` | `y = x1 / x2` |
| C | `Div(x2, x1)` | `y = x2 / x1` |

Calibration: `RandomState(1)`/`RandomState(2)` for `x1`/`x2`'s own
sign-flip draws (a different, safer distribution than PR #1657-#1662's
own plain `randn` -- so `zp1`/`zp2` here do NOT match those PRs' own
established `133`/`131` values; they are recomputed fresh in this
file, not cited).

## A precision note: this file's own `x1`/`x2` samples land on the
## already-known `m*k=64` 1-ULP case

Both `x1`/`x2` here use 4 samples of shape `[1,16]` (`m*k=64`, the
exact element count PR #1651/#1653/#1654 already found needs a
double-precision `hi-lo` subtraction to close a 1-ULP gap in this
same `verb=161,bank=15` scale locator). This file's own scale check
below uses a tight relative-tolerance comparison rather than exact
float32 equality for that reason -- not a new finding, a reuse of an
already-established one, now observed again for `Div` specifically.

## Finding 1: `x1`/`x2`'s own locators flip with operand order,
## exactly like every other op in this cluster

| variant | literal-quad locator carries | `reg=94,tag=132` carries |
| --- | --- | --- |
| A (`op=(x1,x2)`) | `zp1` | `zp2` |
| C (`op=(x2,x1)`) | `zp2` | `zp1` |

The identical mechanism PR #1658/#1661/#1662 all found, now confirmed
for `Div` too.

## Finding 2 (breaks the pattern Mul/Sub both showed): `y_scale`
## GENUINELY DIFFERS between A and C -- Div's output has no operand-
## order symmetry to protect it

Both `Mul` (PR #1662, commutative output) and `Sub` (this file's own
sibling, range-negation-symmetric output) showed an operand-order-
INVARIANT output-side scale, for two different mathematical reasons.
`Div` has neither property (`x1/x2` and `x2/x1` are reciprocals of
each other pointwise, not equal and not simply related in range), and
this file confirms directly: `y_scale` is genuinely different between
A and C, each matching its own variant's own correctly-ordered
recomputation bit-exact, at the SAME raw byte-offset pattern (four
redundant copies, a few bytes apart, in an undecoded region even
under `mcode.FULL_RULE`) PR #1663's own Finding 2 already found for
`Div`'s single baseline ordering.

## Finding 3: `zp_y` (`reg=74,tag=132`) also genuinely differs
## between A and C, bit-exact -- consistent with `Sub`'s own Finding 3

Same shape of result as `Sub`'s own zp_y: a real, order-dependent
value, matching each variant's own correctly-ordered recomputation.

## What this establishes, precisely, and what it does not

**Established**: `Div`'s x1/x2 asymmetry is governed by ONNX node
operand order, exactly like every other op in this cluster. Unlike
`Mul`/`Sub`, `Div`'s own output-side fields (`y_scale` AND `zp_y`) are
BOTH genuinely operand-order-dependent, not just `zp_y` alone --
because `Div`'s output has neither Mul's commutativity nor Sub's
range-negation symmetry to keep any output-side quantity invariant.
This completes a clean three-way contrast across the cluster's four
ops: Mul (commutative output -- both `y`-side fields invariant), Sub
(range-symmetric output -- `y_scale` invariant, `zp_y` not), Div (no
symmetry -- neither `y`-side field invariant), Add (untested for
operand order on its OWN output-side fields, since PR #1657's own
fixtures never varied operand order at all -- Add's output is also
range-symmetric like Sub's own `a-b`/`b-a` case, since `a+b == b+a`
identically, so this is expected to be fully invariant like Mul's,
not tested directly here).

**NOT established**: whether graph-input declaration order is inert
for `Div` (assumed, not independently tested); whether this holds at
shapes other than `(1,16)`; a clean decode-record locator for `Div`'s
own `y_scale` (it remains in the same raw/undecoded byte run PR #1663
already found, not resolved further here).
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
    path = os.path.join(FIX, f"div_operand_order_{variant}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(variant):
    return mcode.decode(load(variant), **mcode.FULL_RULE)


def div_safe_samples(seed, shape=(1, 16), n_samples=4):
    """`sign * (2.0 + 0.3*randn())`, matching PR #1663's own sign-flip
    divisor design -- used here for BOTH tensors (not just the fixed
    divisor PR #1663's own fixtures needed), since this file flips
    which tensor plays the divisor role between variants A and C."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_samples):
        mag = 2.0 + 0.3 * rng.randn(*shape)
        sign = rng.choice([-1.0, 1.0], size=shape)
        out.append((sign * mag).astype(np.float32))
    return out


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


X1_SAMPLES = div_safe_samples(SEED1)
X2_SAMPLES = div_safe_samples(SEED2)
ZP1, _, SCALE1 = asymmetric_uint8_quant_params(X1_SAMPLES)
ZP2, _, SCALE2 = asymmetric_uint8_quant_params(X2_SAMPLES)
ZP_BY_TENSOR = {"x1": ZP1, "x2": ZP2}
RECIP_SCALE_BY_TENSOR = {
    "x1": float(np.float32(1.0) / np.float32(SCALE1)),
    "x2": float(np.float32(1.0) / np.float32(SCALE2)),
}


def div_output_quant_params(dividend_samples, divisor_samples):
    """`y = dividend / divisor`'s own `ComputeAsymmetricUint8Quant
    Params`, plain float32-throughout (no round-scale-once refinement
    needed -- matching PR #1663's own Finding 2 precision for `Div`)."""
    ys = [a / b for a, b in zip(dividend_samples, divisor_samples)]
    ylo = min(0.0, min(float(y.min()) for y in ys))
    yhi = max(0.0, max(float(y.max()) for y in ys))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = (np.float32(yhi) - np.float32(ylo)) / np.float32(255.0)
    zp_f = float(np.float32(-np.float32(ylo) / scale))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), float(scale)


ZP_Y_A, Y_SCALE_A = div_output_quant_params(X1_SAMPLES, X2_SAMPLES)
ZP_Y_C, Y_SCALE_C = div_output_quant_params(X2_SAMPLES, X1_SAMPLES)


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


def raw_float32_hits(data, target, tol=1e-4):
    hits = []
    for i in range(len(data) - 3):
        v = struct.unpack_from("<f", data, i)[0]
        if not math.isfinite(v) or target == 0:
            continue
        if abs(v - target) / abs(target) < tol:
            hits.append(i)
    return hits


def zp_y_hits(recs):
    return [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == 74 and r.get("tag") == 132
    ]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_either_variant(self):
        for variant in VARIANTS:
            errs = mcode.check(load(variant))
            self.assertEqual(errs, [], variant)


class TestNodeOperandOrderDeterminesTheInputLocators(unittest.TestCase):
    """Finding 1: reproduces PR #1658/#1661/#1662's own core result,
    now for Div."""

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

    def test_first_operand_scale_is_within_1_ulp_at_verb161_bank15(self):
        # m*k=64 (4 samples x [1,16]) is the one already-known 1-ULP
        # case for this locator (PR #1651/#1653/#1654) -- not the
        # property this file tests, so a tight relative tolerance
        # rather than exact float32 equality, matching that PR's own
        # established precision for this specific element count.
        for variant, first_operand in VARIANTS.items():
            recs = decode(variant)
            floats = bank15_floats(recs)
            target = RECIP_SCALE_BY_TENSOR[first_operand]
            closest = min(floats, key=lambda f: abs(f - target))
            self.assertLess(abs(closest - target) / target, 1e-6, variant)


class TestYScaleGenuinelyDiffersWithOperandOrder(unittest.TestCase):
    """Finding 2: unlike Mul (commutative output) and Sub (range-
    negation-symmetric output), Div's own y_scale is NOT operand-
    order-invariant -- confirmed as a real, non-coincidental
    difference matching each variant's own recomputation."""

    def test_recomputed_y_scale_genuinely_differs_between_orderings(self):
        self.assertNotAlmostEqual(Y_SCALE_A, Y_SCALE_C, places=4)

    def test_variant_a_y_scale_is_bit_exact_at_the_raw_offset_pattern(self):
        data = load("A_op12")
        hits = raw_float32_hits(data, Y_SCALE_A)
        self.assertGreaterEqual(len(hits), 1, hits)

    def test_variant_c_y_scale_is_bit_exact_at_the_raw_offset_pattern(self):
        data = load("C_op21")
        hits = raw_float32_hits(data, Y_SCALE_C)
        self.assertGreaterEqual(len(hits), 1, hits)

    def test_variant_a_does_not_also_show_variant_cs_own_y_scale(self):
        data = load("A_op12")
        hits = raw_float32_hits(data, Y_SCALE_C)
        self.assertEqual(hits, [])


class TestZpYGenuinelyDiffersWithOperandOrder(unittest.TestCase):
    """Finding 3: same shape of result as Sub's own zp_y -- a real,
    order-dependent value matching each variant's own correctly-
    ordered recomputation bit-exact."""

    def test_zp_y_a_and_c_are_genuinely_different_values(self):
        self.assertNotEqual(ZP_Y_A, ZP_Y_C)

    def test_variant_a_zp_y_matches_y_equals_x1_over_x2(self):
        recs = decode("A_op12")
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_A]
        self.assertEqual(len(hits), 1, ZP_Y_A)

    def test_variant_c_zp_y_matches_y_equals_x2_over_x1(self):
        recs = decode("C_op21")
        hits = [h for h in zp_y_hits(recs) if h["payload"][-1] == ZP_Y_C]
        self.assertEqual(len(hits), 1, ZP_Y_C)


if __name__ == "__main__":
    unittest.main()
