"""Extends `tests/test_axera_add_operand_order.py` (PR #1658)'s own
operand-order disentanglement to `Mul`. PR #1658 found `Add`'s x1/x2
asymmetry (x1 gets the literal-quad `zp_x` + `verb=161,bank=15` scale
treatment, x2 gets only a `reg=94,tag=132` zero point) is governed by
the ONNX node's own OPERAND order, not graph-input declaration order.
PR #1660's own synthesis file (`tests/test_axera_two_live_input_
cluster_synthesis.py`) explicitly flagged this as untested for `Mul`:
"operand-order dependence: not re-tested [for Mul]... (Mul's own
fixtures did not vary operand order)".

## Fixtures

Two fresh `Mul(a, b)` builds, both declaring graph inputs `[x1, x2]`
(graph-input order held fixed -- PR #1658 already showed it has no
effect on either locator's own content for `Add`, and this file does
not re-derive that for `Mul`, only assumes it), varying only the
`Mul` node's own operand order:

| variant | node operands |
| --- | --- |
| A | `Mul(x1, x2)` -- matches PR #1659's own baseline |
| C | `Mul(x2, x1)` -- operand order flipped |

Calibration: `RandomState(1)` for `x1`, `RandomState(2)` for `x2` --
the same seed pair PR #1658/#1659 both used, so `zp1=133`/`zp2=131`
are directly comparable to their own already-established values.

## Finding 1: operand order governs `x1`/`x2`'s own locators
## exactly like Add -- confirmed unambiguously

Flipping the node operand order alone (A -> C) flips which tensor's
value the literal-quad locator carries, and which tensor's value
`reg=94,tag=132` carries -- the identical mechanism PR #1658 found for
`Add`, now independently confirmed for `Mul`:

| variant | literal-quad locator carries | `reg=94,tag=132` carries |
| --- | --- | --- |
| A (`op=(x1,x2)`) | `zp1`=133 | `zp2`=131 |
| C (`op=(x2,x1)`) | `zp2`=131 | `zp1`=133 |

## Finding 2 (new, does not generalize from a naive extrapolation of
## Finding 1): the requant MULTIPLIER (`field in (224,240)`) and
## `zp_y` (`reg=76,tag=132`) are UNAFFECTED by operand order --
## bit-identical between A and C

Unlike Add's own single `y_scale`, which PR #1657's own construction
never tested for operand-order sensitivity, Mul's own output-side
fields (the combined requantization multiplier `y_scale/(x1_scale*
x2_scale)`, PR #1659's own Finding 3; `zp_y`, PR #1659's own Finding
4) are IDENTICAL, bit-for-bit, between variant A and variant C. This
is architecturally sensible and not a decode artifact: `Mul` is
commutative in its OUTPUT value (`x1*x2 == x2*x1`), so `y`'s own
range, and every quantity derived purely from `y`'s own range (its
zero point, and a multiplier that only depends on the *product* of
both input scales, not which one is "first"), cannot depend on
operand order even though the INPUT-side locators (which tensor's
identity gets which treatment) plainly do. This is a genuine
difference from a naive "operand order flips everything" extrapolation
of PR #1658's own Add finding -- operand order flips WHICH TENSOR gets
which locator, not WHAT VALUE ends up in every locator.

## What this establishes, precisely, and what it does not

**Established**: `Mul`'s x1/x2 asymmetry is governed by ONNX node
operand order, exactly like `Add` (PR #1658) -- confirmed by an A/C
flip, the same design PR #1658 used to isolate operand order from
graph-input order (assumed, not re-tested, that graph-input order is
similarly irrelevant for `Mul`, since PR #1658 already established
this pattern for the closely related `Add`). The requant-multiplier
and `zp_y` fields are unaffected by operand order, because `Mul`'s own
output value is commutative -- a genuine asymmetry between the
input-side and output-side fields' own sensitivity to operand order.

**NOT established**: whether graph-input declaration order is
genuinely inert for `Mul` (assumed from Add's own result, not
independently tested here with a full 2x2 design); whether this holds
at shapes other than `(1,16)`; whether `Sub`'s own operand-order
sensitivity matches `Add`'s (plausible, given PR #1661 already found
Sub's encoding is otherwise indistinguishable from Add's, but not
tested here); `Div`'s own operand-order behavior (not tested by any PR
in this arc yet).
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
    path = os.path.join(FIX, f"mul_operand_order_{variant}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(variant):
    return mcode.decode(load(variant), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data this file's own fixture-build script used, matching PR
    #1658/#1659's own seed pair (1, 2) so `zp1`/`zp2` are directly
    comparable to their own already-established values."""
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
    division by 255) done in float64, `scale` rounded to float32 exactly
    ONE time -- cited directly from PR #1659's own module, not
    re-derived here."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


def requant_multiplier(seed1, seed2):
    """`y_scale / (x1_scale * x2_scale)` -- PR #1659's own Finding 3
    formula, cited directly, not re-derived."""
    x1 = calib_samples(seed1)
    x2 = calib_samples(seed2)
    ys = [a * b for a, b in zip(x1, x2)]
    sy = float(round_scale_once(ys))
    sc1 = float(round_scale_once(x1))
    sc2 = float(round_scale_once(x2))
    return np.float32(sy / (sc1 * sc2))


ZP1, _, _ = asymmetric_uint8_quant_params(calib_samples(SEED1))
ZP2, _, _ = asymmetric_uint8_quant_params(calib_samples(SEED2))
ZP_BY_TENSOR = {"x1": ZP1, "x2": ZP2}

_YS = [a * b for a, b in zip(calib_samples(SEED1), calib_samples(SEED2))]
ZP_Y, _, _ = asymmetric_uint8_quant_params(_YS)
MULTIPLIER = requant_multiplier(SEED1, SEED2)


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


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_either_variant(self):
        for variant in VARIANTS:
            errs = mcode.check(load(variant))
            self.assertEqual(errs, [], variant)


class TestZp1AndZp2MatchThePR1658EstablishedValues(unittest.TestCase):
    def test_zp1_is_133_and_zp2_is_131(self):
        self.assertEqual(ZP1, 133)
        self.assertEqual(ZP2, 131)


class TestNodeOperandOrderDeterminesTheX1X2Locator(unittest.TestCase):
    """Mirrors PR #1658's own core finding: whichever tensor is Mul's
    own FIRST operand gets the literal-quad locator; the other gets
    reg=94."""

    def test_first_operand_tensor_carries_the_literal_quad_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            data = load(variant)
            zp_first = ZP_BY_TENSOR[first_operand]
            hits = literal_quad_hits(data, zp_first)
            self.assertEqual(len(hits), 1, (variant, first_operand, zp_first))

    def test_second_operand_tensor_carries_reg94_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            second_operand = "x2" if first_operand == "x1" else "x1"
            recs = decode(variant)
            zp_second = ZP_BY_TENSOR[second_operand]
            hits = reg94_hits(recs, zp_second)
            self.assertEqual(len(hits), 1, (variant, second_operand, zp_second))

    def test_first_operand_tensor_does_not_also_appear_at_reg94(self):
        for variant, first_operand in VARIANTS.items():
            recs = decode(variant)
            zp_first = ZP_BY_TENSOR[first_operand]
            hits = reg94_hits(recs, zp_first)
            self.assertEqual(hits, [], (variant, first_operand))

    def test_second_operand_tensor_does_not_also_appear_in_the_literal_quad(self):
        for variant, first_operand in VARIANTS.items():
            second_operand = "x2" if first_operand == "x1" else "x1"
            data = load(variant)
            zp_second = ZP_BY_TENSOR[second_operand]
            hits = literal_quad_hits(data, zp_second)
            self.assertEqual(hits, [], (variant, second_operand))


class TestOutputSideFieldsAreUnaffectedByOperandOrder(unittest.TestCase):
    """The new finding: unlike the input-side x1/x2 locators, the
    requant multiplier and zp_y are bit-identical between the two
    operand orders -- Mul's own output value is commutative, so no
    quantity derived purely from y's own range can depend on which
    input was declared "first" in the node."""

    def test_requant_multiplier_matches_and_is_identical_across_variants(self):
        target = struct.pack("<f", MULTIPLIER)
        observed = {}
        for variant in VARIANTS:
            recs = decode(variant)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (224, 240)
            ]
            self.assertEqual(len(hits), 2, variant)
            operands = {r["operand"] for r in hits}
            self.assertEqual(operands, {target}, (variant, MULTIPLIER))
            observed[variant] = struct.unpack("<f", hits[0]["operand"])[0]
        self.assertEqual(observed["A_op12"], observed["C_op21"])

    def test_zp_y_matches_and_is_identical_across_variants(self):
        observed = {}
        for variant in VARIANTS:
            recs = decode(variant)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 76 and r.get("tag") == 132
            ]
            bytes_seen = {h["payload"][-1] for h in hits}
            self.assertIn(ZP_Y, bytes_seen, (variant, bytes_seen, ZP_Y))
            observed[variant] = ZP_Y
        self.assertEqual(observed["A_op12"], observed["C_op21"])


if __name__ == "__main__":
    unittest.main()
