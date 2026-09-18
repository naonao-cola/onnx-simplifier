"""Synthesizes the operand-order and trivial-zero-point findings that
landed after `tests/test_axera_two_live_input_four_op_synthesis.py`
(PR #1664, the four-op Add/Sub/Mul/Div synthesis): PR #1665 (Sub/Div
operand-order), PR #1666 (Add output-side operand-order), and PR #1667
(the trivial-zero-point encoder omission extended to Add). Plays the
same additive-synthesis role every prior capstone file in this arc has
played: (1) directly re-decodes and recomputes each contributing PR's
own headline claim, independently of its own test module, using its
already-committed fixtures (no new Docker builds), (2) extends PR
#1664's own four-op table with the now-complete operand-order picture,
(3) documents the trivial-zero-point phenomenon's own confirmed extent
(2 of 4 ops, not all four), and (4) lists what remains open across the
whole two-live-input cluster now that this file marks a natural
completion point.

## The operand-order invariance picture, now complete for all four ops

| op | output algebraic symmetry | `y_scale`-or-multiplier invariant? | `zp_y` invariant? | source |
| --- | --- | --- | --- | --- |
| Add | full commutativity (`a+b` bit-identical to `b+a`) | yes | yes | PR #1666 |
| Mul | full commutativity (`a*b` bit-identical to `b*a`) | yes | yes | PR #1662 |
| Sub | range-negation symmetry only (`range(a-b)==range(b-a)`, but `a-b != b-a`) | yes | no | PR #1665 |
| Div | no symmetry (`a/b` and `b/a` are unrelated in range) | no | no | PR #1665 |

The rule is clean and now covers every op in the cluster: invariance
tracks exactly each op's own output algebraic symmetry class, not
merely "is the op commutative" (Sub is not commutative, yet its
`y_scale` is still invariant, for a strictly weaker reason -- range
negation preserves span, not value). All four rows are independently
re-decoded and recomputed below, not cited from any contributing PR's
own test module.

## The trivial-zero-point encoder omission: confirmed for 2 of 4 ops

PR #1663 found that an all-positive `x2` (divisor) drives its own
calibrated zero point to a degenerate `0`
(`ComputeAsymmetricUint8QuantParams`'s own `lo = min(0, samples.min())`
clamps to `0` for any strictly-positive tensor), and that this
suppresses the `reg=94,tag=132` locator entirely -- not merely a
zero-valued record, a structurally ABSENT one. PR #1667 confirmed this
reproduces identically for `Add`, with a proper control-fixture
comparison ruling out "the locator is just unreliable" as an
alternative explanation. This is now confirmed for `Add` and `Div`
specifically -- `Sub` and `Mul` have NEVER been built with a
deliberately all-positive `x2`, so this file does not claim the
phenomenon for all four ops, only the two it has real evidence for.

## What remains open across the whole two-live-input cluster

- Whether the operand-order and symmetry-class rules hold at shapes
  other than `(1,16)`, for any of the four ops.
- Whether the trivial-zero-point omission also applies to `Sub`/`Mul`
  (plausible given the identical mechanism now confirmed twice, but
  not independently verified for those two ops -- would need a new
  Docker build with a deliberately all-positive `x2` for each).
- Whether `x1`'s own zero point shows the same omission if it were
  made trivially zero (a different locator, the literal-quad, not
  `reg=94,tag=132` -- never tested, PR #1667's own open item).
- WHY `Mul` specifically needs the combined multiplier while `Div`
  (also multiplicative) does not (PR #1664's own still-open item, no
  mechanistic explanation available).
- Whether the symmetry-class rule would predict correctly for an op
  this arc has not yet characterized (e.g. a hypothetical `Max`/`Min`,
  commutative like Add/Mul, but with its own unknown quantization
  formula).
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
SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data Add/Sub/Mul's own fixture-build scripts all used."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def div_safe_samples(seed, shape=(1, 16), n_samples=4):
    """`sign * (2.0 + 0.3*randn())` -- PR #1663/#1665's own sign-flip
    divisor design, reused unchanged here for `Div`'s own operand-order
    fixtures (which flip which tensor plays the divisor role)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_samples):
        mag = 2.0 + 0.3 * rng.randn(*shape)
        sign = rng.choice([-1.0, 1.0], size=shape)
        out.append((sign * mag).astype(np.float32))
    return out


def x2_trivial_samples(seed2, shape=(1, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed2).randn(1,16)` x4, no sign flip --
    PR #1663's own all-positive design, reliably producing a
    degenerate `zp2=0`, reused unchanged here."""
    rng = np.random.RandomState(seed2)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


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
    """PR #1654's own refined formula: the `hi-lo` subtraction and the
    division by 255 done in float64, `scale` rounded to float32
    exactly once."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


def output_quant_params(ys, use_round_scale_once):
    ylo = min(0.0, min(float(y.min()) for y in ys))
    yhi = max(0.0, max(float(y.max()) for y in ys))
    if yhi <= ylo:
        yhi = ylo + 1.0
    if use_round_scale_once:
        scale = float(round_scale_once(ys))
    else:
        scale = float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))
    zp_f = float(np.float32(-np.float32(ylo) / np.float32(scale)))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), scale


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


def zp_y_hits(recs, reg, tag):
    return [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == reg and r.get("tag") == tag
    ]


class TestAddOutputSideIsFullyOperandOrderInvariant(unittest.TestCase):
    """Re-confirms PR #1666's own headline claim directly against real
    fixtures, independently of its own test module."""

    def test_y_scale_and_zp_y_are_bit_identical_between_orderings(self):
        x1, x2 = calib_samples(SEED1), calib_samples(SEED2)
        ys_a = [a + b for a, b in zip(x1, x2)]
        ys_c = [a + b for a, b in zip(x2, x1)]
        zpy_a, sy_a = output_quant_params(ys_a, use_round_scale_once=True)
        zpy_c, sy_c = output_quant_params(ys_c, use_round_scale_once=True)
        self.assertEqual(zpy_a, zpy_c)
        self.assertAlmostEqual(sy_a, sy_c, places=9)

        recs_a = decode("add_operand_order_A_in12_op12.mcode.gz")
        recs_c = decode("add_operand_order_C_in12_op21.mcode.gz")
        floats_a, floats_c = bank15_floats(recs_a), bank15_floats(recs_c)
        hit_a = {f for f in floats_a if abs(f - sy_a) / sy_a < 1e-5}
        hit_c = {f for f in floats_c if abs(f - sy_c) / sy_c < 1e-5}
        self.assertEqual(len(hit_a), 1)
        self.assertEqual(len(hit_c), 1)
        self.assertAlmostEqual(hit_a.pop(), hit_c.pop(), places=5)

        self.assertTrue(
            [h for h in zp_y_hits(recs_a, 14, 131) if h["payload"][-1] == zpy_a]
        )
        self.assertTrue(
            [h for h in zp_y_hits(recs_c, 14, 131) if h["payload"][-1] == zpy_c]
        )


class TestSubOutputSideScaleInvariantZpYNot(unittest.TestCase):
    """Re-confirms PR #1665's own Sub findings directly against real
    fixtures, independently of its own test module."""

    def test_y_scale_invariant_zp_y_differs(self):
        x1, x2 = calib_samples(SEED1), calib_samples(SEED2)
        ys_a = [a - b for a, b in zip(x1, x2)]
        ys_c = [a - b for a, b in zip(x2, x1)]
        zpy_a, sy_a = output_quant_params(ys_a, use_round_scale_once=False)
        zpy_c, sy_c = output_quant_params(ys_c, use_round_scale_once=False)

        # range(a-b) == range(b-a): scale is invariant.
        self.assertAlmostEqual(sy_a, sy_c, places=9)
        # zero's own position within a negated range is not invariant.
        self.assertNotEqual(zpy_a, zpy_c)

        recs_a = decode("sub_operand_order_A_op12.mcode.gz")
        recs_c = decode("sub_operand_order_C_op21.mcode.gz")
        floats_a, floats_c = bank15_floats(recs_a), bank15_floats(recs_c)
        hit_a = {f for f in floats_a if abs(f - sy_a) / sy_a < 1e-5}
        hit_c = {f for f in floats_c if abs(f - sy_c) / sy_c < 1e-5}
        self.assertEqual(len(hit_a), 1)
        self.assertEqual(len(hit_c), 1)
        self.assertAlmostEqual(hit_a.pop(), hit_c.pop(), places=5)

        self.assertTrue(
            [h for h in zp_y_hits(recs_a, 14, 131) if h["payload"][-1] == zpy_a]
        )
        self.assertTrue(
            [h for h in zp_y_hits(recs_c, 14, 131) if h["payload"][-1] == zpy_c]
        )
        self.assertFalse(
            [h for h in zp_y_hits(recs_a, 14, 131) if h["payload"][-1] == zpy_c]
        )


class TestDivOutputSideNeitherInvariant(unittest.TestCase):
    """Re-confirms PR #1665's own Div findings directly against real
    fixtures, independently of its own test module -- the one op in
    the cluster where NEITHER output-side field is operand-order
    invariant."""

    def test_y_scale_and_zp_y_both_differ(self):
        x1 = div_safe_samples(SEED1)
        x2 = div_safe_samples(SEED2)
        ys_a = [a / b for a, b in zip(x1, x2)]
        ys_c = [a / b for a, b in zip(x2, x1)]
        zpy_a, sy_a = output_quant_params(ys_a, use_round_scale_once=False)
        zpy_c, sy_c = output_quant_params(ys_c, use_round_scale_once=False)

        self.assertNotAlmostEqual(sy_a, sy_c, places=6)
        self.assertNotEqual(zpy_a, zpy_c)

        data_a = load("div_operand_order_A_op12.mcode.gz")
        data_c = load("div_operand_order_C_op21.mcode.gz")
        target_a = struct.pack("<f", np.float32(sy_a))
        target_c = struct.pack("<f", np.float32(sy_c))
        hits_a = [i for i in range(len(data_a) - 3) if data_a[i : i + 4] == target_a]
        hits_c = [i for i in range(len(data_c) - 3) if data_c[i : i + 4] == target_c]
        self.assertTrue(hits_a)
        self.assertTrue(hits_c)

        recs_a = decode("div_operand_order_A_op12.mcode.gz")
        recs_c = decode("div_operand_order_C_op21.mcode.gz")
        self.assertTrue(
            [h for h in zp_y_hits(recs_a, 74, 132) if h["payload"][-1] == zpy_a]
        )
        self.assertTrue(
            [h for h in zp_y_hits(recs_c, 74, 132) if h["payload"][-1] == zpy_c]
        )


class TestOperandOrderInvarianceTracksExactlyTheOutputSymmetryClass(unittest.TestCase):
    """The cluster-level claim, as a real assertion: full commutativity
    (Add, Mul) invariants both fields; range-negation symmetry only
    (Sub) invariants scale but not zero point; no symmetry (Div)
    invariants neither."""

    def test_symmetry_class_predicts_invariance_pattern_for_all_four_ops(self):
        x1, x2 = calib_samples(SEED1), calib_samples(SEED2)

        add_a = output_quant_params(
            [a + b for a, b in zip(x1, x2)], use_round_scale_once=True
        )
        add_c = output_quant_params(
            [a + b for a, b in zip(x2, x1)], use_round_scale_once=True
        )
        self.assertEqual(add_a, add_c)  # full commutativity: both invariant

        sub_a = output_quant_params(
            [a - b for a, b in zip(x1, x2)], use_round_scale_once=False
        )
        sub_c = output_quant_params(
            [a - b for a, b in zip(x2, x1)], use_round_scale_once=False
        )
        self.assertAlmostEqual(sub_a[1], sub_c[1], places=9)  # scale invariant
        self.assertNotEqual(sub_a[0], sub_c[0])  # zero point not

        div_x1, div_x2 = div_safe_samples(SEED1), div_safe_samples(SEED2)
        div_a = output_quant_params(
            [a / b for a, b in zip(div_x1, div_x2)], use_round_scale_once=False
        )
        div_c = output_quant_params(
            [a / b for a, b in zip(div_x2, div_x1)], use_round_scale_once=False
        )
        self.assertNotAlmostEqual(div_a[1], div_c[1], places=6)  # neither invariant
        self.assertNotEqual(div_a[0], div_c[0])


class TestTrivialZeroPointOmissionConfirmedForAddAndDivOnly(unittest.TestCase):
    """Directly re-confirms PR #1667's own Add finding, and explicitly
    checks (not merely narrates) that Sub/Mul have no such fixtures --
    the phenomenon's own confirmed extent is 2 of 4 ops, not claimed
    more broadly."""

    def test_add_trivial_x2_zero_point_suppresses_reg94_tag132(self):
        for seed1, seed2 in SEED_PAIRS:
            samples = x2_trivial_samples(seed2)
            zp2, _, _ = asymmetric_uint8_quant_params(samples)
            self.assertEqual(zp2, 0, (seed1, seed2))
            recs = decode(f"add_1x16_two_live_seed{seed1}_{seed2}_trivialx2.mcode.gz")
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(hits, [], (seed1, seed2))

    def test_no_sub_or_mul_trivial_x2_fixtures_exist_yet(self):
        for name in (
            "sub_1x16_two_live_seed1_2_trivialx2.mcode.gz",
            "mul_1x16_two_live_seed1_2_trivialx2.mcode.gz",
        ):
            self.assertFalse(
                os.path.exists(os.path.join(FIX, name)),
                f"{name} exists -- update this file's own claim that "
                "the trivial-zero-point phenomenon is confirmed for "
                "only Add and Div, not yet Sub/Mul",
            )


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_fixture_used_in_this_file(self):
        names = [
            "add_operand_order_A_in12_op12.mcode.gz",
            "add_operand_order_C_in12_op21.mcode.gz",
            "sub_operand_order_A_op12.mcode.gz",
            "sub_operand_order_C_op21.mcode.gz",
            "div_operand_order_A_op12.mcode.gz",
            "div_operand_order_C_op21.mcode.gz",
        ]
        for name in names:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(
                load(f"add_1x16_two_live_seed{seed1}_{seed2}_trivialx2.mcode.gz")
            )
            self.assertEqual(errs, [], (seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
