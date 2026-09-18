"""Synthesizes the two-live-input elementwise cluster (PRs #1657/
#1658/#1659), the same role `tests/test_axera_quant_field_grid_
completion_synthesis.py` (PR #1656) already played for the earlier
single-live-input arc (Conv/Gemm/MatMul, PRs #1636/#1640-#1655).

Every op in the single-live-input arc has exactly ONE live input plus
one compile-time-constant weight. `Add`/`Mul` are this project's first
ops with TWO independently-calibrated LIVE inputs, neither one a
constant (`scripts/axera/README.md`'s own "Differential analysis" work,
PR #1098, only ever profiled `Add`/`Sub`/`Mul`/`Div` against a CONSTANT
broadcast operand -- a different, already-closed question; the real
`AxQuantizedAdd` residual add of two live activations had never had its
own zero-point/scale byte encoding attempted before PR #1657).

This file (1) directly re-decodes and recomputes every headline claim
from all three contributing PRs, independently of their own test
modules, using their already-committed fixtures (no new Docker builds),
(2) lays out one side-by-side Add-vs-Mul comparison table, (3) states
the "additive stores `y_scale` directly, multiplicative stores a
combined requant multiplier instead" hypothesis with EXACTLY the
evidence it has -- two ops, not an established law -- and notes `Sub`
has not yet been tested (still pending as of this file), and (4) lists
what remains open.

## The Add-vs-Mul comparison table

| field | Add (PR #1657) | Mul (PR #1659) |
| --- | --- | --- |
| `x1` zero point | literal quad `02 10 1b <zp> 83 36`, bit-exact | SAME locator, bit-exact |
| `x1` scale (`1/x1_scale`) | `verb=161,bank=15,field in (96,112,128)`, bit-exact | SAME locator, bit-exact |
| `x2` zero point | `reg=94,tag=132`, bit-exact | SAME locator, bit-exact |
| `x2` scale | genuinely absent (well-searched negative) | SAME negative |
| output scale | `y_scale` directly, bit-exact via PR #1654's round-scale-once formula, at the SAME `verb=161,bank=15,field in (96,112,128)` group as `x1`'s own scale | NOT `y_scale` -- a combined requant multiplier `y_scale/(x1_scale*x2_scale)`, bit-exact, at a NEW `field in (224,240)` pair in the same `verb=161,bank=15` group |
| `zp_y` | `reg=14,tag=131` (a constant `4` byte also present at that reg/tag, filtered out) | `reg=76,tag=132` (constant `32`/`26` bytes also present, filtered out) -- different register AND different tag from Add's own |
| operand-order dependence | `x1`/`x2` treatment is governed by ONNX node OPERAND order, not graph-input declaration order (PR #1658, a 2x2 disentangling design) | not re-tested (Mul's own fixtures did not vary operand order) |

Every row for both ops is independently re-decoded and recomputed
below, not cited from either contributing PR's own test module.

## The additive-vs-multiplicative hypothesis: exactly two data points

PR #1659's own diff proposed an architectural story: Add's dequantize-
then-add path only needs `y_scale` itself (both operands are already on
a compatible additive footing after their own zero-point subtraction),
while Mul's dequantize-then-multiply path produces a value scaled by
the PRODUCT of both input scales, so a combined multiplier bridging
that product to the output's own storage scale is the natural quantity
to store instead. This is a real, testable hypothesis -- but it
currently has exactly TWO data points (Add confirms the "additive"
branch, Mul confirms the "multiplicative" branch), not an established
law across two-live-input ops in general. `Sub` (also additive in the
dequantized domain, `y = x1 - x2`) is the natural next test and would
predict Add-style direct `y_scale` storage if the hypothesis holds --
as of this file, no `Sub` two-live-input survey has landed yet (no
`scripts/axera/fixtures/sub_*_two_live_*.mcode.gz` fixtures exist, and
no merged/open PR title matches "sub two-live" in this repository), so
this remains an open, still-pending test of the hypothesis, not a third
confirmed data point.

## What remains open

- `Sub`'s and `Div`'s own two-live-input treatment (not tested by any
  PR in this cluster yet).
- Whether the pattern holds at shapes other than `(1,16)`.
- Whether `x1`'s own byte offset (not just its content) is stable
  across ops and shapes -- PR #1648 already found the single-live-input
  arc's own `verb=161,bank=15` offset is shape-dependent for Gemm; this
  cluster has not re-checked whether that same shape-dependence applies
  to `x1`'s own copy of the locator in a two-live-input op.
- Which specific double-precision sub-step Mul's own requant-multiplier
  formula actually needs (PR #1659's own open item -- every variant it
  tried agreed, so the formula was not surgically isolated the way PR
  #1654 isolated `1/x_scale`'s own `hi-lo` subtraction step).
- Whether Add's `x2`/Mul's `x2` scale is truly absent from the stream,
  or merely not found by the search methods tried (a fixed-point or
  otherwise-encoded representation was never attempted).
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

SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]


def load(op, seed1, seed2):
    path = os.path.join(FIX, f"{op}_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(op, seed1, seed2):
    return mcode.decode(load(op, seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data both PR #1657's Add fixtures and PR #1659's Mul fixtures
    used, independently re-derived here (not imported)."""
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
    exactly ONE time -- independently re-derived here, not imported."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


def zp_x1_literal_quad_hits(data, zp1):
    quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
    return [i for i in range(len(data) - 5) if data[i : i + 6] == quad]


def recip_x1_scale_hits(recs, target_bytes):
    return [
        r
        for r in recs
        if r.get("kind") == "V"
        and r.get("verb") == 161
        and r.get("bank") == 15
        and r.get("field") in (96, 112, 128)
        and r.get("operand") == target_bytes
    ]


def zp_x2_hits(recs, zp2):
    return [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("reg") == 94
        and r.get("tag") == 132
        and r["payload"][-1] == zp2
    ]


class TestAddClusterReconfirmed(unittest.TestCase):
    """Re-decodes and recomputes PR #1657/#1658's own headline claims
    directly against real fixtures, independently of their own test
    modules."""

    def test_x1_zp_and_scale_are_bit_exact_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load("add", seed1, seed2)
            recs = decode("add", seed1, seed2)
            zp1, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            self.assertEqual(len(zp_x1_literal_quad_hits(data, zp1)), 1, (seed1, seed2))
            recip1 = struct.pack("<f", np.float32(np.float32(1.0) / np.float32(s1)))
            hits = recip_x1_scale_hits(recs, recip1)
            self.assertEqual(len(hits), 3, (seed1, seed2))

    def test_x2_zp_is_bit_exact_and_x2_scale_is_absent(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("add", seed1, seed2)
            zp2, _, s2 = asymmetric_uint8_quant_params(calib_samples(seed2))
            self.assertEqual(len(zp_x2_hits(recs, zp2)), 1, (seed1, seed2))
            recip2 = struct.pack("<f", np.float32(np.float32(1.0) / np.float32(s2)))
            allV = [r for r in recs if r.get("kind") == "V" and r.get("operand")]
            self.assertNotIn(recip2, {r["operand"] for r in allV}, (seed1, seed2))

    def test_y_scale_is_directly_bit_exact_via_round_scale_once(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("add", seed1, seed2)
            ys = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            sy32 = round_scale_once(ys)
            target = struct.pack("<f", sy32)
            hits = recip_x1_scale_hits(recs, target)
            self.assertEqual(len(hits), 3, (seed1, seed2))

    def test_zp_y_matches_at_reg14_tag131(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("add", seed1, seed2)
            ys = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 14 and r.get("tag") == 131
            ]
            bytes_seen = {h["payload"][-1] for h in hits}
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen))

    def test_operand_order_not_graph_input_order_determines_treatment(self):
        # Re-confirms PR #1658's own core finding directly: A (in=[x1,x2],
        # op=(x1,x2)) vs C (in=[x1,x2], op=(x2,x1)) share graph-input
        # order but flip operand order, and the literal-quad locator's
        # own carried value flips with them.
        zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(1))
        zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2))
        data_a = _load_operand_order("A_in12_op12")
        data_c = _load_operand_order("C_in12_op21")
        self.assertEqual(len(zp_x1_literal_quad_hits(data_a, zp1)), 1)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_a, zp2)), 0)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_c, zp2)), 1)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_c, zp1)), 0)


def _load_operand_order(variant):
    with gzip.open(
        os.path.join(FIX, f"add_operand_order_{variant}.mcode.gz"), "rb"
    ) as f:
        return f.read()


class TestMulClusterReconfirmed(unittest.TestCase):
    """Re-decodes and recomputes PR #1659's own headline claims
    directly against real fixtures, independently of its own test
    module."""

    def test_x1_and_x2_treatment_is_identical_to_add(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load("mul", seed1, seed2)
            recs = decode("mul", seed1, seed2)
            zp1, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
            self.assertEqual(len(zp_x1_literal_quad_hits(data, zp1)), 1, (seed1, seed2))
            recip1 = struct.pack("<f", np.float32(np.float32(1.0) / np.float32(s1)))
            self.assertEqual(len(recip_x1_scale_hits(recs, recip1)), 3, (seed1, seed2))
            self.assertEqual(len(zp_x2_hits(recs, zp2)), 1, (seed1, seed2))

    def test_y_scale_itself_is_not_stored_but_requant_multiplier_is(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("mul", seed1, seed2)
            x1 = calib_samples(seed1)
            x2 = calib_samples(seed2)
            ys = [a * b for a, b in zip(x1, x2)]
            _, _, sy_plain = asymmetric_uint8_quant_params(ys)
            sy_rso = float(round_scale_once(ys))

            second_pair = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (224, 240)
            ]
            self.assertEqual(len(second_pair), 2, (seed1, seed2))
            observed = struct.unpack("<f", second_pair[0]["operand"])[0]
            self.assertNotAlmostEqual(observed, sy_plain, places=3, msg=(seed1, seed2))
            self.assertNotAlmostEqual(observed, sy_rso, places=3, msg=(seed1, seed2))

            sc1 = float(round_scale_once(x1))
            sc2 = float(round_scale_once(x2))
            mult = np.float32(sy_rso / (sc1 * sc2))
            target = struct.pack("<f", mult)
            hits = [r for r in second_pair if r["operand"] == target]
            self.assertEqual(len(hits), 2, (seed1, seed2, mult))

    def test_zp_y_matches_at_reg76_tag132_a_different_locator_than_add(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("mul", seed1, seed2)
            ys = [a * b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 76 and r.get("tag") == 132
            ]
            bytes_seen = {h["payload"][-1] for h in hits}
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen))
        self.assertNotEqual((76, 132), (14, 131))


class TestAddAndMulDisagreeOnTheOutputScaleStorageStrategy(unittest.TestCase):
    """At the time this file was written, no `Sub` two-live-input
    survey had landed yet, so the additive-vs-multiplicative
    hypothesis below rested on exactly these two ops. `Sub` has since
    landed (PR #1661) and confirmed the additive branch directly --
    see `tests/test_axera_two_live_input_four_op_synthesis.py`'s own
    `TestSubClusterReconfirmed` and `TestHypothesisNowHasFourDataPoints
    NotTwo` for the up-to-date, four-op picture. This class keeps only
    its own still-valid claim (Add's and Mul's own output-scale
    mechanisms genuinely differ); the sibling "no Sub fixtures exist
    yet" existence check that used to live here has been removed as
    obsolete, not weakened -- the claim it protected is now fully
    superseded and re-verified elsewhere, not merely unchecked."""

    def test_add_and_mul_disagree_on_the_output_scale_storage_strategy(self):
        # The hypothesis's own two data points, restated as a direct
        # assertion: Add's y_scale IS found bit-exact via round-scale-
        # once at the plain verb161/bank15 group; Mul's is NOT (its
        # own field=224/240 pair carries a different, multiplier
        # quantity instead). If these ever agreed, the hypothesis
        # would already be falsified by this file's own two ops.
        seed1, seed2 = SEED_PAIRS[0]

        add_recs = decode("add", seed1, seed2)
        ys_add = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
        sy_add = round_scale_once(ys_add)
        add_hits = recip_x1_scale_hits(add_recs, struct.pack("<f", sy_add))
        self.assertEqual(len(add_hits), 3)

        mul_recs = decode("mul", seed1, seed2)
        ys_mul = [a * b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
        _, _, sy_mul_plain = asymmetric_uint8_quant_params(ys_mul)
        mul_second_pair = [
            r
            for r in mul_recs
            if r.get("kind") == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("field") in (224, 240)
        ]
        observed_mul = struct.unpack("<f", mul_second_pair[0]["operand"])[0]
        self.assertNotAlmostEqual(observed_mul, float(sy_mul_plain), places=3)
        self.assertNotAlmostEqual(
            observed_mul, float(round_scale_once(ys_mul)), places=3
        )


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_add_or_mul_fixture(self):
        for op in ("add", "mul"):
            for seed1, seed2 in SEED_PAIRS:
                errs = mcode.check(load(op, seed1, seed2))
                self.assertEqual(errs, [], (op, seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
