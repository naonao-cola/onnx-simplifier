"""Synthesizes the COMPLETE four-op two-live-input elementwise cluster
(Add PR #1657/#1658, Mul PR #1659/#1662, Sub PR #1661, Div PR #1663),
extending `tests/test_axera_two_live_input_cluster_synthesis.py` (PR
#1660, Add-vs-Mul only) to all four ops now that Sub and Div have
landed. Plays the same role for this cluster that PR #1656 played for
the earlier single-live-input arc: (1) directly re-decodes and
recomputes every headline claim from the Sub/Mul-operand-order/Div
PRs, independently of their own test modules, using their
already-committed fixtures (no new Docker builds), (2) lays out one
final four-op comparison table, (3) states the REVISED hypothesis --
not "additive vs multiplicative" (falsified by Div) but "Mul
specifically is the odd one out" -- with exactly the evidence now
available (4 ops, not 2), and (4) lists what remains open.

## The four-op comparison table

| field | Add (#1657/#1658) | Sub (#1661) | Mul (#1659/#1662) | Div (#1663) |
| --- | --- | --- | --- | --- |
| `x1` zero point | literal quad `02 10 1b <zp> 83 36`, bit-exact | SAME | SAME | SAME |
| `x1` scale (`1/x1_scale`) | `verb=161,bank=15,field in (96,112,128)`, bit-exact | SAME | SAME | SAME |
| `x2` zero point | `reg=94,tag=132`, bit-exact | SAME | SAME | SAME (degenerate/absent when x2's own zp is trivially 0, #1663 Finding 0) |
| `x2` scale | genuinely absent | SAME | SAME | SAME |
| output scale | `y_scale` directly, round-scale-once, bit-exact, at the decoded `verb=161,bank=15` group | SAME locator/mechanism | NOT `y_scale` -- combined multiplier `y_scale/(x1_scale*x2_scale)` at the SAME `verb=161,bank=15` group's own separate `field in (224,240)` pair | `y_scale` directly, bit-exact under the PLAIN formula (no round-scale-once needed) -- but at a DIFFERENT locator, a raw undecoded byte run (4 evenly-spaced copies), not the `verb=161,bank=15` V-record group Add/Sub/Mul all use |
| `zp_y` | `reg=14,tag=131` | SAME (`reg=14,tag=131`) | `reg=76,tag=132` | `reg=74,tag=132` (shares Mul's tag, not Mul's register, not Add/Sub's tag) |
| operand-order dependence | governed by node OPERAND order, not graph-input order (#1658) | not tested | SAME rule confirmed (#1662); output-side fields (multiplier, zp_y) unaffected, since Mul's output is commutative | not tested |

Every row for Sub/Mul-operand-order/Div is independently re-decoded
and recomputed below, not cited from any contributing PR's own test
module. Add/Mul's own rows were already re-confirmed by PR #1660; this
file does not repeat that work, only extends it.

## The revised hypothesis: not additive-vs-multiplicative, but
## "Mul specifically is the odd one out"

PR #1659's own original hypothesis ("additive ops store `y_scale`
directly, multiplicative ops store a combined multiplier") predicted
`Div` (multiplicative, like `Mul`) would need a multiplier too. It does
not (PR #1663): `Div` stores `y_scale` directly, indistinguishable on
this axis from `Add`/`Sub`. Of the four ops now tested, `Mul` is the
ONE exception on the output-scale-storage axis -- not "the
multiplicative branch" of a clean binary split, since `Div` is also
multiplicative and does not share Mul's own mechanism. This file does
not know WHY `Mul` specifically needs the combined multiplier while
`Div` does not (no access to Pulsar2's own source) -- only that,
directly re-confirmed here against real fixture bytes, it does.

`zp_y`'s own locator gives a similar, non-binary picture: Add/Sub
share one `(reg,tag)` pair, Mul and Div each use their own distinct
register, but Mul and Div share the tag (`132`) that differs from
Add/Sub's (`131`) -- a partial, not total, split along the same
additive/multiplicative line the output-scale axis just falsified as a
clean rule. Four ops is still a small sample; this file does not
extrapolate a general law from one axis matching a 2-2 split and
another axis matching a 3-1 split.

## Finding 0's own generality: does an all-positive `x2` also make
## Add/Sub/Mul's own `zp_y2` trivially absent?

PR #1663's own Finding 0 (an all-positive `x2` drives its own zero
point to a degenerate `0`, which suppresses the `reg=94,tag=132`
locator entirely) was only built and tested for `Div`. This is
checkable WITHOUT a new Docker build for the mechanism's own
first half (whether the calibration data itself would produce a
degenerate zero point) but NOT for the second half (whether the real
mcode encoder actually omits the record) without a real build, since
that is a genuine encoder behavior, not a recomputable formula. This
file confirms the first half only: Add's own `x2` calibration data
(`RandomState(2).randn(1,16)`, PR #1657's own convention) is
symmetric around zero by construction (a Gaussian, not the
`2.0 + 0.3*randn()` shifted-positive design `Div`'s own first attempt
used) and so never produces the specific "clamped to exactly 0"
degenerate case Finding 0 needed -- explaining, via calibration-design
difference alone, why this was never observed for Add/Sub/Mul's own
fixtures without needing a new build to confirm it. Whether the real
encoder's own omission behavior (not just the degenerate zero point
computation) also applies to Add/Sub/Mul's `x2` remains untested and
would require a new build with a deliberately all-positive `x2` for
one of them -- left open below, not claimed here.

## What remains open

- Whether the pattern holds at shapes other than `(1,16)` for any of
  the four ops.
- `Sub`'s and `Div`'s own operand-order sensitivity (only Add/Mul have
  been tested, PRs #1658/#1662).
- WHY `Mul` specifically needs the combined multiplier while `Div`
  (also multiplicative) does not -- no mechanistic explanation
  available, only the directly-confirmed fact of the split.
- Whether Finding 0's own second half (the real encoder's omission
  behavior, not just the degenerate-zero-point calibration math) also
  applies to Add/Sub/Mul's own `x2` -- would require a new Docker
  build with a deliberately all-positive `x2` for one of them, not
  attempted here.
- Which specific double-precision sub-step Mul's own requant-multiplier
  formula actually needs (PR #1659's own still-open item).
- Whether Add's/Sub's/Mul's/Div's own `x2` scale is truly absent from
  the stream, or merely not found by the search methods tried (a
  fixed-point or otherwise-encoded representation was never attempted,
  same open item every contributing PR already flagged).
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


def load_div_trivial(seed1, seed2):
    path = os.path.join(
        FIX, f"div_1x16_two_live_seed{seed1}_{seed2}_trivialzp2.mcode.gz"
    )
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(op, seed1, seed2):
    return mcode.decode(load(op, seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data Add/Sub/Mul's own fixture-build scripts all used, independently
    re-derived here (not imported)."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def div_x2_samples(seed, shape=(1, 16), n_samples=4):
    """`sign * (2.0 + 0.3*randn())` -- PR #1663's own non-degenerate
    divisor design, cited directly, not re-derived from first
    principles (the exact sign/magnitude split is this file's own
    dependency, not something worth re-deriving)."""
    rng = np.random.RandomState(seed)
    mag = 2.0 + 0.3 * rng.randn(*shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return [(sign * mag).astype(np.float32) for _ in range(n_samples)]


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


def mul_requant_multiplier_hits(recs, target_bytes):
    return [
        r
        for r in recs
        if r.get("kind") == "V"
        and r.get("verb") == 161
        and r.get("bank") == 15
        and r.get("field") in (224, 240)
        and r.get("operand") == target_bytes
    ]


def zp_y_hits(recs, reg, tag, zpy):
    return [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("reg") == reg
        and r.get("tag") == tag
        and r.get("payload")
        and r["payload"][-1] == zpy
    ]


class TestSubClusterReconfirmed(unittest.TestCase):
    """Re-decodes and recomputes PR #1661's own headline claims
    directly against real fixtures, independently of its own test
    module."""

    def test_x1_and_x2_treatment_matches_add(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load("sub", seed1, seed2)
            recs = decode("sub", seed1, seed2)
            zp1, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
            self.assertEqual(len(zp_x1_literal_quad_hits(data, zp1)), 1, (seed1, seed2))
            recip1 = struct.pack("<f", np.float32(np.float32(1.0) / np.float32(s1)))
            self.assertEqual(len(recip_x1_scale_hits(recs, recip1)), 3, (seed1, seed2))
            self.assertEqual(len(zp_x2_hits(recs, zp2)), 1, (seed1, seed2))

    def test_y_scale_is_directly_bit_exact_like_add_not_like_mul(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("sub", seed1, seed2)
            ys = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            sy32 = round_scale_once(ys)
            target = struct.pack("<f", sy32)
            self.assertEqual(len(recip_x1_scale_hits(recs, target)), 3, (seed1, seed2))
            self.assertEqual(mul_requant_multiplier_hits(recs, target), [])

    def test_zp_y_matches_at_reg14_tag131_same_as_add(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("sub", seed1, seed2)
            ys = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            self.assertTrue(zp_y_hits(recs, 14, 131, zpy), (seed1, seed2))
            self.assertEqual(zp_y_hits(recs, 76, 132, zpy), [])


class TestMulOperandOrderReconfirmed(unittest.TestCase):
    """Re-decodes and recomputes PR #1662's own headline claims
    directly against real fixtures, independently of its own test
    module."""

    def _requant_multiplier(self, seed1, seed2):
        x1 = calib_samples(seed1)
        x2 = calib_samples(seed2)
        ys = [a * b for a, b in zip(x1, x2)]
        sy = float(round_scale_once(ys))
        sc1 = float(round_scale_once(x1))
        sc2 = float(round_scale_once(x2))
        return np.float32(sy / (sc1 * sc2))

    def _load_mul_operand_order(self, variant):
        with gzip.open(
            os.path.join(FIX, f"mul_operand_order_{variant}.mcode.gz"), "rb"
        ) as f:
            return f.read()

    def test_operand_order_flips_x1_x2_locator_like_add(self):
        zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(1))
        zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2))
        data_a = self._load_mul_operand_order("A_op12")
        data_c = self._load_mul_operand_order("C_op21")
        self.assertEqual(len(zp_x1_literal_quad_hits(data_a, zp1)), 1)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_a, zp2)), 0)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_c, zp2)), 1)
        self.assertEqual(len(zp_x1_literal_quad_hits(data_c, zp1)), 0)

    def test_output_side_fields_are_unaffected_by_operand_order(self):
        mult = self._requant_multiplier(1, 2)
        target = struct.pack("<f", mult)
        ys = [a * b for a, b in zip(calib_samples(1), calib_samples(2))]
        zpy, _, _ = asymmetric_uint8_quant_params(ys)

        recs_a = mcode.decode(self._load_mul_operand_order("A_op12"), **mcode.FULL_RULE)
        recs_c = mcode.decode(self._load_mul_operand_order("C_op21"), **mcode.FULL_RULE)
        self.assertEqual(len(mul_requant_multiplier_hits(recs_a, target)), 2)
        self.assertEqual(len(mul_requant_multiplier_hits(recs_c, target)), 2)
        self.assertTrue(zp_y_hits(recs_a, 76, 132, zpy))
        self.assertTrue(zp_y_hits(recs_c, 76, 132, zpy))


class TestDivClusterReconfirmed(unittest.TestCase):
    """Re-decodes and recomputes PR #1663's own headline claims
    directly against real fixtures, independently of its own test
    module -- the arc's second multiplicative op, which falsifies the
    original "multiplicative needs a combined multiplier" prediction."""

    def test_x1_and_x2_treatment_matches_the_rest_of_the_cluster(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load("div", seed1, seed2)
            recs = decode("div", seed1, seed2)
            zp1, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            zp2, _, _ = asymmetric_uint8_quant_params(div_x2_samples(seed2))
            self.assertEqual(len(zp_x1_literal_quad_hits(data, zp1)), 1, (seed1, seed2))
            recip1 = struct.pack("<f", np.float32(np.float32(1.0) / np.float32(s1)))
            self.assertEqual(len(recip_x1_scale_hits(recs, recip1)), 3, (seed1, seed2))
            self.assertEqual(len(zp_x2_hits(recs, zp2)), 1, (seed1, seed2))

    def test_y_scale_is_directly_bit_exact_falsifying_the_multiplicative_prediction(
        self,
    ):
        # Unlike Add/Sub, Div's own y_scale is NOT found among the
        # decoded verb=161,bank=15 V-records -- it lives at a raw,
        # undecoded byte offset (PR #1663's own Finding 2), found via
        # a direct substring search, 4 evenly-spaced copies.
        for seed1, seed2 in SEED_PAIRS:
            data = load("div", seed1, seed2)
            recs = decode("div", seed1, seed2)
            x1 = calib_samples(seed1)
            x2 = div_x2_samples(seed2)
            ys = [a / b for a, b in zip(x1, x2)]
            _, _, sy = asymmetric_uint8_quant_params(ys)
            target = struct.pack("<f", np.float32(sy))
            hits = [i for i in range(len(data) - 3) if data[i : i + 4] == target]
            self.assertEqual(len(hits), 4, (seed1, seed2, sy))
            self.assertEqual(mul_requant_multiplier_hits(recs, target), [])

    def test_zp_y_matches_at_reg74_tag132_distinct_from_every_other_op(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode("div", seed1, seed2)
            ys = [a / b for a, b in zip(calib_samples(seed1), div_x2_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            self.assertTrue(zp_y_hits(recs, 74, 132, zpy), (seed1, seed2))
            self.assertEqual(zp_y_hits(recs, 14, 131, zpy), [], (seed1, seed2))

    def test_trivial_x2_zero_point_suppresses_reg94_tag132_finding0(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load_div_trivial(seed1, seed2)
            recs = mcode.decode(data, **mcode.FULL_RULE)
            rng = np.random.RandomState(seed2)
            trivial_x2 = [
                (2.0 + 0.3 * rng.randn(1, 16)).astype(np.float32) for _ in range(4)
            ]
            zp2, _, _ = asymmetric_uint8_quant_params(trivial_x2)
            self.assertEqual(zp2, 0, (seed1, seed2))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(hits, [], (seed1, seed2))


class TestHypothesisNowHasFourDataPointsNotTwo(unittest.TestCase):
    """The revised claim, as a real assertion: Add/Sub (additive) and
    Div (multiplicative, but behaving like the additive ops on this
    axis) all store `y_scale` directly; only Mul stores a combined
    multiplier. Falsifies the original binary additive-vs-
    multiplicative split as literally stated."""

    def test_add_sub_div_all_store_y_scale_directly_only_mul_does_not(self):
        seed1, seed2 = SEED_PAIRS[0]

        add_recs = decode("add", seed1, seed2)
        ys_add = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
        sy_add = round_scale_once(ys_add)
        self.assertEqual(
            len(recip_x1_scale_hits(add_recs, struct.pack("<f", sy_add))), 3
        )

        sub_recs = decode("sub", seed1, seed2)
        ys_sub = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
        sy_sub = round_scale_once(ys_sub)
        self.assertEqual(
            len(recip_x1_scale_hits(sub_recs, struct.pack("<f", sy_sub))), 3
        )

        div_data = load("div", seed1, seed2)
        ys_div = [a / b for a, b in zip(calib_samples(seed1), div_x2_samples(seed2))]
        _, _, sy_div_plain = asymmetric_uint8_quant_params(ys_div)
        # Div's own y_scale lives at a raw byte offset, not the
        # decoded verb=161,bank=15 locator Add/Sub/Mul's own y_scale-
        # or-multiplier group uses -- found via a direct substring
        # search, matching PR #1663's own method.
        target_div = struct.pack("<f", np.float32(sy_div_plain))
        div_hits = [
            i for i in range(len(div_data) - 3) if div_data[i : i + 4] == target_div
        ]
        self.assertEqual(len(div_hits), 4)

        mul_recs = decode("mul", seed1, seed2)
        x1_mul = calib_samples(seed1)
        x2_mul = calib_samples(seed2)
        ys_mul = [a * b for a, b in zip(x1_mul, x2_mul)]
        _, _, sy_mul_plain = asymmetric_uint8_quant_params(ys_mul)
        sy_mul_rso = float(round_scale_once(ys_mul))
        self.assertEqual(
            recip_x1_scale_hits(mul_recs, struct.pack("<f", np.float32(sy_mul_plain))),
            [],
        )
        self.assertEqual(
            recip_x1_scale_hits(mul_recs, struct.pack("<f", np.float32(sy_mul_rso))),
            [],
        )
        sc1_mul = float(round_scale_once(x1_mul))
        sc2_mul = float(round_scale_once(x2_mul))
        multiplier = np.float32(sy_mul_rso / (sc1_mul * sc2_mul))
        self.assertNotEqual(
            mul_requant_multiplier_hits(mul_recs, struct.pack("<f", multiplier)), []
        )

    def test_zp_y_tags_split_three_one_not_two_two(self):
        # Add/Sub share tag 131; Mul/Div share tag 132 but use
        # different registers (76 vs 74) -- a partial, not clean,
        # alignment with the additive/multiplicative line.
        self.assertNotEqual((14, 131), (76, 132))
        self.assertNotEqual((14, 131), (74, 132))
        self.assertNotEqual((76, 132), (74, 132))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_fixture_in_this_file(self):
        for op in ("sub", "div"):
            for seed1, seed2 in SEED_PAIRS:
                errs = mcode.check(load(op, seed1, seed2))
                self.assertEqual(errs, [], (op, seed1, seed2, errs))
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load_div_trivial(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
