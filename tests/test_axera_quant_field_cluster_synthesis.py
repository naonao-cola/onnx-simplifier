"""Synthesizes the cross-op arc that grew out of `tests/test_axera_conv_
binary_cluster_quantization_synthesis.py` (PR #1643): seven PRs (#1643-
#1650) all extending the same first-principles quantization technique --
port `ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_
quantize_matmul.h`) into Python, apply it to the exact `RandomState(seed)`-
derived calibration/weight data a fixture's own build script used, and
compare against the real decoded mcode byte -- across THREE ops (Conv,
Gemm, MatMul) and FOUR fields (an op's own input zero point/scale, and
its own output zero point/scale). This file plays the same synthesis
role `tests/test_axera_reg8_emit_capability_synthesis.py` (PR #1631) and
`tests/test_axera_bank81_e1_cross_op_synthesis.py` (PR #1614) already
played for the reg=8 and bank=0x81/0xe1 clusters: it (1) directly
re-confirms each contributing PR's headline computation against real
fixture bytes, independently of this file's own copy of the formulas,
not merely narrating their docstrings, (2) lays out the complete
(op, field) grid in one place, (3) resolves the one apparent
inconsistency worth flagging across the cluster, and (4) states exactly
what remains open.

## The complete (op, field) grid

| op | `zp_x` | `1/x_scale` | `y_scale` | `zp_y` |
| --- | --- | --- | --- | --- |
| Conv (dilation=3 family) | exact, `reg=54`/`reg=60` (PR #1640) | 4/6 exact, 2/6 within 1 ULP, `verb=161,bank=15` (PR #1641) | close, not exact, `reg=224` (PR #1642) | exact, `reg=232` (PR #1644, and independently again across dilation by PR #1649) |
| Gemm (`M=1,K=16,N=16`) | exact, literal quad `02 10 1b <zp_x> 83 36` (PR #1645) | exact, SAME `verb=161,bank=15` locator (PR #1646) | close, not exact, raw byte run near offset 1636 (PR #1646) | not attempted |
| MatMul (`M=1,K=16,N=16`) | not applicable -- always forced to 0, no literal quad ever fires (`tests/test_axera_zpx_generalizes.py`, PR #1497) | exact, IDENTICAL locator and byte offsets to Gemm's own `(1,16,16)` shape (PR #1647) | close, not exact, same raw-offset pattern as Gemm (PR #1647) | not attempted |

Two further, more recent findings refine this grid rather than filling a
new cell:

- The `verb=161,bank=15` locator's own byte offset is NOT shape-invariant
  -- it shifts with `K` and `M` (not `N`), and `1/x_scale`'s own
  bit-exactness breaks (by exactly 1 float32 ULP) at `M=4`, the first
  time non-bit-exactness was ever seen on `1/x_scale` itself rather than
  `y_scale` (PR #1648, Gemm-only, not re-tested here for Conv/MatMul).
- Conv's `zp_y` (not `y_scale`, which has no rounding step to have a
  "near tie") has its own distance-to-a-rounding-half-boundary, and that
  margin cleanly separates the two dilations where Conv's binary-cluster
  switch is ever observed to flip (2, 3) from the three where it never
  does (1, 4, 8) -- with a difference in the two present dilations' own
  observed flip RATES (16.7% vs. 37.5%) later shown to be statistically
  indistinguishable from sample-size noise at `n=8-12` (Fisher's exact
  `p=0.347`), not a real effect the margin theory needs to explain
  (PR #1649, PR #1650).

## Resolving the one apparent inconsistency: two independent `reg=232`
## decodes, never cross-referenced, dispatched without knowledge of each
## other

PR #1644 decoded `reg=232` as Conv's `zp_y` by varying CALIBRATION SEED
at a fixed `dilation=3`. PR #1649 decoded the same claim -- calling it
"new" in its own diff -- by varying DILATION at a fixed calibration seed
(`RandomState(0)`), because it was dispatched before PR #1644 existed and
never saw it. Both are correct findings on their own terms (different
axes), but they are not merely consistent in FRAMING -- as this file
verifies directly below, they in fact describe the exact same real
fixture: PR #1644's own `GROUP_A_SEED0 = ("conv_dilation3.mcode.gz", 0)`
is byte-identical to PR #1649's own `ALL_BY_DILATION[3][0]` (the first
name in `DILATION3_NAMES`, also `"conv_dilation3.mcode.gz"`). This file
decodes that one shared fixture itself (not trusting either PR's own
citation) and confirms both arrive at the same observed byte.

## What remains open

- A formula for the `verb=161,bank=15` offset shift with `K`/`M` (PR
  #1648 measured it precisely at one shape delta each; no formula was
  derived).
- Why `M>1` specifically breaks `1/x_scale`'s own bit-exactness while
  `K`/`N` changes do not.
- Whether Conv's own `zp_y` rounding-margin mechanism (PR #1649)
  generalizes to Gemm/MatMul's own output-side quantities -- untested,
  since Gemm/MatMul's own `zp_y` has not been attempted at all (next
  row of the grid above).
- MatMul's and Gemm's own `zp_y` (the one cell of the 3x4 grid attempted
  by no PR in this arc).
- Whether the `verb=161,bank=15` locator's shape-sensitivity (PR #1648,
  Gemm-only) also applies to Conv's or MatMul's own copy of it.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg_byte(recs, reg, tag):
    hits = [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == reg and r.get("tag") == tag
    ]
    assert hits, (reg, tag)
    values = {h["payload"][-1] for h in hits}
    assert len(values) == 1, values
    return next(iter(values))


def bank15_scale_like_floats(recs):
    """Conv's own `verb=161,bank=15` records include unrelated 2-byte
    operands elsewhere in the stream (`tests/test_axera_conv_binary_
    cluster_quantization_synthesis.py`'s own `observed_bank15` already
    filters these the same way: restrict to `field in (96,112,128)`
    and a 4-byte float32 operand)."""
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
    assert operands, hits
    return {struct.unpack("<f", op)[0] for op in operands}


def asymmetric_uint8_quant_params(samples):
    """`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_
    quantize_matmul.h`), independently re-ported in this file (not
    shared code with any contributing PR's own test module)."""
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


# ---------------------------------------------------------------------------
# Conv (dilation=3 calibration-seed family): reg=54=zp_x, verb161/bank15=
# 1/x_scale, reg=224=y_scale, reg=232=zp_y.
# ---------------------------------------------------------------------------

CONV_DETERMINISTIC_CASES = {
    1: "conv_dilation3_calibseed1_r0.mcode.gz",
    7: "conv_dilation3_calibseed7_r0.mcode.gz",
    42: "conv_dilation3_calibseed42_r0.mcode.gz",
    100: "conv_dilation3_calibseed100_r0.mcode.gz",
    999: "conv_dilation3_calibseed999_r0.mcode.gz",
}
CONV_GROUP_A_SEED0 = "conv_dilation3.mcode.gz"
CONV_GROUP_B_SEED0 = "conv_dilation3_rebuild0.mcode.gz"


def conv_calib_samples(seed, shape=(1, 4, 16, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def conv_weight():
    rng = np.random.RandomState(0)
    return (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)


def conv_reference(x, w, dilation=3, pad=3):
    cout, cin, kh, kw = w.shape
    _, _, height, width = x.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    w64 = w.astype(np.float64)
    y = np.zeros((1, cout, height, width), dtype=np.float64)
    for oc in range(cout):
        acc = np.zeros((height, width), dtype=np.float64)
        for ic in range(cin):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    oy = kh_i * dilation
                    ox = kw_i * dilation
                    patch = xp[0, ic, oy : oy + height, ox : ox + width]
                    acc += patch * w64[oc, ic, kh_i, kw_i]
        y[0, oc] = acc
    return y.astype(np.float32)


def conv_computed_y_zero_point_and_scale(seed):
    samples = conv_calib_samples(seed)
    w = conv_weight()
    ys_all = [conv_reference(s, w) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))
    zp_f = float(np.float32(-np.float32(ylo) / np.float32(scale)))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), scale


class TestConvClusterAllFourFieldsMatchSimultaneously(unittest.TestCase):
    """Directly re-confirms PR #1640/#1641/#1642/#1644's combined claim:
    all four fields match their respective formulas on every
    deterministic calibration seed, decoded fresh here."""

    def test_zp_x_and_zp_y_are_exact_on_every_deterministic_seed(self):
        for seed, name in CONV_DETERMINISTIC_CASES.items():
            recs = decode(name)
            zp_x, _, x_scale = asymmetric_uint8_quant_params(conv_calib_samples(seed))
            self.assertEqual(reg_byte(recs, 54, 131), zp_x, seed)
            self.assertEqual(reg_byte(recs, 60, 131), zp_x, seed)
            zp_y, _ = conv_computed_y_zero_point_and_scale(seed)
            self.assertEqual(reg_byte(recs, 232, 132), zp_y, seed)
            # PR #1641's own established tolerance for this field:
            # 4/6 of ITS OWN tested seeds were bit-exact, 2/6 within 1
            # float32 ULP -- this arc's own deterministic seeds are a
            # different 5-seed set, so re-asserting bit-exactness on
            # every one of them here is not warranted; a tight relative
            # tolerance re-confirms the same match quality honestly.
            recip = float(np.float32(1.0) / np.float32(x_scale))
            observed = bank15_scale_like_floats(recs)
            closest = min(observed, key=lambda v: abs(v - recip))
            self.assertLess(abs(closest - recip) / recip, 1e-6, (seed, observed))

    def test_y_scale_is_close_but_not_bit_exact(self):
        worst = 0.0
        for seed, name in CONV_DETERMINISTIC_CASES.items():
            recs = decode(name)
            _, y_scale = conv_computed_y_zero_point_and_scale(seed)
            hits = [
                r
                for r in recs
                if r.get("reg") == 224 and r.get("tag") == 129 and r.get("payload")
            ]
            observed = {struct.unpack("<f", h["payload"][-4:])[0] for h in hits}
            self.assertEqual(len(observed), 1, (seed, observed))
            rel_err = abs(next(iter(observed)) - y_scale) / y_scale
            worst = max(worst, rel_err)
            self.assertLess(rel_err, 5e-3, (seed, rel_err))
        self.assertGreater(worst, 1e-5, worst)


class TestReg232CrossCheckBetweenTheSeedArcAndTheDilationArc(unittest.TestCase):
    """Resolves the one apparent inconsistency flagged in this file's
    own module docstring: PR #1644 (varying calibration seed) and PR
    #1649 (varying dilation) both claim `reg=232` = `zp_y`, dispatched
    without either knowing about the other. This directly decodes the
    ONE fixture both arcs actually share -- `conv_dilation3.mcode.gz`,
    seed=0's Group A state -- and confirms it is not just consistent in
    framing but byte-identical."""

    def test_shared_fixture_reg232_matches_both_arcs_own_cited_value(self):
        recs_a = decode(CONV_GROUP_A_SEED0)
        # PR #1644's own KNOWN_REG232["0A"] and PR #1649's own dilation=3
        # table entry both independently cite 124 for this exact fixture.
        self.assertEqual(reg_byte(recs_a, 232, 132), 124)

        recs_b = decode(CONV_GROUP_B_SEED0)
        # PR #1644's own KNOWN_REG232["0B"].
        self.assertEqual(reg_byte(recs_b, 232, 132), 118)

    def test_recomputed_zp_y_matches_group_a_not_group_b(self):
        zp_y, _ = conv_computed_y_zero_point_and_scale(0)
        recs_a = decode(CONV_GROUP_A_SEED0)
        recs_b = decode(CONV_GROUP_B_SEED0)
        self.assertEqual(reg_byte(recs_a, 232, 132), zp_y)
        self.assertNotEqual(reg_byte(recs_b, 232, 132), zp_y)


# ---------------------------------------------------------------------------
# Gemm and MatMul (M=1,K=16,N=16): zp_x (Gemm only), 1/x_scale, y_scale.
# ---------------------------------------------------------------------------

GEMM_MATMUL_SEEDS = [0, 1, 7, 42, 100, 999]
RECIP_OFFSETS = (1456, 1464, 1472)
GEMM_ZP_X = {0: 135, 1: 133, 7: 129, 42: 131, 100: 132, 999: 126}
_GEMM_QUAD_PREFIX = bytes.fromhex("02101b")
_GEMM_QUAD_SUFFIX = bytes.fromhex("8336")
_GEMM_QUAD_OFFSET = 1421


def load_gemm(seed):
    return load(f"gemm_1x16x16_zpxseed{seed}.mcode.gz")


def load_matmul(seed):
    return load(f"matmul_1x16x16_zpxseed{seed}.mcode.gz")


def small_calib_samples(seed, shape=(1, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def gemm_weight():
    rng = np.random.RandomState(0)
    w = (rng.randn(16, 16) * 0.1).astype(np.float32)
    b = (rng.randn(16) * 0.1).astype(np.float32)
    return w, b


def matmul_weight():
    rng = np.random.RandomState(0)
    return (rng.randn(16, 16) * 0.1).astype(np.float32)


def gemm_reference(x, w, b):
    return (x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)).astype(
        np.float32
    )


def matmul_reference(x, w):
    return (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)


def computed_recip_x_scale(seed):
    _, _, x_scale = asymmetric_uint8_quant_params(small_calib_samples(seed))
    return float(np.float32(1.0) / np.float32(x_scale))


def computed_gemm_y_scale(seed):
    samples = small_calib_samples(seed)
    w, b = gemm_weight()
    ys_all = [gemm_reference(s, w, b) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    return float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))


def computed_matmul_y_scale(seed):
    samples = small_calib_samples(seed)
    w = matmul_weight()
    ys_all = [matmul_reference(s, w) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    return float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))


def nearest_float32_hit(data, target, lo, hi, max_rel_err):
    best = None
    for i in range(lo, min(hi, len(data) - 3)):
        val = struct.unpack_from("<f", data, i)[0]
        if not math.isfinite(val) or target == 0:
            continue
        rel_err = abs(val - target) / abs(target)
        if rel_err < max_rel_err and (best is None or rel_err < best[2]):
            best = (i, val, rel_err)
    return best


class TestGemmClusterMatchesAcrossAllThreeAttemptedFields(unittest.TestCase):
    """Directly re-confirms PR #1645/#1646's combined claim: `zp_x`
    (literal quad), `1/x_scale` (verb161/bank15), and `y_scale` (close,
    not exact) all match on every deterministic seed."""

    def test_zp_x_literal_quad_matches_every_seed(self):
        for seed, expected in GEMM_ZP_X.items():
            data = load_gemm(seed)
            zp, _, _ = asymmetric_uint8_quant_params(small_calib_samples(seed))
            self.assertEqual(zp, expected, seed)
            quad = _GEMM_QUAD_PREFIX + bytes([zp]) + _GEMM_QUAD_SUFFIX
            found = [
                i
                for i in range(len(data) - len(quad) + 1)
                if data[i : i + len(quad)] == quad
            ]
            self.assertEqual(found, [_GEMM_QUAD_OFFSET], seed)

    def test_recip_x_scale_matches_every_seed_at_fixed_offsets(self):
        for seed in GEMM_MATMUL_SEEDS:
            data = load_gemm(seed)
            recip = computed_recip_x_scale(seed)
            recs = mcode.decode(data)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("verb") == 161 and r.get("bank") == 15
            ]
            self.assertEqual(sorted(r["at"] for r in hits), list(RECIP_OFFSETS), seed)
            for r in hits:
                self.assertEqual(struct.unpack("<f", r["operand"])[0], recip, seed)

    def test_y_scale_is_close_but_not_bit_exact(self):
        worst = 0.0
        for seed in GEMM_MATMUL_SEEDS:
            data = load_gemm(seed)
            ys = computed_gemm_y_scale(seed)
            hit = nearest_float32_hit(data, ys, 1600, 1670, 3e-3)
            self.assertIsNotNone(hit, seed)
            worst = max(worst, hit[2])
        self.assertGreater(worst, 1e-5, worst)


class TestMatMulClusterMatchesForBothAttemptedFields(unittest.TestCase):
    """Directly re-confirms PR #1647's combined claim: `1/x_scale` is
    bit-exact at the IDENTICAL offsets Gemm's own `(1,16,16)` shape
    used, and `y_scale` is close, not exact -- `zp_x`/`zp_y` are
    correctly not attempted here, per PR #1497's already-established
    negative (MatMul's zero points are always forced to 0)."""

    def test_recip_x_scale_matches_every_seed_at_the_same_offsets_as_gemm(self):
        for seed in GEMM_MATMUL_SEEDS:
            data = load_matmul(seed)
            recip = computed_recip_x_scale(seed)
            recs = mcode.decode(data)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("verb") == 161 and r.get("bank") == 15
            ]
            self.assertEqual(sorted(r["at"] for r in hits), list(RECIP_OFFSETS), seed)
            for r in hits:
                self.assertEqual(struct.unpack("<f", r["operand"])[0], recip, seed)

    def test_y_scale_is_close_but_not_bit_exact(self):
        worst = 0.0
        for seed in GEMM_MATMUL_SEEDS:
            data = load_matmul(seed)
            ys = computed_matmul_y_scale(seed)
            hit = nearest_float32_hit(data, ys, 1600, 1670, 3e-3)
            self.assertIsNotNone(hit, seed)
            worst = max(worst, hit[2])
        self.assertGreater(worst, 1e-5, worst)


class TestFixturesDecodeCleanly(unittest.TestCase):
    """Sanity check on every fixture this synthesis file reuses."""

    def test_no_decode_errors(self):
        names = (
            list(CONV_DETERMINISTIC_CASES.values())
            + [CONV_GROUP_A_SEED0, CONV_GROUP_B_SEED0]
            + [f"gemm_1x16x16_zpxseed{s}.mcode.gz" for s in GEMM_MATMUL_SEEDS]
            + [f"matmul_1x16x16_zpxseed{s}.mcode.gz" for s in GEMM_MATMUL_SEEDS]
        )
        for name in names:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], name)


if __name__ == "__main__":
    unittest.main()
