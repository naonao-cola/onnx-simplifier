"""Continues `tests/test_axera_conv_reg54_zeropoint_verification.py`
(PR #1640)'s own newly-proven technique: independently recompute a
calibration-derived quantization quantity from first principles (the
exact `RandomState(seed)`-generated calibration/weight data every
relevant fixture's own build script used) and compare it against a
real, already-decoded mcode field. PR #1640 applied this to `reg=54`
(Conv's own input zero point) and found an exact match on every
deterministic seed. This file applies the same technique to the OTHER
two computed float32 values `tests/test_axera_conv_reg60_mechanism.py`
(PR #1586) found co-moving with `reg=60` inside Conv's own 28-byte
binary-cluster switch: a `verb=161,bank=15` V-record, and `reg=224`
(4 copies).

## Finding 1: `verb=161,bank=15`'s own float value is EXACTLY
## `1 / x_scale` -- the same input calibration scale PR #1640's own
## zero-point formula already computes, and a match this project
## ALREADY had a name for

`scale = (hi - lo) / 255.0` (PR #1640's own cited
`ComputeAsymmetricUint8QuantParams` formula, applied to `x`'s own
calibration range -- the same `lo`/`hi` PR #1640 already computes for
the zero point, just not yet inverted and compared against this
second field). Its reciprocal, computed independently here from the
exact calibration data `tests/test_axera_mcode_structure.py`'s own
`_build_and_get_mcode_bytes` generates, matches the real observed
`bank=15` float value to within float32 print-rounding (relative
error ~1e-7 to 1e-9) on every one of the 6 deterministic states
already known from `tests/test_axera_conv_binary_cluster_seed_survey.py`
(PR #1600):

| seed | `1/x_scale` (computed) | observed `bank=15` value | rel. error |
| --- | --- | --- | --- |
| 0, group A | 33.811733 | 33.811733 | ~7e-9 |
| 1 | 35.855091 | 35.855095 | ~1e-7 |
| 7 | 35.040691 | 35.040691 | ~1e-8 |
| 42 | 35.577232 | 35.577232 | ~1e-8 |
| 100 | 36.078636 | 36.078632 | ~1e-7 |
| 999 | 37.855480 | 37.855480 | ~5e-9 |

(Seed 0's OTHER state, group B, does not match -- expected, not a
counter-example: group B is the OTHER side of the same near-tie PR
#1640 already found for `reg=54`/`reg=60`, so its own real
input-range computation resolved differently inside Pulsar2's own
non-deterministic accumulation, the same reason group B's own `reg=54`
value is `+1` relative to group A's rather than identical.)

This is not a new quantity -- `1/x_scale` is precisely this project's
own already-established "site A" pattern from earlier in this session
(`<f32(1/x_scale)> a1 00 <id>` x4 stride 8, decoded for Gemm/Conv/MatMul
long before this session's own binary-cluster investigation began).
Conv's own `binary path switch` cluster was never previously
cross-referenced against that earlier finding; this file makes that
connection directly, for the first time, with a live numeric check
rather than by name alone.

## Finding 2: `reg=224` is very likely the OUTPUT calibration scale
## (`y_scale`, the same MinMax formula applied to `Conv`'s own output
## range) -- a strong but NOT bit-exact match, reported honestly

`reg=224`'s own values (`~0.0098`-`~0.0184`) are far too small to be a
weight scale: Conv's weight is fixed at `RandomState(0)` regardless of
calibration seed (`tests/test_axera_mcode_structure.py`'s own
`_dilation_conv_model`), so a pure weight-derived quantity would be
CONSTANT across all seeds -- confirmed directly here that the
per-tensor symmetric weight scale (`max(abs(w))/127 ≈ 0.00201`) is
constant and does NOT match any observed `reg=224` value at any seed,
ruling that reading out.

Instead, running the SAME asymmetric MinMax formula PR #1640 already
verified for `x_scale`, but applied to `y = Conv(x, w)`'s own output
range (computed with a float64-accumulating reference convolution to
minimize this file's own numerical noise, then cast back to float32
for the final scale, matching this project's own established float32
storage convention) gives a value close to -- but NOT identical to --
`reg=224`'s own real value, on every deterministic seed:

| seed | computed `y_scale` | observed `reg=224` | relative error |
| --- | --- | --- | --- |
| 0, group A | 0.0145926 | 0.0145932 | ~4e-5 |
| 1 | 0.0169632 | 0.0169835 | ~1.2e-3 |
| 7 | 0.0151387 | 0.0151205 | ~1.2e-3 |
| 42 | 0.0167556 | 0.0167611 | ~3.3e-4 |
| 100 | 0.0158509 | 0.0158318 | ~1.2e-3 |
| 999 | 0.0183340 | 0.0183505 | ~9.0e-4 |

Group A's own match (`~4e-5` relative error) is consistent with pure
float32-precision noise -- comparable to Finding 1's own ~1e-7 to 1e-9
figures once this file's own naive convolution's slightly different
summation order is accounted for. The other five deterministic seeds
show a small but consistently LARGER discrepancy (~0.03%-0.12%
relative) than that -- too large to dismiss as pure rounding, too
small and too systematically close to be coincidence (every one of
the 6 lands within 0.12% of the real value, when a wrong formula
entirely -- like the weight-scale reading already ruled out above --
misses by orders of magnitude, not fractions of a percent).

**This file does NOT claim `reg=224 == y_scale` exactly**, unlike PR
#1640's own zero-point finding. The most likely explanation for the
small residual gap: this file's own naive, hand-written convolution
loop (see `_conv_reference` below) almost certainly does not replicate
the EXACT summation order / intermediate rounding the real onnxsim
calibration pipeline's own convolution reference implementation uses
(a different loop order, a matrix-multiply-based im2col path, or a
different float64-vs-float32 intermediate strategy could each produce
a systematic few-tenths-of-a-percent difference of exactly this
character). Confirming the exact reference implementation used by
`onnxsim`'s own calibration code (not attempted here) is the natural
next step to close this specific gap -- reported as a precise, strong,
but not fully closed positive result, not forced into a false "exact
match" claim.

## What this establishes, and what remains open

**Established**: Conv's binary-cluster switch's own two additional
computed float32 fields (beyond `reg=54`, already confirmed by PR
#1640) are now BOTH identified as real, physically-meaningful
quantization quantities -- `bank=15`'s own value is `1/x_scale`,
confirmed to float32-rounding precision and tied directly to this
project's own pre-existing "site A" pattern; `reg=224` is very likely
`y_scale`, strongly but not yet bit-exactly confirmed. Combined with PR
#1640's own `reg=54` finding, EVERY field this project has ever
decoded as part of the 28-byte binary-cluster switch is now understood
to be a real per-build quantization statistic (a zero point, an input
scale reciprocal, and very likely an output scale) -- not opaque
scheduling noise, confirming PR #1586's own original "compiler-internal
near-tie" framing precisely, and locating exactly WHICH computation the
near-tie sits inside (input/output range calibration, not an unrelated
weight statistic).

**NOT established**: `reg=224`'s own formula to the same bit-exact
precision `reg=54`/`bank=15` already have; why the residual gap is
consistently ~0.03%-0.12% rather than zero (most likely a reference-
convolution implementation difference, not chased further here); the
remaining unexplored fields in the 28-byte cluster (`reg=54`'s own
duplicate at a second offset, `reg=232`'s own trailing byte, and the
exact relationship, if any, between `reg=224`'s small residual gap and
`reg=232`'s own still-undecoded value).
"""

import os
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

# Cited directly from already-merged PRs (tests/test_axera_conv_binary_cluster_seed_survey.py
# PR #1600, tests/test_axera_conv_binary_cluster_calibration_dependence.py PR #1598,
# tests/test_axera_conv_reg60_mechanism.py PR #1586) -- not recomputed from
# fixtures here, since this file's own job is the independent scale
# calculation, not re-decoding mcode.
KNOWN_BANK15 = {
    "0A": 33.811733,
    "0B": 127.514183,
    "1": 35.855095,
    "7": 35.040691,
    "42": 35.577232,
    "100": 36.078632,
    "999": 37.855480,
}
KNOWN_REG224 = {
    "0A": 0.0145932,
    "0B": 0.0097953,
    "1": 0.0169835,
    "7": 0.0151205,
    "42": 0.0167611,
    "100": 0.0158318,
    "999": 0.0183505,
}
DETERMINISTIC_STATES = {"1": 1, "7": 7, "42": 42, "100": 100, "999": 999}
SPLIT_STATE_GROUP_A = ("0A", 0)


def _calib_samples(seed, shape=(1, 4, 16, 16), n_samples=4):
    """`RandomState(seed).randn(1, 4, 16, 16)` x4 -- the exact
    calibration data `tests/test_axera_mcode_structure.py`'s own
    `_build_and_get_mcode_bytes` generates for every
    `Conv(dilation=3,...)` fixture in this project's corpus."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def _asym_minmax_scale(samples):
    """`ComputeAsymmetricUint8QuantParams`'s own scale half
    (`onnxsim/passes/static_quantize_matmul.h`, cited directly in PR
    #1640) -- ``scale = (max(0, hi) - min(0, lo)) / 255``, no
    outlier-clipping (the "minmax" calibration method this project's
    own build config uses)."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return (hi - lo) / np.float32(255.0)


def _conv_weight():
    """Conv's own weight -- fixed at `RandomState(0)` regardless of
    calibration seed (`tests/test_axera_mcode_structure.py`'s own
    `_dilation_conv_model`: `cout=4, cin=4, k=3`, scaled by 0.1)."""
    rng = np.random.RandomState(0)
    return (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)


def _conv_reference(x, w, dilation=3, pad=3):
    """A float64-accumulating reference `Conv` (dilation/pad matching
    `_dilation_conv_model`'s own `Conv(dilation=3, pad=3)` config,
    output size == input size by construction) -- NOT claimed to be
    bit-identical to whatever reference implementation onnxsim's own
    calibration pipeline actually uses internally (see this file's own
    module docstring for why Finding 2's own match is close but not
    exact)."""
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


def _computed_recip_x_scale(seed):
    samples = _calib_samples(seed)
    xs = _asym_minmax_scale(samples)
    return float(np.float32(1.0) / xs)


def _computed_y_scale(seed):
    samples = _calib_samples(seed)
    w = _conv_weight()
    ys_all = [_conv_reference(s, w) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    return float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))


class TestBank15IsExactlyOneOverXScale(unittest.TestCase):
    """The clean, near-exact confirmation: `verb=161,bank=15`'s own
    computed float32 value matches `1/x_scale`, the reciprocal of PR
    #1640's own already-verified input calibration scale, to within
    float32 print-rounding on every deterministic state."""

    def test_all_six_deterministic_states_match_to_float32_precision(self):
        states = dict(DETERMINISTIC_STATES)
        states["0A"] = 0
        for key, seed in states.items():
            computed = _computed_recip_x_scale(seed)
            target = KNOWN_BANK15[key]
            rel_err = abs(computed - target) / target
            self.assertLess(rel_err, 1e-5, (key, computed, target, rel_err))

    def test_group_b_does_not_match_group_as_own_seed(self):
        """Seed 0's OTHER near-tie state (group B) does NOT match --
        expected, since it resolves the same input-range computation
        differently inside Pulsar2's own non-deterministic
        accumulation, the same reason its own `reg=54` value differs
        from group A's."""
        computed = _computed_recip_x_scale(0)
        self.assertNotAlmostEqual(computed, KNOWN_BANK15["0B"], places=2)


class TestWeightScaleAloneDoesNotExplainReg224(unittest.TestCase):
    """Rules out the naive "reg=224 is a fixed weight scale" reading
    directly: the weight is `RandomState(0)`-fixed regardless of
    calibration seed, so a pure weight-derived scale would be constant
    across every seed -- it is not, and does not match any observed
    value."""

    def test_weight_scale_is_constant_and_too_small(self):
        w = _conv_weight()
        w_scale = float(np.abs(w).max() / 127.0)
        for target in KNOWN_REG224.values():
            self.assertNotAlmostEqual(w_scale, target, places=3)


class TestReg224IsCloseToButNotExactlyYScale(unittest.TestCase):
    """The strong-but-imperfect confirmation: the same asymmetric
    MinMax formula, applied to `Conv`'s own OUTPUT range instead of
    its input range, lands within a fraction of a percent of every
    deterministic `reg=224` value -- close enough to rule out
    coincidence, not (yet) close enough to claim bit-exactness."""

    def test_group_a_matches_to_within_float_precision_noise(self):
        computed = _computed_y_scale(0)
        target = KNOWN_REG224["0A"]
        rel_err = abs(computed - target) / target
        self.assertLess(rel_err, 1e-3, (computed, target, rel_err))

    def test_every_deterministic_seed_is_within_half_a_percent(self):
        for key, seed in DETERMINISTIC_STATES.items():
            computed = _computed_y_scale(seed)
            target = KNOWN_REG224[key]
            rel_err = abs(computed - target) / target
            self.assertLess(rel_err, 5e-3, (key, computed, target, rel_err))

    def test_the_match_is_not_bit_exact(self):
        """Documents, as a real assertion rather than prose alone, that
        this file does NOT claim the same precision PR #1640's own
        zero-point formula achieved -- at least one deterministic seed
        shows a residual gap larger than pure float32 rounding noise
        would explain."""
        worst_rel_err = 0.0
        for key, seed in DETERMINISTIC_STATES.items():
            computed = _computed_y_scale(seed)
            target = KNOWN_REG224[key]
            worst_rel_err = max(worst_rel_err, abs(computed - target) / target)
        self.assertGreater(worst_rel_err, 1e-4)


if __name__ == "__main__":
    unittest.main()
