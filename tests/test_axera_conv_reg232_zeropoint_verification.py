"""Closes the last open row in `tests/test_axera_conv_binary_cluster_
quantization_synthesis.py` (PR #1643)'s own field table: `reg=232`,
the one field in Conv's 28-byte binary-cluster switch that arc left
completely unattempted. Applies the SAME first-principles technique
PR #1640/#1641/#1642 already used for `reg=54`, `verb=161,bank=15`,
and `reg=224` -- port the real, already-documented
`ComputeAsymmetricUint8QuantParams` formula
(`onnxsim/passes/static_quantize_matmul.h`), apply it to the exact
`RandomState(seed)`-derived calibration/weight data each fixture's own
build script used, and compare against the real decoded byte.

## Finding: `reg=232` is EXACTLY the output's own calibrated zero
## point (`zp_y`) -- the missing zero-point counterpart to `reg=224`'s
## own `y_scale`

PR #1642 already established `reg=224` is very likely `y_scale`
(`ComputeAsymmetricUint8QuantParams`'s scale half, applied to `Conv`'s
own output range) -- close but not bit-exact. `ComputeAsymmetricUint8
QuantParams` computes a scale AND a zero point together from the same
`lo`/`hi` range; PR #1642 never computed the zero-point half for the
output side. This file does, using the exact same `zp = round_half_
away_from_zero(-lo/scale)` formula PR #1640 already verified bit-exact
for the INPUT zero point (`reg=54`), just applied to `y = Conv(x, w)`'s
own output range instead of `x`'s input range:

| key | observed `reg=232` | computed `zp_y` | `zp_f` (pre-round) |
| --- | --- | --- | --- |
| 0, group A | 124 | 124 | 124.2020 |
| 0, group B | 118 | 124 (does not match) | 124.2020 |
| 1 | 116 | 116 | 115.6353 |
| 7 | 135 | 135 | 134.7700 |
| 42 | 130 | 130 | 129.7977 |
| 100 | 113 | 113 | 112.8983 |
| 999 | 124 | 124 | 123.8178 |

Bit-exact on every one of the 5 deterministic seeds and on group A --
the same precision PR #1640 found for the input zero point, and
notably TIGHTER than PR #1642's own `reg=224`/`y_scale` match (which
had a consistent small residual gap). This is consistent with zero
points being integers (an 8-bit round, immune to the float32
summation-order noise that makes `y_scale` only approximately
reproducible by this file's own hand-written reference convolution) --
the same reason `reg=54`'s own zero point was bit-exact while
`bank=15`'s float32 value needed a 1e-5 tolerance.

Group B (seed=0's rarer state) does NOT match -- expected, not a
counter-example: this file's own `zp_f` computation for group A and
group B is IDENTICAL (124.2020, since this file's own reference
convolution has no notion of group A/B -- that split lives inside
Pulsar2's own real, non-deterministic accumulation, not in this file's
recomputation), yet the two real decoded builds show DIFFERENT
`reg=232` bytes (124 vs 118). This is the exact same shape of
mismatch PR #1640/#1641 already found and explained for `reg=54`
(group B is `+1`) and `bank=15` (group B is an unrelated fixed
constant): group A and group B are two different resolutions of the
same underlying near-tie inside Pulsar2's own internal calibration
math, which this file's own simplified recomputation cannot
distinguish -- it only recomputes group A's (the common, deterministic)
resolution.

## What this establishes

**Established**: `reg=232` is the output's own calibrated zero point,
bit-exact on every deterministic seed -- closing the last open row in
PR #1643's own field table. Every field in Conv's 28-byte binary-
cluster switch (`reg=54`/`reg=60`, `verb=161,bank=15`, `reg=224`,
`reg=232`) is now accounted for as one of exactly two per-build
statistics -- `x`'s own calibrated zero point and scale, and `y`'s own
calibrated zero point and scale -- computed by the same
`ComputeAsymmetricUint8QuantParams` formula applied twice, once to the
input range and once to the output range.

**NOT established**: this file's own group-B value (why it resolves
to 118 rather than the group-A value of 124, i.e. the internal
mechanics of Pulsar2's own non-deterministic accumulation) -- the same
open item PR #1640/#1641 already left open for their own group-B
mismatches, not newly closed here; whether this exact formula
generalizes to Conv shapes with a different `dilation`, `insz`, or
channel count than the one family this whole arc has ever tested.
"""

import math
import os
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

# Cited directly from already-merged PRs (tests/test_axera_conv_reg224_scale_
# verification.py PR #1642, tests/test_axera_conv_binary_cluster_seed_survey.py
# PR #1600) -- not recomputed from fixtures here, since this file's own job is
# the independent zero-point calculation, not re-decoding mcode.
KNOWN_REG232 = {
    "0A": 124,
    "0B": 118,
    "1": 116,
    "7": 135,
    "42": 130,
    "100": 113,
    "999": 124,
}
DETERMINISTIC_STATES = {"1": 1, "7": 7, "42": 42, "100": 100, "999": 999}
SPLIT_STATE_GROUP_A = ("0A", 0)
SPLIT_STATE_GROUP_B = "0B"


def _calib_samples(seed, shape=(1, 4, 16, 16), n_samples=4):
    """`RandomState(seed).randn(1, 4, 16, 16)` x4 -- the exact
    calibration data `tests/test_axera_mcode_structure.py`'s own
    `_build_and_get_mcode_bytes` generates for every
    `Conv(dilation=3,...)` fixture in this project's corpus."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def _conv_weight():
    """Conv's own weight -- fixed at `RandomState(0)` regardless of
    calibration seed (`tests/test_axera_mcode_structure.py`'s own
    `_dilation_conv_model`: `cout=4, cin=4, k=3`, scaled by 0.1)."""
    rng = np.random.RandomState(0)
    return (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)


def _conv_reference(x, w, dilation=3, pad=3):
    """A float64-accumulating reference `Conv` (dilation/pad matching
    `_dilation_conv_model`'s own `Conv(dilation=3, pad=3)` config) --
    the same reference used by PR #1642's own `y_scale` computation,
    reproduced independently here rather than imported."""
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


def _computed_y_zero_point(seed):
    """`ComputeAsymmetricUint8QuantParams`'s own zero-point half
    (`onnxsim/passes/static_quantize_matmul.h`, cited directly by PR
    #1640), applied to `y = Conv(x, w)`'s own output range instead of
    `x`'s input range -- the same `round_half_away_from_zero` formula
    PR #1640 already verified bit-exact for `reg=54`."""
    samples = _calib_samples(seed)
    w = _conv_weight()
    ys_all = [_conv_reference(s, w) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))
    zp_f = float(np.float32(-np.float32(ylo) / np.float32(scale)))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), zp_f


class TestReg232IsExactlyTheOutputZeroPoint(unittest.TestCase):
    """The clean, bit-exact confirmation: `reg=232`'s own trailing
    byte matches `zp_y`, the zero-point half of `ComputeAsymmetricUint8
    QuantParams` applied to `Conv`'s own output range, on every
    deterministic seed -- the same precision PR #1640 found for the
    input-side zero point, `reg=54`."""

    def test_all_five_deterministic_seeds_match_exactly(self):
        for key, seed in DETERMINISTIC_STATES.items():
            zp, zp_f = _computed_y_zero_point(seed)
            self.assertEqual(zp, KNOWN_REG232[key], (key, zp_f))

    def test_group_a_matches_exactly(self):
        _, seed = SPLIT_STATE_GROUP_A
        zp, zp_f = _computed_y_zero_point(seed)
        self.assertEqual(zp, KNOWN_REG232["0A"], zp_f)

    def test_group_b_does_not_match(self):
        """Seed 0's OTHER near-tie state (group B) does NOT match --
        expected, not a counter-example: this file's own recomputation
        has no notion of group A/B (that split lives inside Pulsar2's
        own real, non-deterministic accumulation), so it can only
        reproduce group A's resolution, the same limitation PR #1640/
        #1641 already documented for their own group-B mismatches."""
        _, seed = SPLIT_STATE_GROUP_A
        zp, _ = _computed_y_zero_point(seed)
        self.assertNotEqual(zp, KNOWN_REG232["0B"])


class TestReg232IsNotADuplicateOfAnyOtherClusterField(unittest.TestCase):
    """Directly re-confirms PR #1643's own `TestReg232RemainsUndecoded
    ByThisArc` negative finding still holds even now that a positive
    formula has been found -- `reg=232`'s own byte is a genuinely
    different computed quantity (`zp_y`), not a byte-level duplicate of
    `reg=54` (`zp_x`), despite both being small integers in a similar
    range."""

    def test_reg232_differs_from_reg54_on_every_deterministic_seed(self):
        # Cited directly from tests/test_axera_conv_reg54_zeropoint_
        # verification.py (PR #1640) -- not recomputed here, since this
        # file's own job is reg=232's formula, not re-verifying reg=54.
        known_reg54 = {"1": 113, "7": 124, "42": 115, "100": 116, "999": 133}
        for key in DETERMINISTIC_STATES:
            self.assertNotEqual(KNOWN_REG232[key], known_reg54[key], key)


if __name__ == "__main__":
    unittest.main()
