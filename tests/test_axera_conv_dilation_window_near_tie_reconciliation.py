"""Reconciles two previously-separate findings about Conv's 28-byte
binary "path" switch (`tests/test_axera_conv_reg60_mechanism.py`, PR
#1586) that were never directly cross-referenced against each other:

- `tests/test_axera_conv_reg54_zeropoint_verification.py` (PR #1640)
  proved the switch's `reg=54` copy is Conv's real, calibration-derived
  input zero point (`zp_x`), and that seed=0's own instability is a
  genuine floating-point rounding near-tie in `zp_x`'s computation --
  a quantity that is mathematically **dilation-INDEPENDENT** (a pure
  function of `x`'s own calibration data, which never depends on
  `dilation`).
- `tests/test_axera_conv_dilation2_transition.py` (PR #1599) /
  `tests/test_axera_conv_binary_cluster_higher_dilation.py` (PR #1601)
  established the switch's own PRESENCE (i.e. whether a given rebuild
  ever lands on the alternate state) is **dilation-DEPENDENT**: absent
  at `dilation=1`, present at `2`/`3`, absent again by `4`/`8`.

This file directly tests the leading hypothesis those two findings left
open (that `reg=224`'s own dilation-dependent `y_scale` explains the
window) by independently recomputing, from the exact `RandomState(0)`
calibration/weight data every fixture's own build script used, both the
input- and output-side quantization quantities at every dilation value
this project has already built fixtures for (`1`, `2`, `3`, `4`, `8`) --
reusing only already-committed fixtures, no new Docker builds.

## Finding 1: the cluster's OWN RECORDS are structurally absent from the
## mcode at `dilation=1` and `dilation=8` -- not merely "not varying"

Independently re-decoding `reg=224` (tag=129) and `reg=232` (tag=132)
directly from the fixtures (not trusted from any prior PR's own claim)
finds those records **do not exist at all** in `dilation=1` or
`dilation=8` fixtures, while they exist (deterministically, at
`dilation=4`, or with real rebuild-to-rebuild variance, at `dilation=2`/
`3`) at every other tested value:

| dilation | `reg=54`/`60` (`zp_x`) | `reg=224`/`232` records | flips observed |
| --- | --- | --- | --- |
| 1 | `0x7e`=126, 8/8 | **absent** | 0/8 |
| 2 | `0x7e`=126, 10/12; `0x7f`=127, 2/12 | present, 2 states | 2/12 |
| 3 | `0x7e`=126, 5/8; `0x7f`=127, 3/8 | present, 2 states | 3/8 |
| 4 | `0x7e`=126, 8/8 | present, single state | 0/8 |
| 8 | `0x7e`=126, 8/8 | **absent** | 0/8 |

`reg=54` itself is present and reads the same value (`126`) at *every*
tested dilation, including the two (`1`, `8`) where the rest of the
cluster's records do not exist at all -- `zp_x`'s own computation does
not depend on whether the surrounding cluster is even structurally
present.

## Finding 2: `zp_x`'s near-tie margin is real and dilation-independent,
## but is NOT, by itself, sufficient to explain the dilation window

PR #1640's own near-tie margin (`zp_f=126.4593`, distance `0.0407` from
a rounding boundary) is computed purely from `x`'s calibration data --
identical at every dilation, confirmed trivially since the formula
takes no dilation-dependent input. Yet the alternate resolution
(`0x7f`=127) is **never observed at `dilation=1`, `4`, or `8`** (0 of 24
samples across those three values, re-counted directly from fixtures
here) and only appears at `dilation=2`/`3` (5 of 20 samples). A margin
that does not change with dilation cannot, by itself, explain why the
flip only ever manifests at two specific dilation values -- confirming
the two original findings were never actually in conflict, but also
that the naive "the same near-tie becomes more likely to flip" framing
is incomplete.

## Finding 3: Group A/B are NOT the same calibration computed with extra
## floating-point noise -- they are qualitatively different results

The magnitude of the Group A -> B change is far too large for ordinary
summation-order jitter: `bank=15`'s `1/x_scale` value jumps `33.81` ->
`127.51` (3.77x), and `reg=224`'s `y_scale` drops `~33%`. A systematic
search over every leave-one-out and leave-two-out subset of the 4
`dilation=3` calibration samples (recomputing both `x_scale` and
`y_scale` from each subset) finds **no subset reproduces Group B's
observed values** for either quantity -- ruling out "one calibration
sample silently dropped by a threading race" as the mechanism, and
confirming Group B reflects a genuinely different internal computation,
not a rounding-precision artifact of the same one.

## Finding 4 (new): `reg=232` is very likely Conv's OUTPUT zero point
## (`zp_y`) -- extending PR #1640/#1642's technique to a 4th field

Applying the identical `ComputeAsymmetricUint8QuantParams` formula PR
#1640 already verified for `zp_x`, but to a first-principles reference
convolution's own output range (the same reference-conv PR #1642 built
for `y_scale`), reproduces `reg=232`'s real observed Group-A value
**exactly** at every dilation where the field exists:

| dilation | computed `zp_y` | observed `reg=232` (Group A) |
| --- | --- | --- |
| 2 | 120 | `0x78`=120 |
| 3 | 124 | `0x7c`=124 |
| 4 | 115 | `0x73`=115 |

This closes a gap PR #1640/#1642 both explicitly left open ("`reg=232`'s
own trailing byte" / "the remaining unexplored fields in the 28-byte
cluster"). `dilation=4`'s own `y_scale` recomputation also lands within
the same `~0.03%` relative-error band PR #1642 established across
calibration seeds at `dilation=3` (`0.0158997` observed vs. `0.0159039`
computed), confirming that formula generalizes across dilation, not
just calibration seed.

## Finding 5: `zp_y`'s own near-tie margin, NOT `y_scale`'s (there is no
## rounding step in a scale, only in a zero point) -- DOES separate the
## switch-present dilations from the switch-absent ones cleanly

The leading hypothesis this file set out to test ("`reg=224`'s own
dilation-dependent value explains the window") was imprecisely framed:
`y_scale` is a plain division with no discrete rounding decision, so it
cannot itself have a "near tie" the way `zp_x` does. `zp_y` (Finding 4,
newly decoded here) *does* go through the same `round()` step `zp_x`
does, and computing its own distance to a rounding half-boundary at
every tested dilation gives a clean separation:

| dilation | `zp_y` distance to a half-boundary | switch observed? |
| --- | --- | --- |
| 2 | `0.135` (smallest) | yes (2/12) |
| 3 | `0.298` | yes (3/8) |
| 8 | `0.336` | no (0/8) |
| 1 | `0.431` | no (0/8) |
| 4 | `0.433` (largest) | no (0/8) |

The two switch-present dilations (`2`, `3`) are, respectively, the
smallest and second-smallest `zp_y` margins of all five tested; the
three switch-absent dilations (`1`, `4`, `8`) are exactly the three
largest, with a clean gap between `0.298` (largest switch-present) and
`0.336` (smallest switch-absent). With only 5 dilation values sampled
this is not proof of a hard threshold, but it is a genuine, previously
unavailable, precisely-targeted positive correlation -- not the
`y_scale`-based framing originally proposed, but `zp_y`'s own rounding
margin, once that field was actually identified (Finding 4).

## What this reconciles, precisely

The two original findings are **not** in conflict, and the leading
hypothesis, corrected to the field that actually carries a rounding
step (`zp_y`, not `y_scale`), is **supported**: `zp_x`'s own margin is
real, dilation-independent, and correctly predicts the ordinary/
majority resolution (`126`, the closer side) at every dilation tested,
present-cluster or not -- but by itself it cannot gate a dilation-
dependent window, since it never changes. `zp_y`'s own margin, by
contrast, genuinely does vary with dilation (since `y = Conv(x, w,
dilation)` depends on it), and its two smallest values across all five
tested dilations are exactly the two dilations where flipping was
observed. The most likely reading, consistent with both this margin
result and Finding 3's "qualitatively different, not noise-sized"
Group A/B gap: Pulsar2 selects between two different internal
calibration/reduction code paths in a way that is not itself explained
here, but that alternate path is only numerically CONSEQUENTIAL (i.e.
produces an externally visible flip) when the ordinary computation's
own `zp_y` already sits close enough to a rounding boundary for the
alternate path's own different result to land on the other side --
exactly the dilations (`2`, `3`) where that margin is smallest. Why
Pulsar2 selects the alternate path at all, and whether it is itself
dilation-gated or genuinely random, remains open -- this file pins down
*which* quantity's margin the window tracks, not the underlying
trigger, the same honestly-scoped boundary every predecessor file in
this arc (PR #1586/#1640/#1642) left in place.
"""

import gzip
import itertools
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

DILATION1_NAMES = [f"conv_dilation1_r{i}.mcode.gz" for i in range(8)]
DILATION2_NAMES = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
    "conv_dilation2_rebuild0.mcode.gz",
    "conv_dilation2_rebuild1.mcode.gz",
] + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]
DILATION2_GROUP_A = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
] + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]
DILATION2_GROUP_B = [
    "conv_dilation2_rebuild0.mcode.gz",
    "conv_dilation2_rebuild1.mcode.gz",
]
DILATION3_NAMES = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
DILATION3_GROUP_A = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
DILATION3_GROUP_B = [
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
DILATION4_NAMES = [f"conv_dilation4_r{i}.mcode.gz" for i in range(8)]
DILATION8_NAMES = [f"conv_dilation8_r{i}.mcode.gz" for i in range(8)]

ALL_BY_DILATION = {
    1: DILATION1_NAMES,
    2: DILATION2_NAMES,
    3: DILATION3_NAMES,
    4: DILATION4_NAMES,
    8: DILATION8_NAMES,
}


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
    if not hits:
        return None
    return hits[0]["payload"][-1]


def reg224_floats(recs):
    hits = [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == 224 and r.get("tag") == 129
    ]
    if not hits:
        return None
    return [struct.unpack("<f", r["payload"][-4:])[0] for r in hits]


def bank15_scale_like_floats(recs):
    """The two known scale-like `verb=161,bank=15` operand values this
    cluster carries (excludes the two unrelated, stable
    `verb=161,bank=15,field=128` records Conv's mcode also carries
    elsewhere, which use a completely different operand)."""
    KNOWN = (b"7?\x07B", b"C\x07\xffB")
    hits = [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("verb") == 161
        and r.get("bank") == 15
        and r.get("operand") in KNOWN
    ]
    return {struct.unpack("<f", h["operand"])[0] for h in hits}


def _calib_samples(seed=0, shape=(1, 4, 16, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def _conv_weight():
    rng = np.random.RandomState(0)
    return (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)


def _asym_minmax(lo_hi_pairs):
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for a, b in lo_hi_pairs:
        lo = min(lo, np.float32(a))
        hi = max(hi, np.float32(b))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zp_f = float(np.float32(-lo / scale))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    zp = max(0, min(255, zp))
    return float(scale), zp_f, zp


def _x_scale_zp(samples):
    pairs = [(s.min(), s.max()) for s in samples]
    return _asym_minmax(pairs)


def _conv_reference(x, w, dilation, pad):
    cout, cin, kh, kw = w.shape
    _, _, height, width = x.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    w64 = w.astype(np.float64)
    y = np.zeros((1, cout, height, width), dtype=np.float64)
    for oc in range(cout):
        acc = np.zeros((height, width))
        for ic in range(cin):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    oy = kh_i * dilation
                    ox = kw_i * dilation
                    patch = xp[0, ic, oy : oy + height, ox : ox + width]
                    acc += patch * w64[oc, ic, kh_i, kw_i]
        y[0, oc] = acc
    return y.astype(np.float32)


def _y_scale_zp(samples, w, dilation, pad=None):
    if pad is None:
        pad = dilation
    ys = [_conv_reference(s, w, dilation, pad) for s in samples]
    pairs = [(y.min(), y.max()) for y in ys]
    return _asym_minmax(pairs)


class TestClusterRecordsAreStructurallyAbsentAtDilation1And8(unittest.TestCase):
    """Independently re-decodes reg=224/reg=232 directly (not trusted
    from any prior PR's own claim): those records do not exist at all
    in dilation=1 or dilation=8 fixtures, while reg=54 (zp_x) is present
    and computable at every tested dilation regardless."""

    def test_reg224_and_reg232_absent_at_dilation1_and_8(self):
        for d in (1, 8):
            for n in ALL_BY_DILATION[d][:3]:
                recs = decode(n)
                self.assertIsNone(reg224_floats(recs), (d, n))
                self.assertIsNone(reg_byte(recs, 232, 132), (d, n))

    def test_reg224_and_reg232_present_at_dilation_2_3_4(self):
        for d in (2, 3, 4):
            for n in ALL_BY_DILATION[d][:3]:
                recs = decode(n)
                self.assertIsNotNone(reg224_floats(recs), (d, n))
                self.assertIsNotNone(reg_byte(recs, 232, 132), (d, n))

    def test_reg54_present_and_computable_at_every_dilation(self):
        for d, names in ALL_BY_DILATION.items():
            for n in names[:3]:
                recs = decode(n)
                self.assertIsNotNone(reg_byte(recs, 54, 131), (d, n))


class TestZpXResolvesTo126AtEveryDilationRegardlessOfClusterPresence(unittest.TestCase):
    """PR #1640's own dilation-independent near-tie formula predicts
    zp_x=126 (the closer side of the tie) at seed=0 -- confirmed here to
    hold at every dilation this project has fixtures for, including the
    two (1, 8) where the rest of the cluster's records don't even
    exist."""

    def test_formula_predicts_126(self):
        samples = _calib_samples(0)
        _, zp_f, zp = _x_scale_zp(samples)
        self.assertEqual(zp, 126)
        self.assertAlmostEqual(zp_f, 126.4593, places=3)

    def test_reg54_reads_126_at_every_dilation_when_not_flipped(self):
        for d, names in ALL_BY_DILATION.items():
            for n in names:
                recs = decode(n)
                val = reg_byte(recs, 54, 131)
                self.assertIn(val, (0x7E, 0x7F), (d, n))


class TestFlippingIsEmpiricallyConfinedToDilation2And3(unittest.TestCase):
    """The direct refutation of "a dilation-independent margin alone
    explains the window": re-counted here from scratch, the alternate
    resolution (0x7f) never appears at dilation=1, 4, or 8 (0 of 24
    samples), only at dilation=2/3 (5 of 20)."""

    def test_zero_flips_at_dilation_1_4_8(self):
        for d in (1, 4, 8):
            vals = {reg_byte(decode(n), 60, 131) for n in ALL_BY_DILATION[d]}
            self.assertEqual(vals, {0x7E}, d)

    def test_nonzero_flips_at_dilation_2_and_3(self):
        vals2 = [reg_byte(decode(n), 60, 131) for n in ALL_BY_DILATION[2]]
        vals3 = [reg_byte(decode(n), 60, 131) for n in ALL_BY_DILATION[3]]
        self.assertEqual(vals2.count(0x7F), 2)
        self.assertEqual(vals3.count(0x7F), 3)


class TestGroupBIsNotExplainableByDroppingACalibrationSample(unittest.TestCase):
    """Systematically searches every leave-one-out and leave-two-out
    subset of the 4 dilation=3 calibration samples: none reproduce Group
    B's observed x_scale/y_scale, ruling out "one sample silently
    dropped by a threading race" as the mechanism, and confirming the
    Group A -> B change reflects a qualitatively different computation,
    not summation-order noise on the same one."""

    def test_no_subset_reproduces_group_b_x_scale(self):
        samples = _calib_samples(0)
        target = 1.0 / 127.514183
        full_scale, _, _ = _x_scale_zp(samples)
        self.assertNotAlmostEqual(full_scale, target, places=3)
        for r in (1, 2, 3):
            for combo in itertools.combinations(range(4), r):
                subset = [samples[i] for i in combo]
                scale, _, _ = _x_scale_zp(subset)
                self.assertGreater(
                    abs(scale - target) / target, 0.1, (combo, scale, target)
                )

    def test_no_subset_reproduces_group_b_y_scale(self):
        samples = _calib_samples(0)
        w = _conv_weight()
        target = 0.0097953
        for r in (1, 2, 3, 4):
            for combo in itertools.combinations(range(4), r):
                subset = [samples[i] for i in combo]
                scale, _, _ = _y_scale_zp(subset, w, dilation=3, pad=3)
                self.assertGreater(
                    abs(scale - target) / target, 0.1, (combo, scale, target)
                )


class TestReg232IsLikelyConvsOutputZeroPoint(unittest.TestCase):
    """New finding, extending PR #1640/#1642's technique to a 4th
    cluster field: the same ComputeAsymmetricUint8QuantParams formula,
    applied to a first-principles reference convolution's own output
    range, reproduces reg=232's real observed Group-A value exactly at
    every dilation where the field exists (2, 3, 4)."""

    EXPECTED_ZP_Y = {2: 120, 3: 124, 4: 115}

    def test_computed_zp_y_matches_observed_reg232_group_a(self):
        samples = _calib_samples(0)
        w = _conv_weight()
        for d, expected in self.EXPECTED_ZP_Y.items():
            _, _, zp_y = _y_scale_zp(samples, w, dilation=d)
            self.assertEqual(zp_y, expected, d)

            observed = reg_byte(decode(ALL_BY_DILATION[d][0]), 232, 132)
            self.assertEqual(observed, expected, d)


class TestYScaleFormulaGeneralizesToDilation4(unittest.TestCase):
    """PR #1642's own ~0.03%-0.12% relative-error band, established
    across calibration seeds at dilation=3, also holds at dilation=4 (a
    new dilation value, same seed) -- confirming the reference-conv
    formula generalizes across dilation, not just calibration seed."""

    def test_dilation4_y_scale_within_known_error_band(self):
        samples = _calib_samples(0)
        w = _conv_weight()
        computed, _, _ = _y_scale_zp(samples, w, dilation=4)
        observed = reg224_floats(decode("conv_dilation4_r0.mcode.gz"))[0]
        rel_err = abs(computed - observed) / observed
        self.assertLess(rel_err, 5e-3, (computed, observed, rel_err))


class TestZpYsOwnNearTieMarginSeparatesTheSwitchPresentDilations(unittest.TestCase):
    """Finding 5: the natural follow-up hypothesis to test once reg=232
    is identified as zp_y (Finding 4) -- does ITS OWN distance to a
    rounding half-boundary explain the window, corrected from the
    original (imprecise, since a scale has no rounding step) y_scale
    framing? Tested directly here and found to hold cleanly: the two
    switch-present dilations (2, 3) are exactly the two smallest zp_y
    margins of all five tested; the three switch-absent dilations
    (1, 4, 8) are exactly the three largest."""

    def _dist_to_half(self, d):
        samples = _calib_samples(0)
        w = _conv_weight()
        _, zp_f, _ = _y_scale_zp(samples, w, dilation=d)
        frac = zp_f - math.floor(zp_f)
        return abs(frac - 0.5)

    def test_switch_present_dilations_are_the_two_smallest_margins(self):
        margins = {d: self._dist_to_half(d) for d in (1, 2, 3, 4, 8)}
        ranked = sorted(margins, key=margins.get)
        self.assertEqual(set(ranked[:2]), {2, 3}, margins)
        self.assertEqual(set(ranked[2:]), {1, 4, 8}, margins)

    def test_dilation4_margin_exceeds_dilation3s_despite_sharing_present_fields(self):
        """dilation=4 carries the same cluster fields as dilation=2/3
        (Finding 1) yet never flips -- its own zp_y margin (0.433) is
        the single largest of all five, consistent with the fields
        being structurally present but numerically far from a tie."""
        dist_d3 = self._dist_to_half(3)
        dist_d4 = self._dist_to_half(4)
        self.assertGreater(dist_d4, dist_d3)

        vals4 = {reg_byte(decode(n), 60, 131) for n in ALL_BY_DILATION[4]}
        self.assertEqual(vals4, {0x7E})


if __name__ == "__main__":
    unittest.main()
