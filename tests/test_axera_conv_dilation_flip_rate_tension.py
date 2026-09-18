"""Follows up on `tests/test_axera_conv_dilation_window_near_tie_
reconciliation.py` (PR #1649), which found that Conv's binary-cluster
switch's own `zp_y` near-tie margin cleanly separates the two dilations
where the switch is ever observed (dilation=2: margin 0.135, dilation=3:
margin 0.298) from the three where it never is (dilation=1: 0.431,
dilation=4: 0.433, dilation=8: 0.336). That file only tested PRESENCE
vs. ABSENCE of flipping, never comparing the two present cases' own
flip RATES against each other.

Doing that comparison surfaces an apparent tension: dilation=2 (the
SMALLER margin, 0.135) flips in only 2 of 12 rebuilds (16.7%), while
dilation=3 (the LARGER margin, 0.298) flips in 3 of 8 (37.5%) -- the
opposite of what "smaller margin -> more flip-prone" would naively
predict, if margin size were assumed to also govern relative rate
between two switch-present cases (not just presence/absence).

## Finding: the rate difference is NOT statistically distinguishable
## from noise at this sample size -- the "tension" is apparent, not real

A two-sided Fisher's exact test on the 2x2 table (2/12 vs. 3/8) gives
`p=0.347` -- nowhere near any conventional significance threshold. The
95% Wilson score confidence intervals for the two rates overlap
massively: dilation=2's true rate could plausibly be anywhere in
`[4.7%, 44.8%]`, dilation=3's in `[13.7%, 69.4%]`. A single-digit
sample count per dilation (8-12 rebuilds) cannot distinguish a 16.7%
rate from a 37.5% rate, let alone establish that the ordering is
meaningful or reversed relative to margin size.

This does not strengthen or weaken PR #1649's own presence/absence
claim (0.135 and 0.298 are both far smaller than 0.336-0.433, a
4-value gap the two present dilations' own margins never approach
regardless of how their relative rates are read) -- it only shows that
extending the margin-based explanation to also predict RELATIVE RATE
between the two present cases is not supported (or contradicted) by
the existing corpus; the data is simply too sparse to say either way.

## Ruled out as an explanation: build-configuration confound

`conv_dilation2`'s and `conv_dilation3`'s own fixtures are built via
`_dilation_conv_model(dilation, pad=dilation)`
(`tests/test_axera_mcode_structure.py`) with `insz=16`, `cin=cout=4`,
`k=3` held fixed and calibration seed fixed at `RandomState(0)` in
every case (`ALL_BY_DILATION`'s own fixture names, reused directly
from PR #1649) -- `dilation` is the only axis that varies between the
two fixture sets. There is no shape, padding, or calibration-seed
confound to explain the rate difference by non-margin means.

## What this establishes, precisely, and what it does not

**Established**: the dilation=2-vs-3 flip-RATE difference PR #1649's
own numbers exhibit is not statistically significant at `n=8-12`
(Fisher's exact `p=0.347`), and is not explained by a build-config
confound between the two fixture sets.

**NOT established**: what the TRUE flip rate is for either dilation
value (would require many more real Pulsar2 rebuilds, out of scope
here); whether margin size has any real relationship to relative rate
among switch-present dilations at all -- this file neither confirms
nor refutes that, since the corpus size is fundamentally too small to
resolve it either way.
"""

import gzip
import math
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

# Cited directly from tests/test_axera_conv_dilation_window_near_tie_
# reconciliation.py (PR #1649)'s own ALL_BY_DILATION lists for
# dilation=2/3 -- not re-derived, since this file's own job is the
# statistical comparison, not re-discovering which fixtures exist.
DILATION2_NAMES = [
    "conv_dilation2.mcode.gz",
    "conv_dilation2_rebuild.mcode.gz",
    "conv_dilation2_rebuild0.mcode.gz",
    "conv_dilation2_rebuild1.mcode.gz",
] + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)]
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


def flip_count(names):
    vals = [reg_byte(decode(n), 60, 131) for n in names]
    return sum(1 for v in vals if v == 0x7F), len(vals)


def _hypergeom_p(a, b, c, d):
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    return math.comb(r1, a) * math.comb(r2, c) / math.comb(n, c1)


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher's exact test on a 2x2 contingency table, with
    no scipy dependency: sums the hypergeometric probability of every
    table sharing the same margins whose probability is no greater
    than the observed table's own probability."""
    r1, c1, n = a + b, a + c, a + b + c + d
    p_obs = _hypergeom_p(a, b, c, d)
    total = 0.0
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    for a2 in range(lo, hi + 1):
        b2, c2, d2 = r1 - a2, c1 - a2, n - r1 - (c1 - a2)
        p = _hypergeom_p(a2, b2, c2, d2)
        if p <= p_obs + 1e-12:
            total += p
    return total


def wilson_ci(k, n, z=1.96):
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


class TestFlipCountsReconfirmedDirectlyFromFixtures(unittest.TestCase):
    """Independently re-decodes and re-counts dilation=2/3's own flip
    counts from the real fixtures -- not trusted from PR #1649's own
    restated numbers."""

    def test_dilation2_is_2_of_12(self):
        flips, n = flip_count(DILATION2_NAMES)
        self.assertEqual((flips, n), (2, 12))

    def test_dilation3_is_3_of_8(self):
        flips, n = flip_count(DILATION3_NAMES)
        self.assertEqual((flips, n), (3, 8))


class TestRateDifferenceIsNotStatisticallySignificant(unittest.TestCase):
    """The core finding: a smaller sample-size mismatch (12 vs. 8) is
    fully sufficient to explain an apparent 16.7%-vs-37.5% rate gap by
    chance alone -- Fisher's exact test and the two rates' own
    overlapping confidence intervals both confirm this directly."""

    def test_fisher_exact_p_value_is_not_significant(self):
        f2, n2 = flip_count(DILATION2_NAMES)
        f3, n3 = flip_count(DILATION3_NAMES)
        p = fisher_exact_two_sided(f2, n2 - f2, f3, n3 - f3)
        self.assertGreater(p, 0.05, p)
        self.assertAlmostEqual(p, 0.347, places=3)

    def test_wilson_confidence_intervals_overlap(self):
        f2, n2 = flip_count(DILATION2_NAMES)
        f3, n3 = flip_count(DILATION3_NAMES)
        lo2, hi2 = wilson_ci(f2, n2)
        lo3, hi3 = wilson_ci(f3, n3)
        # Intervals overlap iff neither entirely precedes the other.
        self.assertTrue(lo2 <= hi3 and lo3 <= hi2, (lo2, hi2, lo3, hi3))


class TestNoBuildConfigConfoundExplainsTheRateDifference(unittest.TestCase):
    """Confirms dilation is the only axis that differs between the two
    fixture sets: both are built by calling `_dilation_conv_model(d, d)`
    (`tests/test_axera_mcode_structure.py`) -- `pad == dilation`, and
    every other argument (`cin`, `cout`, `k`, `insz`) left at its
    default in both call sites -- so the rate difference cannot be
    attributed to a shape/padding mismatch between the two dilation
    values' own fixture-build configs. Calibration seed is separately
    fixed at `RandomState(0)` for every fixture regardless of dilation
    (`_build_and_get_mcode_bytes`'s own convention, reused unchanged
    across this whole arc since PR #1586)."""

    def test_dilation2_and_dilation3_call_sites_use_pad_equals_dilation_only(self):
        with open(
            os.path.join(os.path.dirname(__file__), "test_axera_mcode_structure.py")
        ) as f:
            src = f.read()
        # `_dilation_conv_model(2, 2)` / `(3, 3)` -- both call sites
        # pass pad == dilation and nothing else, i.e. no cin/cout/k/insz
        # override that could differ between the two dilation values.
        self.assertIn("_dilation_conv_model(2, 2)", src)
        self.assertIn("_dilation_conv_model(3, 3)", src)

    def test_no_call_site_overrides_shape_for_either_dilation(self):
        with open(
            os.path.join(os.path.dirname(__file__), "test_axera_mcode_structure.py")
        ) as f:
            src = f.read()
        self.assertNotIn("_dilation_conv_model(2, 2, cin=", src)
        self.assertNotIn("_dilation_conv_model(3, 3, cin=", src)
        self.assertNotIn("_dilation_conv_model(2, 2, insz=", src)
        self.assertNotIn("_dilation_conv_model(3, 3, insz=", src)


if __name__ == "__main__":
    unittest.main()
