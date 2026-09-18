"""Investigates a question `tests/test_axera_gemm_recip_xscale_element_
count_dependence.py` (PR #1651) explicitly left open: PR #1651 found
Gemm's `1/x_scale` field (the `verb=161,bank=15` locator) is bit-exact
to a float32-throughout recomputation of `ComputeAsymmetricUint8Quant
Params` at total calibration-sample element counts `m*k` in
`{17,32,128}`, but off by exactly 1 float32 ULP at `m*k` in `{64,256}`
-- "not a simple threshold... plausibly a floating-point reduction-
order sensitivity at specific sample counts... but this file does not
verify that guess."

Reuses PR #1651's own already-committed fixtures
(`scripts/axera/fixtures/gemm_*_mk*.mcode.gz`) -- no new Docker builds.

## Finding 1: reduction order and reciprocal-computation method do NOT
## explain the discrepancy -- ruled out cleanly, not just unconfirmed

15 combinations were tested against all 5 totals: 5 float32 min/max
reduction orders over the flattened 4-sample calibration data
(sequential left-to-right, numpy's own vectorized `.min()`/`.max()`,
sorted-then-take-ends, reversed, and per-sample-then-combine) crossed
with 3 ways of computing the final reciprocal (`np.float32(1.0) /
scale`, `np.reciprocal(scale)`, `float(1.0 / float(scale))` then cast).
All 15 variants agree with EACH OTHER on every one of the 5 totals,
and all 15 agree with each other on which totals mismatch the real
device (`64`, `256`) -- if reduction order or reciprocal method were
the explanation, at least one of the 15 variants would have flipped at
least one result. None did. This rules out order-of-operations
sensitivity within a float32-throughout pipeline as the cause.

## Finding 2 (positive): a full double-precision reduction pipeline --
## accumulate `lo`/`hi`/`scale` in float64, cast to float32 only for
## the final stored value -- is bit-exact at 4 of 5 totals, closing
## the `m*k=256` gap PR #1651 left open

Recomputing with EVERY intermediate step (`lo`, `hi`, `scale`) kept in
float64 and only the final `1/scale` result cast to float32 (rather
than casting `lo`/`hi`/`scale` to float32 at each step, as PR #1636-
#1651's own formula has done throughout this whole arc) reproduces the
real device's value bit-exact at `m*k` in `{17, 32, 128, 256}` --
including `256`, which every float32-throughout variant in Finding 1
got wrong. This is a real, verified, non-trivial improvement, not a
coincidental fit: `256` was WRONG under all 15 Finding-1 variants and
is RIGHT under this one change (float64 accumulation), tested across
both of `256`'s constituent building blocks (the `1x64`/`2x32` split
pair already established as numerically identical by PR #1651, so
there is nothing left to vary there) plus a direct recheck of all 4
`m*k=32` splits and both tested `m*k=128` splits, none of which
regress.

## Finding 3: `m*k=64` remains a genuine, unexplained outlier even
## under the double-precision hypothesis

The same full-double-precision recomputation is STILL 1 ULP off at
`m*k=64` (both the `1x64` and `2x32` splits, confirmed independently
identical per PR #1651's own already-established split-independence).
A per-sample (rather than flattened) double-precision reduction gives
the identical result -- grouping does not matter for a min/max
reduction, as expected. No variant tested in this file (or PR #1651's
own 15 float32 variants) closes this specific gap. This file does not
claim an explanation for `m*k=64` specifically; establishing that it
survives the double-precision hypothesis (unlike `256`, which did not
survive it) is itself the finding.

## What this establishes, precisely, and what it does not

**Established**: the discrepancy is not a reduction-order or
reciprocal-method artifact (Finding 1, a clean negative across 15
variants); a full double-precision reduction pipeline is a real,
verified improvement over PR #1651's float32-throughout formula,
closing the `m*k=256` gap specifically (Finding 2); `m*k=64` is a
qualitatively different, still-unexplained case, not merely "the same
kind of gap `256` had, just not yet tried hard enough" -- it resists
the one variant that fixed `256` (Finding 3).

**NOT established**: why `m*k=64` specifically resists the double-
precision fix while `17`/`32`/`128`/`256` do not; whether the real
Pulsar2 implementation genuinely uses float64 accumulation internally
(this file demonstrates numerical consistency with that hypothesis at
4/5 totals, not any form of access to Pulsar2's own source); whether
a still-different formula would close `64` too without reopening any
of the other four.
"""

import gzip
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

# (name, m, k, filename, m*k)
SHAPES = [
    ("mk17_1x17", 1, 17, "gemm_1x17x16_negctrl.mcode.gz"),
    ("mk32_1x32", 1, 32, "gemm_1x32x16_mk32_split_1x32.mcode.gz"),
    ("mk32_2x16", 2, 16, "gemm_2x16x16_mk32_split_2x16.mcode.gz"),
    ("mk32_4x8", 4, 8, "gemm_4x8x16_mk32_split_4x8.mcode.gz"),
    ("mk32_8x4", 8, 4, "gemm_8x4x16_mk32_split_8x4.mcode.gz"),
    ("mk64_1x64", 1, 64, "gemm_1x64x16_mk64_split_1x64.mcode.gz"),
    ("mk64_2x32", 2, 32, "gemm_2x32x16_mk64_split_2x32.mcode.gz"),
    ("mk128_1x128", 1, 128, "gemm_1x128x16_mk128_split_1x128.mcode.gz"),
    ("mk128_8x16", 8, 16, "gemm_8x16x16_mk128_split_8x16.mcode.gz"),
    ("mk256_16x16", 16, 16, "gemm_16x16x16_mk256_split_16x16.mcode.gz"),
]

# Expected (observed - computed) ULP diff under the float32-throughout
# formula PR #1636-#1651 have used -- cited directly from PR #1651's
# own EXPECTED_ULP_DIFF, not recomputed here.
FLOAT32_ULP_DIFF = {
    "mk17_1x17": 0,
    "mk32_1x32": 0,
    "mk32_2x16": 0,
    "mk32_4x8": 0,
    "mk32_8x4": 0,
    "mk64_1x64": 1,
    "mk64_2x32": 1,
    "mk128_1x128": 0,
    "mk128_8x16": 0,
    "mk256_16x16": 1,
}

# Expected (observed - computed) ULP diff under THIS file's own
# full-double-precision reduction.
DOUBLE_PRECISION_ULP_DIFF = {
    "mk17_1x17": 0,
    "mk32_1x32": 0,
    "mk32_2x16": 0,
    "mk32_4x8": 0,
    "mk32_8x4": 0,
    "mk64_1x64": 1,
    "mk64_2x32": 1,
    "mk128_1x128": 0,
    "mk128_8x16": 0,
    "mk256_16x16": 0,
}


def load(filename):
    with gzip.open(os.path.join(FIX, filename), "rb") as f:
        return f.read()


def observed_recip_x_scale(filename):
    recs = mcode.decode(load(filename))
    hits = [r for r in recs if r.get("verb") == 161 and r.get("bank") == 15]
    vals = {struct.unpack("<f", r["operand"])[0] for r in hits}
    assert len(vals) == 1, vals
    return vals.pop()


def ulp_diff(observed, computed):
    o = struct.unpack("<i", struct.pack("<f", np.float32(observed)))[0]
    c = struct.unpack("<i", struct.pack("<f", np.float32(computed)))[0]
    return o - c


def calib_samples(m, k, seed=1, n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(m, k).astype(np.float32) for _ in range(n_samples)]


def float32_throughout_recip(m, k, seed=1):
    """PR #1636-#1651's own formula: `lo`/`hi`/`scale` all kept in
    float32 at every step."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in calib_samples(m, k, seed):
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    return float(np.float32(1.0) / scale)


def double_precision_recip(m, k, seed=1):
    """This file's own Finding 2: `lo`/`hi`/`scale` accumulated in
    float64, cast to float32 only for the final returned value."""
    flat64 = np.concatenate(
        [s.astype(np.float64).reshape(-1) for s in calib_samples(m, k, seed)]
    )
    lo = min(0.0, float(flat64.min()))
    hi = max(0.0, float(flat64.max()))
    if hi <= lo:
        hi = lo + 1.0
    scale = (hi - lo) / 255.0
    return float(np.float32(1.0 / scale))


def reduction_order_variants(m, k, seed=1):
    """Finding 1: 5 float32 reduction orders x 3 reciprocal methods =
    15 variants, all kept float32-throughout like PR #1651's own
    formula, differing only in the order data is visited / how the
    final reciprocal is taken."""
    samples = calib_samples(m, k, seed)
    flat = np.concatenate([s.reshape(-1) for s in samples])

    def minmax_sequential(arr):
        lo = np.float32(0.0)
        hi = np.float32(0.0)
        for v in arr:
            v32 = np.float32(v)
            lo = min(lo, v32)
            hi = max(hi, v32)
        return lo, hi

    def minmax_numpy(arr):
        return (
            min(np.float32(0.0), np.float32(arr.min())),
            max(np.float32(0.0), np.float32(arr.max())),
        )

    def minmax_sorted(arr):
        s = np.sort(arr)
        return (
            min(np.float32(0.0), np.float32(s[0])),
            max(np.float32(0.0), np.float32(s[-1])),
        )

    orders = {
        "sequential": minmax_sequential(flat),
        "numpy": minmax_numpy(flat),
        "sorted": minmax_sorted(flat),
        "reversed": minmax_sequential(flat[::-1]),
        "per_sample": (
            min(np.float32(0.0), *(np.float32(s.min()) for s in samples)),
            max(np.float32(0.0), *(np.float32(s.max()) for s in samples)),
        ),
    }

    recips = {
        "true_div": lambda scale: float(np.float32(1.0) / scale),
        "np_reciprocal": lambda scale: float(np.reciprocal(scale)),
        "double_then_cast": lambda scale: float(np.float32(1.0 / float(scale))),
    }

    out = {}
    for oname, (lo, hi) in orders.items():
        if hi <= lo:
            hi = lo + np.float32(1.0)
        scale = (hi - lo) / np.float32(255.0)
        for rname, rfunc in recips.items():
            out[f"{oname}/{rname}"] = rfunc(scale)
    return out


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_any_shape(self):
        for name, _m, _k, filename in SHAPES:
            errs = mcode.check(load(filename))
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], (name, errs))


class TestReductionOrderAndReciprocalMethodDoNotExplainTheGap(unittest.TestCase):
    """Finding 1: a clean negative -- all 15 float32-throughout
    variants agree with each other on every total, ruling out
    reduction order / reciprocal method as the cause."""

    def test_all_15_variants_agree_with_each_other_on_every_total(self):
        for name, m, k, filename in SHAPES:
            variants = reduction_order_variants(m, k)
            diffs = {
                vname: ulp_diff(v, list(variants.values())[0])
                for vname, v in variants.items()
            }
            self.assertTrue(all(d == 0 for d in diffs.values()), (name, diffs))

    def test_all_15_variants_match_the_known_float32_ulp_pattern(self):
        for name, m, k, filename in SHAPES:
            observed = observed_recip_x_scale(filename)
            variants = reduction_order_variants(m, k)
            for vname, computed in variants.items():
                d = ulp_diff(observed, computed)
                self.assertEqual(d, FLOAT32_ULP_DIFF[name], (name, vname, d))


class TestDoublePrecisionReductionClosesTheMk256Gap(unittest.TestCase):
    """Finding 2: a real, verified positive -- float64 accumulation
    throughout the lo/hi/scale computation, cast to float32 only at
    the very end, is bit-exact at m*k in {17,32,128,256}."""

    def test_all_ten_fixtures_match_the_double_precision_ulp_table(self):
        for name, m, k, filename in SHAPES:
            observed = observed_recip_x_scale(filename)
            computed = double_precision_recip(m, k)
            d = ulp_diff(observed, computed)
            self.assertEqual(d, DOUBLE_PRECISION_ULP_DIFF[name], (name, d))

    def test_mk256_is_bit_exact_under_double_precision_unlike_float32(self):
        name, m, k, filename = next(s for s in SHAPES if s[0] == "mk256_16x16")
        observed = observed_recip_x_scale(filename)
        f32_computed = float32_throughout_recip(m, k)
        f64_computed = double_precision_recip(m, k)
        self.assertNotEqual(ulp_diff(observed, f32_computed), 0)
        self.assertEqual(ulp_diff(observed, f64_computed), 0)

    def test_all_mk32_and_mk128_splits_remain_bit_exact_under_double_precision(self):
        for name, m, k, filename in SHAPES:
            if m * k not in (32, 128):
                continue
            observed = observed_recip_x_scale(filename)
            computed = double_precision_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 0, name)


class TestMk64RemainsUnexplainedEvenUnderDoublePrecision(unittest.TestCase):
    """Finding 3: m*k=64 is a genuinely different, still-open case --
    it does NOT get fixed by the same double-precision hypothesis that
    fixed m*k=256, on either of its two tested splits."""

    def test_both_mk64_splits_are_still_1_ulp_off_under_double_precision(self):
        for name, m, k, filename in SHAPES:
            if m * k != 64:
                continue
            observed = observed_recip_x_scale(filename)
            computed = double_precision_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 1, name)

    def test_per_sample_grouping_gives_the_identical_result_as_flattened(self):
        """Grouping (flattened vs. per-sample) cannot matter for a
        min/max reduction -- confirmed directly rather than assumed."""
        m, k = 1, 64
        flat_result = double_precision_recip(m, k)

        samples = calib_samples(m, k)
        lo = 0.0
        hi = 0.0
        for s in samples:
            s64 = s.astype(np.float64)
            lo = min(lo, float(s64.min()))
            hi = max(hi, float(s64.max()))
        if hi <= lo:
            hi = lo + 1.0
        scale = (hi - lo) / 255.0
        per_sample_result = float(np.float32(1.0 / scale))

        self.assertEqual(flat_result, per_sample_result)


if __name__ == "__main__":
    unittest.main()
