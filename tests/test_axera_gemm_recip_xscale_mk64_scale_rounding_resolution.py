"""Closes the one gap `tests/test_axera_gemm_recip_xscale_ulp_
investigation.py` (PR #1653) explicitly left open: PR #1653 found that
a full double-precision reduction pipeline (`lo`/`hi`/`scale`/reciprocal
all in float64, cast to float32 only for the final returned value)
closes the `m*k=256` gap in Gemm's `1/x_scale` field, but `m*k=64`
(both its `1x64` and `2x32` splits) remained a genuine 1-ULP outlier
even under that fix -- "why `m*k=64` specifically resists the
double-precision fix while 17/32/128/256 do not" was left open.

Reuses PR #1651/#1653's own already-committed fixtures
(`scripts/axera/fixtures/gemm_*_mk*.mcode.gz`) -- no new Docker builds.

## Finding: the deciding factor is WHERE the float32 rounding happens,
## not whether double precision is used at all

PR #1653's own `double_precision_recip` keeps `lo`, `hi`, `scale`, AND
the reciprocal `1/scale` all in float64, and casts only the final
returned value to float32. This file tests a different split of the
same computation: compute `lo`, `hi`, and `scale = (hi-lo)/255.0` in
float64 as before, but round `scale` itself to float32 ONE time, then
take the reciprocal as an ordinary float32 division (`float32(1.0) /
scale32`) -- i.e. two float32-precision quantities (`1.0f` and
`scale`) divided by a single float32 operation, rather than a
double-precision division rounded to float32 afterward.

This single change (call it "round-scale-once") is bit-exact on ALL 10
of PR #1651/#1653's own tested fixtures, including BOTH cases that
previously failed:

| `m*k` | float32-throughout (PR #1651) | double-throughout (PR #1653) | round-scale-once (this file) |
| --- | --- | --- | --- |
| 17 | exact | exact | exact |
| 32 (x4 splits) | exact | exact | exact |
| 64 (x2 splits) | **1 ULP off** | **1 ULP off** | **exact** |
| 128 (x2 splits) | exact | exact | exact |
| 256 | **1 ULP off** | exact | exact |

## Isolating exactly which step matters: the `hi - lo` subtraction,
## not the division

A further split test (`variantA`: `hi`/`lo` computed as ordinary
float32 values, their DIFFERENCE taken in float32, THEN promoted to
double only for the division by 255, rounding once) reproduces the
SAME 1-ULP failures at `m*k` in `{64, 256}` that PR #1651/#1653's own
formulas showed -- confirming the subtraction step specifically (not
the division step) is what needs double precision. Only when `hi-lo`
itself is computed with no intermediate float32 rounding (both operands
promoted to float64 before subtracting) does the whole chain become
bit-exact everywhere tested.

This is consistent with an ordinary C++ implementation where `hi`/`lo`
are `float` but get promoted to `double` by the usual arithmetic
conversions in an expression like `double scale = ((double)hi -
(double)lo) / 255.0;` followed by an explicit assignment to a `float
scale` variable (rounding once) and a separate `float recip = 1.0f /
scale;` -- rather than an all-`float` expression (PR #1651's original
formula) or an all-`double` expression including the reciprocal itself
(PR #1653's `double_precision_recip`). This file does not have access
to Pulsar2's actual source and does not claim to have found the literal
C++ expression, only that this specific rounding-point choice is the
one that reproduces the real device's bytes exactly across the full
tested range.

## What this establishes, precisely, and what it does not

**Established**: a formula (`scale` computed via a double-precision
`hi-lo` subtraction and division, rounded to float32 exactly once,
followed by a float32 reciprocal) is bit-exact on all 10 already-tested
Gemm `1/x_scale` fixtures spanning `m*k` in `{17,32,64,128,256}` --
closing PR #1653's own last-open `m*k=64` gap without reopening
`m*k=256` or regressing any of the 17/32/128 cases that were already
exact under simpler formulas. Isolated specifically to the `hi-lo`
subtraction step (not the final division) via a direct A/B comparison.

**NOT established**: whether this is literally Pulsar2's own internal
C++ expression (this file demonstrates numerical consistency with one
plausible compilation of `(hi-lo)/255.0` under normal C++ float-to-
double promotion rules, not source-level access); whether the same
"round scale once, not the reciprocal" formula also explains Conv's own
`1/x_scale` non-bit-exact cases (PR #1641 found 4/6 seeds exact, 2/6
within 1 ULP, on a different fixture family -- not re-tested here);
whether this generalizes past `m*k=256` to even larger totals.
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

# (name, m, k, filename)
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

# Cited directly from PR #1653's own FLOAT32_ULP_DIFF/DOUBLE_PRECISION_
# ULP_DIFF tables -- not recomputed here, since this file's own job is
# testing a THIRD formula, not re-verifying the first two.
FLOAT32_THROUGHOUT_ULP_DIFF = {
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
DOUBLE_THROUGHOUT_ULP_DIFF = {
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
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
    float32 at every step. Re-derived here (not imported) so this
    file's own comparisons are self-contained."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in calib_samples(m, k, seed):
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    return float(np.float32(1.0) / scale)


def double_throughout_recip(m, k, seed=1):
    """PR #1653's own Finding 2 formula: `lo`/`hi`/`scale`/reciprocal
    all accumulated in float64, cast to float32 only for the final
    returned value."""
    flat64 = np.concatenate(
        [s.astype(np.float64).reshape(-1) for s in calib_samples(m, k, seed)]
    )
    lo = min(0.0, float(flat64.min()))
    hi = max(0.0, float(flat64.max()))
    if hi <= lo:
        hi = lo + 1.0
    scale = (hi - lo) / 255.0
    return float(np.float32(1.0 / scale))


def round_scale_once_recip(m, k, seed=1):
    """This file's own new formula: `lo`, `hi`, and `scale = (hi-lo)/
    255.0` computed with no intermediate float32 rounding (both
    operands of every subtraction/division promoted to float64 first),
    `scale` rounded to float32 exactly ONCE, then the reciprocal taken
    as an ordinary float32 division (`float32(1.0) / scale32`) --
    rather than PR #1653's own `double_throughout_recip`, which keeps
    the reciprocal itself in float64 and rounds only the very final
    result."""
    flat64 = np.concatenate(
        [s.astype(np.float64).reshape(-1) for s in calib_samples(m, k, seed)]
    )
    lo = min(0.0, float(flat64.min()))
    hi = max(0.0, float(flat64.max()))
    if hi <= lo:
        hi = lo + 1.0
    scale64 = (hi - lo) / 255.0
    scale32 = np.float32(scale64)
    return float(np.float32(1.0) / scale32)


def float32_subtraction_then_double_division_recip(m, k, seed=1):
    """Isolation variant (`variantA`): `hi`/`lo` are ordinary float32
    values and their DIFFERENCE is taken as a float32 subtraction
    first, only THEN promoted to float64 for the division by 255 and
    rounded once. Distinguishes whether it is the subtraction or the
    division that needs double precision."""
    samples = calib_samples(m, k, seed)
    flat = np.concatenate([s.reshape(-1) for s in samples])
    lo32 = min(np.float32(0.0), np.float32(flat.min()))
    hi32 = max(np.float32(0.0), np.float32(flat.max()))
    diff32 = np.float32(hi32) - np.float32(lo32)
    scale32 = np.float32(float(diff32) / 255.0)
    return float(np.float32(1.0) / scale32)


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_any_shape(self):
        for name, _m, _k, filename in SHAPES:
            errs = mcode.check(load(filename))
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], (name, errs))


class TestPriorFormulasStillShowTheirOwnKnownGaps(unittest.TestCase):
    """Directly re-confirms PR #1651's and PR #1653's own tables still
    hold on a fresh decode, before this file's own new formula is
    compared against them."""

    def test_float32_throughout_matches_its_known_table(self):
        for name, m, k, filename in SHAPES:
            observed = observed_recip_x_scale(filename)
            computed = float32_throughout_recip(m, k)
            self.assertEqual(
                ulp_diff(observed, computed), FLOAT32_THROUGHOUT_ULP_DIFF[name], name
            )

    def test_double_throughout_matches_its_known_table(self):
        for name, m, k, filename in SHAPES:
            observed = observed_recip_x_scale(filename)
            computed = double_throughout_recip(m, k)
            self.assertEqual(
                ulp_diff(observed, computed), DOUBLE_THROUGHOUT_ULP_DIFF[name], name
            )


class TestRoundScaleOnceIsBitExactOnEveryTestedShape(unittest.TestCase):
    """The core new finding: closing PR #1653's own `m*k=64` gap
    without reopening `m*k=256` or regressing 17/32/128."""

    def test_all_ten_fixtures_are_bit_exact(self):
        for name, m, k, filename in SHAPES:
            observed = observed_recip_x_scale(filename)
            computed = round_scale_once_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 0, name)

    def test_both_mk64_splits_specifically_are_now_exact(self):
        for name, m, k, filename in SHAPES:
            if m * k != 64:
                continue
            observed = observed_recip_x_scale(filename)
            computed = round_scale_once_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 0, name)


class TestTheSubtractionStepIsWhatMatters(unittest.TestCase):
    """Isolates the deciding factor: a float32 `hi-lo` subtraction
    (even if the division afterward is done in double) reproduces the
    SAME 1-ULP failures the two prior formulas showed -- so it is
    specifically the subtraction, not the division, that needs
    double-precision inputs to become bit-exact."""

    def test_float32_subtraction_variant_still_fails_at_mk64_and_mk256(self):
        for name, m, k, filename in SHAPES:
            if m * k not in (64, 256):
                continue
            observed = observed_recip_x_scale(filename)
            computed = float32_subtraction_then_double_division_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 1, name)

    def test_float32_subtraction_variant_still_matches_where_others_already_did(self):
        for name, m, k, filename in SHAPES:
            if m * k in (64, 256):
                continue
            observed = observed_recip_x_scale(filename)
            computed = float32_subtraction_then_double_division_recip(m, k)
            self.assertEqual(ulp_diff(observed, computed), 0, name)


if __name__ == "__main__":
    unittest.main()
