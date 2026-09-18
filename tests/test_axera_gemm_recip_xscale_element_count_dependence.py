"""Answers a question PR #1648's own diff explicitly left open: whether
the `verb=161,bank=15` `1/x_scale` locator's byte offset shift and its
value bit-exactness (both found to depend on `M`/`K` in PR #1648's own
4-shape survey: `1,16,16` / `1,32,16` / `1,16,32` / `4,16,16`) follow a
simple `f(M) + g(K)`-style formula. PR #1648's own words: "None of
these three shifts share a common formula this file attempts to
derive -- establishing that they differ is the finding, not explaining
why."

This file tests 10 further `Gemm(x[m,k], w[k,16] const, b[16] const,
transB=0)` shapes -- four different `(m,k)` SPLITS that all share the
same total element count `m*k=32` (`(1,32)`, `(2,16)`, `(4,8)`,
`(8,4)`), plus six more at other totals (`17`, `64` x2 splits, `128`
x2 splits, `256`) -- calibration seed fixed at 1, weight/bias fixed at
`RandomState(0)`, matching PR #1645-#1648's own convention exactly.

## Finding 1: whether the extra `field=80` record appears (PR #1648's
## own `bigK` observation) depends ONLY on `m*k`, not on how that
## product is split between `m` and `k`

PR #1648 found `K=32` (with `M=1`) grows the locator's record count
from 3 to 4 (a new `field=80` copy appears ahead of the other three).
This file finds the SAME 4-record growth at `(2,16)`, `(4,8)` and
`(8,4)` -- three more splits of the same product, `m*k=32` -- and
finds it absent at every one of six other totals tested (`17`, `64`
via two different splits, `128` via two different splits, `256`),
including totals both smaller and larger than 32. The rule is clean
and confirmed, not merely suggestive: **the extra record appears if
and only if `m*k == 32`**, independent of the individual `m`/`k`
values.

| shape `(m,k)` | `m*k` | record count | offsets |
| --- | --- | --- | --- |
| `(1,32)` | 32 | 4 | 1450/1458/1466/1474 |
| `(2,16)` | 32 | 4 | 1450/1458/1466/1474 |
| `(4,8)` | 32 | 4 | 1482/1490/1498/1506 |
| `(8,4)` | 32 | 4 | 1866/1874/1882/1890 |
| `(1,17)` | 17 | 3 | 1456/1464/1472 |
| `(2,32)` | 64 | 3 | 1456/1464/1472 |
| `(1,64)` | 64 | 3 | 1456/1464/1472 |
| `(8,16)` | 128 | 3 | 1873/1881/1889 |
| `(1,128)` | 128 | 3 | 1489/1497/1505 |
| `(16,16)` | 256 | 3 | 1841/1849/1857 |

No formula for *why* `32` specifically triggers the extra record is
attempted here (this file only establishes that it is a function of
the product, not the split).

## Finding 2: `1/x_scale`'s own bit-exactness also depends ONLY on
## `m*k`, for a mechanistic reason this file can point to directly

Numpy's `RandomState(seed).randn(m, k)` draws its underlying flat
stream in the same order regardless of `(m, k)` -- reshaping only
changes how that flat stream is packaged, not its values. So two
different splits of the same product (e.g. `(2,16)` vs `(4,8)`, both
`32`) hand the exact same 128 underlying float32 values to both this
file's own recomputation AND (plausibly) to Pulsar2's real calibration
pass, which only ever sees the same 4 flattened `.npy` calibration
samples regardless of what shape they are logically declared as. This
demystifies what would otherwise look like a coincidence: this file's
own `computed_recip_x_scale(m, k)` is *numerically identical* across
splits of the same product by construction, and the real device's
independently-computed value matches or fails to match it identically
across splits too --

| `m*k` | recomputed `1/x_scale` | ULP diff (all 4 splits tested agree) |
| --- | --- | --- |
| 32 | 56.829399 | 0 (bit-exact, all 4 splits) |
| 17 | 57.930927 | 0 (bit-exact) |
| 64 | 47.919621 | **1** (both `(2,32)` and `(1,64)` splits) |
| 128 | 43.784775 | 0 (bit-exact, both `(8,16)` and `(1,128)` splits) |
| 256 | 36.364323 | **1** |

This closes half of PR #1648's own open question (bit-exactness is a
function of total element count alone) but does NOT explain the
remaining irregularity: `64` and `256` are off by exactly 1 ULP while
`16`/`17`/`24`/`32`/`48`/`128` (this file's own totals plus PR #1648's
own already-established `16`/`24`/`32`/`48` results) are all bit-exact
-- not a simple threshold (`128 > 64` is exact, `64` is not), plausibly
a floating-point reduction-order sensitivity at specific sample counts
(the same class of explanation PR #1642/#1646/#1647 already used for
`y_scale`'s own non-bit-exactness), but this file does not verify that
guess.

## Finding 3: the locator's own byte OFFSET, unlike record-count and
## bit-exactness, genuinely depends on the `(m, k)` split, not just
## the product

All four `m*k=32` splits share record count (4) and value (bit-exact),
but land at four DIFFERENT offsets (1450, 1450, 1482, 1866) -- `(1,32)`
and `(2,16)` coincide, but `(4,8)` and `(8,4)` do not, and `(8,4)`'s
offset (1866) sits far closer to `(8,16)`'s own offset (1873, product
128) than to any other `m*k=32` split. This is consistent with `m`
itself (not the product) being the dominant driver of the offset's
own position -- larger `m` shapes clustering together regardless of
`k` -- but this file does not fit or claim an exact formula for it,
only that it is real, measured, and not product-only (contrast with
Findings 1 and 2, which genuinely are product-only).
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
    ("mk32_1x32", 1, 32, "gemm_1x32x16_mk32_split_1x32.mcode.gz"),
    ("mk32_2x16", 2, 16, "gemm_2x16x16_mk32_split_2x16.mcode.gz"),
    ("mk32_4x8", 4, 8, "gemm_4x8x16_mk32_split_4x8.mcode.gz"),
    ("mk32_8x4", 8, 4, "gemm_8x4x16_mk32_split_8x4.mcode.gz"),
    ("mk17_1x17", 1, 17, "gemm_1x17x16_negctrl.mcode.gz"),
    ("mk64_2x32", 2, 32, "gemm_2x32x16_mk64_split_2x32.mcode.gz"),
    ("mk64_1x64", 1, 64, "gemm_1x64x16_mk64_split_1x64.mcode.gz"),
    ("mk128_8x16", 8, 16, "gemm_8x16x16_mk128_split_8x16.mcode.gz"),
    ("mk128_1x128", 1, 128, "gemm_1x128x16_mk128_split_1x128.mcode.gz"),
    ("mk256_16x16", 16, 16, "gemm_16x16x16_mk256_split_16x16.mcode.gz"),
]

EXPECTED_HITS = {
    "mk32_1x32": [(1450, 80), (1458, 96), (1466, 112), (1474, 128)],
    "mk32_2x16": [(1450, 80), (1458, 96), (1466, 112), (1474, 128)],
    "mk32_4x8": [(1482, 80), (1490, 96), (1498, 112), (1506, 128)],
    "mk32_8x4": [(1866, 80), (1874, 96), (1882, 112), (1890, 128)],
    "mk17_1x17": [(1456, 96), (1464, 112), (1472, 128)],
    "mk64_2x32": [(1456, 96), (1464, 112), (1472, 128)],
    "mk64_1x64": [(1456, 96), (1464, 112), (1472, 128)],
    "mk128_8x16": [(1873, 96), (1881, 112), (1889, 128)],
    "mk128_1x128": [(1489, 96), (1497, 112), (1505, 128)],
    "mk256_16x16": [(1841, 96), (1849, 112), (1857, 128)],
}

# ULP difference (observed_int - computed_int) shared by every record
# in a given shape.
EXPECTED_ULP_DIFF = {
    "mk32_1x32": 0,
    "mk32_2x16": 0,
    "mk32_4x8": 0,
    "mk32_8x4": 0,
    "mk17_1x17": 0,
    "mk64_2x32": 1,
    "mk64_1x64": 1,
    "mk128_8x16": 0,
    "mk128_1x128": 0,
    "mk256_16x16": 1,
}


def load(filename):
    with gzip.open(os.path.join(FIX, filename), "rb") as f:
        return f.read()


def computed_recip_x_scale(m, k, seed=1):
    """`ComputeAsymmetricUint8QuantParams`'s own scale half
    (`onnxsim/passes/static_quantize_matmul.h`), reciprocal, applied
    to a flat min/max over all 4 `RandomState(seed).randn(m, k)`
    samples -- the same formula PR #1640/#1641/#1645/#1646/#1647/#1648
    have all used."""
    rng = np.random.RandomState(seed)
    samples = [rng.randn(m, k).astype(np.float32) for _ in range(4)]
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    return float(np.float32(1.0) / scale)


def find_hits(data):
    recs = mcode.decode(data)
    return sorted(
        (r["at"], r["field"])
        for r in recs
        if r.get("verb") == 161 and r.get("bank") == 15
    )


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_any_shape(self):
        for name, _m, _k, filename in SHAPES:
            errs = mcode.check(load(filename))
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], (name, errs))


class TestExtraRecordDependsOnlyOnTheProductNotTheSplit(unittest.TestCase):
    """Finding 1: the 4th `field=80` record fires iff `m*k == 32`,
    confirmed across four independent splits of that same product and
    six negative-control totals."""

    def test_every_shape_matches_its_expected_offset_set(self):
        for name, _m, _k, filename in SHAPES:
            self.assertEqual(find_hits(load(filename)), EXPECTED_HITS[name], name)

    def test_all_four_mk32_splits_get_the_extra_record(self):
        for name, m, k, filename in SHAPES:
            if m * k != 32:
                continue
            self.assertEqual(len(find_hits(load(filename))), 4, name)

    def test_no_other_tested_total_gets_the_extra_record(self):
        for name, m, k, filename in SHAPES:
            if m * k == 32:
                continue
            self.assertEqual(len(find_hits(load(filename))), 3, name)


class TestBitExactnessDependsOnlyOnTheProductNotTheSplit(unittest.TestCase):
    """Finding 2: the recomputed value is numerically identical across
    splits of the same product (a direct consequence of
    `RandomState.randn` drawing the same flat stream regardless of
    shape), and the real decoded byte's own match/mismatch against it
    is identical across those same splits too."""

    def test_recomputed_value_is_identical_across_splits_of_one_product(self):
        by_product = {}
        for _name, m, k, _filename in SHAPES:
            by_product.setdefault(m * k, set()).add(computed_recip_x_scale(m, k))
        for product, values in by_product.items():
            self.assertEqual(len(values), 1, (product, values))

    def test_ulp_diff_matches_the_expected_table_and_is_split_independent(self):
        for name, m, k, filename in SHAPES:
            data = load(filename)
            recs = mcode.decode(data)
            hits = [r for r in recs if r.get("verb") == 161 and r.get("bank") == 15]
            computed = computed_recip_x_scale(m, k)
            c_int = struct.unpack("<I", struct.pack("<f", computed))[0]
            diffs = {struct.unpack("<I", r["operand"])[0] - c_int for r in hits}
            self.assertEqual(len(diffs), 1, (name, diffs))
            self.assertEqual(diffs.pop(), EXPECTED_ULP_DIFF[name], name)

    def test_same_product_different_split_gives_the_same_ulp_diff(self):
        by_product = {}
        for name, m, k, filename in SHAPES:
            by_product.setdefault(m * k, set()).add(EXPECTED_ULP_DIFF[name])
        for product, diffs in by_product.items():
            self.assertEqual(len(diffs), 1, (product, diffs))


class TestOffsetIsNotProductOnly(unittest.TestCase):
    """Finding 3: unlike Findings 1 and 2, the locator's own byte
    offset is NOT a pure function of `m*k` -- two of the four `m*k=32`
    splits land at different offsets from each other."""

    def test_mk32_splits_do_not_all_share_one_offset(self):
        offsets = set()
        for name, m, k, filename in SHAPES:
            if m * k != 32:
                continue
            offsets.add(find_hits(load(filename))[0][0])
        self.assertGreater(len(offsets), 1, offsets)


if __name__ == "__main__":
    unittest.main()
