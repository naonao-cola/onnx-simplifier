"""Answers a question PR #1646/#1647's own diffs both explicitly left
open: whether the `verb=161,bank=15` `1/x_scale` locator's fixed byte
offset (1456/1464/1472, found identically for both Gemm and MatMul at
the shared `(1,16,16)` shape) is an artifact of that one shape, or
genuinely shape-invariant. PR #1647's own words: "this is consistent
with... the scheduler laying out an identical stream prefix for any
op producing that shape, not evidence the offset is shape-invariant
across other op/shape combinations."

## Fixtures

Four fresh `Gemm(x[m,k], w[k,n] const, b[n] const, transB=0)` builds,
calibration seed fixed at 1 (`RandomState(1).randn(m, k)` x4 samples,
`calibration_method: MinMax` -- the same convention
`_build_and_get_mcode_bytes` uses), weight/bias fixed at `RandomState(0)`
-- one baseline at the exact `(1,16,16)` shape PR #1645/#1646/#1647 all
used, and three single-axis variations (`K` doubled, `N` doubled, `M`
quadrupled), each changing exactly one of `M`/`K`/`N` from the baseline
so any offset/value shift can be attributed to that one axis. Real
`docker run pulsar2:7.0-lite build` invocations, `pulsar2_docker.build()`
called directly with a hand-written config, matching the process
PR #1645/#1646/#1647 all used.

## Finding: the offset moves with `K` and `M`, but NOT with `N` --
## and bit-exactness itself breaks at `M=4`

| shape (`M,K,N`) | offset (field=96 record) | record count | value vs recomputed |
| --- | --- | --- | --- |
| `1,16,16` (baseline) | 1456 | 3 | bit-exact |
| `1,32,16` (`K` x2) | 1450 (-6) | **4** (new field=80 record) | bit-exact |
| `1,16,32` (`N` x2) | 1456 (unchanged) | 3 | bit-exact |
| `4,16,16` (`M` x4) | 1488 (+32) | 3 | **1 float32 ULP off** (`0xb2ad3f42` vs computed `0xb1ad3f42`) |

Two separate, independent axes of "not shape-invariant" here, neither
of them a coincidence:

1. **The offset itself.** `N` alone (which only changes `w`'s/`y`'s own
   width, not `x`'s shape at all) leaves the locator's byte offset
   completely unchanged. `K` alone shifts it by -6 bytes AND grows the
   record count from 3 to 4 (a new `field=80` copy appears ahead of the
   other three) -- plausibly because `x`'s own byte footprint in the
   stream scales with `K` (`x` is `[M,K]`), so a `K` change reflows
   whatever comes between `x`'s own encoding and this locator. `M`
   alone shifts the offset by +32 bytes with no new record appearing.
   None of these three shifts share a common formula this file
   attempts to derive -- establishing that they differ is the finding,
   not explaining why.

2. **The value's own bit-exactness.** Every `M=1` case (baseline, `K`
   doubled, `N` doubled) matches the recomputed `1/x_scale` bit-for-bit
   in `struct.pack("<f", ...)` terms, the same precision PR #1641/
   #1646/#1647 already found. At `M=4`, the SAME `ComputeAsymmetricUint8
   QuantParams`-style min/max-over-all-samples formula this whole arc
   has used gets the value to within 1 ULP but not exactly right --
   the first shape tested in this whole arc where the "close but not
   bit-exact" pattern (previously only seen for `y_scale`, never for
   `1/x_scale`) appears on `1/x_scale` itself. This file does not
   determine why `M>1` breaks bit-exactness (a plausible guess -- some
   per-row or partitioned calibration statistic this file's flat
   `min()`/`max()` over the whole `[M,K]` sample doesn't reproduce --
   is not tested here).

## What this establishes, precisely, and what it does not

**Established**: the `verb=161,bank=15` locator's byte offset is NOT
shape-invariant -- it depends on at least `K` and `M` (not `N`) in ways
this file measures exactly but does not derive a formula for. The
locator's own VALUE precision also depends on shape: bit-exact at
`M=1` regardless of `K`/`N`, off by 1 ULP at `M=4`.

**NOT established**: a formula predicting the offset shift from
`M`/`K`/`N`; why `M>1` specifically breaks bit-exactness while `K`/`N`
changes do not; whether a larger `M` (8, 16, ...) shows a growing or
bounded ULP gap; whether Conv's own `verb=161,bank=15` copy (PR #1641)
shows the same `M`-analogous sensitivity (Conv's calibration axis
structure differs enough -- `NCHW` samples, no direct `M` analogue --
that this file makes no claim about it either way).
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

# (name, m, k, n)
SHAPES = [
    ("baseline", 1, 16, 16),
    ("bigK", 1, 32, 16),
    ("bigN", 1, 16, 32),
    ("bigM", 4, 16, 16),
]

EXPECTED_HITS = {
    "baseline": [(1456, 96), (1464, 112), (1472, 128)],
    "bigK": [(1450, 80), (1458, 96), (1466, 112), (1474, 128)],
    "bigN": [(1456, 96), (1464, 112), (1472, 128)],
    "bigM": [(1488, 96), (1496, 112), (1504, 128)],
}

# Whether every hit's operand bytes are bit-exact to the recomputed value.
EXPECTED_BIT_EXACT = {
    "baseline": True,
    "bigK": True,
    "bigN": True,
    "bigM": False,
}


_FIXTURE_NAMES = {
    "baseline": "gemm_1x16x16_baseline_seed1.mcode.gz",
    "bigK": "gemm_1x32x16_bigK_seed1.mcode.gz",
    "bigN": "gemm_1x16x32_bigN_seed1.mcode.gz",
    "bigM": "gemm_4x16x16_bigM_seed1.mcode.gz",
}


def load(name):
    with gzip.open(os.path.join(FIX, _FIXTURE_NAMES[name]), "rb") as f:
        return f.read()


def computed_recip_x_scale(m, k, seed=1):
    """`ComputeAsymmetricUint8QuantParams`'s own scale half
    (`onnxsim/passes/static_quantize_matmul.h`), reciprocal, applied to
    a flat min/max over all 4 `RandomState(seed).randn(m, k)` samples --
    the same formula PR #1640/#1641/#1645/#1646/#1647 have all used."""
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
        for name, _, _, _ in SHAPES:
            errs = mcode.check(load(name))
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], (name, errs))


class TestOffsetDependsOnKAndMButNotN(unittest.TestCase):
    """The core finding: the locator's own byte offset (and record
    count) shifts with `K`/`M` but is exactly unchanged by `N` alone."""

    def test_every_shape_matches_its_expected_offset_set(self):
        for name, _, _, _ in SHAPES:
            self.assertEqual(find_hits(load(name)), EXPECTED_HITS[name], name)

    def test_n_alone_leaves_the_offset_set_unchanged(self):
        self.assertEqual(find_hits(load("baseline")), find_hits(load("bigN")))

    def test_k_alone_changes_both_offset_and_record_count(self):
        base = find_hits(load("baseline"))
        big_k = find_hits(load("bigK"))
        self.assertNotEqual(base, big_k)
        self.assertEqual(len(big_k), len(base) + 1)

    def test_m_alone_shifts_the_offset_with_no_new_record(self):
        base = find_hits(load("baseline"))
        big_m = find_hits(load("bigM"))
        self.assertEqual(len(big_m), len(base))
        self.assertNotEqual([at for at, _ in big_m], [at for at, _ in base])


class TestValueBitExactnessAlsoDependsOnShape(unittest.TestCase):
    """The second finding: the recomputed `1/x_scale` value matches
    every `M=1` shape's real decoded bytes bit-for-bit, but is off by
    exactly 1 float32 ULP at `M=4` -- shape affects not just where the
    field lives, but whether this arc's formula reproduces it exactly."""

    def test_bit_exactness_matches_the_expected_table_per_shape(self):
        for name, m, k, _n in SHAPES:
            data = load(name)
            recs = mcode.decode(data)
            hits = [r for r in recs if r.get("verb") == 161 and r.get("bank") == 15]
            self.assertTrue(hits, name)
            computed = computed_recip_x_scale(m, k)
            cbytes = struct.pack("<f", computed)
            all_exact = all(r["operand"] == cbytes for r in hits)
            self.assertEqual(all_exact, EXPECTED_BIT_EXACT[name], name)

    def test_bigm_mismatch_is_exactly_one_float32_ulp(self):
        data = load("bigM")
        recs = mcode.decode(data)
        hits = [r for r in recs if r.get("verb") == 161 and r.get("bank") == 15]
        computed = computed_recip_x_scale(4, 16)
        cbytes = struct.pack("<f", computed)
        c_int = struct.unpack("<I", cbytes)[0]
        for r in hits:
            o_int = struct.unpack("<I", r["operand"])[0]
            self.assertEqual(abs(o_int - c_int), 1, (r["at"], r["operand"]))

    def test_all_m1_shapes_are_bit_exact(self):
        for name, m, k, _n in SHAPES:
            if m != 1:
                continue
            data = load(name)
            recs = mcode.decode(data)
            hits = [r for r in recs if r.get("verb") == 161 and r.get("bank") == 15]
            computed = computed_recip_x_scale(m, k)
            cbytes = struct.pack("<f", computed)
            for r in hits:
                self.assertEqual(r["operand"], cbytes, (name, r["at"]))


class TestTheMatchIsNotACoincidentalNearbyByteRun(unittest.TestCase):
    """Specificity check: the recomputed value (bit pattern) does not
    also appear at unrelated offsets far from the located records, on
    any shape -- ruling out the match being a common/generic constant
    that happens to recur throughout the stream."""

    def test_recomputed_value_bytes_are_rare_in_the_stream(self):
        for name, m, k, _n in SHAPES:
            data = load(name)
            computed = computed_recip_x_scale(m, k)
            cbytes = struct.pack("<f", computed)
            all_hits = [i for i in range(len(data) - 3) if data[i : i + 4] == cbytes]
            located = {at for at, _ in EXPECTED_HITS[name]}
            self.assertLessEqual(
                len(all_hits), len(located) + 1, (name, all_hits, located)
            )


if __name__ == "__main__":
    unittest.main()
