"""Answers a question flagged as open in nearly every PR of the
two-live-input elementwise cluster (`tests/test_axera_add_quant_
fields_two_live_inputs.py` PR #1657 and every op/operand-order/
trivial-zero-point PR since): every single fixture in that whole
cluster used the exact same shape, `(1, 16)`, for both `x1`/`x2`.
This file builds `Add(x1, x2)` at two new shapes -- `(1, 32)`
(`m*k=32`) and both `(2, 32)`/`(1, 64)` (two splits of `m*k=64`) --
and checks whether the cluster's own established locators still hold.

## Fixtures

Three fresh `Add(x1[m,k], x2[m,k])` builds beyond the baseline
`(1,16)` (`add_1x16_two_live_seed1_2.mcode.gz`, PR #1657), all at
calibration seed pair `(1,2)` for direct comparability: `add_1x32_
two_live_seed1_2.mcode.gz` (`m*k=32`), `add_2x32_two_live_seed1_2.
mcode.gz` and `add_1x64_two_live_seed1_2.mcode.gz` (two different
splits of the same `m*k=64` total -- chosen deliberately, since the
single-live-input arc's own `tests/test_axera_gemm_recip_xscale_
element_count_dependence.py` (PR #1651) already established that
`m*k` (the total flattened element count), not the individual `m`/`k`
split, drives several of that arc's own shape-dependent effects,
because `numpy.random.RandomState(seed).randn(m,k)` draws the same
flat stream regardless of shape). Real `docker run pulsar2:7.0-lite
build` invocations via `pulsar2_docker.build()`, two separate
`input_configs` entries (one per tensor), matching every prior fixture
in this cluster's own build convention exactly. `mcode.check()`
reports zero errors on all three.

## Finding 1: `x1`'s own literal-quad `zp_x` locator offset is STABLE
## across every tested shape -- unlike the single-live-input arc's own
## Gemm/MatMul locators, which PR #1648 found shift with `K`/`M`

The literal 6-byte quad (`02 10 1b <zp_x1> 83 36`) lands at byte offset
`1092` in all FOUR tested shapes (`(1,16)`, `(1,32)`, `(2,32)`,
`(1,64)`) -- a genuine, positive "offset-invariant" result, in contrast
to the analogous Gemm/MatMul locator PR #1648 found shifts by several
bytes with `K`/`M`. This file does not claim the offset is invariant
at every possible shape, only that it did not move across the four
shapes actually tested here.

## Finding 2: `x1`'s own scale locator gains a 4th record (`field=80`)
## at exactly `m*k=32` -- the SAME mechanism PR #1651 found for Gemm,
## now confirmed to carry over to the two-live-input cluster

At `(1,16)` and both `m*k=64` shapes, `x1`'s own `verb=161,bank=15`
scale group has the usual 3 records (`field in (96,112,128)`). At
`(1,32)` (`m*k=32`) specifically, a 4th record appears at `field=80`
-- exactly the trigger PR #1651 established for Gemm's own `1/x_scale`
group (`the extra record appears if and only if m*k == 32`). This is
not a coincidence unique to Gemm: the SAME product-driven trigger
governs Add's own `x1` locator too.

## Finding 3: `x1`'s own scale VALUE needs PR #1654's round-scale-once
## formula at `m*k=64`, bit-exact under the plain formula at `m*k` in
## `{16,32}` -- the SAME `m*k=64` 1-ULP anomaly PR #1651/#1653/#1654
## established for Gemm, now confirmed for Add's own `x1` too

The plain float32-throughout `ComputeAsymmetricUint8QuantParams`
formula matches `x1`'s own recomputed `1/x1_scale` bit-exact at
`(1,16)` and `(1,32)`, but finds ZERO hits at either `m*k=64` shape;
PR #1654's own "round-scale-once" formula (the `hi-lo` subtraction and
division done in float64, `scale` rounded to float32 exactly once)
closes the gap, bit-exact on both `(2,32)` and `(1,64)` -- identical
results on both splits, consistent with `m*k` (not the split) being
the driver, per PR #1651's own established mechanism.

## Finding 4 (new): `x2`'s own zero-point locator RELOCATES entirely
## from `reg=94,tag=132` to `reg=66,tag=131` at `m*k=64` -- not merely
## a value or offset shift, a full locator identity change

At `(1,16)` and `(1,32)`, `x2`'s own zero point decodes exactly where
every other fixture in this cluster expects it: `reg=94,tag=132`. At
BOTH `m*k=64` shapes (`(2,32)` and `(1,64)`), `reg=94,tag=132` is
entirely ABSENT (`reg=94` itself still fires at other tags, `130` and
`159` -- not a global absence, the same "only the zero-point-carrying
tag is affected" pattern PRs #1663/#1667/#1669 already established for
the *trivial*-zero-point case, but this is a *non*-trivial zero point,
`zp2=100`, not `0` -- a genuinely different trigger). Instead, a NEW
locator, `reg=66,tag=131`, carries `x2`'s own zero point at both
`m*k=64` shapes -- identical `(reg,tag)` pair, identical value (`100`),
on both splits, the same product-driven (not split-driven) consistency
Finding 2/3 already showed. This locator shows the exact same
"filter out an unrelated constant `4` byte" collision this whole
cluster's own output-side locators already need (Add's own `zp_y` at
`reg=14,tag=131`, PR #1657's own Finding 4) -- but here it is an
INPUT-side field (`x2`'s own zero point) using an output-side-style
tag (`131`), at a shape where the usual input-side locator vanishes.

## What this establishes, precisely, and what it does not

**Established**: the two-live-input cluster's own established locators
are NOT uniformly shape-invariant. `x1`'s own literal-quad OFFSET is
stable across all four tested shapes (a genuine positive result, unlike
the single-live-input arc's own Gemm/MatMul finding); `x1`'s own scale
locator's RECORD COUNT and VALUE PRECISION both depend on `m*k`
exactly the way PR #1651/#1654 already established for Gemm (a real
cross-cluster confirmation, not assumed); `x2`'s own zero-point locator
RELOCATES to a different `(reg,tag)` pair entirely at `m*k=64` -- a new
finding, confirmed identical across two different splits of that same
product, not previously observed anywhere in this cluster (which had
only ever tested `m*k=16`).

**NOT established**: whether `reg=66,tag=131` is itself op-generic
(only tested for `Add`, not Sub/Mul/Div); the underlying reason `m*k=64`
specifically triggers a full locator relocation rather than merely a
value/offset shift (no access to Pulsar2's own source); whether other
`m*k` totals (e.g. `128`, `256`, already characterized for Gemm by PR
#1651) show their own distinct relocation patterns for this cluster's
own `x2`/`zp_y` locators (not tested here); `y_scale`'s and `zp_y`'s own
locators, confirmed STABLE (`verb=161,bank=15,field in (96,112,128)`
and `reg=14,tag=131` respectively) across every shape tested here, but
not stress-tested at any `m*k` beyond `64`.
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

# (tag, shape, m*k)
SHAPES = [
    ("1x16", (1, 16), 16),
    ("1x32", (1, 32), 32),
    ("2x32", (2, 32), 64),
    ("1x64", (1, 64), 64),
]


def load(tag):
    path = os.path.join(FIX, f"add_{tag}_two_live_seed1_2.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(tag):
    return mcode.decode(load(tag), **mcode.FULL_RULE)


def calib_samples(seed, shape, n_samples=4):
    """`RandomState(seed).randn(*shape)` x4 -- the exact calibration
    data this file's own fixture-build script used for each `Add`
    input, at each tested shape."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


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
    exactly ONE time -- not re-derived here, cited directly and reused
    as-is."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_shape(self):
        for tag, _, _ in SHAPES:
            errs = mcode.check(load(tag))
            self.assertEqual(errs, [], (tag, errs))


class TestX1LiteralQuadOffsetIsStableAcrossEveryTestedShape(unittest.TestCase):
    def test_offset_is_1092_at_every_shape(self):
        for tag, shape, _ in SHAPES:
            data = load(tag)
            zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(1, shape))
            quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(hits, [1092], (tag, zp1))


class TestX1ScaleLocatorRecordCountDependsOnMK(unittest.TestCase):
    def test_extra_field80_record_fires_only_at_mk32(self):
        for tag, shape, mk in SHAPES:
            recs = decode(tag)
            _, _, s1 = asymmetric_uint8_quant_params(calib_samples(1, shape))
            recip1 = np.float32(np.float32(1.0) / np.float32(s1))
            target = struct.pack("<f", recip1)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == target
            ]
            fields = sorted(r["field"] for r in hits)
            if mk == 32:
                self.assertEqual(fields, [80, 96, 112, 128], (tag, mk))
            elif mk == 16:
                self.assertEqual(fields, [96, 112, 128], (tag, mk))
            # mk == 64 is checked separately below (needs the refined
            # formula to find any hit at all, see the next test class).


class TestX1ScaleValuePrecisionDependsOnMKLikeGemmDid(unittest.TestCase):
    def test_plain_formula_finds_zero_hits_at_mk64(self):
        for tag, shape, mk in SHAPES:
            if mk != 64:
                continue
            recs = decode(tag)
            _, _, s1 = asymmetric_uint8_quant_params(calib_samples(1, shape))
            recip1 = np.float32(np.float32(1.0) / np.float32(s1))
            target = struct.pack("<f", recip1)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == target
            ]
            self.assertEqual(hits, [], tag)

    def test_round_scale_once_is_bit_exact_at_every_shape(self):
        for tag, shape, _ in SHAPES:
            recs = decode(tag)
            s1 = calib_samples(1, shape)
            recip1 = np.float32(np.float32(1.0) / round_scale_once(s1))
            target = struct.pack("<f", recip1)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == target
            ]
            self.assertGreaterEqual(len(hits), 3, tag)
            fields = sorted({r["field"] for r in hits})
            self.assertTrue({96, 112, 128}.issubset(set(fields)), (tag, fields))


class TestX2ZeroPointLocatorRelocatesAtMK64(unittest.TestCase):
    def test_reg94_tag132_present_at_mk16_and_mk32(self):
        for tag, shape, mk in SHAPES:
            if mk not in (16, 32):
                continue
            recs = decode(tag)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2, shape))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(len(hits), 1, (tag, mk))
            self.assertEqual(hits[0]["payload"][-1], zp2, (tag, mk))

    def test_reg94_tag132_absent_at_mk64_but_reg94_still_fires_at_other_tags(self):
        for tag, shape, mk in SHAPES:
            if mk != 64:
                continue
            recs = decode(tag)
            r94_132 = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(r94_132, [], tag)
            r94_any = [r for r in recs if r.get("kind") == "S" and r.get("reg") == 94]
            self.assertGreater(len(r94_any), 0, tag)
            other_tags = {r["tag"] for r in r94_any}
            self.assertNotIn(132, other_tags, tag)

    def test_reg66_tag131_carries_zp2_at_mk64_identically_on_both_splits(self):
        for tag, shape, mk in SHAPES:
            if mk != 64:
                continue
            recs = decode(tag)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2, shape))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 66 and r.get("tag") == 131
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(4, bytes_seen, (tag, bytes_seen))
            self.assertIn(zp2, bytes_seen, (tag, bytes_seen, zp2))

    def test_reg66_tag131_is_absent_at_mk16_and_mk32(self):
        """Specificity check: `reg=66,tag=131` is not just a
        coincidentally-present record at every shape -- confirms it is
        specifically an `m*k=64` phenomenon, not always there."""
        for tag, shape, mk in SHAPES:
            if mk == 64:
                continue
            recs = decode(tag)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2, shape))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S"
                and r.get("reg") == 66
                and r.get("tag") == 131
                and r["payload"][-1] == zp2
            ]
            self.assertEqual(hits, [], (tag, mk, zp2))


class TestYScaleAndZpYLocatorsAreStableAcrossEveryTestedShape(unittest.TestCase):
    def test_y_scale_matches_at_the_same_locator_every_shape(self):
        for tag, shape, _ in SHAPES:
            recs = decode(tag)
            ys = [
                a + b for a, b in zip(calib_samples(1, shape), calib_samples(2, shape))
            ]
            sy32 = round_scale_once(ys)
            target = struct.pack("<f", sy32)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == target
            ]
            self.assertGreaterEqual(len(hits), 3, tag)

    def test_zp_y_matches_at_reg14_tag131_every_shape(self):
        for tag, shape, _ in SHAPES:
            recs = decode(tag)
            ys = [
                a + b for a, b in zip(calib_samples(1, shape), calib_samples(2, shape))
            ]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 14 and r.get("tag") == 131
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(4, bytes_seen, (tag, bytes_seen))
            self.assertIn(zpy, bytes_seen, (tag, bytes_seen, zpy))


if __name__ == "__main__":
    unittest.main()
