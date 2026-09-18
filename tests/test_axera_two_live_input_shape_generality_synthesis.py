"""Synthesizes `tests/test_axera_add_two_live_shape_generality.py` (PR
#1671), the first (and so far only) shape-generality test in the
two-live-input elementwise cluster (Add/Sub/Mul/Div, PRs #1657-#1670,
all of which used exactly one shape, `(1,16)`, for every fixture).
Plays the same additive-synthesis role every prior capstone file in
this arc has played: (1) directly re-decodes and recomputes PR #1671's
own headline claims, independently of its own test module, using its
already-committed fixtures (no new Docker builds), (2) states plainly
that shape-generality within this cluster is NOT a single yes/no
verdict but a field-specific picture, and (3) marks this as a
completion point for `Add` specifically while listing what remains
open for the other three ops.

## The field-specific shape-generality picture (`Add` only, 4 shapes:
## `(1,16)`, `(1,32)`, `(2,32)`, `(1,64)`)

| field | shape-generality result |
| --- | --- |
| `x1` zero point (literal-quad offset) | STABLE -- lands at byte 1092 on every tested shape, a genuine positive contrast to the single-live-input arc's own Gemm/MatMul locators (PR #1648), which shift with `K`/`M` |
| `x1` scale (record count) | DEPENDS ON `m*k` -- a 4th `field=80` record appears only at `m*k=32`, the identical trigger PR #1651 established for Gemm's own `1/x_scale` group |
| `x1` scale (value precision) | DEPENDS ON `m*k` -- bit-exact under the plain formula at `m*k` in `{16,32}`, needs PR #1654's round-scale-once formula at `m*k=64`, the identical pattern PR #1651/#1653/#1654 established for Gemm |
| `x2` zero point | RELOCATES ENTIRELY -- `reg=94,tag=132` (the locator every fixture in this cluster up to now has used) is replaced by a completely different `reg=66,tag=131` pair at `m*k=64`, confirmed identical across two different splits of that same product; not merely a value or offset shift within the same locator |
| `y_scale` | STABLE -- same `verb=161,bank=15,field in (96,112,128)` locator, correctly recomputed via round-scale-once, at every tested shape |
| `zp_y` | STABLE -- same `reg=14,tag=131` locator (after filtering the same unrelated constant-`4` byte this locator has needed since PR #1657's own Finding 4) at every tested shape |

This is deliberately NOT collapsed into a single "shape-generality
holds" or "shape-generality fails" headline: three of six rows are
genuinely stable, two show an established cross-cluster mechanism
(the `m*k`-driven record-count/precision rules PR #1651/#1654 already
found for the single-live-input arc's own Gemm), and one shows a
phenomenon never previously observed anywhere in either cluster (a
full locator identity change, not a value/offset shift). Any reader
asking "does shape-generality hold for this cluster" needs to ask it
per-field, not once.

## What this marks as complete, and what remains open

This file marks shape-generality as tested (with the mixed picture
above) for `Add` specifically, at the four shapes PR #1671 built.

**NOT established, still open**:
- `Sub`/`Mul`/`Div`'s own shape-generality -- every fixture for these
  three ops, across every PR in this cluster (operand-order,
  trivial-zero-point, or baseline), has used `(1,16)` only. Whether
  their own `x1`/`x2`/output-side locators are as shape-stable as
  `Add`'s mostly were, or show their own version of the `reg=66,
  tag=131` relocation, is untested.
- Whether `reg=66,tag=131` is itself `Add`-specific or op-generic (PR
  #1671's own open item).
- Whether other `m*k` totals already characterized for Gemm (`128`,
  `256`, PR #1651) show their own distinct patterns for this cluster's
  `x2`/`zp_y` locators.
- The underlying reason `m*k=64` specifically triggers a full locator
  relocation for `x2` rather than merely a value/offset shift (no
  access to Pulsar2's own source, only the directly-confirmed fact of
  the relocation).
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

# (tag, shape, m*k) -- cited directly from PR #1671's own SHAPES list,
# not re-derived, since this file's own job is re-confirming its
# claims, not rediscovering which fixtures exist.
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
    data PR #1671's own fixture-build script used for each `Add`
    input, at each tested shape (`seed=1` for `x1`, `seed=2` for
    `x2`)."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def asymmetric_uint8_quant_params(samples):
    """`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_
    quantize_matmul.h`), the plain float32-throughout formula this
    whole arc has used since PR #1636/#1640, re-derived here rather
    than imported from PR #1671's own test module."""
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
    """PR #1654's own refined formula: the `hi-lo` subtraction and the
    division by 255 done in float64, `scale` rounded to float32
    exactly once -- cited directly, reused as-is."""
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


class TestX1LocatorOffsetAndScaleAreStableOrMKDependentAsExpected(unittest.TestCase):
    """Re-confirms PR #1671's own Findings 1-3 in one place: the
    literal-quad OFFSET is stable, while the scale locator's own
    record count and value precision both track `m*k`."""

    def test_literal_quad_offset_is_1092_at_every_shape(self):
        for tag, shape, _ in SHAPES:
            data = load(tag)
            zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(1, shape))
            quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(hits, [1092], (tag, zp1))

    def test_extra_field80_record_fires_only_at_mk32(self):
        for tag, shape, mk in SHAPES:
            if mk not in (16, 32):
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
            fields = sorted(r["field"] for r in hits)
            expected = [80, 96, 112, 128] if mk == 32 else [96, 112, 128]
            self.assertEqual(fields, expected, (tag, mk))

    def test_round_scale_once_is_bit_exact_at_mk64_where_plain_formula_fails(self):
        for tag, shape, mk in SHAPES:
            if mk != 64:
                continue
            recs = decode(tag)
            s1_plain = calib_samples(1, shape)
            _, _, plain_scale = asymmetric_uint8_quant_params(s1_plain)
            plain_target = struct.pack(
                "<f", np.float32(np.float32(1.0) / np.float32(plain_scale))
            )
            plain_hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == plain_target
            ]
            self.assertEqual(plain_hits, [], tag)

            refined_target = struct.pack(
                "<f", np.float32(np.float32(1.0) / round_scale_once(s1_plain))
            )
            refined_hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == refined_target
            ]
            self.assertGreaterEqual(len(refined_hits), 3, tag)


class TestX2LocatorRelocatesRatherThanShiftsAtMK64(unittest.TestCase):
    """Re-confirms PR #1671's own new Finding 4: `x2`'s own zero point
    is not merely relocated within the same locator at `m*k=64`, it is
    carried by an entirely different `(reg,tag)` pair."""

    def test_reg94_tag132_carries_zp2_at_mk16_and_mk32(self):
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

    def test_reg94_tag132_is_structurally_absent_at_mk64(self):
        for tag, shape, mk in SHAPES:
            if mk != 64:
                continue
            recs = decode(tag)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(hits, [], tag)

    def test_reg66_tag131_carries_zp2_at_mk64_on_both_splits_not_at_mk16_or_mk32(self):
        for tag, shape, mk in SHAPES:
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
            if mk == 64:
                self.assertEqual(len(hits), 1, (tag, mk, zp2))
            else:
                self.assertEqual(hits, [], (tag, mk, zp2))


class TestOutputSideLocatorsAreStableAcrossEveryTestedShape(unittest.TestCase):
    def test_y_scale_matches_at_the_same_locator_every_shape(self):
        for tag, shape, _ in SHAPES:
            recs = decode(tag)
            ys = [
                a + b for a, b in zip(calib_samples(1, shape), calib_samples(2, shape))
            ]
            target = struct.pack("<f", round_scale_once(ys))
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
                if r.get("kind") == "S"
                and r.get("reg") == 14
                and r.get("tag") == 131
                and r["payload"][-1] == zpy
            ]
            self.assertEqual(len(hits), 1, tag)


if __name__ == "__main__":
    unittest.main()
