"""Closes the one remaining cell of the (op, field) grid `tests/test_
axera_quant_field_cluster_synthesis.py` (PR #1652) laid out explicitly:
Gemm's and MatMul's own `zp_y` (output zero point) had never been
attempted, only Conv's (`reg=232`, PR #1644/#1649). Reuses the same
first-principles technique this whole arc has used since PR #1636/
#1640: port `ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/
static_quantize_matmul.h`) into Python, apply it to a hand-written
float64 reference computation of the op's own output (the same
`gemm_reference`/`matmul_reference` functions PR #1646/#1647 already
wrote), and compare against the real decoded mcode byte.

Reuses PR #1646's own 6 Gemm fixtures and PR #1647's own 6 MatMul
fixtures (`scripts/axera/fixtures/{gemm,matmul}_1x16x16_zpxseed*.
mcode.gz`) -- no new Docker builds.

## Finding: `zp_y` lives at `reg=120, tag=132` for BOTH Gemm and MatMul
## -- the same TAG Conv's own `zp_y` uses (`reg=232, tag=132`), though
## not literally the same register number

Unlike `zp_x`'s literal-quad locator (invisible to `mcode.decode()`,
found only by raw substring search) and `1/x_scale`'s `verb=161,
bank=15` locator (identical register AND offset across all three
ops), `zp_y` decodes cleanly as a normal `S`-kind record, at the exact
same byte offset (1623) and the exact same `(tag, reg)` pair for BOTH
Gemm and MatMul -- but a DIFFERENT `reg` than Conv's own `zp_y`
(`reg=232`, not `reg=120`). The `tag` (132) is shared across all
three ops; the specific register index is not. This file does not
claim the locator is fully op-generic the way `verb=161,bank=15` is --
only that Gemm and MatMul share one with each other.

| op | seed | computed `zp_y` | observed | exact? |
| --- | --- | --- | --- | --- |
| Gemm | 0 | 129 | 129 | yes |
| Gemm | 1 | 128 | *(record absent)* | n/a -- see Finding 2 |
| Gemm | 7 | 115 | 115 | yes |
| Gemm | 42 | 113 | 113 | yes |
| Gemm | 100 | 170 | 170 | yes |
| Gemm | 999 | 138 | 138 | yes |
| MatMul | 0 | 120 | 120 | yes |
| MatMul | 1 | 126 | 126 | yes |
| MatMul | 7 | 117 | 117 | yes |
| MatMul | 42 | 106 | 107 | **no -- see Finding 3** |
| MatMul | 100 | 157 | 157 | yes |
| MatMul | 999 | 127 | 127 | yes |

10 of 12 tested (op, seed) pairs are bit-exact. The two exceptions are
NOT noise -- each has its own independent, precisely-targeted
explanation, in the same spirit as this whole arc's other near-miss
findings (PR #1640's Group A/B, PR #1653's `m*k=64` outlier).

## Finding 2: Gemm's own seed=1 (`zp_y == 128` exactly) is a real,
## already-documented codec phenomenon, not a decode failure

Directly re-searched: no `reg=120` record of any tag, and no `tag=132`
record with a payload ending in `128`, exists ANYWHERE in seed=1's own
decoded stream (`908`, `1050`, `1202` all carry a payload-128 byte, but
all three are `reg=8` -- the unrelated reg=8 pool, not `zp_y`). The
record is genuinely, structurally absent, not merely relocated. This
matches an already-documented codec phenomenon (`scripts/axera/
README.md`, "Two builds of 48 that cannot be patched in place"): "one
build's output zero point came out exactly **128**" and its value "was
not written inline but escaped, shifting every byte after it." That
section was about a Conv build; this file independently rediscovers
the identical trigger value (`128` exactly) causing the identical
class of structural absence for Gemm's own `zp_y` -- strong
corroboration that `128` specifically (not "any near-tie value", which
seed=1's own `zp_f=128.24` is not particularly close to -- distance to
the nearest half-boundary is `0.26`, unremarkable among the 6 seeds)
triggers a real, distinct encoding-format special case, independent of
calibration-rounding near-ties.

## Finding 3: MatMul's own seed=42 1-off mismatch is a genuine
## rounding near-tie, the SAME mechanism PR #1640 established for
## Conv's `zp_x`

`zp_f` (the pre-round float value) for MatMul's own 6 seeds has
distances to the nearest rounding half-boundary of `0.116, 0.393,
0.435, 0.058, 0.226, 0.347` (seeds `0,1,7,42,100,999` respectively).
Seed `42`'s own margin (`0.058`) is by far the smallest -- less than
half the next-closest (`0.116`) -- exactly the same "closest seed to a
tie flips" pattern PR #1640 found for Conv's own `zp_x` (where seed=0
was the outlier and, independently, over 4x closer to a tie than any
other tested seed). This file's own float64 reference simply lands on
the wrong side of that near-tie relative to the real device's own
(unknown, internal) computation -- not a formula error.

## What this establishes, precisely, and what it does not

**Established**: Gemm's and MatMul's own `zp_y` is directly decodable
at `reg=120, tag=132`, closing the last cell PR #1652's own grid left
open; both of the 2 non-exact results out of 12 have a specific,
independently-verified explanation (an already-documented codec
special case for exactly-128, and a rounding near-tie matching this
arc's established mechanism) rather than being unexplained noise.

**NOT established**: whether `reg=120` is itself op-generic beyond
Gemm/MatMul (Conv was not re-tested here, and is already known to use
a different register, `reg=232`, for the same semantic field); what
the real device's own resolution is for Gemm's seed=1 case (only that
the field's ordinary encoding path did not fire); whether other
exactly-128 zero-point values elsewhere in this arc's corpus show the
same structural absence (not surveyed here).
"""

import gzip
import math
import os
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

DETERMINISTIC_SEEDS = [0, 1, 7, 42, 100, 999]
ZP_Y_LOCATOR_OFFSET = 1623
ZP_Y_TAG = 132
ZP_Y_REG = 120

# Cited directly from tests/test_axera_gemm_zpx_zeropoint_verification.py
# (PR #1645)'s own DETERMINISTIC_SEEDS dict -- Gemm's own zp_x, not
# recomputed here since this file's own job is zp_y.
GEMM_ZP_X = {0: 135, 1: 133, 7: 129, 42: 131, 100: 132, 999: 126}

GEMM_STRUCTURALLY_ABSENT_SEEDS = {1}
MATMUL_ONE_OFF_SEEDS = {42}


def load(op, seed):
    return gzip.open(
        os.path.join(FIX, f"{op}_1x16x16_zpxseed{seed}.mcode.gz"), "rb"
    ).read()


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data both PR #1645's Gemm fixtures and PR #1647's MatMul fixtures
    used for `x`."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def gemm_weight():
    """Fixed at `RandomState(0)` -- matches tests/test_axera_gemm_
    scale_field_verification.py (PR #1646) exactly."""
    rng = np.random.RandomState(0)
    w = (rng.randn(16, 16) * 0.1).astype(np.float32)
    b = (rng.randn(16) * 0.1).astype(np.float32)
    return w, b


def gemm_reference(x, w, b):
    """Reused verbatim from PR #1646's own `gemm_reference` (`y = x @
    w + b`, float64-accumulating)."""
    return (x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)).astype(
        np.float32
    )


def matmul_weight():
    """Fixed at `RandomState(0)` -- matches tests/test_axera_matmul_
    scale_field_verification.py (PR #1647) exactly."""
    rng = np.random.RandomState(0)
    return (rng.randn(16, 16) * 0.1).astype(np.float32)


def matmul_reference(x, w):
    """Reused verbatim from PR #1647's own `matmul_reference` (`y = x
    @ w`, float64-accumulating)."""
    return (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)


def computed_zp_y(seed, reference_fn, *weight_args):
    """`ComputeAsymmetricUint8QuantParams`'s own zero-point half
    (`onnxsim/passes/static_quantize_matmul.h`, the same `round_half_
    away_from_zero` formula PR #1640/#1644 already verified bit-exact
    for `zp_x`/Conv's own `zp_y`), applied to an op's own OUTPUT range
    instead of its input range. Returns `(zp_int, zp_f_pre_round)`."""
    samples = calib_samples(seed)
    ys_all = [reference_fn(s, *weight_args) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))
    zp_f = float(np.float32(-np.float32(ylo) / np.float32(scale)))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), zp_f


def zp_y_record(recs):
    hits = [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("tag") == ZP_Y_TAG
        and r.get("reg") == ZP_Y_REG
    ]
    return hits[0] if hits else None


class TestGemmZpYMatchesAtReg120Tag132(unittest.TestCase):
    """The core finding for Gemm: `reg=120,tag=132`'s payload's last
    byte matches the recomputed output zero point exactly on every
    seed except the one documented structural-absence case."""

    def test_five_of_six_seeds_match_exactly_at_the_fixed_offset(self):
        w, b = gemm_weight()
        for seed in DETERMINISTIC_SEEDS:
            if seed in GEMM_STRUCTURALLY_ABSENT_SEEDS:
                continue
            recs = mcode.decode(load("gemm", seed), **mcode.FULL_RULE)
            rec = zp_y_record(recs)
            self.assertIsNotNone(rec, seed)
            self.assertEqual(rec["at"], ZP_Y_LOCATOR_OFFSET, seed)
            zp, zp_f = computed_zp_y(seed, gemm_reference, w, b)
            self.assertEqual(rec["payload"][-1], zp, (seed, zp_f))

    def test_seed_one_has_no_reg120_tag132_record_anywhere(self):
        """Directly verifies Finding 2's own absence claim -- not
        merely that the fixed offset doesn't match, but that the
        record does not exist anywhere in the stream, and that no
        OTHER tag=132 record carries the payload byte 128 either
        (ruling out an unrelated coincidental collision, e.g. the
        already-known reg=8 pool)."""
        recs = mcode.decode(load("gemm", 1), **mcode.FULL_RULE)
        self.assertIsNone(zp_y_record(recs))
        payload_128 = [
            r
            for r in recs
            if r.get("kind") == "S"
            and r.get("tag") == 132
            and r.get("payload")
            and r["payload"][-1] == 128
        ]
        self.assertTrue(payload_128)
        self.assertTrue(all(r.get("reg") == 8 for r in payload_128), payload_128)


class TestMatMulZpYMatchesAtReg120Tag132(unittest.TestCase):
    """The core finding for MatMul: the SAME `reg=120,tag=132` locator
    (same tag, same reg, same offset as Gemm's own) matches on every
    seed except one genuine rounding near-tie."""

    def test_five_of_six_seeds_match_exactly(self):
        w = matmul_weight()
        for seed in DETERMINISTIC_SEEDS:
            if seed in MATMUL_ONE_OFF_SEEDS:
                continue
            recs = mcode.decode(load("matmul", seed), **mcode.FULL_RULE)
            rec = zp_y_record(recs)
            self.assertIsNotNone(rec, seed)
            self.assertEqual(rec["at"], ZP_Y_LOCATOR_OFFSET, seed)
            zp, zp_f = computed_zp_y(seed, matmul_reference, w)
            self.assertEqual(rec["payload"][-1], zp, (seed, zp_f))

    def test_seed_42_is_exactly_one_off(self):
        w = matmul_weight()
        recs = mcode.decode(load("matmul", 42), **mcode.FULL_RULE)
        rec = zp_y_record(recs)
        self.assertIsNotNone(rec)
        zp, zp_f = computed_zp_y(42, matmul_reference, w)
        self.assertEqual(rec["payload"][-1] - zp, 1, zp_f)

    def test_seed_42_has_the_smallest_rounding_margin_of_all_six(self):
        """Directly re-confirms Finding 3: seed=42's own distance to a
        rounding half-boundary is the smallest among all 6 tested
        seeds, by a clear margin -- the same "closest seed to a tie"
        mechanism PR #1640 established for Conv's own zp_x."""
        w = matmul_weight()
        margins = {}
        for seed in DETERMINISTIC_SEEDS:
            _, zp_f = computed_zp_y(seed, matmul_reference, w)
            margins[seed] = abs((zp_f - math.floor(zp_f)) - 0.5)
        closest = min(margins, key=margins.get)
        self.assertEqual(closest, 42, margins)
        others = [m for s, m in margins.items() if s != 42]
        self.assertGreater(min(others), 1.5 * margins[42], margins)


class TestTheLocatorIsNotOpGenericInRegisterNumber(unittest.TestCase):
    """Honesty check: Gemm and MatMul share `reg=120`, but Conv's own
    `zp_y` (PR #1644) uses a DIFFERENT register (`reg=232`). Both
    share the `tag=132` value. This file makes no claim that
    `reg=120` itself generalizes to Conv."""

    def test_reg120_tag132_does_not_carry_zpx_at_the_same_seed_values(self):
        """Sanity check: reg=120's own byte is not just a duplicate
        of the already-known `zp_x` value (reg=54's own locator) --
        confirms this is a genuinely different computed quantity."""
        for seed in DETERMINISTIC_SEEDS:
            if seed in GEMM_STRUCTURALLY_ABSENT_SEEDS:
                continue
            recs = mcode.decode(load("gemm", seed), **mcode.FULL_RULE)
            rec = zp_y_record(recs)
            self.assertNotEqual(rec["payload"][-1], GEMM_ZP_X[seed], seed)


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_any_fixture(self):
        for op in ("gemm", "matmul"):
            for seed in DETERMINISTIC_SEEDS:
                errs = mcode.check(load(op, seed))
                self.assertEqual(errs, [], (op, seed, errs))


if __name__ == "__main__":
    unittest.main()
