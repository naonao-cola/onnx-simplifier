"""Extends `tests/test_axera_gemm_zpx_zeropoint_verification.py` (PR
#1645)'s own "NOT established" list: whether Gemm's `1/x_scale` and
`y_scale` have their own readable byte-level fields, given that Gemm
lacks Conv's 28-byte binary cluster
(`tests/test_axera_gemm_binary_cluster_search_small_shape.py`, PR
#1596). Reuses the same 6 `Gemm(1,16,16)` fixtures PR #1645 already
built (`scripts/axera/fixtures/gemm_1x16x16_zpxseed*.mcode.gz`,
duplicated here since this branch predates that PR's merge), and the
same `ComputeAsymmetricUint8QuantParams` formula
(`onnxsim/passes/static_quantize_matmul.h`) this whole arc has used
since PR #1636/#1640.

## Finding 1: `1/x_scale` is bit-exact, at the SAME `verb=161,bank=15`
## locator Conv already uses

Unlike `zp_x`'s literal-quad search, this field decodes cleanly through
`mcode.decode()`: three `V`-kind records with `verb=161, field in
{96,112,128}, bank=15`, all three carrying the identical little-endian
float32 operand, at fixed offsets 1456/1464/1472 in every one of the 6
tested seeds. This is the exact same `verb=161,bank=15` mechanism
`tests/test_axera_conv_verb161_bank15_verification.py` (PR #1641)
already found for Conv's own `1/x_scale` -- the locator itself
generalizes across ops, not just the technique.

| seed | computed `1/x_scale` | observed (verb=161,bank=15) |
| --- | --- | --- |
| 0 | 52.874458 | 52.874458 |
| 1 | 57.930927 | 57.930927 |
| 7 | 56.218044 | 56.218044 |
| 42 | 66.894928 | 66.894928 |
| 100 | 75.695793 | 75.695793 |
| 999 | 60.406139 | 60.406139 |

Bit-exact (`struct.pack`-identical float32) on all 6 seeds, three
redundant copies each.

## Finding 2: `y_scale` is close but not bit-exact, in a partially
## undecoded raw region -- the same shape of gap PR #1642 found for
## Conv's `reg=224`

A hand-written float64-accumulating reference `Gemm` (`A @ B + C`)
applied to the same calibration data locates a float32 value near
offset 1636 (shifting by up to 1 byte per seed, since the bytes
immediately before it are not fully accounted for by `mcode.decode()`
even under `FULL_RULE` -- three of its four apparent copies fall
inside a clean `S`-kind unit with `tag=129, reg=112`, the fourth inside
a short undecoded `raw` run). The value there is consistently close to
the recomputed `y_scale`, never bit-exact:

| seed | computed `y_scale` | observed (nearest raw hit) | rel. error |
| --- | --- | --- | --- |
| 0 | 0.00768605 | 0.00768638 | 0.0042% |
| 1 | 0.00696061 | 0.00695180 | 0.1265% |
| 7 | 0.00754769 | 0.00754228 | 0.0718% |
| 42 | 0.00678417 | 0.00678822 | 0.0598% |
| 100 | 0.00705196 | 0.00706662 | 0.2078% |
| 999 | 0.00899960 | 0.00899642 | 0.0354% |

Every seed's error is under 0.21%, well inside the same "reference
summation order differs from onnxsim's real implementation" band PR
#1642 documented for Conv's `reg=224` (0.03%-0.12% there; a slightly
wider 0.004%-0.21% here, plausibly because this file's reference `Gemm`
uses a single `@` reduction rather than Conv's nested-loop
accumulation, a different rounding path). No claim of bit-exactness is
made or tested.

## What this establishes, precisely, and what it does not

**Established**: the `verb=161,bank=15` locator for `1/x_scale` is not
Conv-specific -- it reproduces bit-exact, at a fixed offset, across
all 6 tested Gemm seeds, closing half of PR #1645's own open item.

**Established, with a real but non-bit-exact tolerance**: a `y_scale`-
adjacent float32 value exists near offset 1636, consistently within
0.21% of the recomputed value on the same 6 seeds -- evidence for the
field's presence, not a byte-exact decode.

**NOT established**: a clean decode-record locator for `y_scale` (it
sits partly in a raw/undecoded byte run even under `mcode.FULL_RULE`,
unlike `1/x_scale`'s clean three-record locator); why this file's own
hand-written `Gemm` reference has a wider, seed-dependent error band
than PR #1642's Conv reference did; whether either field's offset
stays fixed at Gemm shapes other than `(1,16,16)`, or at MatMul (already
closed by PR #1645/`test_axera_zpx_generalizes.py` for zero points, but
not attempted here for scale fields, since MatMul's own zero point is
always forced to 0 and its float-field layout has not been surveyed).
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

DETERMINISTIC_SEEDS = [0, 1, 7, 42, 100, 999]
RECIP_OFFSETS = (1456, 1464, 1472)
Y_SCALE_SEARCH_WINDOW = (1600, 1670)
Y_SCALE_MAX_REL_ERR = 3e-3


def load(seed):
    path = os.path.join(FIX, f"gemm_1x16x16_zpxseed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data `tests/test_axera_gemm_zpx_zeropoint_verification.py`'s own
    fixture-build script used for Gemm's `x` input."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def gemm_weight():
    """Gemm's own `B`/`C` -- fixed at `RandomState(0)` regardless of
    calibration seed (`w = (rng.randn(16, 16) * 0.1)`, `b =
    (rng.randn(16) * 0.1)`), matching PR #1645's own fixture-build
    convention exactly."""
    rng = np.random.RandomState(0)
    w = (rng.randn(16, 16) * 0.1).astype(np.float32)
    b = (rng.randn(16) * 0.1).astype(np.float32)
    return w, b


def asymmetric_uint8_scale(samples):
    """`ComputeAsymmetricUint8QuantParams`'s own scale half
    (`onnxsim/passes/static_quantize_matmul.h`), applied to Gemm's `x`
    input range."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return (hi - lo) / np.float32(255.0)


def gemm_reference(x, w, b):
    """A float64-accumulating reference `Gemm` (`transB=0`: `y = x @ w
    + b`) -- NOT claimed to be bit-identical to whatever reference
    implementation onnxsim's own calibration pipeline actually uses
    internally (see this file's own module docstring for why Finding
    2's own match is close but not exact)."""
    return (x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)).astype(
        np.float32
    )


def computed_recip_x_scale(seed):
    samples = calib_samples(seed)
    xs = asymmetric_uint8_scale(samples)
    return float(np.float32(1.0) / xs)


def computed_y_scale(seed):
    samples = calib_samples(seed)
    w, b = gemm_weight()
    ys_all = [gemm_reference(s, w, b) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    return float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))


def nearest_float32_hit(data, target, lo, hi, max_rel_err):
    """Scans every byte offset in `[lo, hi)` (not just 4-byte-aligned
    ones -- the codec is a variable-length byte stream, so a field's
    offset can shift by a byte between builds) for the float32 value
    closest to `target`, returning `(offset, value)` or `None` if
    nothing is within `max_rel_err`."""
    best = None
    for i in range(lo, min(hi, len(data) - 3)):
        val = struct.unpack_from("<f", data, i)[0]
        if not math.isfinite(val) or target == 0:
            continue
        rel_err = abs(val - target) / abs(target)
        if rel_err < max_rel_err and (best is None or rel_err < best[2]):
            best = (i, val, rel_err)
    return best


class TestRecipXScaleMatchesTheSameVerb161Bank15LocatorAsConv(unittest.TestCase):
    """The core finding: Gemm's `1/x_scale` is bit-exact at the exact
    same `verb=161,bank=15` locator PR #1641 already found for Conv --
    the locator generalizes across ops, not just the underlying
    technique."""

    def test_all_six_seeds_match_exactly_at_fixed_offsets(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            recip = computed_recip_x_scale(seed)
            recs = mcode.decode(data)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("verb") == 161 and r.get("bank") == 15
            ]
            self.assertEqual(
                sorted(r["at"] for r in hits), list(RECIP_OFFSETS), (seed, hits)
            )
            for r in hits:
                observed = struct.unpack("<f", r["operand"])[0]
                self.assertEqual(observed, recip, (seed, r["at"]))


class TestYScaleIsCloseButNotBitExactNearOffset1636(unittest.TestCase):
    """Directly re-confirms the module docstring's Finding 2: a
    float32 value near offset 1636 tracks the recomputed `y_scale`
    within a few tenths of a percent on every seed, and this file makes
    no bit-exactness claim -- it asserts the gap explicitly stays
    nonzero on at least one seed, the same honesty pattern PR #1642
    used for Conv's own `reg=224`."""

    def test_every_seed_has_a_close_match_in_the_search_window(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            ys = computed_y_scale(seed)
            hit = nearest_float32_hit(
                data, ys, *Y_SCALE_SEARCH_WINDOW, Y_SCALE_MAX_REL_ERR
            )
            self.assertIsNotNone(hit, (seed, ys))
            self.assertLess(hit[2], Y_SCALE_MAX_REL_ERR, (seed, hit))

    def test_the_match_is_not_bit_exact_on_at_least_one_seed(self):
        worst = 0.0
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            ys = computed_y_scale(seed)
            hit = nearest_float32_hit(
                data, ys, *Y_SCALE_SEARCH_WINDOW, Y_SCALE_MAX_REL_ERR
            )
            worst = max(worst, hit[2])
        self.assertGreater(worst, 1e-5, worst)

    def test_weight_scale_alone_does_not_explain_y_scale(self):
        w, _ = gemm_weight()
        w_scale = float(np.abs(w).max() / 127.0)
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            hit = nearest_float32_hit(
                data, w_scale, *Y_SCALE_SEARCH_WINDOW, Y_SCALE_MAX_REL_ERR
            )
            self.assertIsNone(hit, (seed, w_scale))


class TestFixturesDecodeCleanly(unittest.TestCase):
    """Sanity check on the (duplicated) fixtures themselves."""

    def test_no_decode_errors_on_any_seed(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            errs = mcode.check(data)
            self.assertEqual(errs, [], (seed, errs))


if __name__ == "__main__":
    unittest.main()
