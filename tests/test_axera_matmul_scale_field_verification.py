"""Extends `tests/test_axera_gemm_scale_field_verification.py` (PR
#1646)'s own "NOT established" list: whether MatMul's `1/x_scale` and
`y_scale` have their own readable byte-level fields, given that
`tests/test_axera_zpx_generalizes.py` (PR #1497) already closed
MatMul's zero point (always forced to 0, so the literal-quad locator
`zp_x`/`zp_y` use never fires there) without ever surveying MatMul's
own scale fields. Reuses the same first-principles technique this
whole arc has used since PR #1636/#1640: port
`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_quantize_
matmul.h`) into Python, apply it to real seeded calibration data, and
compare against real decoded mcode bytes.

## Fixtures

Six fresh `MatMul(x[1,16], w[16,16] const)` builds -- the closest
possible MatMul analogue of the Gemm shape PR #1645/#1646 used
(`tests/test_axera_bank81_cross_op_check.py`, PR #1608, already
established this "one live tensor + one compile-time-constant weight"
MatMul construction as the correct Gemm-equivalent shape for
cross-op comparisons). Weight fixed at `RandomState(0)` (`w =
(rng.randn(16, 16) * 0.1)`), only `x`'s own calibration data varies
per seed (`RandomState(seed).randn(1, 16)` x4 samples, `calibration_
method: MinMax` -- the same convention `_build_and_get_mcode_bytes`
uses in `tests/test_axera_mcode_structure.py`). Real `docker run
pulsar2:7.0-lite build` invocations, `pulsar2_docker.build()` called
directly with a hand-written config, same as PR #1645/#1646's own
build process. `mcode.check()` reports zero hard errors on all six.

## Finding 1: `1/x_scale` is bit-exact, at the IDENTICAL
## `verb=161,bank=15` locator AND the IDENTICAL byte offsets Gemm uses

Three `V`-kind records with `verb=161, field in {96,112,128}, bank=15`,
all three carrying the same little-endian float32 operand, at fixed
offsets 1456/1464/1472 -- not just the same locator mechanism PR #1641
(Conv) and PR #1646 (Gemm) already found, but the exact same byte
offsets PR #1646 reported for Gemm's own `(1,16,16)` shape. Given both
fixtures share the same `x`/`w` shape (`[1,16]` times `[16,16]`) and
weight seed, this is consistent with -- though this file does not
independently verify -- the scheduler laying out an identical stream
prefix for any op producing that shape, not evidence the offset is
shape-invariant across other op/shape combinations.

| seed | computed `1/x_scale` | observed (verb=161,bank=15) |
| --- | --- | --- |
| 0 | 52.874458 | 52.874458 |
| 1 | 57.930927 | 57.930927 |
| 7 | 56.218044 | 56.218044 |
| 42 | 66.894928 | 66.894928 |
| 100 | 75.695793 | 75.695793 |
| 999 | 60.406139 | 60.406139 |

Bit-exact (`struct.pack`-identical float32) on all 6 seeds, three
redundant copies each -- the same precision PR #1641/#1646 already
found for Conv and Gemm's own `1/x_scale`.

## Finding 2: `y_scale` is close but not bit-exact, at the SAME raw
## offset (1636) Gemm's own `y_scale` used

A hand-written float64-accumulating reference `MatMul` (`x @ w`)
applied to the same calibration data locates a float32 value at a
fixed offset (1636 for 5 of 6 seeds; seed=7 needed a +/-1-byte local
window around that same offset, the same shifting behavior PR #1646
documented for Gemm) consistently close to the recomputed `y_scale`,
never bit-exact:

| seed | computed `y_scale` | observed (nearest hit near offset 1636) | rel. error |
| --- | --- | --- | --- |
| 0 | 0.00803036 | 0.00803068 | 0.0040% |
| 1 | 0.00668782 | 0.00667902 | 0.1317% |
| 7 | 0.00788550 | 0.00787023 | 0.1937% |
| 42 | 0.00728506 | 0.00728912 | 0.0557% |
| 100 | 0.00674331 | 0.00675797 | 0.2173% |
| 999 | 0.00881793 | 0.00881475 | 0.0361% |

Every seed's error is under 0.22% -- inside the same "reference
summation order differs from onnxsim's real implementation" band PR
#1642 (Conv) and PR #1646 (Gemm) both already documented. No claim of
bit-exactness is made or tested.

## What this establishes, precisely, and what it does not

**Established**: the `verb=161,bank=15` locator for `1/x_scale`
generalizes to MatMul too, bit-exact, at a fixed offset, across all 6
tested seeds, closing MatMul's own scale-field gap left open by PR
#1646 -- the SAME locator (and, for this particular shape, the same
byte offsets) now confirmed across all three of Conv, Gemm, and
MatMul.

**Established, with a real but non-bit-exact tolerance**: a `y_scale`-
adjacent float32 value exists near offset 1636 (matching Gemm's own
reported offset for the same shape), consistently within 0.22% of the
recomputed value on the same 6 seeds -- evidence for the field's
presence, not a byte-exact decode, the same caveat PR #1642/#1646
already carry for Conv/Gemm.

**NOT established**: a clean decode-record locator for `y_scale` (this
file only confirms a raw-byte-offset match, the same limitation PR
#1646 found for Gemm -- it was not checked here whether the value
falls inside a `mcode.decode()` record the way `1/x_scale`'s does, or
in an undecoded raw run the way Gemm's did); whether either field's
offset stays fixed at MatMul shapes other than `(1,16,16)`, or whether
the offset coincidence with Gemm survives at a different shape (this
file's own `(1,16,16)` shape was chosen specifically to match PR
#1645/#1646's Gemm shape for a clean side-by-side comparison, so an
identical offset is not itself surprising and is not claimed to
generalize); MatMul's own `zp_y` (output zero point) field, not
attempted here since PR #1645's own literal-quad locator is specific
to a real, non-zero-forced zero point and MatMul's output here is
still an asymmetric-uint8 quantized tensor with its own real zero
point that a different locator might expose -- left for a future PR.
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
    path = os.path.join(FIX, f"matmul_1x16x16_zpxseed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data this file's own fixture-build script used for MatMul's `x`
    input, matching `tests/test_axera_gemm_zpx_zeropoint_verification.
    py`'s own convention."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def matmul_weight():
    """MatMul's own constant `w` -- fixed at `RandomState(0)`
    regardless of calibration seed (`w = (rng.randn(16, 16) * 0.1)`),
    the closest MatMul analogue of Gemm's own `B` weight
    (`tests/test_axera_bank81_cross_op_check.py`, PR #1608)."""
    rng = np.random.RandomState(0)
    return (rng.randn(16, 16) * 0.1).astype(np.float32)


def asymmetric_uint8_scale(samples):
    """`ComputeAsymmetricUint8QuantParams`'s own scale half
    (`onnxsim/passes/static_quantize_matmul.h`), applied to MatMul's
    `x` input range."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return (hi - lo) / np.float32(255.0)


def matmul_reference(x, w):
    """A float64-accumulating reference `MatMul` (`y = x @ w`) -- NOT
    claimed to be bit-identical to whatever reference implementation
    onnxsim's own calibration pipeline actually uses internally (see
    this file's own module docstring for why Finding 2's own match is
    close but not exact)."""
    return (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)


def computed_recip_x_scale(seed):
    samples = calib_samples(seed)
    xs = asymmetric_uint8_scale(samples)
    return float(np.float32(1.0) / xs)


def computed_y_scale(seed):
    samples = calib_samples(seed)
    w = matmul_weight()
    ys_all = [matmul_reference(s, w) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    return float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))


def nearest_float32_hit(data, target, lo, hi, max_rel_err):
    """Scans every byte offset in `[lo, hi)` (not just 4-byte-aligned
    ones -- the codec is a variable-length byte stream, so a field's
    offset can shift by a byte between builds) for the float32 value
    closest to `target`, returning `(offset, value, rel_err)` or
    `None` if nothing is within `max_rel_err`."""
    best = None
    for i in range(lo, min(hi, len(data) - 3)):
        val = struct.unpack_from("<f", data, i)[0]
        if not math.isfinite(val) or target == 0:
            continue
        rel_err = abs(val - target) / abs(target)
        if rel_err < max_rel_err and (best is None or rel_err < best[2]):
            best = (i, val, rel_err)
    return best


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_seed(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            errs = [e for e in mcode.check(data) if not e.startswith("coverage:")]
            self.assertEqual(errs, [], (seed, errs))


class TestRecipXScaleMatchesTheSameVerb161Bank15LocatorAsConvAndGemm(unittest.TestCase):
    """The core finding: the same `verb=161,bank=15` locator PR #1641
    (Conv) and PR #1646 (Gemm) already found reproduces bit-exact for
    MatMul too, at the same fixed offsets Gemm's own `(1,16,16)` shape
    used."""

    def test_all_six_seeds_match_exactly_at_fixed_offsets(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            recs = mcode.decode(data, **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V" and r.get("verb") == 161 and r.get("bank") == 15
            ]
            self.assertEqual(sorted(r["at"] for r in hits), list(RECIP_OFFSETS), seed)
            computed = computed_recip_x_scale(seed)
            for r in hits:
                observed = struct.unpack("<f", r["operand"])[0]
                self.assertEqual(observed, computed, (seed, r))


class TestYScaleIsCloseButNotBitExact(unittest.TestCase):
    """Directly re-confirms Finding 2's own honest caveat as a real
    assertion: a close match exists near a fixed offset, but it is not
    bit-exact on at least one seed -- the same shape of gap PR #1642/
    #1646 already documented for Conv/Gemm."""

    def test_all_six_seeds_have_a_close_match_near_the_fixed_offset(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            target = computed_y_scale(seed)
            hit = nearest_float32_hit(
                data, target, *Y_SCALE_SEARCH_WINDOW, Y_SCALE_MAX_REL_ERR
            )
            self.assertIsNotNone(hit, (seed, target))

    def test_the_match_is_not_bit_exact_on_at_least_one_seed(self):
        worst = 0.0
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            target = computed_y_scale(seed)
            hit = nearest_float32_hit(
                data, target, *Y_SCALE_SEARCH_WINDOW, Y_SCALE_MAX_REL_ERR
            )
            worst = max(worst, hit[2])
        self.assertGreater(worst, 1e-5, worst)


if __name__ == "__main__":
    unittest.main()
