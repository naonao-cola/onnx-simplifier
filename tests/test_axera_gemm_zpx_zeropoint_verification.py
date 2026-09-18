"""Extends `tests/test_axera_conv_reg54_zeropoint_verification.py`
(PR #1640)'s first-principles recomputation technique -- port the real
documented C++ formula `ComputeAsymmetricUint8QuantParams`
(`onnxsim/passes/static_quantize_matmul.h`) into Python, apply it to
the exact `RandomState(seed)`-derived calibration data a build script
used, and compare directly against real decoded mcode bytes -- to
Gemm, which PR #1643's own diff explicitly flagged as a plausible next
candidate.

`tests/test_axera_zpx_generalizes.py` (PR #1497) already established
that Gemm's zero point *can* appear as the same literal 6-byte quad
Mul/Conv use (``02 10 1b <zp_x> 83 36``), but only checked two
hardcoded values (128, 32) for absence -- it never recomputed a real
expected value from seeded calibration data and looked for a match.
This file does exactly that.

## Fixtures

Six fresh `Gemm(A[1,16], B[16,16] const, C[16] const)` builds
(`transB=0`), matching `gemm_1x16x16_binary_cluster_r*`'s own
shape/weight convention exactly (`tests/test_axera_gemm_binary_cluster_search_small_shape.py`,
PR #1596): weight/bias fixed at `RandomState(0)` (`w = (rng.randn(16,
16) * 0.1)`, `b = (rng.randn(16) * 0.1)`, the same construction
`_gemm_model` in `tests/test_axera_mcode_structure.py` uses), only the
input `x`'s own calibration data varies, generated identically to
`_build_and_get_mcode_bytes`'s own convention
(`RandomState(seed).randn(1, 16)` x4 samples, `calibration_method:
MinMax`). Real `docker run pulsar2:7.0-lite build` invocations (no
extension available in this worktree, so the same standalone
Docker-wrapper bootstrap prior sibling PRs in this session already
used -- `pulsar2_docker.build()` called directly with a hand-written
config, not through the full onnxsim CLI). `mcode.check()` reports zero
errors on all six.

## Result: exact, unique match on all 6 tested seeds -- including seed=0

Unlike Conv's own `reg=54`/`reg=60` (PR #1640, which found a genuine
seed=0-only Group A/Group B split from a rounding near-tie), Gemm's own
`zp_x` literal quad shows **no split at all**: seed=0 lands on the same
single, exact value every one of 3 independent decode checks confirmed
(no cross-rebuild non-determinism was observed here, unlike Conv's
28-byte binary cluster -- this file makes no claim about whether that
kind of non-determinism could ever occur for Gemm's literal-quad
mechanism, only that it did not appear in these 6 single-build
samples).

| seed | computed `zp` | observed literal-quad byte | offset |
| --- | --- | --- | --- |
| 0 | 135 | 135 | 1421 |
| 1 | 133 | 133 | 1421 |
| 7 | 129 | 129 | 1421 |
| 42 | 131 | 131 | 1421 |
| 100 | 132 | 132 | 1421 |
| 999 | 126 | 126 | 1421 |

Every seed lands at the exact same byte offset (1421) -- the literal
quad's own position in the stream does not move with `zp_x`'s value,
matching the "fixed 6-byte quad" framing `test_axera_zpx_generalizes.py`
already established. A specificity check (search for the literal quad
at `zp_x - 2` through `zp_x + 2` for each seed) finds hits **only** at
the exact computed value, zero exceptions -- ruling out the literal
quad being some unrelated common byte pattern that happens to overlap
by coincidence.

## What this establishes, precisely, and what it does not

**Established**: the first-principles quantization-recomputation
technique (PR #1636/#1640/#1641/#1642/#1643) generalizes cleanly beyond
Conv to Gemm, via the already-known literal-quad locator -- Gemm's own
real, calibration-derived input zero point is directly readable from
raw mcode bytes with no ambiguity, for a small `Gemm(1,16,16)` shape,
across 6 different calibration seeds.

**NOT established**: whether Gemm's `1/x_scale` or `y_scale` have their
own readable byte-level fields anywhere in the stream (Gemm does not
have Conv's 28-byte binary cluster --
`tests/test_axera_gemm_binary_cluster_search_small_shape.py`, PR #1596,
already found a clean negative for that specific mechanism at this same
shape -- so those two fields, if present at all, would need to be
found via a different locator, not attempted here); whether this same
literal-quad recomputation holds at Gemm's other already-studied larger
shapes (`K=512,N=1000`, whose own record-count instability PR #1592/
#1596 already documented as unfriendly to fixed-offset methods, though
the literal-quad byte pattern itself is a raw substring search, not an
offset-based decode, so it may not actually be affected by that
instability -- not tested here); or MatMul, which
`test_axera_zpx_generalizes.py` already closed for a different reason
(input zero points forced to 0, no literal quad ever fires).
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

FIX = os.path.join(_AXERA_DIR, "fixtures")

import mcode  # noqa: E402

DETERMINISTIC_SEEDS = {0: 135, 1: 133, 7: 129, 42: 131, 100: 132, 999: 126}
EXPECTED_OFFSET = 1421

_CONST_PREFIX = bytes.fromhex("02101b")
_CONST_SUFFIX = bytes.fromhex("8336")


def load(seed):
    path = os.path.join(FIX, f"gemm_1x16x16_zpxseed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def literal_quad(zpx):
    return _CONST_PREFIX + bytes([zpx]) + _CONST_SUFFIX


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


def compute_asymmetric_uint8_zero_point(seed, shape=(1, 16), n_samples=4):
    """Reproduces `ComputeAsymmetricUint8QuantParams`
    (`onnxsim/passes/static_quantize_matmul.h`) exactly, in float32
    arithmetic, against the same `RandomState(seed)`-derived
    calibration data this file's own fixture-build script used for
    Gemm's `x` input (shape `[1, 16]`, 4 samples -- matching
    `_build_and_get_mcode_bytes`'s own convention in
    tests/test_axera_mcode_structure.py). Returns
    `(zero_point_int, zp_f_pre_round, scale)`.
    """
    rng = np.random.RandomState(seed)
    samples = [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zp_f = float(np.float32(-lo / scale))
    # std::round is round-half-away-from-zero, not Python's
    # round-half-to-even.
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    zp = max(0, min(255, zp))
    return zp, zp_f, float(scale)


class TestFormulaMatchesEverySeedExactlyAtTheLiteralQuadLocator(unittest.TestCase):
    """The core finding: the standard MinMax asymmetric-uint8 zero-point
    formula, applied to the exact calibration data this file's own
    fixtures were built with, reproduces the literal-quad byte's real
    observed value exactly for all 6 tested seeds -- including seed=0,
    unlike Conv's own split behavior."""

    def test_all_six_seeds_match_exactly(self):
        for seed, expected in DETERMINISTIC_SEEDS.items():
            data = load(seed)
            zp, zp_f, _ = compute_asymmetric_uint8_zero_point(seed)
            self.assertEqual(zp, expected, (seed, zp_f))
            found = hits(data, literal_quad(zp))
            self.assertEqual(found, [EXPECTED_OFFSET], (seed, zp))

    def test_every_seed_lands_at_the_same_fixed_offset(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            zp, _, _ = compute_asymmetric_uint8_zero_point(seed)
            self.assertEqual(hits(data, literal_quad(zp)), [EXPECTED_OFFSET], seed)


class TestTheMatchIsUniqueNotACoincidentalByteRun(unittest.TestCase):
    """Specificity check: searching for the literal quad at nearby
    (but wrong) zero-point values finds nothing, on every seed -- the
    match is not some unrelated common byte pattern that happens to
    overlap by coincidence."""

    def test_off_by_one_and_two_values_have_no_hits(self):
        for seed, zp in DETERMINISTIC_SEEDS.items():
            data = load(seed)
            for delta in (-2, -1, 1, 2):
                candidate = zp + delta
                if 0 <= candidate <= 255:
                    self.assertEqual(
                        hits(data, literal_quad(candidate)),
                        [],
                        (seed, zp, delta),
                    )


class TestFixturesDecodeCleanly(unittest.TestCase):
    """Sanity check on the fixtures themselves: real, well-formed
    `pulsar2 build` output, not corrupted or truncated files."""

    def test_no_decode_errors_on_any_seed(self):
        for seed in DETERMINISTIC_SEEDS:
            data = load(seed)
            errs = mcode.check(data)
            self.assertEqual(errs, [], (seed, errs))


if __name__ == "__main__":
    unittest.main()
