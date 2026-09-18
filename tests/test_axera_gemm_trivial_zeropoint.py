"""Answers a cross-arc question no prior PR in either the single-live-
input arc (Conv/Gemm/MatMul, PRs #1636/#1640-#1656) or the two-live-
input arc (Add/Sub/Mul/Div, PRs #1657-#1677) has ever tested: does the
"a live input's own trivially-zero calibrated zero point gets no
stored encoding at all" phenomenon -- first found for the two-live-
input cluster's own `x2` (`reg=94,tag=132`, PR #1663's Finding 0) and
`x1` (the literal-quad locator, PR #1670) -- also apply to the single-
live-input arc's own Gemm `zp_x` literal-quad locator
(`tests/test_axera_gemm_zpx_zeropoint_verification.py`, PR #1645)?

Gemm's own literal-quad locator is architecturally the SAME raw byte
pattern (`02 10 1b <zp_x> 83 36`) PR #1670 already tested for `x1`'s
own version of this locator in the two-live-input cluster -- but
Gemm's own single-live-input construction (`Gemm(A, B_const, C_const)`,
one live tensor plus a compile-time-constant weight/bias, no second
live tensor at all) is a genuinely different graph shape from the two-
live-input `Add(x1,x2)` case PR #1670 tested. This file checks whether
the suppression is a property of the VALUE `0` inside Pulsar2's own
encoder (as PR #1670's own diff already concluded for the two-live-
input case), independent of how many live tensors the surrounding op
has, or whether it is specific to the two-live-input architecture in
some way not yet ruled out.

## Fixtures

Three fresh `Gemm(A[1,16], B[16,16] const, C[16] const, transB=0)`
builds, matching `tests/test_axera_gemm_zpx_zeropoint_verification.py`
(PR #1645)'s own shape/weight convention exactly (`w`/`b` fixed at
`RandomState(0)`), but with `A`'s own calibration data drawn from PR
#1663's own all-positive design (`2.0 + 0.3*randn()`, no sign flip --
the exact distribution that reliably produces a degenerate zero point
of exactly `0`) instead of the plain `randn()` every other Gemm fixture
in this project uses. Real `docker run pulsar2:7.0-lite build`
invocations via `pulsar2_docker.build()`, matching this whole arc's own
established convention. `mcode.check()` reports zero errors on all
three.

## Finding: the phenomenon generalizes across arcs -- it is not
## specific to the two-live-input architecture

`A`'s own zero point is exactly `0` on all three seed pairs (identical
calibration math to the two-live-input cluster's own findings,
independent of which op or graph shape consumes the tensor). Searching
for the literal 6-byte quad `02 10 1b 00 83 36` (the trivial byte value
`0` baked into what's otherwise the same pattern PR #1645 established
for Gemm) finds ZERO hits in any of the three fixtures. A control check
against Gemm's own already-committed, non-degenerate fixtures
(`gemm_1x16x16_zpxseed*.mcode.gz`, PR #1645, ordinary `randn()` for
`A`) confirms the quad IS present there (exactly one hit each, at the
already-known `zp` value for that seed, at the same offset `1421` PR
#1645 already established) -- ruling out "the locator is just
generally unreliable for Gemm" as an alternative explanation.

## What this establishes, precisely, and what it does not

**Established**: the trivial-zero-point encoder-suppression phenomenon
is not specific to the two-live-input elementwise cluster (Add/Sub/
Mul/Div) -- it reproduces identically for Gemm, a single-live-input op
with an entirely different graph shape (one live tensor, one compile-
time-constant weight, no second live tensor at all). This is consistent
with the suppression being a genuine, op-and-architecture-independent
property of Pulsar2's own encoder, tied to the VALUE `0` itself,
confirmed now across both major arcs this project has characterized.

**NOT established**: whether this also holds for Conv's or MatMul's own
zero-point locators (not tested here); whether this holds at shapes
other than `(1,16)`; what happens if BOTH Gemm's own `zp_x` AND `zp_y`
(the output zero point, `reg=120,tag=132`, PR #1655) were simultaneously
trivial in the same fixture (not tested -- this file only makes `A`'s
own input-side zero point trivial, the same "only one field at a time"
scope every prior file in this whole trivial-zero-point thread has
used).
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

SEEDS = [1, 7, 100]

_CONST_PREFIX = bytes.fromhex("02101b")
_CONST_SUFFIX = bytes.fromhex("8336")
EXPECTED_OFFSET = 1421


def load_trivial(seed):
    path = os.path.join(FIX, f"gemm_1x16x16_trivialA_seed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def load_control(seed):
    """The already-committed, non-degenerate Gemm fixtures (PR #1645)
    -- ordinary `randn()` for `A`, used here only as a locator-presence
    control."""
    path = os.path.join(FIX, f"gemm_1x16x16_zpxseed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def trivial_samples(seed, shape=(1, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed).randn(1,16)` x4 -- PR #1663's own
    all-positive design, reused here unchanged for Gemm's own live
    input `A`."""
    rng = np.random.RandomState(seed)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """Plain `RandomState(seed).randn(1,16)` x4 -- Gemm's own baseline
    convention (PR #1645), used here for the control fixtures."""
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


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_trivial_fixture(self):
        for seed in SEEDS:
            errs = mcode.check(load_trivial(seed))
            self.assertEqual(errs, [], (seed, errs))


class TestAsOwnZeroPointIsTriviallyZero(unittest.TestCase):
    def test_zp_is_zero_on_every_seed(self):
        for seed in SEEDS:
            zp, zp_f, _ = asymmetric_uint8_quant_params(trivial_samples(seed))
            self.assertEqual(zp, 0, (seed, zp_f))


class TestLiteralQuadLocatorIsAbsentForGemmToo(unittest.TestCase):
    """The headline, cross-arc finding: the same trivial-zero-point
    suppression PR #1663/#1667/#1669/#1670 found for the two-live-
    input cluster's own locators reproduces for Gemm's own single-
    live-input literal-quad locator too."""

    def test_no_literal_quad_exists_in_any_trivial_fixture(self):
        quad = _CONST_PREFIX + bytes([0]) + _CONST_SUFFIX
        for seed in SEEDS:
            data = load_trivial(seed)
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(hits, [], (seed, hits))

    def test_control_fixtures_do_carry_the_literal_quad_at_the_known_offset(self):
        """Rules out "the locator is just unreliable for Gemm" as an
        alternative explanation: Gemm's own already-committed, non-
        degenerate fixtures (ordinary `randn()`, PR #1645) DO carry
        exactly one literal-quad hit each, at their own already-
        established `zp` value and offset."""
        for seed in SEEDS:
            data = load_control(seed)
            zp, _, _ = asymmetric_uint8_quant_params(calib_samples(seed))
            quad = _CONST_PREFIX + bytes([zp]) + _CONST_SUFFIX
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(hits, [EXPECTED_OFFSET], (seed, zp, hits))


if __name__ == "__main__":
    unittest.main()
