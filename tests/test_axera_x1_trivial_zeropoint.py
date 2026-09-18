"""Closes the one item `tests/test_axera_add_trivial_x2_zeropoint.py`
(PR #1667), `tests/test_axera_two_live_input_operand_order_synthesis.py`
(PR #1668), and `tests/test_axera_sub_mul_trivial_x2_zeropoint.py` (PR
#1669) have all repeatedly flagged as open: whether `x1` (not just
`x2`) shows the same "trivially-zero calibrated zero point gets no
stored record" suppression -- `x1`'s own zero point lives at an
architecturally DIFFERENT locator (the literal 6-byte quad
`02 10 1b <zp> 83 36`, a raw byte-substring pattern, not a decode
record like `x2`'s own `reg=94,tag=132`), so this needed its own
separate test rather than an assumption that it behaves the same way.

## Fixtures

Three fresh `Add(x1[1,16], x2[1,16])` builds, `x1` drawn from PR
#1663's own all-positive design (`2.0 + 0.3*randn()`, no sign flip --
the exact distribution that reliably produces a degenerate zero point
of exactly `0`), `x2` drawn from the plain `RandomState(seed2).
randn(1,16)` convention every op in this cluster uses -- the MIRROR
IMAGE of PR #1667's own fixture design (which made `x2` trivial and
kept `x1` plain; this file makes `x1` trivial and keeps `x2` plain).
Same seed pairs as the rest of the cluster (`(1,2)`, `(7,42)`,
`(100,999)`). Real `docker run pulsar2:7.0-lite build` invocations via
`pulsar2_docker.build()`, two separate `input_configs` entries.
`mcode.check()` reports zero errors on all three.

## Finding: the phenomenon generalizes to `x1`'s own literal-quad
## locator too -- a raw byte-substring search finds ZERO hits, not
## just the trailing byte value being trivially `0`

Directly re-decoded and recomputed: `x1`'s own zero point is exactly
`0` on all three seed pairs (identical calibration math to `x2`'s own
Finding 0 in PR #1663/#1667/#1669, independent of which input position
the tensor occupies). Searching for the literal 6-byte quad
`02 10 1b 00 83 36` (the trivial byte value `0` baked into what's
otherwise the same pattern PR #1657 established) finds ZERO hits in
any of the three fixtures -- not a decode-record absence like `x2`'s
own `reg=94,tag=132` case, but a raw substring search finding nothing.
A control check against the CLUSTER's own already-committed,
non-degenerate `Add` fixtures (`add_1x16_two_live_seed*.mcode.gz`, PR
#1657, ordinary `RandomState(seed1).randn(1,16)` for `x1`) confirms
the quad IS present there (exactly one hit each, at the already-known
`zp1` value for that seed) -- ruling out "the quad pattern's own
trailing-byte-zero case is just rare/unlucky" as an alternative
explanation; the suppression is a real, structural absence.

This means BOTH of the two-live-input cluster's own input-side zero-
point locators -- `x1`'s literal-quad and `x2`'s `reg=94,tag=132`
decode record -- independently suppress their own stored value when
that value is trivially `0`, despite being architecturally distinct
encoding mechanisms (a raw byte pattern vs. a decoded record). This is
consistent with the suppression being a property of the underlying
VALUE `0` inside Pulsar2's own encoder, applied uniformly regardless
of which specific wire format a given field happens to use -- not an
accident of one particular locator's own implementation.

## What this establishes, precisely, and what it does not

**Established**: `x1`'s own zero point shows the identical trivial-
zero suppression `x2`'s own `reg=94,tag=132` locator already showed
for all four ops (PRs #1663/#1667/#1669) -- confirmed here for `Add`
specifically, at `x1`'s own architecturally distinct literal-quad
locator, with a proper control-fixture comparison.

**NOT established**: whether this also holds for `x1` at `Sub`/`Mul`/
`Div` (not tested here -- plausible given the identical mechanism now
confirmed at both of the cluster's own locator types, but not
independently verified for `x1` at those three ops); whether this
holds at shapes other than `(1,16)`; whether BOTH `x1` and `x2` being
simultaneously trivial in the same fixture produces any different
behavior than either alone (not tested -- each prior fixture in this
whole trivial-zero-point thread has only ever made ONE of the two
inputs trivial at a time).
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

SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]


def load_trivial(seed1, seed2):
    path = os.path.join(
        FIX, f"add_1x16_two_live_seed{seed1}_{seed2}_trivialx1.mcode.gz"
    )
    with gzip.open(path, "rb") as f:
        return f.read()


def load_control(seed1, seed2):
    """The already-committed, non-degenerate `Add` fixtures (PR #1657)
    -- ordinary `randn` `x1`, used here only as a locator-presence
    control."""
    path = os.path.join(FIX, f"add_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def x1_trivial_samples(seed1, shape=(1, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed1).randn(1,16)` x4 -- PR #1663's own
    all-positive design, reused here unchanged but applied to `x1`
    instead of `x2`."""
    rng = np.random.RandomState(seed1)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """Plain `RandomState(seed).randn(1,16)` x4 -- the cluster's own
    baseline convention, used here for the control fixtures' `x1`."""
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
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load_trivial(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


class TestX1sOwnZeroPointIsTriviallyZero(unittest.TestCase):
    def test_zp1_is_zero_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            samples = x1_trivial_samples(seed1)
            zp1, zp1_f, _ = asymmetric_uint8_quant_params(samples)
            self.assertEqual(zp1, 0, (seed1, seed2, zp1_f))


class TestLiteralQuadLocatorIsAbsentForX1Too(unittest.TestCase):
    """The headline finding: the same trivial-zero-point suppression
    PR #1663/#1667/#1669 found for `x2`'s own `reg=94,tag=132` decode
    record reproduces for `x1`'s own architecturally distinct literal-
    quad raw byte pattern too."""

    def test_no_literal_quad_exists_in_any_trivial_x1_fixture(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load_trivial(seed1, seed2)
            quad = bytes.fromhex("02101b00" + "8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(hits, [], (seed1, seed2))

    def test_control_fixtures_do_carry_the_literal_quad(self):
        """Rules out "the trailing-byte-zero case is just rare" as an
        alternative explanation: the cluster's own already-committed,
        non-degenerate `Add` fixtures (ordinary `randn` `x1`, PR
        #1657) DO carry exactly one literal-quad hit each, at their
        own already-established `zp1` value."""
        for seed1, seed2 in SEED_PAIRS:
            data = load_control(seed1, seed2)
            zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(seed1))
            quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(len(hits), 1, (seed1, seed2, zp1))


if __name__ == "__main__":
    unittest.main()
