"""Extends `tests/test_axera_div_quant_fields_two_live_inputs.py` (PR
#1663)'s own Finding 0 -- an all-positive `x2` drives its own
calibrated zero point to a degenerate `0` (`ComputeAsymmetricUint8
QuantParams`'s own `lo = min(0, samples.min())` clamps to `0` for any
strictly-positive tensor), which suppresses the `reg=94,tag=132`
locator this whole two-live-input cluster relies on for `x2`'s own
zero point -- to `Add`, closing the one item `tests/test_axera_two_
live_input_four_op_synthesis.py` (PR #1664) left explicitly open:
"Whether Finding 0's own second half (the real encoder's omission
behavior, not just the degenerate-zero-point calibration math) also
applies to Add/Sub/Mul's own `x2` -- would require a new Docker build
with a deliberately all-positive `x2` for one of them, not attempted
here."

## Fixtures

Three fresh `Add(x1[1,16], x2[1,16])` builds, `x1` drawn from the
plain `RandomState(seed1).randn(1,16)` convention every op in this
cluster uses (PR #1657), `x2` drawn from PR #1663's own all-positive
first-attempt design (`2.0 + 0.3*randn()`, no sign flip) -- the exact
distribution that reliably produced a degenerate `zp2=0` for `Div`.
Same seed pairs as the rest of the cluster (`(1,2)`, `(7,42)`,
`(100,999)`). Real `docker run pulsar2:7.0-lite build` invocations via
`pulsar2_docker.build()`, two separate `input_configs` entries (one
per tensor), matching every prior fixture in this cluster's own build
convention. `mcode.check()` reports zero errors on all three.

## Finding: the phenomenon generalizes cleanly to `Add` -- it is a
## real, op-independent encoder behavior tied to the VALUE being
## trivially `0`, not anything Div-specific

Directly re-decoded and recomputed: `x2`'s own zero point is exactly
`0` on all three seed pairs (identical calibration math to Div's own
Finding 0, independent of which op consumes the tensor), and the
`reg=94,tag=132` locator is entirely absent from all three decoded
streams -- not zero-valued, structurally missing, the same class of
absence PR #1663 found for Div. A control check against the CLUSTER's
own already-committed, non-degenerate `Add` fixtures
(`add_1x16_two_live_seed*.mcode.gz`, PR #1657, ordinary
`RandomState(seed2).randn(1,16)` for `x2`) confirms the locator IS
present there (exactly one `reg=94,tag=132` hit per fixture) --
ruling out "the locator is just generally unreliable for Add" as an
alternative explanation. `reg=94` itself still fires at other tags
(`130`, `159`) in the trivial-x2 fixtures, exactly matching Div's own
observation that `reg=94` isn't globally absent, only its
`tag=132`/zero-point-carrying copy.

## What this establishes, precisely, and what it does not

**Established**: the "a live input's own trivially-zero calibrated
zero point gets no stored `reg=94,tag=132` record at all" phenomenon
is not Div-specific -- it reproduces identically for `Add`, confirming
it is a genuine, op-independent property of the real Pulsar2 encoder
tied to the VALUE `0` itself (consistent with the same "degenerate
value gets a trivial/absent encoding" pattern `scripts/axera/
README.md`'s own "Differential analysis" section already documents
for compile-time CONSTANT operands, now confirmed twice for a LIVE
input's own zero point too).

**NOT established**: whether this also holds for `Sub`/`Mul` (not
tested here -- plausible given the identical mechanism now confirmed
for both `Add` and `Div`, but not independently verified for those two
specific ops); whether `x1` (not just `x2`) shows the same suppression
if IT were made trivially zero (the literal-quad `zp_x` locator, not
`reg=94,tag=132`, carries `x1`'s own zero point -- an architecturally
different locator this file does not test); whether this holds at
shapes other than `(1,16)`.
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
        FIX, f"add_1x16_two_live_seed{seed1}_{seed2}_trivialx2.mcode.gz"
    )
    with gzip.open(path, "rb") as f:
        return f.read()


def load_control(seed1, seed2):
    """The already-committed, non-degenerate `Add` fixtures (PR #1657)
    -- same tensor `x1` construction, ordinary (not all-positive) `x2`
    calibration, used here only as a locator-presence control."""
    path = os.path.join(FIX, f"add_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def x2_trivial_samples(seed2, shape=(1, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed2).randn(1,16)` x4 -- the exact
    all-positive design `tests/test_axera_div_quant_fields_two_live_
    inputs.py` (PR #1663) established as reliably producing a
    degenerate `zp2=0`, reused here unchanged for `x2`."""
    rng = np.random.RandomState(seed2)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


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


class TestX2sOwnZeroPointIsTriviallyZero(unittest.TestCase):
    def test_zp2_is_zero_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            samples = x2_trivial_samples(seed2)
            zp2, zp2_f, _ = asymmetric_uint8_quant_params(samples)
            self.assertEqual(zp2, 0, (seed1, seed2, zp2_f))


class TestReg94Tag132LocatorIsAbsentForAddToo(unittest.TestCase):
    """The headline finding: the same phenomenon PR #1663 found for
    `Div` reproduces identically for `Add` -- a real, op-independent
    encoder behavior, not something specific to `Div`."""

    def test_no_reg94_tag132_record_exists_in_any_trivial_add_fixture(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = mcode.decode(load_trivial(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(hits, [], (seed1, seed2))

    def test_reg94_still_fires_at_other_tags_not_globally_absent(self):
        """Confirms `reg=94` itself isn't simply missing from the
        stream -- only its `tag=132`/zero-point-carrying copy is,
        matching PR #1663's own identical observation for Div."""
        for seed1, seed2 in SEED_PAIRS:
            recs = mcode.decode(load_trivial(seed1, seed2), **mcode.FULL_RULE)
            reg94_all = [r for r in recs if r.get("kind") == "S" and r.get("reg") == 94]
            tags = {r.get("tag") for r in reg94_all}
            self.assertNotIn(132, tags, (seed1, seed2, tags))
            self.assertTrue(tags, (seed1, seed2))

    def test_control_fixtures_do_carry_the_locator(self):
        """Rules out "the locator is just generally unreliable for
        Add" as an alternative explanation: the cluster's own
        already-committed, non-degenerate `Add` fixtures (ordinary
        `randn` `x2`, PR #1657) DO carry exactly one `reg=94,tag=132`
        record each."""
        for seed1, seed2 in SEED_PAIRS:
            recs = mcode.decode(load_control(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2, hits))


if __name__ == "__main__":
    unittest.main()
