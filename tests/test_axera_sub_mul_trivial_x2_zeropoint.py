"""Extends `tests/test_axera_add_trivial_x2_zeropoint.py` (PR #1667)'s
own "NOT established" list: whether the "trivially-zero calibrated
zero point gets no stored `reg=94,tag=132` record" phenomenon (first
found for `Div`, PR #1663's Finding 0; confirmed op-independent for
`Add`, PR #1667) also holds for `Sub` and `Mul` -- the two remaining
untested ops in the four-op two-live-input cluster. This file confirms
it does, for both.

## Fixtures

Six fresh builds (`Sub(x1,x2)` x3 seed pairs, `Mul(x1,x2)` x3 seed
pairs), `x1` drawn from the plain `RandomState(seed1).randn(1,16)`
convention every op in this cluster uses, `x2` drawn from PR #1663's
own all-positive design (`2.0 + 0.3*randn()`, no sign flip) -- the
exact distribution that reliably produced a degenerate `zp2=0` for
both `Div` and `Add`. Same seed pairs as the rest of the cluster
(`(1,2)`, `(7,42)`, `(100,999)`). Real `docker run pulsar2:7.0-lite
build` invocations via `pulsar2_docker.build()`, two separate
`input_configs` entries (one per tensor), matching every prior fixture
in this cluster's own build convention. `mcode.check()` reports zero
errors on all six.

## Finding: the phenomenon generalizes cleanly to Sub AND Mul --
## confirming it for all FOUR ops in the cluster

Directly re-decoded and recomputed: `x2`'s own zero point is exactly
`0` on all three seed pairs for both ops (identical calibration math,
independent of which op consumes the tensor -- the formula never
looks at the op at all), and the `reg=94,tag=132` locator is entirely
absent from every one of the 6 decoded streams -- not zero-valued,
structurally missing, the same class of absence PR #1663/#1667 already
found for Div/Add. A control check against each op's own
already-committed, non-degenerate fixtures (`sub_1x16_two_live_seed*`/
`mul_1x16_two_live_seed*.mcode.gz`, PRs #1661/#1659, ordinary
`RandomState(seed2).randn(1,16)` for `x2`) confirms the locator IS
present there (exactly one hit each) -- ruling out "the locator is
just generally unreliable" for either op. `reg=94` itself still fires
at other tags (`130`, `159`) in every trivial-x2 fixture, exactly
matching Div's and Add's own observation that `reg=94` isn't globally
absent, only its `tag=132`/zero-point-carrying copy.

## What this establishes, precisely, and what it does not

**Established**: the "a live input's own trivially-zero calibrated
zero point gets no stored `reg=94,tag=132` record at all" phenomenon
is confirmed for all FOUR ops in the two-live-input cluster (Div
PR #1663, Add PR #1667, Sub and Mul here) -- a genuine, fully
op-independent property of the real Pulsar2 encoder tied to the VALUE
`0` itself, not to any one op's own arithmetic.

**NOT established**: whether `x1` (not just `x2`) shows the same
suppression if IT were made trivially zero (the literal-quad `zp_x`
locator, not `reg=94,tag=132`, carries `x1`'s own zero point -- an
architecturally different locator this file does not test, the same
open item PR #1667 already flagged); whether this holds at shapes
other than `(1,16)`.
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
OPS = ["sub", "mul"]


def load_trivial(op, seed1, seed2):
    path = os.path.join(
        FIX, f"{op}_1x16_two_live_seed{seed1}_{seed2}_trivialx2.mcode.gz"
    )
    with gzip.open(path, "rb") as f:
        return f.read()


def load_control(op, seed1, seed2):
    """The already-committed, non-degenerate fixtures (PR #1659/#1661)
    -- same tensor `x1` construction, ordinary (not all-positive) `x2`
    calibration, used here only as a locator-presence control."""
    path = os.path.join(FIX, f"{op}_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def x2_trivial_samples(seed2, shape=(1, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed2).randn(1,16)` x4 -- the exact
    all-positive design PR #1663/#1667 both established as reliably
    producing a degenerate `zp2=0`, reused here unchanged."""
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
        for op in OPS:
            for seed1, seed2 in SEED_PAIRS:
                errs = mcode.check(load_trivial(op, seed1, seed2))
                self.assertEqual(errs, [], (op, seed1, seed2, errs))


class TestX2sOwnZeroPointIsTriviallyZero(unittest.TestCase):
    def test_zp2_is_zero_on_every_op_and_seed_pair(self):
        for op in OPS:
            for seed1, seed2 in SEED_PAIRS:
                samples = x2_trivial_samples(seed2)
                zp2, zp2_f, _ = asymmetric_uint8_quant_params(samples)
                self.assertEqual(zp2, 0, (op, seed1, seed2, zp2_f))


class TestReg94Tag132LocatorIsAbsentForSubAndMulToo(unittest.TestCase):
    """The headline finding: the same phenomenon PR #1663/#1667 found
    for Div/Add reproduces identically for Sub and Mul -- a real,
    fully op-independent encoder behavior, confirmed now for all four
    ops in the cluster."""

    def test_no_reg94_tag132_record_exists_in_any_trivial_fixture(self):
        for op in OPS:
            for seed1, seed2 in SEED_PAIRS:
                recs = mcode.decode(load_trivial(op, seed1, seed2), **mcode.FULL_RULE)
                hits = [
                    r
                    for r in recs
                    if r.get("kind") == "S"
                    and r.get("reg") == 94
                    and r.get("tag") == 132
                ]
                self.assertEqual(hits, [], (op, seed1, seed2))

    def test_reg94_still_fires_at_other_tags_not_globally_absent(self):
        """Confirms `reg=94` itself isn't simply missing from the
        stream -- only its `tag=132`/zero-point-carrying copy is,
        matching PR #1663/#1667's own identical observation."""
        for op in OPS:
            for seed1, seed2 in SEED_PAIRS:
                recs = mcode.decode(load_trivial(op, seed1, seed2), **mcode.FULL_RULE)
                reg94_all = [
                    r for r in recs if r.get("kind") == "S" and r.get("reg") == 94
                ]
                tags = {r.get("tag") for r in reg94_all}
                self.assertNotIn(132, tags, (op, seed1, seed2, tags))
                self.assertTrue(tags, (op, seed1, seed2))

    def test_control_fixtures_do_carry_the_locator(self):
        """Rules out "the locator is just generally unreliable" as an
        alternative explanation: the cluster's own already-committed,
        non-degenerate Sub/Mul fixtures (ordinary `randn` `x2`, PRs
        #1661/#1659) DO carry exactly one `reg=94,tag=132` record
        each."""
        for op in OPS:
            for seed1, seed2 in SEED_PAIRS:
                recs = mcode.decode(load_control(op, seed1, seed2), **mcode.FULL_RULE)
                hits = [
                    r
                    for r in recs
                    if r.get("kind") == "S"
                    and r.get("reg") == 94
                    and r.get("tag") == 132
                ]
                self.assertEqual(len(hits), 1, (op, seed1, seed2, hits))


if __name__ == "__main__":
    unittest.main()
