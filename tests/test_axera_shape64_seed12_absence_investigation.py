"""Answers the deepest open question `tests/test_axera_shape64_x2_
locator_correction_synthesis.py` (PR #1676) left in the `m*k=64`
`x2`-zero-point-locator correction thread (PRs #1673-#1676): WHY does
calibration seed pair `(1,2)` specifically trigger `reg=94,tag=132`'s
own structural absence for `Add`/`Mul`/`Sub` (at shape `m*k=64`), when
the other two standard seed pairs (`(7,42)`, `(100,999)`) do not?

PR #1673 already ruled out a rounding-near-tie explanation for `Mul`.
This file tests a different, more basic question first: does the
absence track `x1`'s own seed, `x2`'s own seed, or genuinely the PAIR
`(1,2)` as a whole? It finds a clean, decisive answer: it is `x2`'s own
seed (`2`) alone, completely independent of `x1`.

## Method

Three fresh `Add(x1[2,32], x2[2,32])` builds, deliberately crossing
`x1`'s and `x2`'s own seeds independently of the cluster's own
standard pairing convention (which has always kept `x1`'s seed and
`x2`'s seed changing together, one pair at a time -- `(1,2)`, `(7,42)`,
`(100,999)` -- so no prior fixture in this whole cluster could
distinguish "x1's seed matters" from "x2's seed matters" from "the
specific pair matters"):

| build | `x1` seed | `x2` seed | `reg=94,tag=132` |
| --- | --- | --- | --- |
| baseline `(1,2)` (PR #1671/#1674, already committed) | `1` | `2` | **absent** |
| `x1seed7_x2seed2` (this file) | `7` | `2` | **absent** |
| `x1seed100_x2seed2` (this file) | `100` | `2` | **absent** |
| `x1seed1_x2seed42` (this file) | `1` | `42` | present, bit-exact |
| baseline `(7,42)` (PR #1674, already committed) | `7` | `42` | present, bit-exact |

Real `docker run pulsar2:7.0-lite build` invocations via
`pulsar2_docker.build()`, matching this cluster's own established
two-`input_configs` convention exactly (`x1`'s own seed and `x2`'s own
seed independently controllable via two separate calibration tar
files). `mcode.check()` reports zero errors on all three fresh builds.

## Finding: the absence tracks `x2`'s own seed (`2`) ALONE, not `x1`'s
## seed and not the `(1,2)` pair specifically

Three different `x1` seeds (`1`, `7`, `100`) all produce the identical
absence when paired with `x2`'s own seed `2` -- `x1`'s own seed has no
observable effect on whether `reg=94,tag=132` fires. Conversely,
keeping `x1`'s own seed fixed at `1` (the exact value from the
original absent baseline) but changing ONLY `x2`'s own seed to `42`
restores the locator, present and bit-exact. This rules out both "the
`(1,2)` pair specifically" and "`x1`'s own seed" as drivers, and
narrows the cause to a real property of `x2`'s own calibration data at
seed `2` -- independent of whichever op consumes it (Add/Mul/Sub all
showed the identical `(1,2)`-triggers-absence pattern in PRs
#1673-#1675, and `x2`'s own `RandomState(2).randn(2,32)` draw is
identical regardless of op, exactly the same "shared calibration
data, op-independent effect" shape PR #1663/#1667/#1669/#1670's own
trivial-zero-point phenomenon already showed for a different trigger
condition).

This does NOT explain WHY seed `2`'s own calibration data specifically
triggers the absence (not a rounding near-tie, PR #1673 already ruled
that out for `Mul`; not a trivially-zero `zp2`, since `zp2=100` at seed
`2` is an ordinary, non-degenerate value) -- only that the trigger is
squarely a property of `x2`'s own data, not `x1`'s, not the pairing,
and not the consuming op.

## What this establishes, precisely, and what it does not

**Established**: the `reg=94,tag=132` absence at `m*k=64` is driven by
`x2`'s own seed specifically (confirmed: seed `2` triggers absence
regardless of `x1`'s own seed among three tested values; seed `42`
does not trigger absence when `x1`'s own seed is held at the exact
value from the original absent case) -- not by `x1`'s own seed, and
not by the `(1,2)` pair as a joint unit.

**NOT established**: WHY `x2`'s own seed-`2` calibration data
specifically triggers the absence (no access to Pulsar2's own source);
whether some other property of seed `2`'s own draw (distinguishable
from seed `42`'s) can be identified without exhaustively trying many
more seeds (not attempted here -- this file answers "which tensor's
seed matters", not "what about that seed's own data matters", a
narrower and more tractable question than the one PR #1676 originally
posed); whether the same "x2's seed alone matters" mechanism also
explains why `Div` never shows the absence at any tested seed (Div's
own `x2` uses a different, sign-flip calibration design -- PR #1663 --
so its own seed-`2` draw is not the same underlying data as Add's/
Mul's/Sub's, and this file does not test Div).
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg_hits(recs, reg, tag):
    return [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == reg and r.get("tag") == tag
    ]


def calib_samples(seed, shape=(2, 32), n_samples=4):
    """`RandomState(seed).randn(2, 32)` x4 -- the exact calibration
    data every op's own `x1`/`x2` uses in this whole `m*k=64` thread
    (PRs #1671/#1673-#1676)."""
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
    def test_no_decode_errors_on_any_crossed_seed_build(self):
        for name in [
            "add_2x32_x1seed7_x2seed2.mcode.gz",
            "add_2x32_x1seed100_x2seed2.mcode.gz",
            "add_2x32_x1seed1_x2seed42.mcode.gz",
        ]:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


class TestAbsenceTracksX2sOwnSeedRegardlessOfX1(unittest.TestCase):
    """The core finding: pairing x2's own seed 2 with three DIFFERENT
    x1 seeds all produce the absence -- x1's own seed is irrelevant."""

    def test_x2_seed2_is_absent_with_x1_seed_7(self):
        recs = decode("add_2x32_x1seed7_x2seed2.mcode.gz")
        hits = reg_hits(recs, 94, 132)
        self.assertEqual(hits, [])

    def test_x2_seed2_is_absent_with_x1_seed_100(self):
        recs = decode("add_2x32_x1seed100_x2seed2.mcode.gz")
        hits = reg_hits(recs, 94, 132)
        self.assertEqual(hits, [])


class TestPresenceReturnsWhenOnlyX2sSeedChangesX1HeldFixed(unittest.TestCase):
    """The disentangling counter-case: x1's own seed held at the EXACT
    value (1) from the original absent baseline, but x2's own seed
    changed to 42 (a seed that is NOT 2) -- the locator returns,
    bit-exact, proving x1's own seed was never the driver."""

    def test_x2_seed42_is_present_and_bit_exact_even_with_x1_seed_1(self):
        recs = decode("add_2x32_x1seed1_x2seed42.mcode.gz")
        zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(42))
        hits = reg_hits(recs, 94, 132)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["payload"][-1], zp2)


if __name__ == "__main__":
    unittest.main()
