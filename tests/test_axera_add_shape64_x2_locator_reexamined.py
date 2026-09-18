"""Corrects `tests/test_axera_add_two_live_shape_generality.py` (PR
#1671)'s own "Finding 4" for `Add`: at `m*k=64`, `x2`'s own zero-point
locator (`reg=94,tag=132`) goes absent, and PR #1671 reported a NEW
locator (`reg=66,tag=131`) "relocates" to carry `x2`'s own zero point
instead -- "identical (reg,tag) pair, identical value (100), on both
splits" of `m*k=64`.

`tests/test_axera_mul_shape64_x2_locator_reexamined.py` (PR #1673)
already found this exact claim does NOT hold for `Mul`: `reg=66,
tag=131`'s own payload is a FIXED CONSTANT (`0x64` = `100`), unrelated
to `x2`'s own real zero point, and PR #1671's own match at seed pair
`(1,2)` was a coincidence (`x2`'s own real zero point happened to also
be `100` there, since `x2`'s calibration data -- `RandomState(seed2).
randn(*shape)` -- is identical regardless of which op consumes it).

This file directly re-checks PR #1671's own `Add` claim at the two
seed pairs it never tested (`(7,42)`, `(100,999)`) and finds the
IDENTICAL correction applies: `reg=66,tag=131` carries the same fixed
constant at all three seed pairs, never tracking `x2`'s own real,
seed-varying zero point (`100`/`103`/`130`). PR #1671's own claim was
the exact same n=1-seed false positive PR #1673 already diagnosed for
`Mul`, now confirmed for `Add` too, with real hardware verification at
the two untested seed pairs.

## Fixtures

Two fresh `Add(x1[2,32], x2[2,32])` builds at `m*k=64` (PR #1671's own
`(2,32)` split), at the cluster's other two standard seed pairs
(`(7,42)`, `(100,999)` -- PR #1671 only ever tested `(1,2)` at this
shape, for any op). Real `docker run pulsar2:7.0-lite build`
invocations, matching this cluster's own established two-
`input_configs` convention exactly. `mcode.check()` reports zero
errors on both.

## Finding 1: `reg=66,tag=131`'s own payload never changes for `Add`
## either -- confirmed at all three now-tested seed pairs

The trailing payload byte at `reg=66,tag=131` is `0x64` (`100`) at
EVERY one of the three `Add` `m*k=64` fixtures now built (`(1,2)`,
PR #1671's own original; `(7,42)`/`(100,999)`, built here) -- a second
record at the same `(reg,tag)` pair also always carries an unrelated
constant `4` byte (the same "filter out a constant-4 byte" collision
this cluster's own output-side locators already needed, e.g. `zp_y`'s
own `reg=14,tag=131`, PR #1657's Finding 4). Neither hit tracks `x2`'s
own real, seed-varying zero point (`100`/`103`/`130`).

## Finding 2 (corrects PR #1671): `reg=94,tag=132`'s own presence at
## `m*k=64` is SEED-DEPENDENT for `Add` too, not a clean `m*k`-driven
## rule -- the identical shape PR #1673 already found for `Mul`

| seed pair | `zp2` (real, recomputed) | `reg=94,tag=132` | `reg=66,tag=131` payload |
| --- | --- | --- | --- |
| `(1,2)` | `100` | **absent** (PR #1671's own original finding) | `100` (coincidental match) |
| `(7,42)` | `103` | present, bit-exact | `100` (unrelated) |
| `(100,999)` | `130` | present, bit-exact | `100` (unrelated) |

Two of three tested seed pairs show `reg=94,tag=132` present and
bit-exact, matching the cluster's own baseline behavior at `m*k=16`;
only `(1,2)` -- the one seed pair PR #1671 happened to test -- shows
it absent. This is the exact same 2-of-3-present pattern PR #1673
found for `Mul` at the identical shape and seed pairs.

## What this corrects, precisely, and what remains open

**CORRECTED**: PR #1671's own "Finding 4" ("`x2`'s own zero-point
locator relocates to `reg=66,tag=131` at `m*k=64`" for `Add`) does NOT
hold under a multi-seed test -- the match it found at its one tested
seed pair (`(1,2)`) was a coincidence, identical in kind to the one PR
#1673 already diagnosed and corrected for `Mul`. `reg=66,tag=131` is a
fixed constant, not a relocated zero-point encoding, for BOTH ops now
directly verified. `tests/test_axera_two_live_input_shape_generality_
synthesis.py` (PR #1672) also restates PR #1671's now-corrected claim
as established fact in its own comparison table -- that restatement is
likewise superseded by this file, not independently re-checked here
since this file's own job is the `Add`-side hardware verification, not
re-auditing every downstream citation.

**NOT established**: WHY `reg=94,tag=132` goes absent for seed pair
`(1,2)` specifically, for either `Add` or `Mul` (PR #1673 already
ruled out a rounding near-tie as the cause for `Mul`; not independently
re-checked for `Add` here, though the identical seed pair producing
the identical absence for both ops is suggestive that whatever the
cause is, it is a property of `x2`'s own shared calibration data --
`RandomState(2)` at shape `(2,32)` -- not of either op's own
arithmetic); what `reg=66,tag=131`'s own constant `100` actually
represents; whether `Sub`/`Div` show the identical seed-dependent
absence and fixed-constant pattern at `m*k=64` (not tested here).
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

SHAPE64_SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]
FIXED_CONSTANT_PAYLOAD = bytes.fromhex("e01b64")
UNRELATED_CONSTANT_PAYLOAD = bytes.fromhex("0c8004")


def load_shape64(seed1, seed2):
    path = os.path.join(FIX, f"add_2x32_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def calib_samples(seed, shape, n_samples=4):
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
    def test_no_decode_errors_on_any_shape64_seed_pair(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            errs = mcode.check(load_shape64(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


class TestReg66Tag131IsAFixedConstantForAddToo(unittest.TestCase):
    """The core correction: this locator's own payload never varies
    across all three now-tested seed pairs, even though `zp2` genuinely
    does (100/103/130) -- it cannot be carrying `x2`'s own real zero
    point."""

    def test_both_payloads_are_fixed_across_all_three_seed_pairs(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            recs = mcode.decode(load_shape64(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 66 and r.get("tag") == 131
            ]
            payloads = sorted(h["payload"] for h in hits)
            self.assertEqual(
                payloads,
                sorted([FIXED_CONSTANT_PAYLOAD, UNRELATED_CONSTANT_PAYLOAD]),
                (seed1, seed2, payloads),
            )

    def test_constant_value_does_not_equal_zp2_except_by_coincidence(self):
        zp2_values = []
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2, (2, 32)))
            zp2_values.append(zp2)
        self.assertEqual(zp2_values, [100, 103, 130])
        self.assertEqual(FIXED_CONSTANT_PAYLOAD[-1], 100)
        self.assertEqual(zp2_values[0], FIXED_CONSTANT_PAYLOAD[-1])
        self.assertNotEqual(zp2_values[1], FIXED_CONSTANT_PAYLOAD[-1])
        self.assertNotEqual(zp2_values[2], FIXED_CONSTANT_PAYLOAD[-1])


class TestReg94Tag132PresenceAtMK64IsSeedDependentForAdd(unittest.TestCase):
    """Corrects PR #1671's own framing of `m*k=64` as a clean trigger
    for `x2`'s own zero-point locator going absent: for `Add`, it is
    present and bit-exact at 2 of 3 tested seed pairs, absent only at
    the one PR #1671 itself happened to test -- the identical pattern
    PR #1673 already found for `Mul`."""

    def test_present_and_bit_exact_at_two_of_three_seed_pairs(self):
        present_seeds = [(7, 42), (100, 999)]
        for seed1, seed2 in present_seeds:
            recs = mcode.decode(load_shape64(seed1, seed2), **mcode.FULL_RULE)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2, (2, 32)))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2))
            self.assertEqual(hits[0]["payload"][-1], zp2, (seed1, seed2, zp2))

    def test_absent_at_the_one_seed_pair_pr1671_tested(self):
        recs = mcode.decode(load_shape64(1, 2), **mcode.FULL_RULE)
        hits = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
        ]
        self.assertEqual(hits, [])

    def test_no_record_anywhere_carries_zp2_at_the_absent_seed_pair(self):
        """The absent case is a genuine unencoded value, not a
        relocation this file failed to search hard enough for: no
        `S`-kind record of any (reg,tag) carries the payload byte
        `100` except the fixed `reg=66,tag=131` constant (already
        shown unrelated)."""
        recs = mcode.decode(load_shape64(1, 2), **mcode.FULL_RULE)
        zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(2, (2, 32)))
        hits = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("payload") and r["payload"][-1] == zp2
        ]
        self.assertEqual(len(hits), 1, hits)
        self.assertEqual((hits[0]["reg"], hits[0]["tag"]), (66, 131))


if __name__ == "__main__":
    unittest.main()
