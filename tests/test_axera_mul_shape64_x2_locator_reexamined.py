"""Re-examines `tests/test_axera_add_two_live_shape_generality.py` (PR
#1671)'s own "Finding 4": at `m*k=64`, `x2`'s own zero-point locator
(`reg=94,tag=132`) goes absent for `Add`, and a NEW locator
(`reg=66,tag=131`) was reported to carry `x2`'s own zero point instead
-- confirmed, PR #1671 said, "identical (reg,tag) pair, identical
value (100), on both splits" of `m*k=64`.

This file tests whether that "relocation" is op-generic (does `Mul`
show it too?), the exact question PR #1671's own diff left open. It
finds something more precise, and more important, than a yes/no
answer: **`reg=66,tag=131`'s own payload is a FIXED CONSTANT (`0x64`
= `100`), completely independent of calibration seed and of `x2`'s own
real zero point** -- confirmed across six real fixtures (three
already-committed `Mul` baselines at `m*k=16`, PR #1659, plus three
fresh `Mul` builds at `m*k=64` with three DIFFERENT seed pairs built
here). PR #1671's own claim used only ONE seed pair ((1,2)) at
`m*k=64`, where `x2`'s own real zero point (`zp2=100`) happened to
COINCIDE numerically with this constant -- a false positive from an
n=1 seed sample, not a genuine relocation.

## Fixtures

Three fresh `Mul(x1[2,32], x2[2,32])` builds at `m*k=64` (PR #1671's
own `(2,32)` split), at the cluster's three standard seed pairs
(`(1,2)`, `(7,42)`, `(100,999)`) -- PR #1671 only ever tested `(1,2)`
at this shape, for any op. Real `docker run pulsar2:7.0-lite build`
invocations via `pulsar2_docker.build()`, matching this cluster's own
established two-`input_configs` convention exactly. `mcode.check()`
reports zero errors on all three.

## Finding 1: `reg=66,tag=131`'s own payload never changes -- not with
## seed, not with shape, not with op (as far as tested)

The trailing payload byte at `reg=66,tag=131` is `0x64` (`100`) in
EVERY ONE of six real fixtures checked: `Mul`'s own three
already-committed `m*k=16` baselines (`mul_1x16_two_live_seed{1_2,
7_42,100_999}.mcode.gz`, PR #1659, where `x2`'s own real zero point is
131/129/... -- never 100) AND all three of this file's own fresh
`m*k=64` builds. A field whose value tracks real, seed-varying
calibration data cannot behave this way; this is a fixed constant, not
a zero-point encoding.

## Finding 2 (corrects PR #1671): `reg=94,tag=132`'s own presence at
## `m*k=64` is SEED-DEPENDENT for `Mul`, not a clean `m*k`-driven rule

| seed pair | `zp2` (real, recomputed) | `reg=94,tag=132` | `reg=66,tag=131` payload |
| --- | --- | --- | --- |
| `(1,2)` | `100` | **absent** | `100` (coincidental match) |
| `(7,42)` | `103` | present, bit-exact | `100` (unrelated) |
| `(100,999)` | `130` | present, bit-exact | `100` (unrelated) |

Two of three tested seed pairs show `reg=94,tag=132` present and
bit-exact, matching the cluster's own baseline behavior; only `(1,2)`
shows it absent -- the same seed pair PR #1671's own diff happened to
build its `m*k=64` fixtures with. At `(1,2)`, an exhaustive scan of
every `S`-kind record's own payload byte finds NOTHING matching `zp2`
(`100`) anywhere in the stream except the fixed `reg=66,tag=131`
constant (which, per Finding 1, is unrelated) -- `x2`'s own real zero
point is genuinely unencoded there, with no relocation target found,
not "relocated."

## What this establishes, precisely, and what it does not

**Established**: `reg=66,tag=131` is a fixed constant (`100`),
independent of seed/shape/(as far as tested)op -- PR #1671's own
Finding 4 ("x2's own zero-point locator relocates to reg=66,tag=131 at
m*k=64") does not hold under a multi-seed test; the match it found was
a coincidence specific to its own one tested seed pair.
`reg=94,tag=132`'s own presence at `Mul`'s `m*k=64` is genuinely
seed-dependent (2 of 3 tested seeds present, 1 of 3 absent), not a
clean, always-triggered `m*k`-driven rule the way the `field=80`
extra-record mechanism (PR #1651) is.

**NOT established**: WHY `reg=94,tag=132` goes absent for seed pair
`(1,2)` specifically but not the other two (checked and ruled out: not
a rounding near-tie -- `zp2`'s own distance to a half-boundary is
`~0.30` for `(1,2)`, not meaningfully closer to a tie than `(7,42)`'s
own `~0.29`); whether the same absence pattern (and its own seed
dependence) holds for `Add` too (PR #1671's own single-seed-pair
fixtures cannot be re-checked for this without a fresh build, not done
here); what `reg=66,tag=131`'s own constant `100` actually represents
(unexplored -- a fixed configuration/format value, not per-tensor
calibration data).
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
BASELINE_SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]
FIXED_CONSTANT_PAYLOAD = bytes.fromhex("e01b64")


def load_shape64(seed1, seed2):
    path = os.path.join(FIX, f"mul_2x32_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def load_baseline(seed1, seed2):
    """The already-committed, non-degenerate `m*k=16` Mul fixtures (PR
    #1659) -- used here only to confirm `reg=66,tag=131`'s own payload
    is fixed regardless of shape too, not just regardless of seed."""
    path = os.path.join(FIX, f"mul_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
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


class TestReg66Tag131IsAFixedConstantNotXTwosZeroPoint(unittest.TestCase):
    """The core correction: this locator's own payload never varies,
    across six real fixtures spanning two shapes and six distinct
    calibration seeds -- it cannot be carrying `x2`'s own real zero
    point, which does vary (131/129/... at m*k=16; 100/103/130 at
    m*k=64, per the recomputed table in this file's own module
    docstring)."""

    def test_payload_is_fixed_at_shape64_across_all_three_seed_pairs(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            recs = mcode.decode(load_shape64(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 66 and r.get("tag") == 131
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2, hits))
            self.assertEqual(hits[0]["payload"], FIXED_CONSTANT_PAYLOAD, (seed1, seed2))

    def test_payload_is_fixed_at_baseline_shape_across_all_three_seed_pairs(self):
        for seed1, seed2 in BASELINE_SEED_PAIRS:
            recs = mcode.decode(load_baseline(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 66 and r.get("tag") == 131
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2, hits))
            self.assertEqual(hits[0]["payload"], FIXED_CONSTANT_PAYLOAD, (seed1, seed2))

    def test_constant_value_does_not_equal_zp2_except_by_coincidence(self):
        """`zp2` genuinely varies across seed pairs at m*k=64 (100 /
        103 / 130) while the constant stays at 100 -- only the first
        pair coincides, confirming the match PR #1671 found was not
        systematic."""
        zp2_values = []
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2, (2, 32)))
            zp2_values.append(zp2)
        self.assertEqual(zp2_values, [100, 103, 130])
        self.assertEqual(FIXED_CONSTANT_PAYLOAD[-1], 100)
        # Only the FIRST pair's zp2 happens to equal the constant.
        self.assertEqual(zp2_values[0], FIXED_CONSTANT_PAYLOAD[-1])
        self.assertNotEqual(zp2_values[1], FIXED_CONSTANT_PAYLOAD[-1])
        self.assertNotEqual(zp2_values[2], FIXED_CONSTANT_PAYLOAD[-1])


class TestReg94Tag132PresenceAtMK64IsSeedDependentForMul(unittest.TestCase):
    """Corrects PR #1671's own framing of `m*k=64` as a clean trigger
    for `x2`'s own zero-point locator going absent: for `Mul`, it is
    present and bit-exact at 2 of 3 tested seed pairs, absent only at
    the one PR #1671 itself happened to test."""

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

    def test_zp2_is_not_a_rounding_near_tie_at_the_absent_seed_pair(self):
        """Rules out PR #1640's own near-tie mechanism as the
        explanation: `(1,2)`'s own distance to a rounding half-boundary
        is not meaningfully closer than `(7,42)`'s own (both ~0.29-0.30),
        unlike a genuine near-tie case (PR #1640's own seed=0, at
        ~0.04)."""
        _, zp_f_absent, _ = asymmetric_uint8_quant_params(calib_samples(2, (2, 32)))
        _, zp_f_present, _ = asymmetric_uint8_quant_params(calib_samples(42, (2, 32)))
        dist_absent = abs((zp_f_absent - math.floor(zp_f_absent)) - 0.5)
        dist_present = abs((zp_f_present - math.floor(zp_f_present)) - 0.5)
        self.assertLess(abs(dist_absent - dist_present), 0.05)

    def test_no_record_anywhere_carries_zp2_at_the_absent_seed_pair(self):
        """The absent case is a genuine unencoded value, not a
        relocation this file failed to search hard enough for: no
        `S`-kind record of any (reg,tag) carries the payload byte
        `100` except the fixed constant at `reg=66,tag=131` (already
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
