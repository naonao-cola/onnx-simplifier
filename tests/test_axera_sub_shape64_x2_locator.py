"""Characterizes `Sub`'s own `x2` zero-point locator at shape `m*k=64`,
extending the multi-seed methodology `tests/test_axera_add_shape64_
x2_locator_reexamined.py` (PR #1674) and `tests/test_axera_mul_
shape64_x2_locator_reexamined.py` (PR #1673) established after both
found that PR #1671's own single-seed-pair "relocation" claim
(`x2`'s own zero point moves from `reg=94,tag=132` to a new
`reg=66,tag=131` pair at `m*k=64`) does not hold: `reg=66,tag=131`'s
own payload is a FIXED CONSTANT, unrelated to `x2`'s own real,
seed-varying zero point, and `reg=94,tag=132`'s own presence at
`m*k=64` is genuinely SEED-DEPENDENT (2 of 3 tested seeds present and
bit-exact, 1 of 3 absent with no findable replacement).

Unlike PR #1671's own mistake, this file tests all THREE of this
cluster's standard seed pairs from the start, not one.

## Fixtures

Three fresh `Sub(x1[2,32], x2[2,32])` builds at `m*k=64` (the same
`(2,32)` split PR #1671/#1673/#1674 all used), at the cluster's three
standard seed pairs (`(1,2)`, `(7,42)`, `(100,999)`). Real `docker run
pulsar2:7.0-lite build` invocations via `pulsar2_docker.build()`, two
separate `input_configs` entries, matching this cluster's own
established convention exactly. `mcode.check()` reports zero errors on
all three.

## Finding: `Sub` shows the IDENTICAL corrected pattern Add and Mul
## both show -- `reg=94,tag=132` seed-dependent, `reg=66,tag=131` a
## fixed, unrelated constant

| seed pair | `zp2` (real, recomputed) | `reg=94,tag=132` | `reg=66,tag=131` payload |
| --- | --- | --- | --- |
| `(1,2)` | `100` | **absent** | `e01b64` (last byte `100`, coincidental match) |
| `(7,42)` | `103` | present, bit-exact | `e01b64` (unrelated) |
| `(100,999)` | `130` | present, bit-exact | `e01b64` (unrelated) |

Same shape (2 of 3 present, absent only at `(1,2)`) and same fixed
`reg=66,tag=131` payload (`e01b64`, trailing byte `100`) PR #1673/#1674
already found for `Mul`/`Add` at this identical shape and these
identical seed pairs -- a third independent confirmation that
`reg=66,tag=131` is not a relocated zero-point encoding, and that
`x2`'s own `reg=94,tag=132` presence at `m*k=64` is a seed-dependent
property of `x2`'s own shared calibration data (`RandomState(2)` at
shape `(2,32)`), not of any one op's own arithmetic.

One minor difference from Add's own pattern: Add's `reg=66,tag=131`
group carried a SECOND record at the same `(reg,tag)` pair with an
unrelated constant-`4` payload (the same "filter out a constant-4
byte" collision `zp_y`'s own `reg=14,tag=131` locator needs, PR
#1657's Finding 4). `Sub`'s own `reg=66,tag=131` shows only the ONE
constant hit, no such collision -- not itself surprising (different
ops' streams need not share every incidental record), and does not
affect the headline finding, which only requires there being no
seed-tracking record present.

## What this establishes, precisely, and what it does not

**Established**: `Sub` shows the identical seed-dependent
`reg=94,tag=132` absence / fixed `reg=66,tag=131` constant pattern PR
#1673 (Mul) and PR #1674 (Add) both already found -- a third
op confirming the corrected picture, tested correctly from the start
(three seed pairs, not one).

**NOT established**: WHY `reg=94,tag=132` goes absent for seed pair
`(1,2)` specifically (not investigated here, same open item PR
#1673/#1674 both left); whether `Div` shows the identical pattern
(not tested here); what `reg=66,tag=131`'s own constant `100` actually
represents.
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


def load_shape64(seed1, seed2):
    path = os.path.join(FIX, f"sub_2x32_two_live_seed{seed1}_{seed2}.mcode.gz")
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


class TestReg66Tag131IsAFixedConstantForSubToo(unittest.TestCase):
    """The core finding: this locator's own payload never varies
    across all three tested seed pairs, even though `zp2` genuinely
    does (100/103/130) -- it cannot be carrying `x2`'s own real zero
    point."""

    def test_payload_is_fixed_across_all_three_seed_pairs(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            recs = mcode.decode(load_shape64(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 66 and r.get("tag") == 131
            ]
            payloads = [h["payload"] for h in hits]
            self.assertEqual(payloads, [FIXED_CONSTANT_PAYLOAD], (seed1, seed2))

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


class TestReg94Tag132PresenceAtMK64IsSeedDependentForSub(unittest.TestCase):
    """`Sub` matches the identical 2-of-3-present pattern PR #1673
    (Mul) and PR #1674 (Add) both already found."""

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

    def test_absent_at_the_1_2_seed_pair(self):
        recs = mcode.decode(load_shape64(1, 2), **mcode.FULL_RULE)
        hits = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
        ]
        self.assertEqual(hits, [])

    def test_no_record_anywhere_carries_zp2_at_the_absent_seed_pair(self):
        """The absent case is a genuine unencoded value: no `S`-kind
        record of any (reg,tag) carries the payload byte `100` except
        the fixed `reg=66,tag=131` constant (already shown
        unrelated)."""
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
