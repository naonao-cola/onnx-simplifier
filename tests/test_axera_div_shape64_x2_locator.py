"""Characterizes `Div`'s own `x2` zero-point locator at shape
`m*k=64`, extending the multi-seed methodology `tests/test_axera_add_
shape64_x2_locator_reexamined.py` (PR #1674), `tests/test_axera_mul_
shape64_x2_locator_reexamined.py` (PR #1673), and `tests/test_axera_
sub_shape64_x2_locator.py` to the fourth and final op in this
cluster.

Add/Mul/Sub all show an identical, now-corrected pattern at `m*k=64`:
`reg=94,tag=132` (the locator that normally carries `x2`'s own zero
point) is present and bit-exact at 2 of the 3 standard seed pairs, and
absent (with no findable replacement) at exactly one, `(1,2)`;
`reg=66,tag=131` carries a FIXED CONSTANT, unrelated to `x2`'s own
real zero point (the coincidental match at `(1,2)` that originally
misled PR #1671 into reporting a false "relocation").

## Finding: `Div` does NOT show the same seed-dependent absence --
## `reg=94,tag=132` is present and bit-exact at ALL THREE tested seed
## pairs

Unlike Add/Mul/Sub, `Div`'s own `reg=94,tag=132` never goes absent at
`m*k=64`, not even at `(1,2)` -- a genuine, real difference from the
other three ops, not a gap in this file's own search. `reg=66,tag=131`
is still present (two records: the same `e01b64` constant Add/Mul/Sub
all showed, plus a second, single-byte `0x20` payload that also never
tracks `zp2`) -- so the fixed-constant locator exists for `Div` too,
it just isn't needed as a fallback the way it apparently was for the
other three ops at this one seed pair.

## Fixtures

Three fresh `Div(x1[2,32], x2[2,32])` builds at `m*k=64` (the same
`(2,32)` split PR #1671/#1673/#1674 all used for Add/Mul), at the
cluster's three standard seed pairs. `x1` drawn from the plain
`RandomState(seed1).randn(2,32)` convention every op in this cluster
uses; `x2` drawn from PR #1663's own established sign-flip divisor
design (`sign * (2.0 + 0.3*randn())`), reused unchanged -- NOT applied
to `x1` (an earlier draft of this file's own build script mistakenly
used the sign-flip design for both tensors; re-verified against PR
#1663's own `test_axera_div_quant_fields_two_live_inputs.py` and
rebuilt correctly before writing these assertions; the corrected
rebuild produced byte-identical results to the mistaken one, so the
mistake did not silently change this file's own conclusions, but the
committed fixtures use the correct, established convention). Real
`docker run pulsar2:7.0-lite build` invocations, matching this
cluster's own established two-`input_configs` convention. `mcode.
check()` reports zero errors on all three.

## What this establishes, precisely, and what it does not

**Established**: `Div`'s own `reg=94,tag=132` locator is present and
bit-exact at every one of the three tested seed pairs -- it does NOT
share Add's/Mul's/Sub's own seed-dependent absence at `m*k=64`, a
genuine op-specific difference on this axis. `reg=66,tag=131` still
carries the same fixed constant (`e01b64`, trailing byte `100`) the
other three ops showed, confirming that piece of the picture is
op-generic even though the absence-triggering condition is not.

**NOT established**: WHY `Div` differs from the other three ops here
(no access to Pulsar2's own source); whether this generalizes to
other `m*k` totals; what `reg=66,tag=131`'s own second payload
(`0x20`) represents.
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
SECOND_CONSTANT_PAYLOAD = bytes.fromhex("20")


def load_shape64(seed1, seed2):
    path = os.path.join(FIX, f"div_2x32_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def x2_div_safe_samples(seed, shape=(2, 32), n_samples=4):
    """`sign * (2.0 + 0.3*randn())` -- PR #1663's own sign-flip
    divisor design, reused unchanged for `x2` only."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_samples):
        mag = 2.0 + 0.3 * rng.randn(*shape)
        sign = rng.choice([-1.0, 1.0], size=shape)
        out.append((sign * mag).astype(np.float32))
    return out


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


class TestReg94Tag132IsPresentAtEveryTestedSeedPairUnlikeOtherOps(unittest.TestCase):
    """The headline, op-specific finding: `Div` does not show the
    seed-dependent absence Add/Mul/Sub all showed at this shape."""

    def test_present_and_bit_exact_at_all_three_seed_pairs(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            recs = mcode.decode(load_shape64(seed1, seed2), **mcode.FULL_RULE)
            zp2, _, _ = asymmetric_uint8_quant_params(x2_div_safe_samples(seed2))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2))
            self.assertEqual(hits[0]["payload"][-1], zp2, (seed1, seed2, zp2))


class TestReg66Tag131StillCarriesTheSameFixedConstant(unittest.TestCase):
    def test_fixed_constant_present_at_every_seed_pair(self):
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
                sorted([FIXED_CONSTANT_PAYLOAD, SECOND_CONSTANT_PAYLOAD]),
                (seed1, seed2, payloads),
            )

    def test_neither_payload_tracks_zp2_at_any_seed_pair(self):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            zp2, _, _ = asymmetric_uint8_quant_params(x2_div_safe_samples(seed2))
            self.assertNotEqual(FIXED_CONSTANT_PAYLOAD[-1], zp2, (seed1, seed2))
            self.assertNotEqual(SECOND_CONSTANT_PAYLOAD[-1], zp2, (seed1, seed2))


if __name__ == "__main__":
    unittest.main()
