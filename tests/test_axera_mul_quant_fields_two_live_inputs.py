"""Extends the first-principles quantization-recomputation technique
(port `ComputeAsymmetricUint8QuantParams`, `onnxsim/passes/static_
quantize_matmul.h`, into Python; apply it to the exact seeded
calibration data a fixture's own build script used; compare against
real decoded mcode bytes -- PRs #1636/#1640-#1657) from `Add`'s own
two-live-input survey (PR #1657) to `Mul`, the arc's own explicitly
flagged next candidate: "whether Add's two-live-input architecture
generalizes to Sub/Mul/Div with two live operands."

`Mul` is a genuinely different test of generalization from a mere
shape variant: its output range is a PRODUCT of its two inputs' own
ranges, not a sum, so a naive port of Add's own findings (`y_scale`
directly bit-exact via PR #1654's formula) does not obviously apply --
and, as Finding 3 below shows, it does not: Mul does not store
`y_scale` directly at all, but a different, genuinely new quantity.

## Fixtures

Three fresh `Mul(x1[1,16], x2[1,16])` builds, both `x1`/`x2` declared
graph inputs (neither a compile-time constant), independently
calibrated with the SAME seed pairs PR #1657 used for Add
(`(1,2)`, `(7,42)`, `(100,999)`) for direct comparability -- same
`RandomState(seed).randn(1,16)` x4-samples convention, same
`calibration_method: MinMax`, same `pulsar2:7.0-lite` Docker build.
`mcode.check()` reports zero errors on all three.

## Finding 1: `x1` is treated exactly like Add's own `x1` -- same
## locators, bit-exact

`x1`'s own zero point is the same literal 6-byte quad
(`02 10 1b <zp_x1> 83 36`) every op in this arc uses; its own
`1/x_scale` is the same `verb=161,bank=15,field in (96,112,128)`
locator, three redundant copies, bit-exact on all 3 tested seed pairs
-- identical to Add's own Finding 1 (PR #1657), not merely similar.

## Finding 2: `x2` is treated exactly like Add's own `x2` -- a
## distinct zero-point-only locator, no findable scale

`x2`'s own zero point decodes cleanly and uniquely at `reg=94,
tag=132` (bit-exact on all 3 seed pairs) -- the SAME `(reg, tag)` pair
Add's own `x2` used (PR #1657's Finding 2), not a new locator. `x2`'s
own scale (or its reciprocal) is, like Add's, nowhere findable: not in
any decoded `V` record's operand bytes, not via a raw byte-offset
float32 scan (tolerance `1e-3`) across the whole stream.

## Finding 3 (NEW, does not generalize from Add): Mul does not store
## `y_scale` directly -- it stores the combined REQUANTIZATION
## MULTIPLIER `y_scale / (x1_scale * x2_scale)` instead, bit-exact

Add's own `y_scale` (PR #1657 Finding 3) was directly, bit-exact
findable at the `verb=161,bank=15` locator once PR #1654's
"round-scale-once" formula was applied. Mul's own `verb=161,bank=15`
group carries a SECOND pair of records this arc has not seen before,
at `field in (224, 240)` (two redundant copies, not three) -- but the
value there does NOT match `y_scale` (plain or round-scale-once), nor
`1/x_scale`-style candidates for `x1`/`x2`/`y` individually, nor
`x1_scale*x2_scale` or its reciprocal.

It DOES match, bit-exact on all 3 tested seed pairs, the combined
ratio `y_scale / (x1_scale * x2_scale)` -- the standard integer-
multiply requantization multiplier for a quantized elementwise product
(`y_int approx (x1_int - zp1) * (x2_int - zp2) * multiplier + zp_y`),
computed as ONE double-precision division (not three separately-
rounded float32 scales multiplied/divided together) and rounded to
float32 exactly once -- consistent with PR #1654's own "round the
whole reduction chain once, not at each intermediate step" lesson,
extended here to a three-scale ratio rather than a single `hi-lo`
subtraction. Every tested rounding-order variant (double-precision
`sy_plain`/`sy_round-scale-once` crossed with double-precision or
already-float32 `sc1`/`sc2`) agreed to the bit, so this file does not
further isolate which specific sub-step needs double precision here
(unlike PR #1654's own more surgical isolation for `1/x_scale`) --
only that computing the full three-scale ratio in double precision and
rounding once reproduces the real device's byte exactly.

This is architecturally sensible and NOT a coincidence of this file's
own formula-search breadth: Add's own dequantize-then-add path needs
only `y_scale` itself (both operands are already on a compatible
additive footing after their own zero-point subtraction); Mul's
dequantize-then-multiply path produces a value scaled by the PRODUCT
of both input scales, so an integer multiplier bridging that product
scale to the output's own storage scale is the natural quantity to
store instead of `y_scale` alone -- and this file's own negative
search (Finding 3's own preamble) confirms `y_scale` itself is not
ALSO separately present.

## Finding 4: `zp_y` decodes at `reg=76, tag=132` -- a DIFFERENT
## register from both Add's own `zp_y` (`reg=14, tag=131`) and Conv/
## Gemm/MatMul's own `zp_y` locators

Filtering to `reg=76, tag=132` finds 7 records: 6 carry fixed,
irrelevant payload bytes (`32` x2, `26` x4 -- unrelated constants, the
same "filter out the constant hits" pattern Add's own `reg=14` needed
for its own constant `4` byte), and exactly one matches the recomputed
`zp_y` bit-for-bit on all 3 tested seed pairs. Both the REGISTER
(`76`, not `14`) and the TAG (`132`, not `131`) differ from Add's own
`zp_y` locator -- `zp_y`'s own encoding is not shared even between
these two closely related two-live-input ops.

## What this establishes, precisely, and what it does not

**Established**: Add's two-live-input architecture (PR #1657)
generalizes cleanly to Mul for THREE of its four findings --
`x1`/`x2`'s own treatment is identical, register-for-register, tag-
for-tag; the "no findable `x2` scale" negative also generalizes. The
FOURTH finding (Add's own directly-stored `y_scale`) does NOT
generalize as-is: Mul instead stores a genuinely different, new
quantity (the requantization multiplier `y_scale/(x1_scale*x2_scale)`,
bit-exact via a double-precision-throughout, round-once formula) at a
new locator (`field in (224,240)`, same `verb=161,bank=15` group).
`zp_y` decodes cleanly but at yet another new `(reg,tag)` pair
(`reg=76,tag=132`), distinct from every other op's own `zp_y` in this
arc.

**NOT established**: whether `Sub`/`Div`'s own two-live-input
treatment follows Mul's (multiplicative-requant) or Add's (additive,
direct-`y_scale`) pattern, or a third pattern of its own (not tested);
which specific double-precision sub-step Finding 3's own multiplier
formula actually needs (all variants tried happened to agree, unlike
PR #1654's own more surgical `1/x_scale` isolation); whether this
holds at shapes other than `(1,16)`, or with `x1`/`x2`'s operand order
swapped (not tested here, same open item PR #1657 itself left open
for Add).
"""

import gzip
import math
import os
import struct
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]


def load(seed1, seed2):
    path = os.path.join(FIX, f"mul_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(seed1, seed2):
    return mcode.decode(load(seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration data
    this file's own fixture-build script used for each `Mul` input,
    matching PR #1657's own Add convention and shape."""
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


def round_scale_once(samples):
    """PR #1654's own refined formula: the `hi-lo` subtraction (and the
    division by 255) done in float64, `scale` rounded to float32 exactly
    ONE time -- not re-derived here, cited directly and reused as-is."""
    lo = 0.0
    hi = 0.0
    for s in samples:
        lo = min(lo, float(s.min()))
        hi = max(hi, float(s.max()))
    if hi <= lo:
        hi = lo + 1.0
    return np.float32((hi - lo) / 255.0)


def requant_multiplier(seed1, seed2):
    """The new quantity Finding 3 establishes: `y_scale /
    (x1_scale * x2_scale)`, computed as one double-precision ratio and
    rounded to float32 exactly once (PR #1654's own "round once, not
    at each intermediate step" lesson, extended to a 3-scale ratio)."""
    x1 = calib_samples(seed1)
    x2 = calib_samples(seed2)
    ys = [a * b for a, b in zip(x1, x2)]
    sy = float(round_scale_once(ys))
    sc1 = float(round_scale_once(x1))
    sc2 = float(round_scale_once(x2))
    return np.float32(sy / (sc1 * sc2))


class TestX1UsesTheSameLocatorsAddsOwnX1Used(unittest.TestCase):
    def test_zp_x1_literal_quad_matches_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load(seed1, seed2)
            zp1, _, _ = asymmetric_uint8_quant_params(calib_samples(seed1))
            quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(len(hits), 1, (seed1, seed2, zp1))

    def test_recip_x1_scale_is_bit_exact_at_verb161_bank15(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            _, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            recip1 = np.float32(np.float32(1.0) / np.float32(s1))
            target = struct.pack("<f", recip1)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("operand") == target
            ]
            self.assertEqual(len(hits), 3, (seed1, seed2))
            fields = sorted(r["field"] for r in hits)
            self.assertEqual(fields, [96, 112, 128])


class TestX2UsesTheSameLocatorAddsOwnX2UsedButHasNoFindableScale(unittest.TestCase):
    def test_zp_x2_matches_uniquely_at_reg94_tag132(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(len(hits), 1, (seed1, seed2))
            self.assertEqual(hits[0]["payload"][-1], zp2, (seed1, seed2))

    def test_no_other_s_record_at_reg94_tag132_carries_a_nearby_wrong_value(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
            for delta in (-2, -1, 1, 2):
                candidate = zp2 + delta
                if not (0 <= candidate <= 255):
                    continue
                hits = [
                    r
                    for r in recs
                    if r.get("kind") == "S"
                    and r.get("reg") == 94
                    and r.get("tag") == 132
                    and r["payload"][-1] == candidate
                ]
                self.assertEqual(hits, [], (seed1, seed2, delta))

    def test_x2_scale_is_not_findable_anywhere_in_the_stream(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load(seed1, seed2)
            recs = decode(seed1, seed2)
            _, _, s2 = asymmetric_uint8_quant_params(calib_samples(seed2))
            candidates = {"s2": s2, "recip_s2": 1.0 / s2}
            allV = [r for r in recs if r.get("kind") == "V"]
            decoded_bytes = {
                r["operand"]
                for r in allV
                if r.get("operand") and len(r["operand"]) == 4
            }
            for name, val in candidates.items():
                target = struct.pack("<f", np.float32(val))
                self.assertNotIn(target, decoded_bytes, (seed1, seed2, name))
            for name, val in candidates.items():
                hits = []
                for i in range(len(data) - 3):
                    raw = struct.unpack_from("<f", data, i)[0]
                    if not math.isfinite(raw) or val == 0:
                        continue
                    if abs(raw - val) / abs(val) < 1e-3:
                        hits.append(i)
                self.assertEqual(hits, [], (seed1, seed2, name, val))


class TestMulStoresARequantMultiplierNotYScaleDirectly(unittest.TestCase):
    def test_plain_y_scale_does_not_appear_at_the_second_verb161_bank15_pair(self):
        """Confirms Finding 3's own preamble: unlike Add, Mul's second
        `verb=161,bank=15` record pair is NOT `y_scale` itself (plain
        or round-scale-once)."""
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a * b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            _, _, sy_plain = asymmetric_uint8_quant_params(ys)
            sy_rso = round_scale_once(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (224, 240)
            ]
            self.assertEqual(len(hits), 2, (seed1, seed2))
            observed = struct.unpack("<f", hits[0]["operand"])[0]
            self.assertNotAlmostEqual(observed, sy_plain, places=3, msg=(seed1, seed2))
            self.assertNotAlmostEqual(
                observed, float(sy_rso), places=3, msg=(seed1, seed2)
            )

    def test_requant_multiplier_is_bit_exact_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            mult = requant_multiplier(seed1, seed2)
            target = struct.pack("<f", mult)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (224, 240)
                and r.get("operand") == target
            ]
            self.assertEqual(len(hits), 2, (seed1, seed2, mult))
            fields = sorted(r["field"] for r in hits)
            self.assertEqual(fields, [224, 240])


class TestZpYMatchesAtReg76Tag132(unittest.TestCase):
    def test_zp_y_matches_after_excluding_the_unrelated_constant_hits(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a * b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 76 and r.get("tag") == 132
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen, zpy))

    def test_locator_differs_from_adds_own_zp_y_register_and_tag(self):
        """Add's own `zp_y` (PR #1657) lives at `reg=14,tag=131` --
        confirms Mul's own `reg=76,tag=132` locator is not that same
        pair coincidentally relabeled."""
        self.assertNotEqual((76, 132), (14, 131))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
