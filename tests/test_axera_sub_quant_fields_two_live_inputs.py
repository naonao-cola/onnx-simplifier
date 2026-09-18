"""Extends the two-live-input quantization survey (`tests/test_axera_
add_quant_fields_two_live_inputs.py`, PR #1657; `tests/test_axera_
mul_quant_fields_two_live_inputs.py`, PR #1659) to `Sub`, the arc's own
explicitly flagged next candidate: PR #1659's own diff proposed that
Add's directly-stored `y_scale` (vs. Mul's combined requantization
multiplier) tracks whether the op is additive or multiplicative in the
dequantized domain -- `Sub` (`y = x1 - x2`, additive) should therefore
behave like `Add`, not like `Mul`, a concrete and testable prediction
this file confirms.

## Fixtures

Three fresh `Sub(x1[1,16], x2[1,16])` builds, both `x1`/`x2` declared
graph inputs (neither a compile-time constant), independently
calibrated with the SAME seed pairs PR #1657/#1659 used
(`(1,2)`, `(7,42)`, `(100,999)`) for direct comparability -- same
`RandomState(seed).randn(1,16)` x4-samples convention, same
`calibration_method: MinMax`, same `pulsar2:7.0-lite` Docker build.
`mcode.check()` reports zero errors on all three.

## Finding 1: `x1`/`x2` are treated exactly like Add's and Mul's own --
## identical locators, bit-exact

`x1`'s own zero point is the same literal 6-byte quad
(`02 10 1b <zp_x1> 83 36`) every op in this arc uses; its own
`1/x_scale` is the same `verb=161,bank=15,field in (96,112,128)`
locator. `x2`'s own zero point decodes at the SAME `reg=94,tag=132`
pair Add's and Mul's own `x2` used; its own scale is, like theirs,
nowhere findable (exact-bytes `V`-record scan plus a raw byte-offset
float32 scan, tolerance `1e-3`, both negative for `x2_scale` and
`1/x2_scale`). All bit-exact/negative on all 3 tested seed pairs.

## Finding 2 (confirms the additive-vs-multiplicative prediction):
## `y_scale` is DIRECTLY stored, bit-exact, at the SAME locator and
## SAME formula Add used -- NOT a requantization multiplier

`Sub`'s own `verb=161,bank=15` group carries a second, co-located
record pair (`field in (96,112,128)`, distinct operand from `1/x1_
scale`) -- exactly where Add's own `y_scale` lived (PR #1657 Finding
3), and NOT at Mul's own separate `field in (224,240)` locator (that
field pair does not exist at all in this file's own fixtures -- checked
directly, not assumed). The plain float32-throughout formula is
bit-exact on 2 of 3 seed pairs but 1 ULP off at `(1,2)` -- the same
single-seed-pair miss Add's own Finding 3 hit; PR #1654's own
"round-scale-once" formula (the `hi-lo` subtraction and division done
in float64, `scale` rounded to float32 exactly once) closes it,
bit-exact on all 3 tested seed pairs. This is not a coincidental
re-use of Add's own formula -- it is independently re-derived and
re-verified against `Sub`'s own real fixture bytes here.

## Finding 3 (confirms the prediction further): `zp_y` decodes at the
## IDENTICAL locator Add used -- `reg=14, tag=131` -- not Mul's own
## separate `reg=76,tag=132`

Filtering to `reg=14,tag=131` finds exactly the same two-hit pattern
Add's own Finding 4 found: one record with a fixed, irrelevant payload
byte of `4`, and one matching the recomputed `zp_y` bit-for-bit on all
3 tested seed pairs. This is a STRONGER form of agreement with Add than
Mul showed: Mul shared `x1`/`x2`'s own locators with Add but used
entirely different registers for both its own multiplier field and its
own `zp_y` (`field in (224,240)` and `reg=76,tag=132` respectively) --
`Sub` shares ALL FOUR of Add's own locators, register-for-register,
tag-for-tag, formula-for-formula.

## What this establishes, precisely, and what it does not

**Established**: `Sub`'s two-live-input encoding is, in every field
tested, indistinguishable from `Add`'s own -- not merely "also
additive" in the abstract, but literally the same registers, tags, and
formulas, bit-exact on all 3 tested seed pairs. This confirms PR
#1659's own additive-vs-multiplicative prediction directly: the
op-specific difference PR #1659 found for `Mul` (a combined
requantization multiplier, at a `Mul`-specific locator) does NOT
generalize to every two-live-input op -- it is specific to
multiplicative ops, and `Sub`, being additive like `Add`, does not
need it.

**NOT established**: whether `Div` (multiplicative, like `Mul`) also
needs its own combined multiplier, or reuses Mul's own formula/locator
verbatim (not tested here); whether this holds at shapes other than
`(1,16)`; whether Sub's own operand order (the arc's established
"first operand gets x1-style treatment" rule, PR #1658) generalizes
identically here (not re-tested, assumed consistent given every other
finding in this file matched Add's own exactly).
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
    path = os.path.join(FIX, f"sub_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(seed1, seed2):
    return mcode.decode(load(seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration data
    this file's own fixture-build script used for each `Sub` input,
    matching PR #1657's/#1659's own Add/Mul convention and shape."""
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


class TestX1UsesTheSameLocatorsAddAndMulBothUsed(unittest.TestCase):
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


class TestX2UsesTheSameLocatorButHasNoFindableScale(unittest.TestCase):
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


class TestYScaleIsDirectlyStoredLikeAddNotLikeMul(unittest.TestCase):
    def test_no_mul_style_requant_multiplier_field_group_exists(self):
        """Confirms Sub does NOT carry Mul's own separate `field in
        (224,240)` requantization-multiplier group at all."""
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            hits = [
                r
                for r in recs
                if r.get("kind") == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (224, 240)
            ]
            self.assertEqual(hits, [], (seed1, seed2))

    def test_plain_float32_formula_is_one_ulp_off_at_the_first_seed_pair(self):
        """Confirms the starting point this finding improves on, the same
        single-seed-pair miss Add's own Finding 3 (PR #1657) hit."""
        recs = decode(1, 2)
        ys = [a - b for a, b in zip(calib_samples(1), calib_samples(2))]
        _, _, sy = asymmetric_uint8_quant_params(ys)
        target = struct.pack("<f", np.float32(sy))
        hits = [
            r
            for r in recs
            if r.get("kind") == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("field") in (96, 112, 128)
            and r.get("operand") == target
        ]
        self.assertEqual(hits, [])

    def test_round_scale_once_is_bit_exact_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            sy32 = round_scale_once(ys)
            target = struct.pack("<f", sy32)
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


class TestZpYMatchesAtReg14Tag131LikeAddNotLikeMul(unittest.TestCase):
    def test_zp_y_matches_after_excluding_the_unrelated_constant_hit(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 14 and r.get("tag") == 131
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(4, bytes_seen, (seed1, seed2, bytes_seen))
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen, zpy))

    def test_no_reg76_tag132_locator_mul_used_carries_zp_y_here(self):
        """Confirms Sub does NOT use Mul's own separate `zp_y` locator
        (`reg=76,tag=132`) -- a further, independent check that Sub
        follows Add's own encoding, not Mul's."""
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a - b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S"
                and r.get("reg") == 76
                and r.get("tag") == 132
                and r.get("payload")
                and r["payload"][-1] == zpy
            ]
            self.assertEqual(hits, [], (seed1, seed2))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
