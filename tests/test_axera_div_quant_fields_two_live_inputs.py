"""Extends the two-live-input quantization survey (`tests/test_axera_
add_quant_fields_two_live_inputs.py`, PR #1657; `tests/test_axera_
mul_quant_fields_two_live_inputs.py`, PR #1659; `tests/test_axera_
sub_quant_fields_two_live_inputs.py`, PR #1661) to `Div`, the arc's own
explicitly flagged next candidate. `Div` (`y = x1 / x2`) is
multiplicative like `Mul` in the dequantized domain, so PR #1659's own
additive-vs-multiplicative hypothesis would predict a combined
requantization quantity rather than a direct `y_scale` -- this file
finds that prediction does NOT hold as stated: `Div` stores `y_scale`
DIRECTLY, like `Add`/`Sub`, not a combined multiplier like `Mul`,
revising the hypothesis (see "What this establishes" below).

## A calibration-design note: `x2` (the divisor) needed a different
## distribution than every other op in this arc used

Every prior two-live-input file drew both inputs from plain
`RandomState(seed).randn(1,16)`. For `Div`, `x2` is a divisor -- values
near zero would blow up `y=x1/x2`'s own calibration range to a
degenerate span, and (more importantly, discovered while writing this
file, see below) drove `x2`'s own zero point to a degenerate `0`.
`x2` here is instead `sign * (2.0 + 0.3*randn())` with `sign` an
independent `+-1` coin flip per element -- roughly half the elements
land near `+2`, half near `-2`, straddling zero for a genuine
non-degenerate zero point while no individual element ever lands
within five sigma (`~1.1`) of zero itself.

## Finding 0 (methodological, not about Div's own semantics): an
## all-positive `x2` makes its own zero point degenerate (exactly `0`),
## which suppresses the `reg=94,tag=132` locator entirely

The first attempt used an all-positive `x2` (`2.0 + 0.3*randn()`, no
sign flip) to sidestep the near-zero-divisor risk. Every one of 3 real
builds under that design gave `x2` a zero point of exactly `0`
(`ComputeAsymmetricUint8QuantParams`'s own `lo = min(0, samples.min())`
clamps to `0` for any strictly-positive tensor, by construction, not a
Div-specific quirk) -- and in every one of those 3 builds, NO
`reg=94,tag=132` record exists anywhere in the stream at all (checked
directly: `reg=94` fires at other tags, never `tag=132`). This is the
same shape of "degenerate value gets a trivial/absent encoding" pattern
`scripts/axera/README.md`'s own "Differential analysis" section
documents for CONSTANT operands (e.g. `Add`'s own trivial `{0,1,2}`
fast path) -- here observed for the first time for a LIVE input's own
zero point, not a compile-time constant. These 3 builds are kept,
renamed `*_trivialzp2.mcode.gz`, as real evidence for this specific
claim (`TestReg94Tag132LocatorIsAbsentWhenX2sOwnZeroPointIsTriviallyZero`
below) rather than discarded. The main 3 fixtures below use the
sign-flip design instead, giving each seed pair a genuine, non-trivial
`x2` zero point, which is what the rest of this file characterizes.

## Fixtures

Three fresh `Div(x1[1,16], x2[1,16])` builds (sign-flip `x2` design
above), same seed pairs PR #1657/#1659/#1661 used (`(1,2)`, `(7,42)`,
`(100,999)`) for direct comparability, same `RandomState(seed)`
convention for `x1` and for `x2`'s own magnitude/sign draws, same
`calibration_method: MinMax`, same `pulsar2:7.0-lite` Docker build.
`mcode.check()` reports zero errors on all three (and on the 3
trivial-zp2 fixtures).

## Finding 1: `x1`/`x2` are treated like every other op in this
## cluster -- same locators, bit-exact (given a non-degenerate `x2`)

`x1`'s own zero point is the same literal 6-byte quad
(`02 10 1b <zp_x1> 83 36`); its own `1/x_scale` is the same
`verb=161,bank=15,field in (96,112,128)` locator, 3 redundant copies.
`x2`'s own zero point decodes at the SAME `reg=94,tag=132` pair
Add/Mul/Sub all used, bit-exact on all 3 tested seed pairs (once `x2`'s
own zero point is non-degenerate -- see Finding 0). `x2`'s own scale is,
like every other op's, nowhere findable (exact-bytes `V`-record scan
plus a raw byte-offset float32 scan, tolerance `1e-4`, both negative for
`x2_scale` and `1/x2_scale`).

## Finding 2 (revises PR #1659's own prediction): `y_scale` is
## DIRECTLY stored, bit-exact under the PLAIN float32 formula (no
## round-scale-once needed) -- Div does NOT use Mul's combined-
## multiplier mechanism, despite also being multiplicative

PR #1659's own diff predicted `Div` (multiplicative, like `Mul`) would
need a combined requantization quantity bridging `x1`'s and `x2`'s own
scales to `y`'s, the same way `Mul` needed `y_scale/(x1_scale*x2_scale)`.
It does not: no `field in (224,240)` group (`Mul`'s own multiplier
locator) exists anywhere in any of the 3 tested fixtures -- checked
directly. Instead, `y_scale` itself (the plain `ComputeAsymmetricUint8
QuantParams` scale from `y=x1/x2`'s own calibration range, no
round-scale-once refinement needed) appears bit-exact, 4 REDUNDANT
copies (not 3, unlike `x1`'s own triple), at a raw, undecoded byte run
(near offset 1549-1553, shifting by a few bytes between seed pairs --
the same class of gap PR #1634's `tail_vector`-adjacent regions and
Gemm/MatMul's own `y_scale` raw runs (PR #1646/#1647) already showed;
`mcode.decode()` under `FULL_RULE` does not resolve this region into
named records). This is a DIRECT match to `Add`/`Sub`'s own mechanism,
not `Mul`'s.

## What this establishes, precisely, and what it does not

**Established**: `x1`/`x2`'s own treatment generalizes to `Div`,
register-for-register, tag-for-tag, with every other tested op in this
cluster. `Div` stores `y_scale` DIRECTLY, bit-exact (even more cleanly
than `Add`/`Sub`, which needed PR #1654's round-scale-once formula for
one seed pair each -- `Div`'s own 3 tested seed pairs are all bit-exact
under the plain formula). `zp_y` decodes cleanly at `reg=74,tag=132`
(after excluding a fixed, irrelevant `16` payload byte also present at
that `(reg,tag)` pair -- the same constant-filtering pattern Add's own
`reg=14` and Mul's own `reg=76` needed) -- a locator distinct from
every other op's own `zp_y` (Add/Sub: `reg=14,tag=131`; Mul:
`reg=76,tag=132`), though it does share `Mul`'s own TAG (`132`), unlike
Add/Sub's `131`.

This means PR #1659's own additive-vs-multiplicative hypothesis, as
literally stated ("multiplicative ops need a combined requant
multiplier"), does NOT hold for `Div` -- the arc's second multiplicative
op behaves like the ADDITIVE ops on this specific axis (`y_scale`
stored directly), not like `Mul`. The cleaner, revised story this
file's own evidence supports: `Mul`'s combined multiplier is specific
to `Mul` (or to some other property `Div` does not share -- not
`Div`'s own multiplicative-ness alone, which `Div` also has), not a
general "multiplicative ops need a combined requant quantity" rule.
`Div` DOES differ from `Add`/`Sub` on `zp_y`'s own tag (sharing `Mul`'s
`132` instead of Add/Sub's `131`) -- a partial, not total, alignment
with either camp.

**NOT established**: WHY `Mul` specifically needs a combined
multiplier while `Div` (also multiplicative) does not -- this file
does not have Pulsar2's own source and only reports what each op's
real mcode bytes show, not a mechanistic explanation for the split;
whether this holds at shapes other than `(1,16)`; whether `Div`'s own
operand order (dividend vs. divisor) is governed by ONNX node operand
order the same way PR #1658 found for `Add` (not re-tested here);
whether the `reg=94,tag=132`-absent-when-trivial phenomenon (Finding 0)
also applies to `Add`/`Mul`/`Sub`'s own `x2` if IT were made
degenerate (their own fixtures never had a degenerate `x2`, so this is
inferred by mechanism, not independently confirmed for those ops).
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
    path = os.path.join(FIX, f"div_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def load_trivial(seed1, seed2):
    path = os.path.join(
        FIX, f"div_1x16_two_live_seed{seed1}_{seed2}_trivialzp2.mcode.gz"
    )
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(seed1, seed2):
    return mcode.decode(load(seed1, seed2), **mcode.FULL_RULE)


def x1_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data this file's own fixture-build script used for `x1`, matching
    PR #1657's/#1659's/#1661's own Add/Mul/Sub convention and shape."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def x2_samples(seed, shape=(1, 16), n_samples=4):
    """`sign * (2.0 + 0.3*randn())`, `sign` an independent `+-1` coin
    flip per element -- see this file's own module docstring for why
    plain `randn()` (used by every other op in this cluster) does not
    work for a divisor: it drives `x2`'s own zero point to a
    degenerate `0` (Finding 0), not merely a near-zero-division risk."""
    rng = np.random.RandomState(seed)
    mag = 2.0 + 0.3 * rng.randn(*shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return [(sign * mag).astype(np.float32) for _ in range(n_samples)]


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


class TestX1UsesTheSameLocatorsEveryOtherOpInThisClusterUsed(unittest.TestCase):
    def test_zp_x1_literal_quad_matches_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load(seed1, seed2)
            zp1, _, _ = asymmetric_uint8_quant_params(x1_samples(seed1))
            quad = bytes.fromhex("02101b") + bytes([zp1]) + bytes.fromhex("8336")
            hits = [i for i in range(len(data) - 5) if data[i : i + 6] == quad]
            self.assertEqual(len(hits), 1, (seed1, seed2, zp1))

    def test_recip_x1_scale_is_bit_exact_at_verb161_bank15(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            _, _, s1 = asymmetric_uint8_quant_params(x1_samples(seed1))
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
            zp2, _, _ = asymmetric_uint8_quant_params(x2_samples(seed2))
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
            zp2, _, _ = asymmetric_uint8_quant_params(x2_samples(seed2))
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
            _, _, s2 = asymmetric_uint8_quant_params(x2_samples(seed2))
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
                    if abs(raw - val) / abs(val) < 1e-4:
                        hits.append(i)
                self.assertEqual(hits, [], (seed1, seed2, name, val))


class TestReg94Tag132LocatorIsAbsentWhenX2sOwnZeroPointIsTriviallyZero(
    unittest.TestCase
):
    """Finding 0: an all-positive `x2` (this file's own first, since-
    replaced calibration attempt) drives `x2`'s own zero point to a
    degenerate `0`, and the `reg=94,tag=132` locator this whole cluster
    otherwise relies on for `x2`'s own zero point is entirely absent in
    that case -- kept as real, separately-built evidence rather than
    discarded once the main fixtures moved to a non-degenerate design."""

    def test_all_three_trivial_fixtures_decode_cleanly(self):
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load_trivial(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))

    def test_x2_zero_point_is_trivially_zero_in_the_trivial_fixtures(self):
        # The all-positive design this file's own module docstring
        # describes: `mag = 2.0 + 0.3*randn()`, no sign flip.
        for seed1, seed2 in SEED_PAIRS:
            rng = np.random.RandomState(seed2)
            samples = [
                (2.0 + 0.3 * rng.randn(1, 16)).astype(np.float32) for _ in range(4)
            ]
            zp2, _, _ = asymmetric_uint8_quant_params(samples)
            self.assertEqual(zp2, 0, (seed1, seed2))

    def test_no_reg94_tag132_record_exists_in_any_trivial_fixture(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = mcode.decode(load_trivial(seed1, seed2), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 94 and r.get("tag") == 132
            ]
            self.assertEqual(hits, [], (seed1, seed2))


class TestYScaleIsDirectlyStoredNotLikeMulsMultiplier(unittest.TestCase):
    def test_no_mul_style_requant_multiplier_field_group_exists(self):
        """Confirms Div does NOT carry Mul's own separate `field in
        (224,240)` requantization-multiplier group -- the concrete
        check behind Finding 2's own "revises the hypothesis" claim."""
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

    def test_plain_y_scale_is_bit_exact_at_a_raw_byte_offset_four_times(self):
        for seed1, seed2 in SEED_PAIRS:
            data = load(seed1, seed2)
            ys = [a / b for a, b in zip(x1_samples(seed1), x2_samples(seed2))]
            _, _, sy = asymmetric_uint8_quant_params(ys)
            target = struct.pack("<f", np.float32(sy))
            hits = [i for i in range(len(data) - 3) if data[i : i + 4] == target]
            self.assertEqual(len(hits), 4, (seed1, seed2, sy))
            # the 4 copies are evenly spaced (7 bytes apart), matching
            # this file's own module-docstring description.
            gaps = {b - a for a, b in zip(hits, hits[1:])}
            self.assertEqual(gaps, {7}, (seed1, seed2, hits))


class TestZpYMatchesAtReg74Tag132ADistinctLocatorFromEveryOtherOp(unittest.TestCase):
    def test_zp_y_matches_after_excluding_the_unrelated_constant_hit(self):
        """`reg=74,tag=132` carries two records: one with a fixed,
        irrelevant payload byte of `16` (the same "filter out the
        constant hit" pattern Add's own `reg=14` and Mul's own
        `reg=76` needed, PRs #1657/#1659), and one matching the
        recomputed `zp_y` bit-for-bit."""
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a / b for a, b in zip(x1_samples(seed1), x2_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 74 and r.get("tag") == 132
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(16, bytes_seen, (seed1, seed2, bytes_seen))
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen, zpy))

    def test_no_reg14_tag131_locator_add_sub_used_carries_zp_y_here(self):
        """Confirms Div does NOT use Add/Sub's own `zp_y` locator
        (`reg=14,tag=131`) -- Div's own `zp_y` tag (`132`) actually
        matches Mul's, not Add/Sub's, a partial (not total) alignment
        with the "multiplicative" camp this file's own module docstring
        discusses."""
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a / b for a, b in zip(x1_samples(seed1), x2_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S"
                and r.get("reg") == 14
                and r.get("tag") == 131
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
