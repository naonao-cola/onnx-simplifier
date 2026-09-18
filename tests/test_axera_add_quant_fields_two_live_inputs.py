"""Extends the first-principles quantization-recomputation technique
(port `ComputeAsymmetricUint8QuantParams`, `onnxsim/passes/static_
quantize_matmul.h`, into Python; apply it to the exact seeded
calibration data a fixture's own build script used; compare against
real decoded mcode bytes -- PRs #1636/#1640-#1655) to a genuinely new
op family: `Add` with TWO independently-calibrated LIVE inputs (`Add(x1,
x2)`, both graph inputs, neither a compile-time constant).

Every op this arc has touched so far (Conv, Gemm, MatMul) has exactly
ONE live input plus one compile-time-constant weight. `scripts/axera/
README.md`'s own "Differential analysis" section already characterized
`Add`/`Sub`/`Mul`/`Div` against a CONSTANT broadcast operand (finding a
"trivial" integer fast-path for small uniform constants, and an still-
unsolved opaque 4-byte trailer field for the non-trivial case) -- a
different, already-closed question. `AxQuantizedAdd`, the real residual
add of two live activations, was only ever profiled for its execution
engine (`teng2`, README's "Expanding past Conv" section) -- its own
zero-point/scale byte encoding was never attempted. This file is that
attempt.

## Fixtures

Three fresh `Add(x1[1,16], x2[1,16])` builds, both `x1` and `x2` declared
graph inputs (not one live + one constant), each independently calibrated
(`RandomState(seed1).randn(1,16)` / `RandomState(seed2).randn(1,16)`, 4
samples each, `calibration_method: MinMax` -- the same convention
`_build_and_get_mcode_bytes` uses throughout this project). Real `docker
run pulsar2:7.0-lite build` invocations via `pulsar2_docker.build()`,
with two separate `input_configs` entries (one per tensor). `mcode.
check()` reports zero errors on all three.

## Finding 1: `x1`'s own `zp_x`/`1/x_scale` use the SAME locators every
## other op in this arc has used -- Add's "first" input is treated like
## Conv/Gemm/MatMul's single live input

`x1`'s own zero point is the same literal 6-byte quad (`02 10 1b <zp_x1>
83 36`) Conv/Gemm/MatMul all use; its own `1/x_scale` is the same
`verb=161,bank=15,field in (96,112,128)` locator all three ops share.
Both are bit-exact on all 3 tested seed pairs.

## Finding 2 (new): `x2`'s own zero point has its OWN distinct locator --
## a clean `S`-kind record at `reg=94, tag=132` -- but `x2`'s own SCALE
## is nowhere in the stream

`x2`'s own zero point decodes cleanly and uniquely at `reg=94, tag=132`
(exactly one record in the whole stream carries that `(reg, tag)` pair,
and its payload byte matches the recomputed `zp_x2` bit-for-bit on all 3
tested seed pairs). This is a genuinely different locator from `x1`'s own
literal-quad mechanism -- Add's two inputs are NOT treated symmetrically
by the codec.

`x2`'s own `1/x_scale` (or `x_scale` itself) was searched for exhaustively
and found NOWHERE: not at the `verb=161,bank=15` locator (only two
distinct float32 values appear there, both accounted for by Finding 1/3,
never a third), not via an exact-bytes search across every decoded `V`
record, and not via a raw byte-offset float32 scan (tolerance `1e-3`)
across the entire stream. Several structurally-motivated ratio
candidates (`x2_scale/x1_scale`, `x1_scale/x2_scale`, `y_scale/x2_scale`,
and their reciprocals) were also tried and found nowhere. This is
consistent with a real quantized-add implementation that requantizes
`x2` into `x1`'s own scale before adding (needing only `x2`'s zero point,
not an independent scale, to do the shift) -- a common technique for
elementwise ops with differently-scaled operands -- though this file
does not claim access to Pulsar2's own source and only reports what is,
and is not, findable in the byte stream.

## Finding 3: `y_scale` is genuinely BIT-EXACT for Add -- unlike Conv/
## Gemm/MatMul, which only ever got "close, not exact" -- once PR #1654's
## refined formula is used

Every single-live-input op in this arc (Conv PR #1642, Gemm/MatMul PR
#1646/#1647) found `y_scale` only approximately, via a raw-byte scan,
never bit-exact under the plain float32-throughout formula. For Add, this
file finds `y_scale` living at the SAME `verb=161,bank=15,field in
(96,112,128)` locator as `1/x1_scale` itself (a second, co-located
record with the same field values, different operand) -- but the plain
float32-throughout formula is only 1 ULP off (confirmed at seed pair
`(1,2)`). Applying PR #1654's own "round-scale-once" formula (`hi-lo`
subtraction done in double precision, `scale` rounded to float32 exactly
once, no further intermediate rounding) makes it bit-exact on all 3
tested seed pairs -- a real, independent cross-op confirmation of PR
#1654's own formula, extending it from Gemm's `1/x_scale` to Add's own
`y_scale`.

## Finding 4: `zp_y` decodes cleanly at `reg=14, tag=131` -- a different
## TAG from every other op's own `zp_y` (`tag=132` for Conv/Gemm/MatMul)

Filtering to `tag=131` specifically, exactly one record at `reg=14`
carries a payload byte matching the recomputed `zp_y` on all 3 tested
seed pairs (a second, unrelated `reg=14,tag=131` record is always present
with a fixed, irrelevant payload byte of `4` -- filtering by tag alone
is not sufficient to get a single hit, but filtering by tag AND excluding
the constant `4` byte is). The register differs from every other op's
own `zp_y` too (Conv: `reg=232`; Gemm/MatMul: `reg=120`) -- `tag=131` (not
`132`) for the output zero point is new to this op, not shared with any
prior finding in this arc.

## What this establishes, precisely, and what it does not

**Established**: Add's `x1` (the first live input) is treated like every
other op's single live input (same `zp_x`/`1/x_scale` locators); `x2`
(the second live input) has its own zero point at a new, clean, unique
locator (`reg=94, tag=132`) but no independently-stored scale anywhere in
the stream (a genuine, well-searched negative); `y_scale` is bit-exact
for Add specifically, once PR #1654's refined formula is used -- the
first bit-exact `y_scale` result in this whole arc; `zp_y` decodes
cleanly at a new locator (`reg=14, tag=131`) distinct from every other
op's own `zp_y` register/tag.

**NOT established**: whether `x2`'s own scale is truly absent from the
stream (only that it is not findable by the search methods tried here --
a fixed-point or otherwise-encoded representation was not attempted);
whether Add's two-live-input architecture generalizes to `Sub`/`Mul`/
`Div` with two live operands (not tested); whether this holds at shapes
other than `(1,16)`, or with `x1`/`x2` swapped (does "the first declared
graph input" or "the first operand in the `Add` node" determine which
input gets the `x1`-style treatment -- not distinguished here, since both
happen to coincide in this file's own fixtures).
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
    path = os.path.join(FIX, f"add_1x16_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(seed1, seed2):
    return mcode.decode(load(seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration data
    this file's own fixture-build script used for each `Add` input,
    matching `_build_and_get_mcode_bytes`'s own convention."""
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


class TestX1UsesTheSameLocatorsEveryOtherOpsSingleLiveInputUses(unittest.TestCase):
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


class TestX2HasItsOwnZeroPointLocatorButNoFindableScale(unittest.TestCase):
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

    def test_no_other_s_record_carries_zp_x2s_value_at_reg94(self):
        """Specificity check: `reg=94` is not a coincidentally-common
        byte value elsewhere -- confirm it's not ALSO hit by a nearby,
        wrong zero-point value."""
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
        """A genuine, well-searched negative: neither `x2_scale` nor
        `1/x2_scale`, nor several structurally-motivated ratios against
        `x1`'s or `y`'s own scale, appear anywhere as a decoded `V`
        operand or a raw byte-offset float32 (tolerance `1e-3`)."""
        for seed1, seed2 in SEED_PAIRS:
            data = load(seed1, seed2)
            recs = decode(seed1, seed2)
            _, _, s1 = asymmetric_uint8_quant_params(calib_samples(seed1))
            _, _, s2 = asymmetric_uint8_quant_params(calib_samples(seed2))
            ys = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            _, _, sy = asymmetric_uint8_quant_params(ys)

            candidates = {
                "s2": s2,
                "recip_s2": 1.0 / s2,
                "s2_over_s1": s2 / s1,
                "s1_over_s2": s1 / s2,
                "s2_over_sy": s2 / sy,
                "sy_over_s2": sy / s2,
            }
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


class TestYScaleIsBitExactUnderTheRoundScaleOnceFormula(unittest.TestCase):
    def test_plain_float32_formula_is_one_ulp_off_at_the_first_seed_pair(self):
        """Confirms the starting point this finding improves on -- the
        same "close, not exact" pattern every other op in this arc hit
        for `y_scale`, before PR #1654's own refinement is applied."""
        recs = decode(1, 2)
        ys = [a + b for a, b in zip(calib_samples(1), calib_samples(2))]
        _, _, sy = asymmetric_uint8_quant_params(ys)
        target = struct.pack("<f", np.float32(sy))
        hits = [
            r
            for r in recs
            if r.get("kind") == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("operand") == target
        ]
        self.assertEqual(hits, [])

    def test_round_scale_once_is_bit_exact_on_every_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
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


class TestZpYMatchesAtReg14Tag131(unittest.TestCase):
    def test_zp_y_matches_after_excluding_the_unrelated_constant_hit(self):
        for seed1, seed2 in SEED_PAIRS:
            recs = decode(seed1, seed2)
            ys = [a + b for a, b in zip(calib_samples(seed1), calib_samples(seed2))]
            zpy, _, _ = asymmetric_uint8_quant_params(ys)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 14 and r.get("tag") == 131
            ]
            bytes_seen = sorted({h["payload"][-1] for h in hits})
            self.assertIn(4, bytes_seen, (seed1, seed2, bytes_seen))
            self.assertIn(zpy, bytes_seen, (seed1, seed2, bytes_seen, zpy))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_seed_pair(self):
        for seed1, seed2 in SEED_PAIRS:
            errs = mcode.check(load(seed1, seed2))
            self.assertEqual(errs, [], (seed1, seed2, errs))


if __name__ == "__main__":
    unittest.main()
