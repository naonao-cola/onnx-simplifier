"""Synthesizes the `m*k=64` `x2`-zero-point-locator correction thread
(PRs #1673/#1674/#1675, all extending `tests/test_axera_add_two_live_
shape_generality.py`, PR #1671) into one final, verified record for
the two-live-input elementwise cluster (Add/Sub/Mul/Div).

## Background: a caught mistake, not a new investigation

PR #1671's own "Finding 4" claimed that at shape `m*k=64`, `x2`'s own
zero-point locator (`reg=94,tag=132`) "relocates" to a new `reg=66,
tag=131` pair for `Add` -- "identical (reg,tag) pair, identical value
(100), on both splits" of `m*k=64`. That claim was built from exactly
ONE calibration seed pair, `(1,2)`.

- PR #1673 built `Mul` at `m*k=64` across all THREE of this cluster's
  standard seed pairs and found the claim does not hold: `reg=66,
  tag=131`'s own payload is a FIXED CONSTANT (`0x64` = `100`), the same
  value at every seed and at the unrelated `m*k=16` baseline shape too
  -- it does not track `x2`'s own real, seed-varying zero point at all.
  PR #1671's own `(1,2)` match was a coincidence: `x2`'s own real zero
  point at that one seed pair happens to also equal `100`.
- PR #1674 re-ran the identical multi-seed check for `Add` itself (the
  op PR #1671's own claim was actually about) and found the identical
  correction applies there too.
- PR #1675 extended the (now correctly multi-seed-from-the-start)
  characterization to `Sub` and `Div`: `Sub` matches Add/Mul's own
  corrected pattern exactly; `Div` genuinely differs (see the table
  below).

This file does not re-derive any of that -- it directly re-decodes and
recomputes each contributing PR's own already-committed fixtures, independently
of their own test modules, to confirm the corrected record is
consistent across all four ops in one place.

## The corrected picture, one table, all four ops

| op | `reg=94,tag=132` at `m*k=64` | `reg=66,tag=131` payload |
| --- | --- | --- |
| Add | seed-dependent: present+bit-exact at 2/3 seeds, absent at `(1,2)` | fixed constant `100`, same at every seed |
| Mul | seed-dependent: present+bit-exact at 2/3 seeds, absent at `(1,2)` | fixed constant `100`, same at every seed |
| Sub | seed-dependent: present+bit-exact at 2/3 seeds, absent at `(1,2)` | fixed constant `100`, same at every seed |
| Div | present+bit-exact at ALL 3 tested seeds, never absent | fixed constant `100` still present (plus an unrelated `0x20` byte), but not needed as a fallback |

Three of four ops share one pattern exactly (down to which specific
seed pair triggers the absence); `Div` is a genuine, directly-confirmed
exception on the presence axis, while still sharing the "`reg=66,
tag=131` is an unrelated fixed constant" half of the picture with the
other three.

## A note on the arc's own self-correcting discipline

PR #1671's original single-seed claim was wrong, and it was caught by
straightforward follow-up work that simply tested more seeds -- not by
doubting the methodology in general. This is worth stating plainly for
future readers of this whole quantization-field arc: a claim built
from one calibration seed pair is provisional until a second (or
third) seed pair confirms it, exactly as this correction thread
demonstrates. This is NOT a reason to distrust every other single-seed
finding in this arc -- most of this arc's own bit-exactness claims
(e.g. the `ComputeAsymmetricUint8QuantParams` zero-point/scale
formulas themselves, PRs #1636/#1640-#1655) were independently
cross-validated multiple different ways (multiple ops, multiple
shapes, control fixtures, specificity/near-miss checks), not resting
on a single coincidental byte match the way PR #1671's own Finding 4
did. The lesson here is specific -- treat a claim resting on exactly
one data point as provisional -- not a blanket invalidation of the
arc's own methodology.

## What remains genuinely open

- WHY `(1,2)` specifically triggers the `reg=94,tag=132` absence for
  Add/Sub/Mul but `Div` never shows it at any of the three tested seed
  pairs (no access to Pulsar2's own source; PR #1673 already ruled out
  a rounding-near-tie explanation for `Mul`).
- What `reg=66,tag=131`'s own constant value (`100`) actually
  represents (a fixed configuration/format byte, not calibration
  data -- never decoded further).
- Whether this whole pattern (both the seed-dependent absence and the
  fixed-constant fallback) holds at `m*k` totals other than `64`, or at
  shapes beyond the `(1,16)`/`(1,32)`/`(2,32)`/`(1,64)` family this
  whole cluster has tested so far.
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
ABSENT_SEED_PAIR = (1, 2)
PRESENT_SEED_PAIRS = [(7, 42), (100, 999)]
FIXED_CONSTANT_PAYLOAD = bytes.fromhex("e01b64")

# (op, needs_div_safe_x2)
OPS_WITH_SEED_DEPENDENT_ABSENCE = ["add", "mul", "sub"]


def load(op, seed1, seed2):
    path = os.path.join(FIX, f"{op}_2x32_two_live_seed{seed1}_{seed2}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(op, seed1, seed2):
    return mcode.decode(load(op, seed1, seed2), **mcode.FULL_RULE)


def calib_samples(seed, shape=(2, 32), n_samples=4):
    """`RandomState(seed).randn(2, 32)` x4 -- the exact calibration
    data every op's own `x2` uses in this whole correction thread,
    except `Div`'s (see `div_safe_x2_samples`)."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def div_safe_x2_samples(seed, shape=(2, 32), n_samples=4):
    """`sign * (2.0 + 0.3*randn())` -- PR #1663's own sign-flip divisor
    design, reused unchanged for `Div`'s own `x2` (a plain zero-centered
    draw would risk a near-zero divisor)."""
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


def reg_hits(recs, reg, tag):
    return [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == reg and r.get("tag") == tag
    ]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_op_or_seed_pair(self):
        for op in ["add", "mul", "sub", "div"]:
            for seed1, seed2 in SHAPE64_SEED_PAIRS:
                errs = mcode.check(load(op, seed1, seed2))
                self.assertEqual(errs, [], (op, seed1, seed2, errs))


class TestReg66Tag131IsAFixedConstantAcrossAllFourOps(unittest.TestCase):
    """The core correction, directly re-confirmed for every op in the
    cluster: this locator's own payload never tracks x2's real,
    seed-varying zero point."""

    def test_fixed_constant_present_at_every_op_and_seed_pair(self):
        for op in ["add", "mul", "sub", "div"]:
            for seed1, seed2 in SHAPE64_SEED_PAIRS:
                recs = decode(op, seed1, seed2)
                hits = reg_hits(recs, 66, 131)
                payloads = [h["payload"] for h in hits]
                self.assertIn(
                    FIXED_CONSTANT_PAYLOAD, payloads, (op, seed1, seed2, payloads)
                )

    def test_constant_does_not_track_the_real_seed_varying_zero_point(self):
        zp2_by_seed = {}
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
            zp2_by_seed[(seed1, seed2)] = zp2
        self.assertEqual(zp2_by_seed, {(1, 2): 100, (7, 42): 103, (100, 999): 130})
        # Only the (1,2) pair's own zp2 coincides with the constant --
        # confirming PR #1671's own match was the coincidence, not the
        # rule.
        self.assertEqual(FIXED_CONSTANT_PAYLOAD[-1], zp2_by_seed[(1, 2)])
        self.assertNotEqual(FIXED_CONSTANT_PAYLOAD[-1], zp2_by_seed[(7, 42)])
        self.assertNotEqual(FIXED_CONSTANT_PAYLOAD[-1], zp2_by_seed[(100, 999)])


class TestReg94Tag132IsSeedDependentForAddMulSubButNotDiv(unittest.TestCase):
    """The corrected presence picture: three ops share the identical
    seed-dependent absence (always at (1,2) specifically); Div is a
    real, directly-confirmed exception."""

    def test_add_mul_sub_are_present_and_bit_exact_at_the_two_non_absent_seeds(self):
        for op in OPS_WITH_SEED_DEPENDENT_ABSENCE:
            for seed1, seed2 in PRESENT_SEED_PAIRS:
                recs = decode(op, seed1, seed2)
                zp2, _, _ = asymmetric_uint8_quant_params(calib_samples(seed2))
                hits = reg_hits(recs, 94, 132)
                self.assertEqual(len(hits), 1, (op, seed1, seed2))
                self.assertEqual(hits[0]["payload"][-1], zp2, (op, seed1, seed2, zp2))

    def test_add_mul_sub_are_absent_at_exactly_the_1_2_seed_pair(self):
        for op in OPS_WITH_SEED_DEPENDENT_ABSENCE:
            recs = decode(op, *ABSENT_SEED_PAIR)
            hits = reg_hits(recs, 94, 132)
            self.assertEqual(hits, [], op)

    def test_div_is_present_and_bit_exact_at_every_tested_seed_pair_including_1_2(
        self,
    ):
        for seed1, seed2 in SHAPE64_SEED_PAIRS:
            recs = decode("div", seed1, seed2)
            zp2, _, _ = asymmetric_uint8_quant_params(div_safe_x2_samples(seed2))
            hits = reg_hits(recs, 94, 132)
            self.assertEqual(len(hits), 1, (seed1, seed2))
            self.assertEqual(hits[0]["payload"][-1], zp2, (seed1, seed2, zp2))


if __name__ == "__main__":
    unittest.main()
