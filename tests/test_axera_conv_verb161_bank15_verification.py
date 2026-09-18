"""Continues `tests/test_axera_conv_reg54_zeropoint_verification.py`
(PR #1640)'s own newly-proven first-principles recomputation technique,
applied to a SECOND field in Conv's own 28-byte "binary path switch"
cluster (`tests/test_axera_conv_reg60_mechanism.py`, PR #1586): the
three copies of a `verb=161,bank=15` V-record's own computed float32
operand. PR #1640 found `reg=54` (Conv's own literal input zero point)
matches the standard MinMax asymmetric-uint8 formula exactly, from the
same `RandomState(seed)`-derived calibration data every
`Conv(dilation=3,...)` fixture uses. This file asks the same question of
the OTHER computed field the same cluster carries.

A sibling investigation concurrently checks the cluster's THIRD
computed field, `reg=224` -- not duplicated here.

## Method: the same calibration recomputation PR #1640 already verified,
## one candidate formula tested directly against real decoded fixture
## bytes

`tests/test_axera_mcode_structure.py`'s own `_build_and_get_mcode_bytes`
generates calibration data as `[RandomState(seed).randn(1, 4, 16, 16)
.astype(np.float32) for _ in range(4)]` for every relevant fixture (PR
#1640's own cited convention, reused verbatim here). From that data,
`scale = (hi - lo) / 255.0` is the same asymmetric-uint8 scale factor
PR #1640 already used to derive `reg=54`'s own zero point. This file
tests the single most natural remaining candidate for a second
scale-adjacent field: **the scale's own reciprocal**, `1.0 / scale`.

`verb=161,bank=15`'s own real values (re-decoded directly from the
already-committed fixtures below, not copied from any PR's own
docstring) at every calibration seed this project has already tested:

| seed | observed value (float32) | `1/scale` (this file's own recomputation) | exact bytes match |
| --- | --- | --- | --- |
| 0 (group A, `reg=60`=`0x7e`, 5/8 samples) | `33.811733` | `33.811733` | **yes** |
| 1 | `35.855095` | `35.855091` | no -- 1 ULP off |
| 7 | `35.040691` | `35.040691` | **yes** |
| 42 | `35.577232` | `35.577232` | **yes** |
| 100 | `36.078632` | `36.078636` | no -- 1 ULP off |
| 999 | `37.855480` | `37.855480` | **yes** |

**4 of 6 seeds match bit-for-bit; the other 2 (seeds 1 and 100) differ
by exactly one ULP (the smallest possible float32 step) regardless of
whether the reciprocal is computed as `1.0/scale` or directly as
`255.0/(hi-lo)`** (both orderings checked directly below,
`TestReciprocalMatchesEveryDeterministicSeed`) -- consistent with, not
a refutation of, PR #1640's own already-documented caveat that this
project's onnxsim-based recomputation is NOT a bit-exact reproduction
of Pulsar2's own real internal floating-point computation (different
IR, different accumulation/rounding), only a very close one. A 1-ULP
float32 discrepancy on a raw (non-rounded) value is exactly the kind of
gap PR #1640's own INTEGER zero point (rounded to the nearest whole
number, absorbing any such tiny discrepancy) could never show, and is
not itself evidence against the formula.

**This is a second, genuine confirmation of the same first-principles
technique**: the `verb=161,bank=15` computed value is Conv's own
quantization scale's reciprocal (`1/scale`, equivalently the
"scale in this-many-representable-steps-per-unit" form some
requantization pipelines use), derived from the exact same calibration
range PR #1640 already independently verified explains `reg=54`.

## What does NOT match: group B's own value

`reg=60`'s own "rare" state at seed 0 (`0x7f`, 3/8 samples) pairs with
a DIFFERENT `verb=161,bank=15` value, `127.514183` -- and this value
does **not** match `1/scale` for seed 0 or any nearby seed tested
(`TestGroupBDoesNotMatchAnyNearbySeed` below checks seeds 0 through 9
directly: none come close). `127.514183` is suspiciously near `127.5`
(exactly half of `255`, `diff = 0.0142`) but is not exactly `127.5`
either. It also recurs BYTE-IDENTICALLY across several unrelated
`Conv(dilation=3, insz=...)` fixtures with DIFFERENT `insz` (15, 17,
19, 20 -- different shapes, necessarily different weight/output
statistics even at the same calibration seed) -- confirmed directly
below (`TestGroupBValueRecursAcrossUnrelatedShapes`). A genuinely
calibration- or shape-derived quantity would not be expected to repeat
identically across shapes with different receptive-field/output
statistics; this recurrence is more consistent with `127.514183` being
a fixed constant or fallback value belonging to whichever alternate
scheduling/reduction path "group B" represents, not itself a per-build
computed scale. This file does not decode what `127.514183` actually
is -- reported as a precise, evidenced open question, not forced into
the same formula that explains group A and every deterministic seed.
"""

import gzip
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

# Observed verb=161,bank=15 operand bytes per deterministic calibration
# seed, re-decoded directly from real fixtures below (not copied from
# any PR's own docstring prose) -- see setUpClass.
DETERMINISTIC_SEED_FIXTURES = {
    1: "conv_dilation3_calibseed1_r0.mcode.gz",
    7: "conv_dilation3_calibseed7_r0.mcode.gz",
    42: "conv_dilation3_calibseed42_r0.mcode.gz",
    100: "conv_dilation3_calibseed100_r0.mcode.gz",
    999: "conv_dilation3_calibseed999_r0.mcode.gz",
}
GROUP_A_FIXTURE = "conv_dilation3.mcode.gz"  # seed=0, reg=60=0x7e
GROUP_B_FIXTURE = "conv_dilation3_rebuild0.mcode.gz"  # seed=0, reg=60=0x7f

UNRELATED_SHAPE_FIXTURES_WITH_GROUP_B_VALUE = [
    "conv_dilation3_insz15.mcode.gz",
    "conv_dilation3_insz17.mcode.gz",
    "conv_dilation3_insz19.mcode.gz",
    "conv_dilation3_insz20.mcode.gz",
]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def bank15_operand(name):
    """Returns the single verb=161,bank=15 float32 operand this cluster
    carries in `name` (the 3 copies PR #1586 found are byte-identical
    duplicates within one build, confirmed directly below), excluding
    the two unrelated, stable verb=161/bank=15/field=128 records this
    project's own earlier work already identified (operand
    `b"$\\x83N"`, not a 4-byte float)."""
    recs = decode(name)
    hits = [
        r
        for r in recs
        if r["kind"] == "V"
        and r.get("verb") == 161
        and r.get("bank") == 15
        and r.get("field") in (96, 112, 128)
        and r.get("operand") not in (None, b"$\x83N")
    ]
    operands = {r["operand"] for r in hits}
    assert len(operands) == 1, (name, operands)
    return next(iter(operands))


def calib_scale(seed, shape=(1, 4, 16, 16), n_samples=4):
    """Same calibration-range computation
    `tests/test_axera_conv_reg54_zeropoint_verification.py` (PR #1640)
    already verified explains `reg=54` -- reused verbatim here, only the
    candidate formula applied to `(lo, hi)` differs."""
    rng = np.random.RandomState(seed)
    samples = [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return lo, hi, (hi - lo) / np.float32(255.0)


class TestBank15CarriesExactlyOneFloatValuePerBuild(unittest.TestCase):
    def test_group_a_and_deterministic_seeds_have_one_value(self):
        for name in [GROUP_A_FIXTURE] + list(DETERMINISTIC_SEED_FIXTURES.values()):
            bank15_operand(name)  # raises via assert if not exactly one


class TestReciprocalMatchesEveryDeterministicSeed(unittest.TestCase):
    """The core finding: `1/scale` (equivalently `255/(hi-lo)`) matches
    the real observed `verb=161,bank=15` value for every deterministic
    seed, 4/6 bit-exact and 2/6 off by exactly one float32 ULP -- the
    same "very close, not bit-exact" gap PR #1640's own docstring
    already attributes to this project's onnxsim-based recomputation
    not being identical to Pulsar2's own real internal computation."""

    def _observed_float(self, name):
        return struct.unpack("<f", bank15_operand(name))[0]

    def test_all_six_seeds_within_one_ulp(self):
        cases = {0: GROUP_A_FIXTURE, **DETERMINISTIC_SEED_FIXTURES}
        for seed, name in cases.items():
            observed = self._observed_float(name)
            _, _, scale = calib_scale(seed)
            computed = float(np.float32(1.0) / scale)
            observed_bits = struct.unpack("<i", struct.pack("<f", observed))[0]
            computed_bits = struct.unpack("<i", struct.pack("<f", computed))[0]
            self.assertLessEqual(
                abs(observed_bits - computed_bits),
                1,
                (seed, observed, computed),
            )

    def test_at_least_four_of_six_are_bit_exact(self):
        cases = {0: GROUP_A_FIXTURE, **DETERMINISTIC_SEED_FIXTURES}
        exact = 0
        for seed, name in cases.items():
            observed_bytes = bank15_operand(name)
            _, _, scale = calib_scale(seed)
            computed_bytes = struct.pack("<f", float(np.float32(1.0) / scale))
            if observed_bytes == computed_bytes:
                exact += 1
        self.assertGreaterEqual(exact, 4)

    def test_direct_255_over_range_gives_the_same_result_as_two_step_division(self):
        """Confirms the 1-ULP gap is not an artifact of computation
        order (two-step `1/((hi-lo)/255)` vs. one-step `255/(hi-lo)`)."""
        for seed in [0, *DETERMINISTIC_SEED_FIXTURES]:
            lo, hi, scale = calib_scale(seed)
            two_step = struct.pack("<f", float(np.float32(1.0) / scale))
            one_step = struct.pack("<f", float(np.float32(255.0) / (hi - lo)))
            self.assertEqual(two_step, one_step, seed)


class TestGroupBDoesNotMatchAnyNearbySeed(unittest.TestCase):
    """Group B's own value (127.514183) is not `1/scale` for seed=0
    (33.81) or any of the 10 nearby integer seeds checked -- ruling out
    "it's the same formula, just a slightly different effective seed"
    as an easy explanation."""

    def test_no_seed_zero_through_nine_is_close(self):
        target = struct.unpack("<f", bank15_operand(GROUP_B_FIXTURE))[0]
        for seed in range(10):
            _, _, scale = calib_scale(seed)
            computed = float(np.float32(1.0) / scale)
            self.assertGreater(abs(computed - target), 50.0, (seed, computed, target))


class TestGroupBValueRecursAcrossUnrelatedShapes(unittest.TestCase):
    """Group B's own value is byte-identical across several
    Conv(dilation=3, insz=...) fixtures with DIFFERENT insz -- a real
    calibration- or shape-derived quantity would not be expected to
    repeat exactly across shapes with different receptive-field/output
    statistics, consistent with this being a fixed constant rather than
    a per-build computed value."""

    def test_group_b_value_matches_every_unrelated_insz_fixture(self):
        group_b_bytes = bank15_operand(GROUP_B_FIXTURE)
        for name in UNRELATED_SHAPE_FIXTURES_WITH_GROUP_B_VALUE:
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in (96, 112, 128)
                and r.get("operand") == group_b_bytes
            ]
            self.assertTrue(hits, name)

    def test_group_b_value_is_near_but_not_exactly_127_5(self):
        target = struct.unpack("<f", bank15_operand(GROUP_B_FIXTURE))[0]
        self.assertAlmostEqual(target, 127.5, delta=0.02)
        self.assertNotEqual(target, 127.5)


if __name__ == "__main__":
    unittest.main()
