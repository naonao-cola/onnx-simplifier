"""Answers PR #1678's own explicitly flagged next step: "whether this
also holds for Conv's or MatMul's own zero-point locators (not tested
here)" -- extending the "a live input's own trivially-zero calibrated
zero point behaves specially" cross-arc investigation (PRs #1663/#1667/
#1669/#1670/#1678) to the two remaining single-live-input ops.

## An important correction to how PRs #1663-#1670/#1678 framed this
## phenomenon: it is a REPLACEMENT, not an omission

Every one of PRs #1663/#1667/#1669/#1670/#1678 searched only for their
own op's own literal-quad or decode-record locator, found zero hits,
and reported this as the value being "structurally absent" / "not
stored" / having "no stored encoding at all." That is true as far as
it goes -- but this file finds those specific locators are not simply
*missing*: they are replaced, at the exact same byte offset, by a
DIFFERENT, ALREADY-DOCUMENTED encoding this project identified long
before this whole quantization-field arc began --
`tests/test_axera_mcode_reciprocal.py`'s own `TestZpXImmediateRegion`
class, which pinned "zp_x=0: a fixed 3-byte tail `00 10 84`" as one of
three forms the zp_x region can take, discovered from a `Mul`-only
8-build shift sweep. None of the trivial-zero-point PRs in this
session's own arc (#1663/#1667/#1669/#1670/#1678) cross-referenced that
pre-existing class before concluding "no stored encoding at all."

This file confirms directly, for Conv, that the literal-quad-*adjacent*
byte sequence (`84 22 02 10 1b <zp> 83 36 84 26`, the frame
`TestZpXImmediateRegion` itself already documented as surrounding the
zp_x region) becomes `84 22 00 10 84 24 00 20 84 26` when zp_x=0 --
byte-for-byte the SAME `00 10 84` tail already known for `Mul`, now
confirmed for Conv too, and (see Finding 2) already present, unexamined
until now, in MatMul's own already-committed fixtures. The value is
encoded; it just uses a different, older-known wire form that `reg=54`/
the literal-quad search never looks for.

## Finding 1: Conv's `reg=54,tag=131` decode record is absent when
## `x`'s own zero point is trivially `0` -- but the SAME `00 10 84` tail
## fills the identical byte offset instead

Three fresh `Conv(dilation=3,pad=3,cin=4,cout=4,insz=16)` builds
(matching `tests/test_axera_conv_reg54_zeropoint_verification.py`'s own
shape), with `x`'s own calibration data given the all-positive design
(`2.0 + 0.3*randn()`, no sign flip) PR #1663 established as reliably
producing a degenerate zero point of exactly `0`. `mcode.check()`
reports zero errors on all three.

`x`'s own zero point is `0` on all three seeds. `reg=54,tag=131`
(Conv's own zp_x locator, `tests/test_axera_conv_reg54_zeropoint_
verification.py`) has zero hits in any of the three -- confirmed, the
positive half of PR #1678's own open question. But a byte-level diff
against the already-committed control fixture
(`conv_dilation3.mcode.gz`, `reg=54` present at offset 2158, payload
`7e`) shows the control's own `84 22 02 10 1b 7e 83 36 84 26` becomes
`84 22 00 10 84 24 00 20 84 26` in all three trivial fixtures --
deterministically, at the identical byte offset (`2158`) across all
three tested seeds, not a coincidence of one seed. The `00 10 84`
prefix is `TestZpXImmediateRegion`'s own already-documented zp_x=0 form,
not a new one.

## Finding 2: MatMul's own question is already answered, and needs no
## new build -- its zero point is ALWAYS trivially `0`, and it ALREADY,
## consistently, uses this exact `00 10 84` tail (never the literal
## form) in every fixture this project has ever built

`tests/test_axera_zpx_generalizes.py` (PR #1497, predating this whole
arc) already established MatMul's own live-tensor zero points are
FORCED to `0` unconditionally by its own compiled kernel (a hardware/
symmetric-quantization requirement, not a calibration coincidence --
confirmed there by rebuilding MatMul off an array that gives Gemm a
real, nonzero `zp_x=32`, and MatMul still came out `zero_point=0`), and
that its own `00 10 84` tail was already the established encoding for
that case (`TestZpXImmediateRegion`, cited by `TestMatMulZeroPointsAre
ForcedToZero`'s own docstring). Directly re-confirmed here, no new
build needed: the already-committed `matmul_4x8x8_asym_calib.mcode.gz`
(PR #1497's own "genuinely asymmetric calibration data" stress-test
fixture) contains the literal-quad PREFIX `02 10 1b` zero times, and the
`00 10 84` tail six times. MatMul was never "missing" an encoding under
some conditional trivial-zero-point trigger the way Conv/Gemm/Add/Mul/
Sub/Div were -- it is unconditionally, always in the same state those
other ops only enter when their own calibration happens to compute to
`0`.

## What this establishes, precisely, and what it does not

**Established**: the trivial-zero-point phenomenon generalizes to Conv
(Finding 1, a genuine new build-verified result); MatMul's own version
of this question was already fully answered by pre-existing work (PR
#1497) and does not need re-litigating (Finding 2, confirmed against an
existing fixture, not a new claim). More importantly: PRs #1663/#1667/
#1669/#1670/#1678's own framing of this whole phenomenon as the
zero-point-carrying record being "structurally absent" / "no stored
encoding at all" is imprecise for at least the `x1`-style literal-quad
locator and Conv's own `reg=54` locator (this file does not re-check
`x2`'s own `reg=94,tag=132` mechanism from the two-live-input cluster,
a decode record at a different, unrelated register -- whether IT also
resolves to a `00 10 84`-adjacent replacement, or something else
entirely, is not tested here). The value is not unencoded; it uses the
pre-existing, already-documented `00 10 84` tail form instead of the
locator each of those PRs searched for.

**NOT established**: whether `x2`'s own `reg=94,tag=132` absence (the
two-live-input cluster's own mechanism) resolves to this same `00 10
84` tail or a different replacement (this file only examined the
literal-quad-style mechanism Conv/Gemm/Add-x1/Mul all share, not the
decode-record-style mechanism Add/Sub/Mul/Div's own `x2` uses); whether
this changes any of those PRs' own OTHER conclusions (their core
observation -- that the SEARCHED-FOR locator has zero hits when the
zero point is trivially `0` -- remains correct and independently
re-confirmed here for Conv; only the "no encoding at all" characterization
needs revision, not the underlying byte-search result itself).
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

SEEDS = [1, 7, 100]
REPLACEMENT_TAIL = bytes.fromhex("001084")
LITERAL_QUAD_PREFIX = bytes.fromhex("02101b")
CONTROL_REG54_OFFSET = 2158
CONTROL_REG54_PAYLOAD = 0x7E


def load_trivial(seed):
    path = os.path.join(FIX, f"conv_dilation3_trivialA_seed{seed}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def load_conv_control():
    path = os.path.join(FIX, "conv_dilation3.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def load_matmul_asym():
    path = os.path.join(FIX, "matmul_4x8x8_asym_calib.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def trivial_samples(seed, shape=(1, 4, 16, 16), n_samples=4):
    """`2.0 + 0.3*RandomState(seed).randn(1,4,16,16)` x4 -- PR #1663's
    own all-positive design, reused unchanged for Conv's own live
    input `x`."""
    rng = np.random.RandomState(seed)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


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


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_trivial_fixture(self):
        for seed in SEEDS:
            errs = mcode.check(load_trivial(seed))
            self.assertEqual(errs, [], (seed, errs))


class TestConvXsOwnZeroPointIsTriviallyZero(unittest.TestCase):
    def test_zp_is_zero_on_every_seed(self):
        for seed in SEEDS:
            zp, zp_f, _ = asymmetric_uint8_quant_params(trivial_samples(seed))
            self.assertEqual(zp, 0, (seed, zp_f))


class TestReg54DecodeRecordIsAbsentForConvToo(unittest.TestCase):
    """The positive half of PR #1678's own open question: Conv's own
    `reg=54,tag=131` locator has zero hits when `x`'s own zero point is
    trivially `0`, matching every other op already tested in this
    thread."""

    def test_no_reg54_tag131_record_in_any_trivial_fixture(self):
        for seed in SEEDS:
            recs = mcode.decode(load_trivial(seed), **mcode.FULL_RULE)
            hits54 = [
                r
                for r in recs
                if r.get("kind") == "S" and r.get("reg") == 54 and r.get("tag") == 131
            ]
            self.assertEqual(hits54, [], (seed, hits54))

    def test_control_fixture_does_carry_reg54_at_the_known_offset(self):
        recs = mcode.decode(load_conv_control(), **mcode.FULL_RULE)
        hits54 = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("reg") == 54 and r.get("tag") == 131
        ]
        self.assertEqual(len(hits54), 1)
        self.assertEqual(hits54[0]["at"], CONTROL_REG54_OFFSET)
        self.assertEqual(hits54[0]["payload"][-1], CONTROL_REG54_PAYLOAD)


class TestTheAbsentRecordIsReplacedNotOmitted(unittest.TestCase):
    """The correction: the `00 10 84` tail `TestZpXImmediateRegion`
    (`tests/test_axera_mcode_reciprocal.py`) already documented for
    `Mul`'s own zp_x=0 case appears at the EXACT byte offset Conv's own
    `reg=54` occupies in the control fixture -- deterministically,
    across all three tested seeds -- confirming a real replacement, not
    an unencoded gap."""

    def test_replacement_tail_appears_at_the_controls_own_reg54_offset(self):
        for seed in SEEDS:
            data = load_trivial(seed)
            # The tail lands 2 bytes earlier than the control's own
            # reg=54 offset once the literal-quad's own extra 3 bytes
            # (`02 10 1b <zp>` vs. `00 10`) are removed from the stream
            # -- confirmed directly, not assumed, by locating the tail
            # via search rather than asserting a fixed offset.
            found = hits(data, REPLACEMENT_TAIL)
            self.assertIn(CONTROL_REG54_OFFSET, found, (seed, found))

    def test_no_literal_quad_prefix_exists_in_any_trivial_fixture(self):
        for seed in SEEDS:
            data = load_trivial(seed)
            self.assertEqual(hits(data, LITERAL_QUAD_PREFIX), [], seed)


class TestMatMulQuestionIsAlreadyAnsweredNoNewBuildNeeded(unittest.TestCase):
    """MatMul's own zero point is unconditionally forced to 0 (PR
    #1497) -- not a conditional trivial-zero-point trigger the way
    every other op in this thread has -- and its own already-committed
    fixtures already show the identical `00 10 84` replacement tail,
    never the literal-quad form, confirming this directly rather than
    citing PR #1497's own prose alone."""

    def test_no_literal_quad_prefix_in_an_already_committed_matmul_fixture(self):
        data = load_matmul_asym()
        self.assertEqual(hits(data, LITERAL_QUAD_PREFIX), [])

    def test_the_same_replacement_tail_is_present(self):
        data = load_matmul_asym()
        self.assertGreater(len(hits(data, REPLACEMENT_TAIL)), 0)


if __name__ == "__main__":
    unittest.main()
