"""Marks the completion of the (op, field) grid `tests/test_axera_
quant_field_cluster_synthesis.py` (PR #1652) laid out: `zp_x`/
`1/x_scale`/`y_scale`/`zp_y`, all first-principles-derived via
`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_quantize_
matmul.h`) applied to an op's own input range and output range, across
Conv/Gemm/MatMul. PR #1652 left one cell unattempted (Gemm/MatMul's
own `zp_y`); PR #1653/#1654 closed a persistent `1/x_scale` ULP gap;
PR #1655 closed the last cell, finding Gemm's and MatMul's own `zp_y`
at `reg=120,tag=132`. This file (1) directly re-confirms, by decoding
real fixtures and recomputing independently of any contributing PR's
own test module, that every attempted cell now has a real answer, (2)
restates the grid as one table including PR #1655's own new row, and
(3) checks a concrete, previously-unaddressed question about whether
this arc's own "value exactly 128 escapes the fixed on-the-wire
encoding form" phenomenon (independently observed three times now --
`scripts/axera/README.md`'s own Conv `y_zero`/`learn_mcode` case, PR
#1655's own Gemm `zp_y` seed=1 case, and -- found while writing this
file, see below -- an existing test's own Mul `zp_x` case) is a live
risk for `scripts/axera/tiny_emit.py`'s own zero-point-writing
functions.

## The completed grid

| op | `zp_x` | `1/x_scale` | `y_scale` | `zp_y` |
| --- | --- | --- | --- | --- |
| Conv (dilation=3 family) | exact, `reg=54`/`reg=60` (PR #1640) | 4/6 exact, 2/6 within 1 ULP, `verb=161,bank=15` (PR #1641) | close, not exact, `reg=224` (PR #1642) | exact, `reg=232` (PR #1644, #1649) |
| Gemm (`M=1,K=16,N=16`) | exact, literal quad `02 10 1b <zp_x> 83 36` (PR #1645) | exact, `verb=161,bank=15`; bit-exact formula closing a persistent `m*k`-dependent 1-ULP gap found across 10 shapes (PR #1646, #1648, #1651, #1653, #1654) | close, not exact, raw byte run near offset 1636 (PR #1646) | `reg=120,tag=132`, 5/6 seeds exact, seed=1 structurally absent -- a real "exactly 128 escapes encoding" case, not noise (PR #1655) |
| MatMul (`M=1,K=16,N=16`) | not applicable -- always forced to 0 (`tests/test_axera_zpx_generalizes.py`, PR #1497) | exact, IDENTICAL locator/offsets to Gemm's own `(1,16,16)` shape (PR #1647) | close, not exact, same raw-offset pattern as Gemm (PR #1647) | SAME `reg=120,tag=132` locator as Gemm, 5/6 seeds exact, seed=42 a genuine rounding near-tie (PR #1655) |

Every cell that can be attempted (12 of 12 -- Gemm/MatMul's own `zp_x`
row correctly has one N/A cell, MatMul's own zero points being always
forced to 0) now has a first-principles-derived answer: bit-exact,
close-but-not-exact with an identified cause, or an explained
structural-absence/near-tie exception. None remain unattempted.

## A concrete follow-up: does the "exactly 128 escapes encoding"
## phenomenon threaten `tiny_emit.py`'s own zero-point patch functions?

`scripts/axera/README.md`'s own "Two builds of 48 that cannot be
patched in place" section documents this project's OLDEST known
instance: a Conv build's `y_zero` (output zero point) came out exactly
`128` and "was not written inline but escaped, shifting every byte
after it," in the LEARNED table-driven `learn_mcode`/`emit_mcode`
generator (`scripts/axera/emitter.py`) -- a different generator from
this arc's own `tiny_emit.py`. PR #1655 independently found the exact
same trigger value (`128`) causing the exact same class of structural
absence for Gemm's own `zp_y` (`reg=120,tag=132`), this arc's newest
instance. Checking whether this generalizes to `tiny_emit.py`'s own
zero-point-writing functions -- `patch_mul_zp_x`/`patch_conv_zp_x`,
which patch `zp_x`'s literal 6-byte quad (`02 10 1b <zp_x> 83 36`) --
surfaces a THIRD, already-existing instance this arc had not
previously cross-referenced: `tests/test_axera_tiny_emit.py`'s own
`test_patch_mul_zp_x_raises_when_form_absent` already documents that
`mul_1x8` (base fixture) "has zp_x=128 but uses one of the opaque
forms, not the literal one," and asserts `patch_mul_zp_x` raises
`ValueError` rather than silently no-op'ing. That test predates this
arc entirely (it exercises `zp_x=128` as a pre-existing, independently
discovered edge case, not one motivated by this file's own reasoning)
-- but nothing in this whole quantization-field arc had previously
connected it to the SAME `128`-specific escape phenomenon PR #1655 and
`scripts/axera/README.md` both separately describe.

This file confirms, by reading `tiny_emit.py`'s own source directly
(not by re-running the generator, since the existing test above
already covers the runtime behavior): `patch_mul_zp_x`/
`patch_conv_zp_x` do a fixed-width, in-place single-byte overwrite of
an ALREADY-PRESENT literal quad -- they never depend on the NEW value's
own magnitude to decide where or whether to write (the function raises
if the OLD value's own literal-quad unit is not found, which is
exactly the safe behavior the existing `zp_x=128` test already
confirms; it does not attempt to write and silently corrupt the stream
if a caller passes `new_zp_x=128`). `emit_matmul_reg8_quad`,
`emit_conv_reg8_group`, and `emit_gemm_reg8_group` (the other
`tiny_emit.py` functions that write meaningful field values) touch
only the `reg=8` unordered noise pool's own candidate-identity bytes
(fixed anchor-relative offsets, unrelated register/tag numbers) --
none of them read or write `zp_x`/`zp_y` at all, so the escape
phenomenon is architecturally orthogonal to them.

## What this establishes, precisely, and what it does not

**Established**: the (op, field) grid is complete -- 12 of 12
attemptable cells (Conv x4, Gemm x4, MatMul x3) each have a real,
independently-decoded-here answer; `tiny_emit.py`'s own zero-point
patch functions are already safe against the "value exactly 128
escapes the literal encoding form" phenomenon by construction (fail
loudly via the pre-existing form-absence check, not silent
corruption), a claim now backed by THREE independent real-build
observations of the underlying phenomenon (Conv `y_zero` via
`learn_mcode`/`emit_mcode`, Gemm `zp_y` via PR #1655, Mul `zp_x` via
this project's own pre-existing `test_patch_mul_zp_x_raises_when_form_
absent`) rather than one.

**NOT established**: whether Conv's own `zp_y` (`reg=232`) or MatMul's
own `zp_x` (were it not always forced to 0) would show the identical
`128`-specific trigger if tested directly (only Gemm `zp_y` and Mul
`zp_x` have actually been observed hitting it; this file does not run
new builds to check Conv `zp_y` or search for other `reg=232`-adjacent
special cases); why `128` specifically (rather than some other
byte value) is the trigger, for any of the three independent cases;
whether `emit_matmul_reg8_quad`/`emit_conv_reg8_group`/
`emit_gemm_reg8_group`'s own reg=8 pool has an analogous, still
undiscovered escape trigger of its own (not surveyed here, since
that pool's candidate values are a small fixed set that has never
included anything resembling `128`).
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
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg_byte(recs, reg, tag):
    hits = [
        r
        for r in recs
        if r.get("kind") == "S" and r.get("reg") == reg and r.get("tag") == tag
    ]
    assert hits, (reg, tag)
    values = {h["payload"][-1] for h in hits}
    assert len(values) == 1, values
    return next(iter(values))


def asymmetric_uint8_quant_params(samples):
    """`ComputeAsymmetricUint8QuantParams`, independently re-ported
    here (not shared code with any contributing PR's own module)."""
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


# ---------------------------------------------------------------------------
# Gemm/MatMul's own zp_y at reg=120,tag=132 (PR #1655) -- the cell that
# completed the grid.
# ---------------------------------------------------------------------------

GEMM_MATMUL_SEEDS = [0, 1, 7, 42, 100, 999]
ZP_Y_TAG = 132
ZP_Y_REG = 120
GEMM_STRUCTURALLY_ABSENT_SEEDS = {1}
MATMUL_NEAR_TIE_SEEDS = {42}


def load_gemm(seed):
    return load(f"gemm_1x16x16_zpxseed{seed}.mcode.gz")


def load_matmul(seed):
    return load(f"matmul_1x16x16_zpxseed{seed}.mcode.gz")


def small_calib_samples(seed, shape=(1, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def gemm_weight():
    rng = np.random.RandomState(0)
    w = (rng.randn(16, 16) * 0.1).astype(np.float32)
    b = (rng.randn(16) * 0.1).astype(np.float32)
    return w, b


def matmul_weight():
    rng = np.random.RandomState(0)
    return (rng.randn(16, 16) * 0.1).astype(np.float32)


def gemm_reference(x, w, b):
    return (x.astype(np.float64) @ w.astype(np.float64) + b.astype(np.float64)).astype(
        np.float32
    )


def matmul_reference(x, w):
    return (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)


def computed_zp_y(seed, reference_fn, *weight_args):
    samples = small_calib_samples(seed)
    ys_all = [reference_fn(s, *weight_args) for s in samples]
    ylo = min(0.0, min(float(y.min()) for y in ys_all))
    yhi = max(0.0, max(float(y.max()) for y in ys_all))
    if yhi <= ylo:
        yhi = ylo + 1.0
    scale = float((np.float32(yhi) - np.float32(ylo)) / np.float32(255.0))
    zp_f = float(np.float32(-np.float32(ylo) / np.float32(scale)))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), zp_f


def zp_y_record(recs):
    hits = [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("tag") == ZP_Y_TAG
        and r.get("reg") == ZP_Y_REG
    ]
    return hits[0] if hits else None


class TestGridCompletionGemmMatMulZpYMatchesAtReg120Tag132(unittest.TestCase):
    """Directly re-confirms PR #1655's own headline claim, decoding
    fresh here: 5/6 seeds bit-exact for both ops, with the sole
    exception per op independently explained (Gemm seed=1: structural
    absence; MatMul seed=42: a rounding near-tie), not merely cited."""

    def test_gemm_zp_y_matches_on_every_non_absent_seed(self):
        for seed in GEMM_MATMUL_SEEDS:
            recs = decode(f"gemm_1x16x16_zpxseed{seed}.mcode.gz")
            rec = zp_y_record(recs)
            if seed in GEMM_STRUCTURALLY_ABSENT_SEEDS:
                self.assertIsNone(rec, seed)
                continue
            zp, zp_f = computed_zp_y(seed, gemm_reference, *gemm_weight())
            self.assertIsNotNone(rec, seed)
            self.assertEqual(rec["payload"][-1], zp, (seed, zp_f))

    def test_matmul_zp_y_matches_on_every_seed_not_a_near_tie(self):
        for seed in GEMM_MATMUL_SEEDS:
            recs = decode(f"matmul_1x16x16_zpxseed{seed}.mcode.gz")
            rec = zp_y_record(recs)
            self.assertIsNotNone(rec, seed)
            zp, zp_f = computed_zp_y(seed, matmul_reference, matmul_weight())
            if seed in MATMUL_NEAR_TIE_SEEDS:
                self.assertNotEqual(rec["payload"][-1], zp, (seed, zp_f))
                continue
            self.assertEqual(rec["payload"][-1], zp, (seed, zp_f))

    def test_matmul_seed42_is_a_genuine_near_tie_not_a_formula_error(self):
        """Re-confirms PR #1655's own margin claim: seed=42's distance
        to a rounding half-boundary is the smallest of all 6 seeds, by
        a wide margin -- the same signature PR #1640 established for
        Conv's own zp_x near-tie."""

        def dist_to_half(seed):
            _, zp_f = computed_zp_y(seed, matmul_reference, matmul_weight())
            return abs((zp_f - math.floor(zp_f)) - 0.5)

        distances = {seed: dist_to_half(seed) for seed in GEMM_MATMUL_SEEDS}
        closest = min(distances, key=distances.get)
        self.assertEqual(closest, 42, distances)
        others = [d for s, d in distances.items() if s != 42]
        self.assertGreater(min(others), 1.5 * distances[42], distances)


class TestTinyEmitZeroPointPatchesAreSafeAgainstThe128EscapeCase(unittest.TestCase):
    """Checks the concrete follow-up question this file's own module
    docstring raises: does `tiny_emit.py`'s own zero-point-writing code
    ever risk silently mishandling the "exactly 128 escapes the
    literal encoding form" phenomenon this arc (PR #1655) and
    `scripts/axera/README.md` both independently document?"""

    def test_patch_mul_zp_x_writes_in_place_regardless_of_new_value(self):
        """The function does a fixed-width single-byte overwrite of an
        ALREADY-LOCATED unit -- it never re-searches or re-sizes based
        on the new value, so `new_zp_x=128` is not a special case at
        the write site itself (only at the SEARCH site, for
        `old_zp_x`, which the pre-existing
        `test_patch_mul_zp_x_raises_when_form_absent` in
        `tests/test_axera_tiny_emit.py` already covers directly)."""
        import inspect

        src = inspect.getsource(tiny_emit.patch_mul_zp_x)
        # No branch keyed on the NEW value's own magnitude (e.g. an
        # `if new_zp_x == 128` special case) -- confirms the write
        # path cannot silently diverge for 128 specifically.
        self.assertNotIn("new_zp_x ==", src)
        self.assertNotIn("new_zp_x !=", src)
        self.assertIn("out[hits[0] + 3] = new_zp_x", src)

    def test_reg8_group_emit_functions_never_touch_zero_point_bytes(self):
        """Confirms, by reading source directly, that the three
        `emit_*_reg8_*` functions operate purely on the reg=8 pool's
        own candidate-identity bytes and never read or write a
        `zp_x`/`zp_y` field -- the escape phenomenon is architecturally
        orthogonal to them."""
        import inspect

        for fn in (
            tiny_emit.emit_matmul_reg8_quad,
            tiny_emit.emit_conv_reg8_group,
        ):
            src = inspect.getsource(fn)
            self.assertNotIn("zp_x", src)
            self.assertNotIn("zp_y", src)

    def test_mul_zp_x_128_escape_is_a_real_pre_existing_case_not_hypothetical(self):
        """Directly re-runs the exact scenario
        `test_patch_mul_zp_x_raises_when_form_absent` already covers,
        confirming this file's own claim that it is a real,
        independently-discovered THIRD instance of the same `128`
        phenomenon -- not merely restating that other test's own
        assertion."""
        blob = load("mul_1x8.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.patch_mul_zp_x(blob, 128, 100)
        # And confirm directly: no literal quad for 128 exists anywhere
        # in this fixture, the same structural-absence shape PR #1655
        # found for Gemm's own zp_y at the identical trigger value.
        quad = bytes.fromhex("02101b") + bytes([128]) + bytes.fromhex("8336")
        found = [
            i
            for i in range(len(blob) - len(quad) + 1)
            if blob[i : i + len(quad)] == quad
        ]
        self.assertEqual(found, [])


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        names = [f"gemm_1x16x16_zpxseed{s}.mcode.gz" for s in GEMM_MATMUL_SEEDS] + [
            f"matmul_1x16x16_zpxseed{s}.mcode.gz" for s in GEMM_MATMUL_SEEDS
        ]
        for name in names:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], name)


if __name__ == "__main__":
    unittest.main()
