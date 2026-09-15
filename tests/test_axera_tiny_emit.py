"""Offline tests for scripts/axera/tiny_emit.py.

No Docker, no card: the tinygrad pattern match runs against a real
tinygrad graph (skipped if tinygrad is not installed), and the scale
field map is pinned against the committed neg_1x8 fixture. Device
execution of emitted streams is documented in the module docstring and
the README, not asserted here.
"""

import gzip
import os
import struct
import sys

import numpy as np
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import tiny_emit  # noqa: E402
from mcode import (  # noqa: E402
    FULL_RULE,
    check,
    decode,
    encode,
    stream_bounds,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _blob(name):
    with gzip.open(os.path.join(_FIXTURES, name + ".mcode.gz"), "rb") as f:
        return f.read()


def test_trace_neg_matches_mul_by_minus_one():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_neg(-t)
    assert got["shape"] == [1, 8]


def test_trace_neg_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_neg(t + t)
    with pytest.raises(ValueError):
        tiny_emit.trace_neg(t * 2.0)


def test_minmax_scale_matches_pulsar2_to_1e10():
    rng = np.random.default_rng(0)
    samples = [(rng.uniform(-2.5, 2.5, (1, 8))).astype(np.float32) for _ in range(8)]
    got = tiny_emit.minmax_scale(samples)
    # Pulsar2's recorded output scale for this exact calibration.
    assert got == pytest.approx(0.0194994397, rel=1e-6)


def test_find_scale_words_locates_the_quadruple():
    blob = _blob("neg_1x8")
    scale = float(np.float32(0.0076893349))
    assert tiny_emit.find_scale_words(blob, scale) == [1288, 1295, 1302, 1309]


def test_emit_neg_round_trips_and_stays_clean():
    blob = _blob("neg_1x8")
    out = tiny_emit.emit_neg(
        blob, float(np.float32(0.0076893349)), float(np.float32(0.0132596185))
    )
    lo, hi = stream_bounds(out)
    assert encode(decode(out, start=lo, end=hi, **FULL_RULE)) == out[lo:hi]
    assert check(out) == []


# Mul input-side scales. Source of truth for the values is
# tests/test_axera_mcode_reciprocal.py (fixture builds' quant JSON);
# w2 (2x/2x ranges, same short site-B form as base) was scratched in
# ~/npu-scratch/t9-mul/mul_1x8_w2 and is committed as a fixture here.
_BASE = (0.007799775805324316, 0.0076893349178135395, 0.007151617668569088)
_W2 = (0.015599551610648632, 0.015378669835627079, 0.028606470674276352)
_X10 = (0.07799775898456573, 0.0007689335034228861, 0.007151617202907801)
_X01 = (0.0007725197938270867, 0.07746022194623947, 0.005844127852469683)


def _assert_slot_run(blob, value, stride, width=4):
    """The float32 word for ``value`` forms a clean x4 stride run."""
    pat = struct.pack("<f", value)[:width]
    assert len(tiny_emit._strided_run(blob, pat, stride)) == 4


def _assert_family_matches(patched, target, value, stride, width=4):
    _assert_slot_run(patched, value, stride, width)
    _assert_slot_run(target, value, stride, width)


def test_patch_mul_short_to_short_matches_w2_slots():
    """base -> w2 (both short site-B form): every input-side family lands
    on w2's own slot bytes, and the patched stream stays structurally
    clean. Full-stream equality is explicitly out of scope: the output
    quads (z 4x, emitter.py's domain), the manifest string table (x/y
    name order swaps build to build), the magnitude-adaptive S-unit
    programs and one input-driven single (1836: c9 -> cd) do not
    transplant -- see the transplant analysis in the PR."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8"), _BASE, _W2)
    target = _blob("mul_1x8_w2")
    nx, ny, nz = _W2
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_family_matches(patched, target, 1.0 / ny, 6, width=3)
    assert check(patched) == []


def test_patch_mul_full_to_full_matches_x01_slots():
    """x10 -> x01 (both full site-B form): same contract as above."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8_recip_x10"), _X10, _X01)
    target = _blob("mul_1x8_recip_x01")
    nx, ny, nz = _X01
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_family_matches(patched, target, 1.0 / ny, 7)
    assert check(patched) == []


def test_patch_mul_preserves_reference_site_b_form():
    """base (short B) -> x01 scales (full B in its own build): sites A and
    C still land on x01's slot bytes, while site B keeps the reference's
    short form carrying the new value -- patching rewrites values, it
    does not recompile programs."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8"), _BASE, _X01)
    target = _blob("mul_1x8_recip_x01")
    nx, ny, nz = _X01
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_slot_run(patched, 1.0 / ny, 6, width=3)
    assert check(patched) == []
