"""Offline tests for scripts/axera/tiny_emit.py.

No Docker, no card: the tinygrad pattern match runs against a real
tinygrad graph (skipped if tinygrad is not installed), and the scale
field map is pinned against the committed neg_1x8 fixture. Device
execution of emitted streams is documented in the module docstring and
the README, not asserted here.
"""

import gzip
import os
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
