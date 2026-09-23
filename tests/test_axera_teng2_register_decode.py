"""Regression coverage for the teng2 register-write reader (no device needed)."""

import math
import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import teng2_register_decode as trd  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures", "teng2_register_decode")


def _mc(name):
    return trd.load_mcode(os.path.join(_FIX, name))


@pytest.mark.parametrize(
    "name,scale",
    [
        ("relu_range1p0.axmodel.gz", 1.0 / 255),
        ("relu_range4p0.axmodel.gz", 4.0 / 255),
        (
            "relu_zp128.axmodel.gz",
            1.0 / 255,
        ),  # same range as 1p0, only zero-point differs
    ],
)
def test_scale_registers_recover_calibration(name, scale):
    mc = _mc(name)
    s = trd.scale_registers(mc)
    # three per-lane copies of each multiplier
    assert len(s["quant"]) == 3
    assert len(s["dequant"]) == 3
    for q in s["quant"]:
        assert math.isclose(q, 1.0 / scale, rel_tol=1e-4)
    for d in s["dequant"]:
        assert math.isclose(d, scale, rel_tol=1e-4)
    imp = trd.implied_scales(mc)
    assert math.isclose(imp["input_scale"], scale, rel_tol=1e-4)
    assert math.isclose(imp["output_scale"], scale, rel_tol=1e-4)


def test_zero_point_is_not_in_the_scale_registers():
    # zp128 and range1p0 share input/output scale; the zero point lives in a
    # variable-length short unit, not these registers, so the scale registers
    # must be byte-identical between them.
    a = trd.scale_registers(_mc("relu_range1p0.axmodel.gz"))
    b = trd.scale_registers(_mc("relu_zp128.axmodel.gz"))
    assert a == b


def test_register_writes_are_well_formed():
    writes = trd.segment_register_writes(_mc("relu_range1p0.axmodel.gz"))
    assert writes, "expected some register writes in segment 2"
    for w in writes:
        assert 0 <= w["reg"] <= 0xFFFF
        assert (
            0 <= w["width"] <= 4
        )  # a trailing segment-marker record has a 0-byte operand
        assert w["kind"] in ("V", "W")
    # every scale register appears exactly twice (quant occ 0, dequant occ 1)
    for reg in trd.SCALE_REGS:
        occ = sorted(w["occurrence"] for w in writes if w["reg"] == reg)
        assert occ == [0, 1]


def test_non_elementwise_segment_raises():
    # a made-up empty stream has no scale registers
    with pytest.raises(ValueError):
        trd.implied_scales(b"\x00" * 4096)
