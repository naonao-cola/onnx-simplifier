"""Regression coverage for the binary elementwise op register reader. No
Docker/device required -- decodes committed Pulsar2 7.0-lite builds of
``Add``/``Sub``/``Mul``/``Div(x[1,16,8,8], z[1,16,8,8])`` and checks the
recovered scales against each build's own ``quant_axmodel.json`` values."""

import json
import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from binary_op_register_decode import (  # noqa: E402
    MUL_DIVISOR_REGS,
    X_QUANT_REGS,
    decode,
    float_runs,
    load,
)

_FIX = os.path.join(_AXERA_DIR, "fixtures", "binary_op_registers")
_SCALES = json.load(open(os.path.join(_FIX, "scales.json")))
_NAMES = ["add_c1", "add_c2", "add_c3", "sub_c3", "mul_c3", "div_c3"]


def _decode(name):
    mc, params = load(os.path.join(_FIX, f"{name}.axmodel.gz"))
    return decode(mc, params, name.split("_")[0]), mc


@pytest.mark.parametrize("name", _NAMES)
def test_recovered_scales_match_the_builds_quantization(name):
    d, _ = _decode(name)
    for t in ("x", "z", "y"):
        want = _SCALES[name][t]["scale"]
        got = d["scales"][t]
        assert got == pytest.approx(want, rel=5e-4), (name, t)


@pytest.mark.parametrize("name", _NAMES)
def test_x_quant_lanes_1_to_3_on_0f60_0f70_0f80(name):
    # Lane 0's copy is a W / address-carrying record that mcode.decode does
    # not always tokenize as a value write; lanes 1-3 are plain V records.
    _, mc = _decode(name)
    x_run = next(r for r in float_runs(mc) if set(r["regs"]) & set(X_QUANT_REGS))
    assert x_run["count"] in (3, 4)
    assert [r for r in x_run["regs"] if r is not None][-3:] == [0x0F60, 0x0F70, 0x0F80]


def test_cv_program_only_for_unequal_ratio_add_and_sub():
    assert _decode("add_c1")[0]["cv_program"] is False
    for name in ("add_c2", "add_c3", "sub_c3"):
        assert _decode(name)[0]["cv_program"] is True, name
    for name in ("mul_c3", "div_c3"):
        assert _decode(name)[0]["cv_program"] is False, name


def test_q15_header_shared_word_vs_per_input_words():
    assert _decode("add_c1")[0]["q15"] == {"shared": 0.5}
    q = _decode("add_c3")[0]["q15"]
    s = _SCALES["add_c3"]
    assert q["x_over_y"] == pytest.approx(
        s["x"]["scale"] / s["y"]["scale"], abs=1 / 32768
    )
    assert q["z_over_y"] == pytest.approx(
        s["z"]["scale"] / s["y"]["scale"], abs=1 / 32768
    )
    # Sub stores z/y positive; the subtraction is applied elsewhere.
    q = _decode("sub_c3")[0]["q15"]
    s = _SCALES["sub_c3"]
    assert q["z_over_y"] == pytest.approx(
        s["z"]["scale"] / s["y"]["scale"], abs=1 / 32768
    )


def test_mul_divisor_is_y_over_x_times_z_on_0fd0_block():
    d, mc = _decode("mul_c3")
    s = _SCALES["mul_c3"]
    want = s["y"]["scale"] / (s["x"]["scale"] * s["z"]["scale"])
    assert d["mul_divisor"] == pytest.approx(want, rel=1e-5)
    run = next(r for r in float_runs(mc) if set(r["regs"]) & set(MUL_DIVISOR_REGS))
    assert [r for r in run["regs"] if r is not None][-3:] == [0x0FE0, 0x0FF0, 0x1000]
    assert _decode("div_c3")[0]["mul_divisor"] is None


def test_compressed_z_form_falls_back_to_q15():
    # add_c2 stores 1/z_scale in a shorter compressed form this reader does
    # not decode; z_scale then comes from the Q15 z/y word.
    d, _ = _decode("add_c2")
    assert d["z_quant"] is None
    assert d["scales"]["z"] == pytest.approx(_SCALES["add_c2"]["z"]["scale"], rel=5e-4)


_DEVICE = json.load(open(os.path.join(_FIX, "device_patch_results.json")))


def _slope(key, cond, lane):
    s = _DEVICE[key]["conds"][cond][str(lane)]
    return s.get("fit_a", s.get("ratio"))


def test_committed_device_runs_all_passed_health_checks():
    for key, e in _DEVICE.items():
        for cond, v in e["conds"].items():
            assert "FAULT" not in v, (key, cond)
            assert v["health_ok"] is True, (key, cond)


@pytest.mark.parametrize(
    "key,cond,want",
    [
        ("add_c3:q_x_lane1", "x_only", 0.5),  # 1/x_scale copy: x's lane 1 only
        ("add_c3:q_z_lane1", "z_only", 0.5),  # 1/z_scale short unit: z's lane 1 only
        ("add_c3:deq_y_lane1", "both", 2.0),  # y_scale copy: lane 1 output
        ("sub_c3:q_x_lane1", "x_only", 0.5),
        ("sub_c3:q_z_lane1", "z_only", 0.5),
        ("mul_c3:q_x_lane1", "both", 0.5),
        ("mul_c3:q_z_copy2", "both", 0.5),
        ("mul_c3:requant_0fe0", "both", 0.5),  # doubled divisor halves lane 1
        ("mul_c3:deq_y_copy2", "both", 2.0),
        ("div_c3:q_x_lane1", "both", 0.5),
        ("div_c3:q_z_copy2", "z_only", 2.0),  # halved 1/z doubles the quotient
        ("div_c3:deq_y_copy2", "both", 2.0),
    ],
)
def test_per_lane_register_patch_touches_exactly_lane_1(key, cond, want):
    assert _slope(key, cond, 1) == pytest.approx(want, abs=0.02)
    for lane in (0, 2, 3):
        assert _slope(key, cond, lane) == pytest.approx(1.0, abs=1e-3), lane


@pytest.mark.parametrize(
    "key,halved,kept,offset",
    [
        ("add_c3:q15_x_over_y", "x_only", "z_only", -0.252),
        ("add_c3:q15_z_over_y", "z_only", "x_only", -1.505),
        ("sub_c3:q15_x_over_y", "x_only", "z_only", -0.251),
        (
            "sub_c3:q15_z_over_y",
            "z_only",
            "x_only",
            1.506,
        ),  # sign flip: z is subtracted
    ],
)
def test_q15_word_scales_one_input_in_every_lane(key, halved, kept, offset):
    for lane in range(4):
        s = _DEVICE[key]["conds"][halved][str(lane)]
        assert s["fit_a"] == pytest.approx(0.5, abs=0.01)
        assert s["fit_b"] == pytest.approx(offset, abs=0.01)
        assert _DEVICE[key]["conds"][kept][str(lane)]["fit_a"] == pytest.approx(
            1.0, abs=0.01
        )


def test_shared_q15_word_scales_both_inputs_without_cv_program():
    for cond in ("x_only", "z_only"):
        for lane in range(4):
            s = _DEVICE["add_c1:q15_x_over_y"]["conds"][cond][str(lane)]
            assert s["fit_a"] == pytest.approx(0.5, abs=0.01)
            assert s["fit_b"] == pytest.approx(-1.004, abs=0.01)


def test_unsupported_op_rejected():
    mc, params = load(os.path.join(_FIX, "add_c3.axmodel.gz"))
    with pytest.raises(ValueError):
        decode(mc, params, "relu")
