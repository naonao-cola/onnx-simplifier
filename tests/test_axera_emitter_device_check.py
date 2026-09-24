"""Emitters on the AX650: models emitted at calibrations Pulsar2 never built,
checked against a quantized onnxruntime reference
(``scripts/axera/emitter_device_check.py``,
``docs/axera-emitter-device-check.md``).

The no-device tests check the harness itself: every case builds, its emitter
accepts the target calibration, and the Q/DQ reference means what it says.
The device tests run each case (native control, emitted model, native health
run) and need the card (``AXCL_LXD_VM=axcl-vm`` or a local AXCL install)."""

import os
import sys

import numpy as np
import pytest
from onnx import parser

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

pytest.importorskip("onnxruntime")

import emitter_device_check as edc  # noqa: E402

_CASES = edc.cases()


def _case_id(make):
    return make().name


@pytest.fixture(scope="module")
def built():
    return [make() for make in _CASES]


def test_case_names_are_unique(built):
    names = [c.name for c in built]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("i", range(len(_CASES)))
def test_emitter_accepts_the_target_calibration(built, i):
    case = built[i]
    model = case.emitted()
    assert [v.name for v in model.graph.input] == [
        v.name for v in case.native.graph.input
    ]
    feeds, _ = edc.feeds_for(case, case.target_cfg)
    ref = edc.qdq_reference(case.float_model, case.target_cfg, feeds)
    assert ref[0].size > 0


def test_qdq_reference_quantizes_inputs_and_outputs():
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> g (float[4] x) => (float[4] y) '
        "{ y = Relu (x) }"
    )
    model.graph.node[0].name = "r"
    cfg = edc.QConfig(
        inputs={("r", "x"): (0.5, 10, 0, 255)}, outputs={"y": (0.25, 0, 0, 255)}
    )
    x = np.array([-7.0, 0.2, 0.26, 100.0], np.float32)
    (y,) = edc.qdq_reference(model, cfg, {"x": x})
    # x -> clip(round(x/0.5)+10) dequantizes to [-5, 0, 0.5, 100]; relu; y at
    # 0.25 clamps 100 to 255 * 0.25
    np.testing.assert_allclose(y, [0.0, 0.0, 0.5, 63.75])


def test_lsb_error_is_in_output_steps():
    ref = np.array([0.0, 1.0, 2.0], np.float32)
    dev = np.array([0.0, 1.5, 2.0], np.float32).tobytes()
    res = edc.lsb_error(dev, ref, 0.5)
    assert res["max_lsb"] == pytest.approx(1.0)
    assert res["frac_over_1lsb"] == 0.0


def test_conv_concat_ratio_change_is_refused():
    # Moving a 3x3 Conv's Concat input off 2x its source's scale ran 189 LSB
    # off on the device; recalibrate must refuse it. (The first-round stage3
    # conv1 template that showed it is replaced by a step-real one without a
    # power-of-two ratio, so the guard is checked on its own.)
    old = {"src": (0.0125, 0.0), "cat": (0.025, 0.0), "w": (0.003, 0.0)}
    edc.mre._check_fixed_ratios(old, {**old, "src": (0.01, 0.0), "cat": (0.02, 0.0)})
    with pytest.raises(edc.mre.CalibrationError, match="breaks that ratio"):
        edc.mre._check_fixed_ratios(old, {**old, "cat": (0.026, 0.0)})


def test_log_emits_the_output_scale_lanes():
    # the step's Log moves s_y too; the dequantize lanes must follow it
    import misc_op_record_emit as misc

    sc, zp, _ = edc._misc_step("Log:16x1000")
    lanes = misc.lane_values("Log", sc)
    assert "s_y" in lanes


def test_recorded_device_results_are_within_tolerance():
    import json

    with open(os.path.join(edc.OWN, "device_results.json")) as f:
        rec = json.load(f)
    assert len(rec["results"]) == len(_CASES)
    for res in rec["results"]:
        assert res["health_after"], res
        got = res["emitted"]
        if "max_lsb" in got:
            assert got["max_lsb"] <= 2.0 + 1e-3 and got["frac_over_1lsb"] <= 1e-5, res
        else:
            assert got["mismatches"] == 0, res


_ON_DEVICE = pytest.mark.skipif(
    not edc.device_available(), reason="needs the AX650 card (AXCL_LXD_VM)"
)


@_ON_DEVICE
@pytest.mark.parametrize("i", range(len(_CASES)))
def test_emitted_model_on_device(built, i):
    res = edc.check_case(built[i])
    assert "error" not in res, res
    assert res["health_after"], res
    got, ctl = res["emitted"], res["control"]
    assert "error" not in got, res
    if "max_lsb" in got:
        # Within 1 LSB of the reference, like the native templates; a chained
        # requantize (the 3x3 Conv) may add one rounding tie: 1 element in
        # 802,816 at 2 LSB on the device (docs/axera-emitter-device-check.md).
        assert got["max_lsb"] <= max(2.0, ctl["max_lsb"]) + 1e-3, res
        assert got["frac_over_1lsb"] <= 1e-5, res
    else:
        assert got["mismatches"] == 0, res
