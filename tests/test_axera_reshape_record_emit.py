"""No-device checks for ``scripts/axera/reshape_record_emit.py``.

The emitter fits record-level polynomials from committed Pulsar2 builds of a
non-fused ``Reshape -> Relu`` family and predicts the MCode at another size.
Every check compares the prediction with a compiler-built model that was not
used for the fit, byte for byte outside the segment-0 rebuild-noise window
(blob offsets 301-326). See ``docs/axera-reshape-decompressed.md``."""

import os
import struct
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import reshape_record_emit as rre  # noqa: E402
import short_unit_codec as suc  # noqa: E402

_HELD_OUT = os.path.join(_AXERA_DIR, "fixtures", "reshape_record_emit")
_FIX = os.path.join(_AXERA_DIR, "fixtures")
_NOISE = range(301, 327)


def _mcode(path):
    return rre.mcode_of(rre.load_axmodel(path))


def _diffs(got, want):
    assert len(got) == len(want)
    return [i for i in range(len(got)) if got[i] != want[i] and i not in _NOISE]


def _scale(mc):
    for raw in suc.decode_segments(mc):
        slots = rre.scale_lanes(raw)
        if slots:
            k = slots[8]
            return struct.unpack("<f", raw[k * 8 + 4 : k * 8 + 8])[0]
    raise AssertionError("no scale lanes")


@pytest.mark.parametrize("name", sorted(rre.FAMILIES))
def test_leave_one_fixture_out(name):
    for param, rel in rre.FAMILIES[name].fixtures:
        rules = rre.family_rules(name, exclude=[param])
        got = rre.predict_mcode(rules, param)
        assert _diffs(got, _mcode(os.path.join(_FIX, rel))) == [], param


@pytest.mark.parametrize(
    "name, param, fixture",
    [
        ("square_weight_fold", 48, "square_C48"),
        ("square_weight_fold", 52, "square_C52"),
        # The Co values docs/axera-reshape-dma.md recorded as layout changes.
        ("cin8_weight_fold", 36, "cin8_co36"),
        ("cin8_weight_fold", 60, "cin8_co60"),
    ],
)
def test_held_out_native_builds(name, param, fixture):
    got = rre.predict_mcode(rre.family_rules(name), param)
    assert _diffs(got, _mcode(os.path.join(_HELD_OUT, fixture + ".axmodel.gz"))) == []


def test_real_resnet18_weight_fold_from_smaller_sizes():
    # [64,64,3,3] -> [1,64,64,9] as the ResNet18 step uses it, built with the
    # step-shape calibration recipe; the rule is fitted without C=64.
    want = _mcode(os.path.join(_HELD_OUT, "real_w64a_rs_relu.axmodel.gz"))
    rules = rre.family_rules("square_weight_fold", exclude=[64])
    plain = rre.predict_mcode(rules, 64)
    record_diffs = rre.compare_mcode(plain, want)["record_diffs"]
    # Only the 16 calibration scale lanes differ (another calibration) ...
    assert sorted({reg for _, _, reg in record_diffs}) == [
        0x0F50 + 0x10 * i for i in range(8)
    ]
    assert len(record_diffs) == 16
    # ... and with its scale the program is the native one.
    assert _diffs(rre.predict_mcode(rules, 64, scale=_scale(want)), want) == []


def test_scale_override_other_calibration():
    want = _mcode(os.path.join(_HELD_OUT, "cin8_co20.axmodel.gz"))
    rules = rre.family_rules("cin8_weight_fold")
    assert (
        len(rre.compare_mcode(rre.predict_mcode(rules, 20), want)["record_diffs"]) == 16
    )
    assert _diffs(rre.predict_mcode(rules, 20, scale=_scale(want)), want) == []


def test_emit_axmodel_relabels_graph(tmp_path):
    out = tmp_path / "w.axmodel"
    model = rre.emit_axmodel("square_weight_fold", 52, str(out))
    assert out.exists()
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(model.graph.input[0]) == [52, 52, 3, 3]
    assert dims(model.graph.output[0]) == [1, 52, 52, 9]
    attrs = {a.name: a.s for a in model.graph.node[0].attribute}
    assert b"[1, 52, 52, 9]" in attrs["outputs_info"]
    want = _mcode(os.path.join(_HELD_OUT, "square_C52.axmodel.gz"))
    assert _diffs(rre.mcode_of(model), want) == []


@pytest.mark.parametrize(
    "name, param",
    [("square_weight_fold", 68), ("square_weight_fold", 128), ("cin8_weight_fold", 12)],
)
def test_refuses_outside_measured_group(name, param):
    with pytest.raises(ValueError, match="outside the measured"):
        rre.emit_axmodel(name, param)


def test_fit_rejects_mixed_structures():
    a = _mcode(os.path.join(_HELD_OUT, "square_C48.axmodel.gz"))
    b = _mcode(os.path.join(_HELD_OUT, "cin8_co36.axmodel.gz"))
    with pytest.raises(ValueError):
        rre.fit_rules([(48, a), (36, b), (52, a)])


def test_fit_poly_needs_a_spare_point():
    assert rre.fit_poly([1, 2, 3], [1, 4, 9]) is None
    assert rre.fit_poly([1, 2, 3, 4], [1, 4, 9, 16]) == [0, 0, 1]
    assert rre.fit_poly([1, 2], [5, 7]) is None
    assert rre.fit_poly([1, 2, 3], [5, 7, 9]) == [3, 2]
