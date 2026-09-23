"""No-device checks for ``scripts/axera/misc_op_record_emit.py`` on committed
compiler-built fixtures: emitted records must equal held-out native builds."""

import gzip
import itertools
import json
import os
import sys

import onnx
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import misc_op_record_emit as mre  # noqa: E402

FIXTURES = os.path.join(AXERA, "fixtures")
with open(os.path.join(FIXTURES, "misc_op_record_emit", "held_out.json")) as _f:
    HELD = json.load(_f)
INDEX = mre.load_index()


def _mcode(rel: str) -> bytes:
    with gzip.open(os.path.join(FIXTURES, rel), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return bytes(mre.mcode_initializer(model).raw_data)


@pytest.mark.parametrize(
    "key", sorted(k for k, v in INDEX.items() if v["op"] != "GreaterCast")
)
def test_own_calibration_is_identity(key):
    # Each template's lanes and zero-point records are exactly what the
    # formulas give for its own calibration.
    meta = INDEX[key]
    mc = _mcode(meta["file"])
    old = (meta["scales"], meta["zero_points"])
    shifted = {"x": meta["scales"]["x"] * 1.5, "y": meta["scales"]["y"] * 0.75}
    moved = mre.retarget(mc, meta["op"], *old[:1], shifted, old[1], old[1])
    assert mre.normalized_records(moved) != mre.normalized_records(mc)
    back = mre.retarget(moved, meta["op"], shifted, old[0], old[1], old[1])
    assert mre.normalized_records(back) == mre.normalized_records(mc)
    assert mre.retarget(mc, meta["op"], old[0], old[0], old[1], old[1]) == mc


SQRT = sorted(k for k in HELD if "sqrt_512x512x3x3" in k)
REDUCESUM = sorted(k for k in HELD if "reducesum" in k)


@pytest.mark.parametrize("src,dst", list(itertools.permutations(SQRT, 2)))
def test_sqrt_512x512x3x3_matches_held_out_builds(src, dst):
    a, b = HELD[src], HELD[dst]
    got = mre.retarget(_mcode(src), "Sqrt", a["scales"], b["scales"])
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


@pytest.mark.parametrize("src,dst", list(itertools.permutations(REDUCESUM, 2)))
def test_reducesum_matches_held_out_builds(src, dst):
    # s1 -> asym moves a 0x1b10 record between stages (zp_y == zp_x vs 0).
    a, b = HELD[src], HELD[dst]
    got = mre.retarget(
        _mcode(src),
        "ReduceSum",
        a["scales"],
        b["scales"],
        a["zero_points"],
        b["zero_points"],
    )
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


def test_greater_cast_is_calibration_free():
    s1 = _mcode("teng_register_census/gtcast_s1.axmodel.gz")
    asym = _mcode("teng_register_census/gtcast_asym.axmodel.gz")
    assert s1 != asym
    assert mre.normalized_records(s1) == mre.normalized_records(asym)
    model = mre.emit_model("GreaterCast:1x64x56x56")
    assert bytes(mre.mcode_initializer(model).raw_data) == s1
    with pytest.raises(ValueError):
        mre.emit_model("GreaterCast:1x64x56x56", scales={"x": 0.1})


def test_emit_model_retargets_template():
    key = "ReduceSum:16x1x64x576:axes0:k0"
    meta = INDEX[key]
    scales = {"x": 0.01, "y": 0.2}
    zps = {"x": 120, "y": 135}
    mc = bytes(mre.mcode_initializer(mre.emit_model(key, scales, zps)).raw_data)
    want = mre.lane_values("ReduceSum", scales)
    runs = {v for _, v in mre.lane_runs(mre._chunks(mre.suc.decode_segments(mc)[2]))}
    assert set(want.values()) <= runs
    assert not set(mre.lane_values("ReduceSum", meta["scales"]).values()) & runs


@pytest.mark.parametrize(
    "scales,zps",
    [
        ({"x": 0.5, "y": 2.0}, None),  # 1/s_x == s_y: lanes indistinguishable
        ({"x": 0.02, "y": 0.02}, {"x": 0, "y": 0}),  # zp_x == 0 is unmeasured
        ({"x": 0.02, "y": 0.03}, {"x": 128, "y": 5}),  # net record insertion
    ],
)
def test_reducesum_refuses_unmeasured_targets(scales, zps):
    meta = INDEX["ReduceSum:1x1x64x3136:axes2:k0"]
    mc = _mcode(meta["file"])
    with pytest.raises(ValueError):
        mre.retarget(
            mc,
            "ReduceSum",
            meta["scales"],
            scales,
            meta["zero_points"],
            zps or meta["zero_points"],
        )


def test_sqrt_refuses_zero_point_change():
    meta = INDEX["Sqrt:512x512x3x3"]
    with pytest.raises(ValueError):
        mre.retarget(
            _mcode(meta["file"]),
            "Sqrt",
            meta["scales"],
            meta["scales"],
            meta["zero_points"],
            {"x": 3, "y": 0},
        )


def test_unknown_template_is_an_error():
    with pytest.raises(ValueError):
        mre.emit_model("ReduceSum:16x1x128x1152:axes0:k0")


def test_step_node_keys(tmp_path):
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 13]>
        g (float[16,1,64,576] a, float[16,64,56,56] x, float[512,512,3,3] w)
            => (float[1,64,576] r, float[16,64,56,56] c, float[512,512,3,3] s) {
            ax = Constant <value = int64[1] {0}> ()
            r = ReduceSum <keepdims = 0> (a, ax)
            zero = Constant <value = float {0.0}> ()
            g = Greater (x, zero)
            c = Cast <to = 1> (g)
            s = Sqrt (w)
        }
        """
    )
    path = str(tmp_path / "step.onnx")
    onnx.save(model, path)
    assert mre.step_node_keys(path) == [
        ("ReduceSum", "ReduceSum:16x1x64x576:axes0:k0"),
        ("Greater", "GreaterCast:16x64x56x56"),
        ("Cast", "GreaterCast:16x64x56x56"),
        ("Sqrt", "Sqrt:512x512x3x3"),
    ]
    cov = mre.coverage(path)
    assert cov["ReduceSum"] == {"nodes": 1, "covered": 1, "missing": {}}
    assert cov["Sqrt"]["covered"] == 1
    assert cov["Greater"]["missing"] == {"GreaterCast:16x64x56x56": 1}
