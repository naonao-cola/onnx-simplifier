"""No-device checks for ``scripts/axera/teng_register_census.py``.

Fixtures are real Pulsar2 7.0-lite AX650 builds (gzipped ``.axmodel``) of the
census's standalone ops under engineered calibrations, plus each build's
quantization scales (``scales.json``, read from the build's own
``quant_axmodel.onnx``). See ``docs/axera-teng-register-census.md``.
"""

import json
import os
import struct
import sys

import onnx
import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import teng_register_census as census  # noqa: E402

_FIX = os.path.join(_AXERA, "fixtures", "teng_register_census")
with open(os.path.join(_FIX, "scales.json")) as _f:
    _SCALES = json.load(_f)


def _build(name):
    model = census.load_axmodel(os.path.join(_FIX, name + ".axmodel.gz"))
    tensors = {k: (v[0], v[1]) for k, v in _SCALES[name].items()}
    return {
        "mc": census.mcode_of(model),
        "npu_params": census._npu_params(model),
        "params": {"tensors": tensors, "ops": []},
    }


def _builds(op, variants):
    return {v: _build(f"{op}_{v}") for v in variants}


@pytest.mark.parametrize("op", sorted(census.OPS))
def test_every_census_graph_is_a_valid_model(op):
    model = census._model(op)
    onnx.checker.check_model(model, full_check=True)
    assert [i.name for i in model.graph.input] == list(census.OPS[op]["inputs"])


def test_register_writes_cover_v_w_and_short_units_outside_noise():
    writes = census.register_writes(_build("sqrt_p4")["mc"])
    kinds = {w["kind"] for w in writes}
    assert {"V", "W", "S"} <= kinds
    assert all(
        not census.NOISE_WINDOW[0] <= w["at"] < census.NOISE_WINDOW[1] for w in writes
    )
    # Occurrence counts per (segment, register).
    seen = {}
    for w in writes:
        key = (w["seg"], w["reg"])
        assert w["occurrence"] == seen.get(key, 0)
        seen[key] = w["occurrence"] + 1
    # A full 32-bit register write occupies an 8-byte V record.
    v = [w for w in writes if w["kind"] == "V" and w["reg"] == 0x0F60][0]
    assert v["width"] == 4 and v["length"] == 8


def test_scale_monomials_names_and_values():
    monos = census.scale_monomials({"x": (0.5, 0), "y": (4.0, 0)})
    assert monos["1/s_x"] == 2.0
    assert monos["s_x/s_y"] == 0.125
    assert monos["s_x*s_y"] == 2.0
    assert len(monos) == 8  # 3**2 - 1


def test_match_register_float_and_fixed_point():
    params = [
        {"tensors": {"x": (0.25, 0), "y": (0.5, 0)}},
        {"tensors": {"x": (0.125, 0), "y": (0.5, 0)}},
    ]
    as_f32 = [
        struct.unpack("<I", struct.pack("<f", 1 / p["tensors"]["x"][0]))[0]
        for p in params
    ]
    assert "f32 1/s_x" in census.match_register(as_f32, params)
    q15 = [round(p["tensors"]["x"][0] / p["tensors"]["y"][0] * 32768) for p in params]
    assert "round(s_x/s_y * 2^15)" in census.match_register(q15, params)


def test_sqrt_quant_short_unit_dequant_and_divisor_block():
    located = census.locate(_builds("sqrt", ["p1", "p4", "p9"]))
    # 1/s_x: the four per-lane quantize copies at 0x0f50..0x0f80.
    inv = located["1/s_x"]
    assert inv["found"] == 3 and inv["copies"] == {"p1": 4, "p4": 4, "p9": 4}
    assert {"seg2:W:0x0f50", "seg2:V:0x0f60", "seg2:V:0x0f70", "seg2:V:0x0f80"} <= set(
        inv["locations"]
    )
    # s_x: three 7-byte compressed short units.
    sx = located["s_x"]
    assert sx["found"] == 3 and sx["encodings"] == ["S7"]
    # s_y: the 0x0fd0..0x1000 block (Mul's divisor block), in the two variants
    # where s_y differs from s_x.
    sy = located["s_y"]
    assert {"seg2:W:0x0fd0", "seg2:V:0x0fe0", "seg2:V:0x0ff0", "seg2:V:0x1000"} <= set(
        sy["locations"]
    )
    assert sy["discriminating"] == 2


def test_reducesum_quant_requant_ratio_and_dequant():
    located = census.locate(_builds("reducesum", ["s1", "s4", "asym"]))
    assert located["1/s_x"]["discriminating"] == 3
    ratio = located["s_x/s_y"]
    assert ratio["discriminating"] == 3 and ratio["encodings"] == ["S7"]
    assert set(ratio["copies"].values()) == {4}
    sy = located["s_y"]
    assert sy["discriminating"] == 3
    assert "seg2:V:0x0f50" in sy["locations"]


def test_div_dequant_in_short_units_and_no_second_input_multiplier():
    builds = _builds("div", ["s1", "r3", "asym"])
    located = census.locate(builds)
    assert located["1/s_x"]["found"] == 3
    sy = located["s_y"]
    assert sy["found"] == 3 and sy["encodings"] == ["S7"]
    assert set(sy["copies"].values()) == {4}
    # With z's scale different from x's (s1, r3; asym happens to make them
    # equal), 1/s_z is not written anywhere.
    for v in ("s1", "r3"):
        tensors = builds[v]["params"]["tensors"]
        assert tensors["z"][0] != tensors["x"][0]
        want = 1.0 / tensors["z"][0]
        for w in census.register_writes(builds[v]["mc"]):
            if w["width"] == 4:
                assert not abs(census._f32(w["value"]) - want) <= 1e-6 * want


def test_greater_cast_program_is_calibration_invariant():
    builds = _builds("gtcast", ["s1", "asym"])
    assert all(not b["params"]["tensors"] for b in builds.values())
    # Segment 0 differs between the two builds even with no quantization at all,
    # so segment-0 differences are build noise, not calibration.
    assert builds["s1"]["mc"] != builds["asym"]["mc"]
    assert census.identical_outside(builds, (0,))
