"""No-device checks for the elementwise scale-retarget emitter.

Each committed oracle is a native Pulsar2 build at a calibration held out from
the template; emitting that calibration from the template must reproduce it
byte for byte (outside the known 301-325 MCode noise window)."""

import gzip
import json
import os
import sys

import onnx
import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import elementwise_scale_emit as E  # noqa: E402

_ORACLES = os.path.join(E.TEMPLATE_DIR, "oracles")
with open(os.path.join(_ORACLES, "index.json")) as _f:
    _ORACLE_INDEX = json.load(_f)


def _masked(model: onnx.ModelProto) -> bytes:
    init = E._mcode_initializer(model)
    data = bytearray(init.raw_data)
    data[301:326] = bytes(25)
    init.raw_data = bytes(data)
    return model.SerializeToString()


@pytest.mark.parametrize("oracle", sorted(_ORACLE_INDEX))
def test_emit_reproduces_held_out_native_build(tmp_path, oracle):
    meta = _ORACLE_INDEX[oracle]
    out = tmp_path / "emitted.axmodel"
    E.emit(meta["op"], meta["shape"], meta["scale"], meta["zero_point"], str(out))
    with gzip.open(os.path.join(_ORACLES, oracle), "rb") as f:
        native = onnx.load_model_from_string(f.read())
    emitted = onnx.load(str(out), load_external_data=False)
    assert _masked(emitted) == _masked(native)


def test_scale_floats_match_known_pulsar2_encoding():
    # 1/255 and 1.8/255 builds: quant/dequant words read from compiled models.
    q, d = E.scale_floats(0.003921568859368563)
    assert (q.hex(), d.hex()) == ("ffff7e43", "8180803b")
    q, d = E.scale_floats(0.007058823481202126)
    assert (q.hex(), d.hex()) == ("abaa0d43", "b44de73b")


@pytest.mark.parametrize(
    "lo,hi,scale_bits,zp",
    [
        (0.0, 1.0, 0x3B808081, 0),
        (-0.5, 0.5, 0x3B808081, 128),
        (-0.1, 0.9, 0x3B808080, 26),
        (-3.0, 1.0, 0x3C808081, 191),
    ],
)
def test_minmax_params_matches_pulsar2(lo, hi, scale_bits, zp):
    scale, zero_point = E.minmax_params(lo, hi)
    assert E._f32_bits(scale) == scale_bits
    assert zero_point == zp


def test_emit_refuses_unvalidated_template(tmp_path):
    with pytest.raises(ValueError, match="no validated template"):
        E.emit("Relu", [16, 64, 56, 56], 0.01, 37, str(tmp_path / "x.axmodel"))
    with pytest.raises(ValueError, match="no validated template"):
        E.emit("Relu", [2, 64, 56, 56], 0.01, 0, str(tmp_path / "x.axmodel"))


def test_emit_from_reference_refuses_zero_point_change(tmp_path):
    ref = tmp_path / "ref.axmodel"
    model, meta = E.load_template("Relu", [16, 64, 56, 56], 0)
    onnx.save(model, str(ref))
    with pytest.raises(ValueError, match="zero point change"):
        E.emit_from_reference(
            str(ref), meta["scale"], 0, 0.01, 5, str(tmp_path / "o.axmodel")
        )


def test_retarget_refuses_wrong_template_scale():
    model, meta = E.load_template("Relu", [16, 64, 56, 56], 0)
    mc = bytes(E._mcode_initializer(model).raw_data)
    with pytest.raises(ValueError, match="whole groups of four"):
        E.retarget_scale(mc, meta["scale"] * 1.5, 0.01)
