"""Regression coverage for the untiled AX650 Transpose emitter and its field fit."""

import base64
import gzip
import json
import os
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import transpose_fields as tf  # noqa: E402
from transpose_emit import (  # noqa: E402
    _DIR,
    _NOISE_END,
    _NOISE_START,
    emit_transpose_axmodel,
    measured_blocks,
)


def _neu(model):
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


def _index():
    with open(os.path.join(_DIR, "index.json")) as f:
        return json.load(f)


def _oracles():
    with gzip.open(os.path.join(_DIR, "oracles.json.gz"), "rt") as f:
        return {
            tuple(int(v) for v in key.split(",")): base64.b64decode(value)
            for key, value in json.load(f).items()
        }


def _mask(blob):
    data = bytearray(blob)
    data[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    return bytes(data)


def test_index_covers_more_than_one_r_and_q():
    blocks = measured_blocks()
    assert len({r for r, _ in blocks}) > 1
    assert len({q for _, q in blocks}) > 3


@pytest.mark.parametrize("entry", _index(), ids=lambda e: f"R{e['R']}-q{e['q']}")
def test_emitted_models_match_compiler_built_held_out_shapes(tmp_path, entry):
    oracles = _oracles()
    assert entry["held_out"]["exact"] == len(entry["held_out"]["shapes"])
    for r, c in entry["held_out"]["shapes"]:
        out = tmp_path / f"t_{r}_{c}.axmodel"
        emit_transpose_axmodel(str(out), shape=(1, 1, r, c))
        model = onnx.load(str(out), load_external_data=False)
        assert _mask(_neu(model)) == _mask(oracles[(r, c)])
        assert [
            d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
        ] == [
            1,
            1,
            r,
            c,
        ]
        assert [
            d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim
        ] == [
            1,
            1,
            c,
            r,
        ]
        attrs = {a.name: a for a in model.graph.node[0].attribute}
        assert json.loads(attrs["outputs_info"].s) == {"y": ["FP32", [1, 1, c, r]]}


def test_emit_covers_every_c_of_a_measured_block(tmp_path):
    (r, q) = measured_blocks()[0]
    for c in range(8 * (q - 1) + 1, 8 * q):
        out = tmp_path / f"t{c}.axmodel"
        emit_transpose_axmodel(str(out), shape=(1, 1, r, c))
        model = onnx.load(str(out), load_external_data=False)
        assert model.graph.input[0].type.tensor_type.shape.dim[3].dim_value == c


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16, 64),  # C multiple of 8
        (1, 1, 16, 4000),  # block not measured
        (1, 1, 17, 20),  # R not measured
        (2, 1, 16, 20),  # batch
        (1, 1, 16),  # rank
        (1, 1, 16, 0),
        (1, 1, 16, True),
    ],
)
def test_emit_rejects_unmeasured_shapes(tmp_path, shape):
    with pytest.raises(ValueError):
        emit_transpose_axmodel(str(tmp_path / "bad.axmodel"), shape=shape)


def test_field_fit_rejects_mixed_length_builds():
    with pytest.raises(ValueError, match="length"):
        tf.fit({(16, 17): b"\x00" * 10, (16, 18): b"\x00" * 11})


def test_field_fit_finds_a_planted_size_field():
    def blob(c):
        data = bytearray(64)
        value = 4 * 16 * c - 1
        data[10], data[11] = value & 255, value >> 8
        return bytes(data)

    train = {(16, c): blob(c) for c in (17, 18, 19, 21, 23)}
    # the high byte of 4RC-1 and of 4RC agree on every one of these shapes
    with pytest.raises(ValueError, match="ambiguous"):
        tf.fit(train)
    fields = tf.fit(train, tie_break=True)
    assert fields == {10: ("4RC-1", 0), 11: ("4RC-1", 1)}
    assert tf.predict(train[(16, 17)], fields, (16, 22)) == blob(22)
    assert tf.score(train[(16, 17)], fields, {(16, 22): blob(22)}) == (1, [])
