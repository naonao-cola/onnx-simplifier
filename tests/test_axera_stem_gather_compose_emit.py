"""Regression coverage for the composed (14-chunk) stem Gather+Mul emitter.

No Docker/device needed -- see docs/axera-stem-gather-rechunk.md for the
Pulsar2 build and the AX8850 device verification this template is based on.
"""

import gzip
import os
import struct
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from stem_gather_compose_emit import _TEMPLATE, _TOTAL, emit  # noqa: E402


def _words(model):
    table = bytes(
        next(i.raw_data for i in model.graph.initializer if i.name == "npu_params")
    )
    return struct.unpack(f"<{len(table) // 4}I", table)


def _mcode(model):
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )


def test_emit_round_trips_indices(tmp_path):
    indices = [(i * 7) % 50176 for i in range(_TOTAL)]
    out = emit(str(tmp_path / "emitted.axmodel"), indices=indices)

    emitted = onnx.load(out, load_external_data=False)
    assert list(_words(emitted)[:_TOTAL]) == indices


def test_emit_preserves_mcode_and_tail_from_the_template(tmp_path):
    with gzip.open(_TEMPLATE, "rb") as f:
        template = onnx.load_model_from_string(f.read())
    indices = [0] * _TOTAL
    out = emit(str(tmp_path / "emitted.axmodel"), indices=indices)
    emitted = onnx.load(out, load_external_data=False)

    assert _mcode(emitted) == _mcode(template)
    assert _words(emitted)[_TOTAL:] == _words(template)[_TOTAL:]


@pytest.mark.parametrize(
    "indices",
    [
        [0, 1, 2],  # wrong length
        [50176] + [0] * (_TOTAL - 1),  # one index out of range
        [-1] + [0] * (_TOTAL - 1),
        [0.0] * _TOTAL,  # not integers
        [True] + [0] * (_TOTAL - 1),  # bool is not a plain int here
    ],
)
def test_emit_rejects_invalid_indices(tmp_path, indices):
    with pytest.raises(ValueError):
        emit(str(tmp_path / "bad.axmodel"), indices=indices)


def test_emit_rejects_a_reference_with_the_wrong_structure(tmp_path, monkeypatch):
    import stem_gather_compose_emit as mod

    def fake_load_template():
        with gzip.open(_TEMPLATE, "rb") as f:
            model = onnx.load_model_from_string(f.read())
        model.graph.node[0].output[0] = "not_y"
        return model

    monkeypatch.setattr(mod, "_load_template", fake_load_template)
    with pytest.raises(ValueError, match="input 'x' to output 'y'"):
        emit(str(tmp_path / "bad.axmodel"), indices=[0] * _TOTAL)
