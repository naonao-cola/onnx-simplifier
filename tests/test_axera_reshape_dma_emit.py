"""Coverage for the weight-fold Reshape DMA emitter (``Cin=8`` family)."""

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

import reshape_dma_emit as em  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "reshape_dma")


def _load(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _mcode(model):
    return bytes(em._neu(model).raw_data)


def _outside_noise(blob):
    data = bytearray(blob)
    data[em._NOISE[0] : em._NOISE[1]] = bytes(em._NOISE[1] - em._NOISE[0])
    return bytes(data)


def _dims(value_info):
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def test_exact_set_is_the_validated_one():
    assert em.EXACT_CO == {34, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 62, 64}


@pytest.mark.parametrize("co", [34, 42, 58, 62])
def test_emitter_reproduces_compiler_built_held_out_models(tmp_path, co):
    """Held-out Co: the Co=40 template, retargeted, equals Pulsar2's own build."""
    built = _load(f"w1_co{co}_oracle.axmodel.gz")
    target = tmp_path / "out.axmodel"

    em.emit_weight_fold_axmodel(str(target), co=co)

    emitted = onnx.load(str(target), load_external_data=False)
    assert _outside_noise(_mcode(emitted)) == _outside_noise(_mcode(built))
    for name in ("npu_params", "npu_dyn_params"):
        got = next(i for i in emitted.graph.initializer if i.name == name)
        want = next(i for i in built.graph.initializer if i.name == name)
        assert got.raw_data == want.raw_data
    assert _dims(emitted.graph.input[0]) == [co, 8, 3, 3]
    assert _dims(emitted.graph.output[0]) == [1, co, 8, 9]
    attrs = {
        a.name: onnx.helper.get_attribute_value(a)
        for a in emitted.graph.node[0].attribute
    }
    assert json.loads(attrs["outputs_info"]) == {"y": ["FP32", [1, co, 8, 9]]}
    assert attrs["outputs_info"] == next(
        onnx.helper.get_attribute_value(a)
        for a in built.graph.node[0].attribute
        if a.name == "outputs_info"
    )


@pytest.mark.parametrize("co", sorted(em.EXACT_CO))
def test_every_exact_co_changes_only_the_fitted_fields(tmp_path, co):
    template = _mcode(_load("w1_co40.axmodel.gz"))
    predicted = em.predict_mcode(template, co)
    changed = {i for i in range(len(template)) if predicted[i] != template[i]}
    allowed = {offset for offset, _ in em.FIELDS}
    for offset in em.SIZE_OFFSETS:
        allowed |= {offset, offset + 1}
    assert changed <= allowed


def test_size_words_are_the_tensor_byte_size():
    template = _mcode(_load("w1_co40.axmodel.gz"))
    for co in (40, 62):
        out = em.predict_mcode(template, co)
        for offset in em.SIZE_OFFSETS:
            assert int.from_bytes(out[offset : offset + 2], "little") == 4 * 8 * 9 * co


@pytest.mark.parametrize(
    "co",
    [
        36,  # even, but the program changes layout
        60,  # even, but the program changes layout
        41,  # odd: a different instruction form
        53,
        66,  # >= 66 changes layout
        80,
        8,
        128,
        -40,
    ],
)
def test_refuses_unmeasured_or_layout_changing_co(tmp_path, co):
    with pytest.raises(ValueError, match="measured-exact"):
        em.emit_weight_fold_axmodel(str(tmp_path / "bad.axmodel"), co=co)


@pytest.mark.parametrize("co", [40.0, True, "40"])
def test_rejects_non_integer_co(tmp_path, co):
    with pytest.raises(ValueError, match="integer"):
        em.emit_weight_fold_axmodel(str(tmp_path / "bad.axmodel"), co=co)


def test_rejects_other_input_channels(tmp_path):
    with pytest.raises(ValueError, match="cin=8"):
        em.emit_weight_fold_axmodel(str(tmp_path / "bad.axmodel"), co=40, cin=16)
