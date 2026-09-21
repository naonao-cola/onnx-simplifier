"""Regression coverage for the evidence-scoped fused-Reshape emitter."""

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

import reshape_emit  # noqa: E402
from reshape_emit import (  # noqa: E402
    FUSED_AFTER,
    FUSED_BEFORE,
    NOT_FUSED,
    emit_fused_reshape_axmodel,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "reshape")
_NOISE = slice(301, 326)


def _dims(shape):
    return "x".join(str(d) for d in shape)


def _init(model, name):
    return next(i for i in model.graph.initializer if i.name == name)


def _canonical(model):
    """Serialized model with the compiler-noise window of the MCode zeroed."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    mcode = _init(model, "subgraph_npu_0_b1_neu")
    data = bytearray(mcode.raw_data)
    data[_NOISE] = bytes(len(data[_NOISE]))
    mcode.raw_data = bytes(data)
    return model.SerializeToString()


def _load_gz(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return onnx.load_model_from_string(f.read())


def _dims_of(value_info):
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _pair_id(pair):
    return f"{_dims(pair[0])}-to-{_dims(pair[1])}"


@pytest.mark.parametrize("pair", sorted(FUSED_BEFORE), ids=_pair_id)
def test_before_relabels_input_and_keeps_relu_at_output_shape(tmp_path, pair):
    source, target = pair
    out = tmp_path / "e.axmodel"

    emit_fused_reshape_axmodel(source, target, str(out), position="before")

    emitted = onnx.load(str(out), load_external_data=False)
    template = _load_gz(f"relu_{_dims(target)}.axmodel.gz")
    assert _dims_of(emitted.graph.input[0]) == list(source)
    assert _dims_of(emitted.graph.output[0]) == list(target)
    attrs = {
        a.name: onnx.helper.get_attribute_value(a)
        for a in emitted.graph.node[0].attribute
    }
    assert json.loads(attrs["outputs_info"]) == {"y": ["FP32", list(target)]}
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == bytes(
        _init(template, "subgraph_npu_0_b1_neu").raw_data
    )
    assert bytes(_init(emitted, "npu_params").raw_data) == bytes(40)


@pytest.mark.parametrize("pair", sorted(FUSED_AFTER), ids=_pair_id)
def test_after_relabels_output_and_keeps_relu_at_input_shape(tmp_path, pair):
    source, target = pair
    out = tmp_path / "e.axmodel"

    emit_fused_reshape_axmodel(source, target, str(out), position="after")

    emitted = onnx.load(str(out), load_external_data=False)
    template = _load_gz(f"relu_{_dims(source)}.axmodel.gz")
    assert _dims_of(emitted.graph.input[0]) == list(source)
    assert _dims_of(emitted.graph.output[0]) == list(target)
    attrs = {
        a.name: onnx.helper.get_attribute_value(a)
        for a in emitted.graph.node[0].attribute
    }
    assert json.loads(attrs["outputs_info"]) == {"y": ["FP32", list(target)]}
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == bytes(
        _init(template, "subgraph_npu_0_b1_neu").raw_data
    )


@pytest.mark.parametrize(
    "position,source,target",
    [
        ("before", (1, 8, 4, 4), (1, 1, 8, 16)),
        ("before", (16, 4, 4, 4), (16, 1, 4, 16)),
        ("before", (1, 1, 8, 48), (1, 8, 6, 8)),
        ("before", (1, 64), (64,)),
        ("after", (1, 8, 4, 4), (1, 1, 8, 16)),
        ("after", (1, 64), (64,)),
    ],
)
def test_emitted_model_equals_compiler_built_reshape_relu(
    tmp_path, position, source, target
):
    oracle = _load_gz(
        f"oracle_{position}_{_dims(source)}_to_{_dims(target)}.axmodel.gz"
    )
    out = tmp_path / "e.axmodel"

    emit_fused_reshape_axmodel(source, target, str(out), position=position)

    # Whole-model equality (graph, dims, metadata, both initializers, MCode)
    # outside the known 301-325 compiler-noise window.
    assert _canonical(onnx.load(str(out), load_external_data=False)) == _canonical(
        oracle
    )


def test_registry_has_a_fixture_for_every_measured_pair():
    for source, target in FUSED_BEFORE:
        assert os.path.exists(
            os.path.join(_FIXTURES, f"relu_{_dims(target)}.axmodel.gz")
        )
    for source, target in FUSED_AFTER:
        assert os.path.exists(
            os.path.join(_FIXTURES, f"relu_{_dims(source)}.axmodel.gz")
        )


def test_fused_and_not_fused_tables_are_disjoint_and_preserve_numel():
    assert not (FUSED_BEFORE & NOT_FUSED)
    for source, target in FUSED_BEFORE | FUSED_AFTER | NOT_FUSED:
        assert reshape_emit._numel(source) == reshape_emit._numel(target)


def test_resnet18_step_reshape_families_are_covered_by_the_tables():
    # Only the bias flatten fuses; every convolution/weight Reshape family of
    # the ResNet18 step was measured to need real DMA MCode.
    assert ((1, 64), (64,)) in FUSED_BEFORE
    assert ((1, 512), (512,)) in FUSED_BEFORE
    for pair in [
        ((16, 64, 56, 56), (16, 1, 64, 3136)),
        ((16, 1, 64, 3136), (16, 64, 56, 56)),
        ((16, 1, 64, 28224), (16, 1, 576, 3136)),
        ((64, 64, 3, 3), (1, 64, 64, 9)),
        ((1, 64, 64, 9), (1, 1, 64, 576)),
        ((1, 64, 576), (64, 64, 3, 3)),
        ((512, 512, 3, 3), (1, 512, 512, 9)),
    ]:
        assert pair in NOT_FUSED


@pytest.mark.parametrize(
    "source,target",
    [((1, 4, 4, 9), (1, 1, 4, 36)), ((64, 64, 3, 3), (1, 64, 64, 9))],
)
def test_pairs_measured_to_need_real_reshape_mcode_are_rejected(
    tmp_path, source, target
):
    with pytest.raises(ValueError, match="real DMA program"):
        emit_fused_reshape_axmodel(source, target, str(tmp_path / "bad.axmodel"))


@pytest.mark.parametrize(
    "source,target,position,match",
    [
        ((1, 8, 4, 4), (1, 8, 16), "before", "unmeasured"),
        ((1, 8, 4, 4), (1, 1, 8, 15), "before", "element counts"),
        ((1, 4, 4, 4), (1, 1, 4, 16), "after", "unmeasured"),
        ((1, 8, 4, 4), (1, 1, 8, 16), "sideways", "position"),
        ((), (1,), "before", "non-empty"),
        ((1, 0), (0,), "before", "positive"),
        ((1, True), (1,), "before", "positive"),
        ("18", (8,), "before", "sequence"),
    ],
)
def test_invalid_or_unmeasured_requests_are_rejected(
    tmp_path, source, target, position, match
):
    with pytest.raises(ValueError, match=match):
        emit_fused_reshape_axmodel(
            source, target, str(tmp_path / "bad.axmodel"), position=position
        )


def test_template_with_unexpected_parameter_table_is_rejected(tmp_path, monkeypatch):
    model = _load_gz("relu_1x1x8x16.axmodel.gz")
    _init(model, "npu_params").raw_data = bytes([1]) + bytes(39)
    with gzip.open(tmp_path / "relu_1x1x8x16.axmodel.gz", "wb") as f:
        f.write(model.SerializeToString())
    monkeypatch.setattr(reshape_emit, "_FIXTURES", str(tmp_path))
    with pytest.raises(ValueError, match="npu_params"):
        emit_fused_reshape_axmodel(
            (1, 8, 4, 4), (1, 1, 8, 16), str(tmp_path / "bad.axmodel")
        )
