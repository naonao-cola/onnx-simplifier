import gzip
import os
import sys

import onnx
import pytest

_AXERA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "axera")
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import template_model_generator as tmg  # noqa: E402

_TEMPLATE = os.path.join(
    _AXERA, "fixtures", "compose_gather_reshape_matmul_transpose_add.axmodel.gz"
)


def _load_template():
    with gzip.open(_TEMPLATE, "rb") as stream:
        return onnx.load_model_from_string(stream.read())


def test_same_topology_is_generated_without_compiler(tmp_path):
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    template = tmp_path / "template.axmodel"
    output = tmp_path / "generated.axmodel"
    model = _load_template()
    onnx.save(model, source)
    onnx.save(model, template_source)
    onnx.save(model, template)

    sig = tmg.generate(str(source), str(template_source), str(template), str(output))
    assert sig == tmg.signature(model)
    assert (
        onnx.load(str(output), load_external_data=False).SerializeToString()
        == model.SerializeToString()
    )


def test_gz_template_can_be_reused_directly(tmp_path):
    model = _load_template()
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    output = tmp_path / "generated.axmodel.gz"
    onnx.save(model, str(source))
    onnx.save(model, str(template_source))

    tmg.generate(str(source), str(template_source), _TEMPLATE, str(output))

    with gzip.open(output, "rb") as stream:
        assert onnx.load_model_from_string(stream.read()).SerializeToString() == (
            model.SerializeToString()
        )


def test_topology_mismatch_is_refused(tmp_path):
    model = _load_template()
    source = onnx.ModelProto()
    source.CopyFrom(model)
    source.graph.node[0].attribute[0].s = b"different"
    paths = []
    for name, value in (
        ("source.onnx", source),
        ("template.onnx", model),
        ("template.axmodel", model),
    ):
        path = tmp_path / name
        onnx.save(value, path)
        paths.append(str(path))
    with pytest.raises(ValueError, match="topology"):
        tmg.generate(paths[0], paths[1], paths[2], str(tmp_path / "out.axmodel"))


def test_compiled_template_io_mismatch_is_refused(tmp_path):
    model = _load_template()
    source = tmp_path / "source.onnx"
    template_source = tmp_path / "template_source.onnx"
    template = tmp_path / "template.axmodel"
    output = tmp_path / "out.axmodel"
    broken = onnx.ModelProto()
    broken.CopyFrom(model)
    broken.graph.input[0].name = "wrong_input"
    onnx.save(model, str(source))
    onnx.save(model, str(template_source))
    onnx.save(broken, str(template))

    with pytest.raises(ValueError, match="IO"):
        tmg.generate(str(source), str(template_source), str(template), str(output))
