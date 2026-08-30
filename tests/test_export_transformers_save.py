# Unit tests for onnxsim.transformers_export._save: the inline-vs-external-data
# save behavior export_transformers_model uses for every graph it produces.
# Deliberately independent of torch/transformers/optimum (unlike
# test_export_transformers.py) -- _save itself only touches onnx/protobuf, so
# this exercises it directly with a small hand-built model, no export or
# network involved.

import os

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.transformers_export import _save


def _model(body, initializer=(), opset=17, ir_version=9):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _tiny_model():
    w = onnx.numpy_helper.from_array(
        np.random.RandomState(0).rand(4, 4).astype(np.float32), "W"
    )
    return _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Add(X, W)
        }
        """,
        initializer=[w],
    )


def test_save_default_keeps_weights_inline(tmp_path):
    model = _tiny_model()
    path = str(tmp_path / "model.onnx")

    _save(model, path, force_external_data=False)

    assert not os.path.exists(path + ".data")
    reloaded, _pool = onnxsim.load_model(path)
    w = onnx.numpy_helper.to_array(
        next(i for i in reloaded.graph.initializer if i.name == "W")
    )
    np.testing.assert_array_equal(
        w, onnx.numpy_helper.to_array(model.graph.initializer[0])
    )


def test_save_force_external_data_writes_companion_file(tmp_path):
    model = _tiny_model()
    path = str(tmp_path / "model.onnx")
    # Snapshot the expected value before _save(): onnx.save's own
    # save_as_external_data mutates the passed-in model's initializers in
    # place (clears raw_data, points them at the external file), so
    # model.graph.initializer[0] is no longer a plain in-memory tensor once
    # _save() returns.
    expected_w = onnx.numpy_helper.to_array(model.graph.initializer[0]).copy()

    _save(model, path, force_external_data=True)

    assert os.path.exists(path + ".data")
    reloaded, _pool = onnxsim.load_model(path)  # resolves external data by default
    w = onnx.numpy_helper.to_array(
        next(i for i in reloaded.graph.initializer if i.name == "W")
    )
    np.testing.assert_array_equal(w, expected_w)


def test_save_force_external_data_overwrites_stale_companion_file(tmp_path):
    # onnx.save(..., save_as_external_data=True, all_tensors_to_one_file=True)
    # errors if its target .data file already exists -- _save must clear a
    # stale one first, e.g. from a previous run over the same output_dir.
    path = str(tmp_path / "model.onnx")
    with open(path + ".data", "wb") as f:
        f.write(b"stale")

    _save(_tiny_model(), path, force_external_data=True)

    with open(path + ".data", "rb") as f:
        assert f.read() != b"stale"


@pytest.mark.parametrize("force_external_data", [False, True])
def test_save_roundtrips_regardless_of_mode(tmp_path, force_external_data):
    model = _tiny_model()
    path = str(tmp_path / "model.onnx")

    _save(model, path, force_external_data=force_external_data)

    reloaded, _pool = onnxsim.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
