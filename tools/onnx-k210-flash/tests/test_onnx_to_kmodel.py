"""Tests for onnx_to_kmodel.py.

nncase (K210-targeting, 1.9.0) only ships wheels for Python <=3.10 -- see
the script's own module docstring. Tests that need it are skipped (not
failed) when it isn't importable, which is the expected case under this
repo's own Python (3.11+); run them for real under a 3.10 venv:

    python3.10 -m venv .venv && .venv/bin/pip install nncase==1.9.0.20230322 numpy onnx pytest
    .venv/bin/pytest onnx-k210-flash/tests/test_onnx_to_kmodel.py -v

`_pin_batch_dim` itself needs only onnx (no nncase), so it's tested
unconditionally.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from onnx_to_kmodel import _pin_batch_dim, convert_to_kmodel  # noqa: E402


def _make_model_with_batch_dim(batch_dim_param: str | None, batch_dim_value: int) -> onnx.ModelProto:
    # A trivial Identity graph -- enough to exercise _pin_batch_dim's shape
    # handling without needing a real network.
    dim = onnx.TensorShapeProto.Dimension()
    if batch_dim_param is not None:
        dim.dim_param = batch_dim_param
    else:
        dim.dim_value = batch_dim_value
    input_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, None)
    input_type.tensor_type.shape.dim.extend([dim, helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [4]).tensor_type.shape.dim[0]])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
    )
    graph.input[0].type.CopyFrom(input_type)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    return model


def test_pin_batch_dim_replaces_symbolic_dim_param():
    model = _make_model_with_batch_dim("batch", 0)
    fixed = _pin_batch_dim(model)
    dim0 = fixed.graph.input[0].type.tensor_type.shape.dim[0]
    assert not dim0.HasField("dim_param")
    assert dim0.dim_value == 1


def test_pin_batch_dim_replaces_zero_dim_value():
    # Some exporters use a literal 0 instead of a symbolic dim_param for
    # "dynamic" -- the actual bug this function exists to work around
    # (see the module docstring's "Two real gotchas" section).
    model = _make_model_with_batch_dim(None, 0)
    fixed = _pin_batch_dim(model)
    dim0 = fixed.graph.input[0].type.tensor_type.shape.dim[0]
    assert dim0.dim_value == 1


def test_pin_batch_dim_leaves_a_fixed_positive_batch_alone():
    model = _make_model_with_batch_dim(None, 4)
    fixed = _pin_batch_dim(model)
    assert fixed.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 4


def test_convert_to_kmodel_real_hf_model(tmp_path):
    """The actual end-to-end check: run a real Hugging Face model through
    nncase's PTQ compile pipeline and confirm the output is a well-formed,
    runnable kmodel -- not just "didn't crash".

    Requires network access (downloads the model once) and nncase.
    """
    nncase = pytest.importorskip("nncase")

    import urllib.request
    onnx_path = tmp_path / "resnet8_cifar10_fp32.onnx"
    url = "https://huggingface.co/ketiswp/mlcommons-ResNet8-CIFAR10-fp32-onnx/resolve/main/model.onnx"
    try:
        urllib.request.urlretrieve(url, onnx_path)
    except Exception as e:
        pytest.skip(f"no network access to fetch the test model: {e}")

    kmodel_bytes = convert_to_kmodel(onnx_path, samples_count=4)

    # nncase's own kmodel file format magic -- the ASCII identifier "KMDL"
    # packed as a little-endian uint32, so the literal byte order is
    # reversed ("LDMK") -- not an ONNX/TFLite magic.
    assert kmodel_bytes[:4] == b"LDMK"
    assert len(kmodel_bytes) > 1000  # a real compiled model, not an error stub

    # Run it for real through nncase's own simulator -- proves the kmodel
    # is not just well-formed bytes but actually executes and produces the
    # expected output shape.
    sim = nncase.Simulator()
    sim.load_model(kmodel_bytes)
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
    sim.run()
    out = sim.get_output_tensor(0).to_numpy()
    assert out.shape == (1, 10)  # CIFAR-10's 10 classes
