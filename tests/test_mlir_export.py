"""Tests for emitting Torch-dialect MLIR from (simplified) ONNX models.

onnxsim can hand its cleaned-up ``ModelProto`` to torch-mlir's pure-Python ONNX
importer and print the resulting Torch-dialect MLIR (``onnxsim.export_mlir`` /
the ``--emit-mlir`` CLI flag, implemented in ``onnxsim/mlir_export.py``).

torch-mlir is heavy and not part of onnxsim's test requirements, so -- exactly
like ``tests/test_modelopt_integration.py`` -- the whole module is skipped when
it is not installed. The dedicated ``mlir-integration`` CI workflow installs
torch-mlir and runs these tests; the regular build-and-test matrix skips them.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# Skip the entire module unless torch-mlir's ONNX importer is available. Checked
# before importing onnxsim so a torch-mlir-less run skips cleanly.
pytest.importorskip(
    "torch_mlir.extras.onnx_importer",
    reason="torch-mlir is not installed",
)

import onnxsim  # noqa: E402  (imported after the torch-mlir availability check)
from onnxsim import mlir_export  # noqa: E402


def _relu_model() -> onnx.ModelProto:
    """A minimal, universally-supported graph: a single Relu over one input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _foldable_model() -> onnx.ModelProto:
    """Add(input, const_a + const_b) -- the inner Add folds to one constant.

    Simplifying this collapses ``const_a + const_b`` into a single initializer,
    so exporting the *simplified* model exercises the "MLIR reflects onnxsim's
    output, not the original graph" path.
    """
    const_a = helper.make_tensor(
        "a", TensorProto.FLOAT, [3], np.array([1, 2, 3], np.float32).tolist()
    )
    const_b = helper.make_tensor(
        "b", TensorProto.FLOAT, [3], np.array([4, 5, 6], np.float32).tolist()
    )
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
    nodes = [
        helper.make_node("Add", ["a", "b"], ["ab"]),
        helper.make_node("Add", ["x", "ab"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "foldadd", [x], [y], initializer=[const_a, const_b]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def test_has_torch_mlir_true_here():
    # The module-level importorskip guarantees torch-mlir is importable, so the
    # capability probe must agree.
    assert mlir_export.has_torch_mlir() is True


def test_export_returns_torch_dialect_mlir():
    text = onnxsim.export_mlir(_relu_model())
    assert isinstance(text, str) and text
    # A well-formed Torch-dialect module wraps the graph in a func and lowers
    # Relu to a torch op.
    assert "func.func" in text
    assert "torch." in text


def test_export_writes_file_matching_return(tmp_path):
    out = tmp_path / "relu.mlir"
    text = onnxsim.export_mlir(_relu_model(), str(out))
    assert out.read_text() == text


def test_export_of_simplified_model():
    model = _foldable_model()
    simplified, ok = onnxsim.simplify(model)
    assert ok
    # The redundant const+const Add is folded away by onnxsim.
    assert [n.op_type for n in simplified.graph.node].count("Add") == 1
    text = onnxsim.export_mlir(simplified)
    assert "func.func" in text
    assert "torch." in text


def test_convert_matches_export():
    model = _relu_model()
    assert onnxsim.export_mlir(model) == mlir_export.convert_to_torch_mlir(model)


def test_unsupported_target_rejected():
    # "torch" and "onnx" are supported; anything else is rejected up front.
    with pytest.raises(ValueError, match="Unsupported MLIR target"):
        onnxsim.export_mlir(_relu_model(), target="stablehlo")
