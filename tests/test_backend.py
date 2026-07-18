"""Tests for the inference backend fallback (onnxsim/onnxsim#441).

When onnxruntime is not installed, onnxsim should fall back to onnx's
reference evaluator instead of requiring / auto-installing onnxruntime.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

import onnxsim
from onnxsim import backend


def _make_foldable_model() -> onnx.ModelProto:
    """A model whose ``a + b`` can be constant-folded, then added to input."""
    a = numpy_helper.from_array(np.ones((2, 2), np.float32), "a")
    b = numpy_helper.from_array(np.ones((2, 2), np.float32) * 2, "b")
    add_const = helper.make_node("Add", ["a", "b"], ["c"])
    add_input = helper.make_node("Add", ["c", "x"], ["y"])
    graph = helper.make_graph(
        [add_const, add_input],
        "foldable",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
        [a, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    return model


def test_run_model_produces_correct_result():
    model = _make_foldable_model()
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    outputs = backend.run_model(model, {"x": x})
    np.testing.assert_allclose(outputs["y"], x + 3.0)


def test_reference_evaluator_run_model(monkeypatch):
    # Force the onnxruntime-less code path.
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    assert not backend.has_onnxruntime()

    model = _make_foldable_model()
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    # Inference now goes through onnx.reference.ReferenceEvaluator.
    outputs = backend.run_model(model, {"x": x})
    np.testing.assert_allclose(outputs["y"], x + 3.0)


def test_simplify_without_onnxruntime(monkeypatch):
    # Force the onnxruntime-less code path for both constant folding
    # (PyModelExecutor.Run) and correctness checking (model_checking).
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)

    model = _make_foldable_model()
    opt, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    # ``a + b`` is folded into a single constant, leaving only the input Add.
    assert len(opt.graph.node) == 1


def test_reference_evaluator_rejects_custom_lib(monkeypatch):
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    model = _make_foldable_model()
    with pytest.raises(ValueError):
        backend.run_model(
            model, {"x": np.zeros((2, 2), np.float32)}, custom_lib="does_not_exist.so"
        )
