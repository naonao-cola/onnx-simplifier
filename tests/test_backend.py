"""Tests for the inference backend fallback (onnxsim/onnxsim#441).

When onnxruntime is not installed, onnxsim should fall back to onnx's
reference evaluator instead of requiring / auto-installing onnxruntime.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

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
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
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


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_simplify_with_explicit_provider():
    # ``simplify`` threads ``providers`` down to the constant-folding executor.
    # An explicit CPU provider must fold identically to the default.
    model = _make_foldable_model()
    opt, check_ok = onnxsim.simplify(
        model, check_n=3, providers=["CPUExecutionProvider"]
    )
    assert check_ok
    assert len(opt.graph.node) == 1


def test_simplify_with_unavailable_provider_raises():
    # Requesting a provider the installed onnxruntime does not offer surfaces a
    # clear error rather than silently folding on the CPU.
    if not backend.has_onnxruntime():
        pytest.skip("requires onnxruntime")

    import onnxruntime as rt

    if "CUDAExecutionProvider" in rt.get_available_providers():
        pytest.skip("CUDA provider is available; cannot test the unavailable path")
    model = _make_foldable_model()
    with pytest.raises(ValueError, match="not available"):
        onnxsim.simplify(model, providers=["CUDAExecutionProvider"])


def test_reference_evaluator_rejects_custom_lib(monkeypatch):
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    model = _make_foldable_model()
    with pytest.raises(ValueError):
        backend.run_model(
            model, {"x": np.zeros((2, 2), np.float32)}, custom_lib="does_not_exist.so"
        )


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_explicit_cpu_provider_produces_correct_result():
    # Passing an explicit provider list is honoured and gives the same result as
    # the default (which is CPU).
    model = _make_foldable_model()
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    outputs = backend.run_model(model, {"x": x}, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(outputs["y"], x + 3.0)


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_provider_options_tuple_form():
    # onnxruntime also accepts ``(name, options)`` tuples; onnxsim passes them
    # through unchanged.
    model = _make_foldable_model()
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    outputs = backend.run_model(
        model, {"x": x}, providers=[("CPUExecutionProvider", {})]
    )
    np.testing.assert_allclose(outputs["y"], x + 3.0)


@pytest.mark.skipif(
    not backend.has_onnxruntime(), reason="requires onnxruntime for provider selection"
)
def test_unavailable_provider_raises():
    # A requested provider the installed onnxruntime does not offer (e.g. CUDA on
    # a CPU-only wheel) fails loudly instead of silently falling back to CPU.
    import onnxruntime as rt

    if "CUDAExecutionProvider" in rt.get_available_providers():
        pytest.skip("CUDA provider is available; cannot test the unavailable path")
    model = _make_foldable_model()
    with pytest.raises(ValueError, match="not available"):
        backend.run_model(
            model,
            {"x": np.zeros((2, 2), np.float32)},
            providers=["CUDAExecutionProvider"],
        )


def test_reference_evaluator_rejects_non_cpu_provider(monkeypatch):
    # Without onnxruntime the pure-Python reference evaluator cannot honour a
    # non-CPU provider, so asking for one is an error rather than a silent no-op.
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    model = _make_foldable_model()
    with pytest.raises(ValueError, match="require onnxruntime"):
        backend.run_model(
            model,
            {"x": np.zeros((2, 2), np.float32)},
            providers=["CUDAExecutionProvider"],
        )
    # The reference evaluator still accepts an explicit CPU provider.
    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    outputs = backend.run_model(model, {"x": x}, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(outputs["y"], x + 3.0)
