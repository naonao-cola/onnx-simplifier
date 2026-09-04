"""Tests for the inference backend fallback (onnxsim/onnxsim#441).

When onnxruntime is not installed, onnxsim should fall back to onnx's
reference evaluator instead of requiring / auto-installing onnxruntime.
"""

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim
from onnxsim import backend


def _make_foldable_model() -> onnx.ModelProto:
    """A model whose ``a + b`` can be constant-folded, then added to input."""
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 18]>
        foldable (float[2,2] x) => (float[2,2] y)
        <float[2,2] a = {1.0, 1.0, 1.0, 1.0}, float[2,2] b = {2.0, 2.0, 2.0, 2.0}>
        {
          c = Add(a, b)
          y = Add(c, x)
        }
        """
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


def _make_diamond_model() -> onnx.ModelProto:
    """``a`` feeds two consumers (``b`` and ``c``) before they merge into
    ``y`` -- exercises that ``_last_use_indices`` tracks the *last* consumer
    of a multi-consumer value, not just its first.
    """
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 18]>
        diamond (float[4] x) => (float[4] y)
        {
          a = Relu(x)
          b = Neg(a)
          c = Sigmoid(a)
          y = Add(b, c)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def _make_if_model() -> onnx.ModelProto:
    """A model with an ``If`` node -- forces the pruned reference path's
    ``_has_subgraphs`` guard to bail out to the plain, always-correct
    ``ReferenceEvaluator.run``.
    """
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 18]>
        agraph (bool cond, float[2] x) => (float[2] y)
        {
          y = If (cond) <
            then_branch = g1 () => (float[2] then_out) { then_out = Identity(x) },
            else_branch = g2 () => (float[2] else_out) { else_out = Neg(x) }
          >
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def test_reference_pruning_diamond_graph_correct(monkeypatch):
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    model = _make_diamond_model()
    x = np.array([1.0, -2.0, 0.5, -0.5], dtype=np.float32)
    outputs = backend.run_model(model, {"x": x})

    a = np.maximum(x, 0)
    expected = -a + 1 / (1 + np.exp(-a))
    np.testing.assert_allclose(outputs["y"], expected, rtol=1e-6)


def test_reference_pruning_matches_unpruned(monkeypatch):
    # Same diamond model, run once through the pruning path and once through
    # plain ReferenceEvaluator.run directly -- the results must be identical,
    # not just individually plausible.
    from onnx.reference import ReferenceEvaluator

    model = _make_diamond_model()
    x = np.array([3.0, -1.0, 2.0, -4.0], dtype=np.float32)

    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    pruned = backend.run_model(model, {"x": x})

    sess = ReferenceEvaluator(model)
    unpruned = sess.run(list(sess.output_names), {"x": x})
    np.testing.assert_array_equal(pruned["y"], unpruned[0])


def test_reference_pruning_if_node_falls_back_and_is_correct(monkeypatch):
    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    model = _make_if_model()

    x = np.array([1.0, 2.0], dtype=np.float32)
    then_out = backend.run_model(model, {"cond": np.array(True), "x": x})
    np.testing.assert_allclose(then_out["y"], x)

    else_out = backend.run_model(model, {"cond": np.array(False), "x": x})
    np.testing.assert_allclose(else_out["y"], -x)


def test_has_subgraphs():
    assert backend._has_subgraphs(_make_diamond_model().graph) is False
    assert backend._has_subgraphs(_make_if_model().graph) is True


def test_last_use_indices_tracks_last_not_first_consumer():
    graph = _make_diamond_model().graph
    last_use = backend._last_use_indices(graph)
    # nodes, in order: a=Relu(x) [0], b=Neg(a) [1], c=Sigmoid(a) [2], y=Add(b,c) [3]
    assert last_use["x"] == 0
    assert last_use["a"] == 2  # last consumer is c's node, not b's
    assert last_use["b"] == 3
    assert last_use["c"] == 3
    assert "y" not in last_use  # a graph output, never consumed by another node


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
