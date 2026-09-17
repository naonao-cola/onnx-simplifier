"""Tests for onnx_legalizer.py.

Each test builds a tiny one-node graph using the op under test (real ONNX
semantics, via `onnx.parser`), runs it through `onnx.reference.ReferenceEvaluator`
for a ground-truth output, then legalizes the graph and checks two things:
the exotic op is gone (so nncase's importer would no longer choke on it),
and the legalized graph's own `ReferenceEvaluator` output matches the
original's -- proving the decomposition is mathematically equivalent, not
just "some other ops that also run".
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import parser
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from onnx_legalizer import SUPPORTED_OPS, legalize  # noqa: E402


def _model(body: str, *, opset: int = 20, ir_version: int = 9) -> onnx.ModelProto:
    return parser.parse_model(f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}')


def _op_types(model: onnx.ModelProto) -> set[str]:
    return {node.op_type for node in model.graph.node}


def _assert_legalizes_to_equivalent_graph(model: onnx.ModelProto, op_type: str, feeds: dict[str, np.ndarray], *, atol=1e-5):
    reference_output = ReferenceEvaluator(model).run(None, feeds)

    legalized = legalize(model)
    assert op_type not in _op_types(legalized), f"{op_type} should have been legalized away"
    onnx.checker.check_model(legalized)

    legalized_output = ReferenceEvaluator(legalized).run(None, feeds)
    for ref, got in zip(reference_output, legalized_output):
        np.testing.assert_allclose(got, ref, atol=atol, rtol=1e-4)


def test_supported_ops_matches_registered_legalizers():
    assert "Gelu" in SUPPORTED_OPS
    assert "Relu" not in SUPPORTED_OPS  # nncase already supports Relu -- nothing to legalize


def test_leaves_already_supported_ops_alone():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Relu(x) }")
    legalized = legalize(model)
    assert _op_types(legalized) == {"Relu"}


def test_reciprocal():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Reciprocal(x) }")
    x = np.array([1.0, 2.0, -4.0, 0.5], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Reciprocal", {"x": x})


def test_gelu_exact():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Gelu(x) }")
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Gelu", {"x": x})


def test_gelu_tanh_approximate():
    model = _model('agraph (float[4] x) => (float[4] y) { y = Gelu<approximate="tanh">(x) }')
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Gelu", {"x": x})


def test_swish_default_alpha():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Swish(x) }")
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Swish", {"x": x})


def test_swish_custom_alpha():
    model = _model('agraph (float[4] x) => (float[4] y) { y = Swish<alpha=1.702>(x) }')
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Swish", {"x": x})


def test_mish():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Mish(x) }")
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "Mish", {"x": x})


def test_or():
    model = _model("agraph (bool[4] a, bool[4] b) => (bool[4] y) { y = Or(a, b) }")
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    _assert_legalizes_to_equivalent_graph(model, "Or", {"a": a, "b": b})


def test_xor():
    model = _model("agraph (bool[4] a, bool[4] b) => (bool[4] y) { y = Xor(a, b) }")
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    _assert_legalizes_to_equivalent_graph(model, "Xor", {"a": a, "b": b})


def test_isnan():
    model = _model("agraph (float[4] x) => (bool[4] y) { y = IsNaN(x) }")
    x = np.array([1.0, np.nan, -1.0, np.nan], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "IsNaN", {"x": x})


def test_isinf_both_directions():
    model = _model("agraph (float[5] x) => (bool[5] y) { y = IsInf(x) }")
    x = np.array([1.0, np.inf, -np.inf, np.nan, -1.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "IsInf", {"x": x})


def test_isinf_positive_only():
    model = _model('agraph (float[5] x) => (bool[5] y) { y = IsInf<detect_negative=0>(x) }')
    x = np.array([1.0, np.inf, -np.inf, np.nan, -1.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "IsInf", {"x": x})


def test_isinf_neither_direction():
    model = _model('agraph (float[5] x) => (bool[5] y) { y = IsInf<detect_negative=0, detect_positive=0>(x) }')
    x = np.array([1.0, np.inf, -np.inf, np.nan, -1.0], dtype=np.float32)
    _assert_legalizes_to_equivalent_graph(model, "IsInf", {"x": x})


def test_mean_variance_normalization_default_axes():
    # opset 17, not the file's usual 20: the legalizer emits ReduceMean with
    # `axes` as an attribute, matching nncase's own importer (src/importer/onnx/ops/reduce.cpp
    # only ever reads `axes` as an attribute, never as opset 18's new second input) --
    # under opset 20 that attribute form is itself invalid per onnx's own checker.
    model = _model("agraph (float[2,3,4,4] x) => (float[2,3,4,4] y) { y = MeanVarianceNormalization(x) }", opset=17)
    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 4, 4).astype(np.float32)
    _assert_legalizes_to_equivalent_graph(model, "MeanVarianceNormalization", {"x": x}, atol=1e-3)


def test_legalize_is_a_noop_the_second_time():
    # Calling legalize() on an already-legalized graph shouldn't find
    # anything left to rewrite (and, in particular, shouldn't crash on
    # its own output).
    model = _model("agraph (float[4] x) => (float[4] y) { y = Gelu(x) }")
    once = legalize(model)
    twice = legalize(once)
    assert _op_types(twice) == _op_types(once)


def test_legalizes_a_node_nested_among_supported_ops():
    # The op under test isn't the only node in the graph -- make sure
    # legalization only touches what it needs to and the rest of the
    # graph's wiring survives.
    model = _model("""
        agraph (float[4] x) => (float[4] y)
        {
            relu_out = Relu(x)
            gelu_out = Gelu(relu_out)
            y = Identity(gelu_out)
        }
    """)
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    reference_output = ReferenceEvaluator(model).run(None, {"x": x})

    legalized = legalize(model)
    assert _op_types(legalized) >= {"Relu", "Identity"}
    assert "Gelu" not in _op_types(legalized)
    onnx.checker.check_model(legalized)

    legalized_output = ReferenceEvaluator(legalized).run(None, {"x": x})
    np.testing.assert_allclose(legalized_output[0], reference_output[0], atol=1e-5, rtol=1e-4)
