"""Tests for ``onnxsim.apply_ibert_gelu_cpp`` -- the C++-backed port of
``onnxsim.apply_ibert_gelu`` (I-BERT's own "i-GELU", see
``onnxsim/passes/ibert_gelu.h``). Unlike every other ``*_cpp`` port in
this repo, this is not a weight quantizer: it replaces every standalone
``Erf`` node with the paper's closed-form second-order polynomial
approximation (``sign(x) * (a*(clip(|x|, max=-b)+b)**2 + c)``), the piece
of GELU's standard ``0.5*x*(1+Erf(x/sqrt(2)))`` export decomposition an
integer-only accelerator can't evaluate directly. Mirrors
``tests/test_ibert_gelu.py``'s own test names/structure, translated to
this ``_cpp`` port, plus a direct Python-cross-check.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.ibert_gelu import apply_ibert_gelu

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _erf_model():
    return _model(
        """
        g (float[N] X) => (float[N] Y)
        {
          Y = Erf(X)
        }
        """
    )


def _gelu_decomposed_model():
    # The standard export decomposition: 0.5 * x * (1 + erf(x / sqrt(2))).
    return _model(
        """
        g (float[N] X) => (float[N] Y)
        {
          Sqrt2 = Constant<value = float[1] {1.4142135}>()
          Half = Constant<value = float[1] {0.5}>()
          One = Constant<value = float[1] {1.0}>()
          Scaled = Div(X, Sqrt2)
          Erfed = Erf(Scaled)
          Shifted = Add(Erfed, One)
          Weighted = Mul(X, Shifted)
          Y = Mul(Weighted, Half)
        }
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def test_cpp_ibert_gelu_replaces_erf_node():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu_cpp(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Erf" for n in q.graph.node)
    assert any(n.op_type == "Sign" for n in q.graph.node)


def test_cpp_ibert_gelu_polynomial_approximates_real_erf_closely():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu_cpp(model)

    x = np.linspace(-4.0, 4.0, 401).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    # This port's own compile-time constants match ibert_gelu.py's own
    # numeric min-max fit exactly (~0.021 worst-case absolute error
    # against real erf over this exact range) -- allow a little headroom
    # for float32 rounding in the ONNX graph.
    assert np.max(np.abs(float_out - approx_out)) < 0.025


def test_cpp_ibert_gelu_exact_at_zero_and_saturates_at_extremes():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu_cpp(model)

    x = np.array([0.0, 5.0, -5.0], dtype=np.float32)
    (approx_out,) = _run(q, {"X": x})
    assert approx_out[0] == pytest.approx(0.0, abs=1e-6)
    # Far from zero the polynomial saturates at +-1, matching erf's own
    # asymptotic behavior (a*(-b+b)**2 + c == c == 1.0).
    assert approx_out[1] == pytest.approx(1.0, abs=1e-6)
    assert approx_out[2] == pytest.approx(-1.0, abs=1e-6)


def test_cpp_ibert_gelu_end_to_end_on_decomposed_gelu():
    model = _gelu_decomposed_model()
    q = onnxsim.apply_ibert_gelu_cpp(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Erf" for n in q.graph.node)

    x = np.linspace(-3.0, 3.0, 61).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    # GELU(x) = 0.5*x*(1+erf(x/sqrt2)); erf's own approximation error
    # scales by 0.5*|x|, so allow a correspondingly larger absolute bound.
    assert np.max(np.abs(float_out - approx_out)) < 0.5 * 3.0 * 0.025 + 1e-3

    zero_idx = len(x) // 2
    assert x[zero_idx] == pytest.approx(0.0, abs=1e-6)
    assert approx_out[zero_idx] == pytest.approx(0.0, abs=1e-5)


def test_cpp_ibert_gelu_behaves_identically_to_python_port():
    # Both sides build the exact same five ONNX ops (Abs/Clip/Add/Mul/
    # Sign) from the exact same three float32 constants -- see
    # passes/ibert_gelu.h's own docstring on why this port is expected to
    # be numerically identical to the Python one, not merely comparable.
    model = _gelu_decomposed_model()
    py_q = apply_ibert_gelu(model)
    cpp_q = onnxsim.apply_ibert_gelu_cpp(model)

    x = np.linspace(-6.0, 6.0, 201).astype(np.float32)
    (py_out,) = _run(py_q, {"X": x})
    (cpp_out,) = _run(cpp_q, {"X": x})
    np.testing.assert_allclose(py_out, cpp_out, rtol=1e-5, atol=1e-6)


def test_cpp_ibert_gelu_noop_when_no_erf_present():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_ibert_gelu_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_ibert_gelu_multiple_erf_nodes_all_replaced():
    model = _model(
        """
        g (float[N] X, float[N] W) => (float[N] Y)
        {
          Ex = Erf(X)
          Ew = Erf(W)
          Y = Add(Ex, Ew)
        }
        """
    )
    q = onnxsim.apply_ibert_gelu_cpp(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Erf" for n in q.graph.node)
    assert sum(1 for n in q.graph.node if n.op_type == "Sign") == 2
