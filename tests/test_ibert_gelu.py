"""Tests for ``onnxsim.apply_ibert_gelu`` (I-BERT's own "i-GELU", see
``onnxsim/ibert_gelu.py``) -- replaces every standalone ``Erf`` node with
the paper's closed-form second-order polynomial approximation
(``sign(x) * (a*(clip(|x|, max=-b)+b)**2 + c)``), the piece of GELU's
standard ``0.5*x*(1+Erf(x/sqrt(2)))`` export decomposition an
integer-only accelerator can't evaluate directly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def test_ibert_gelu_replaces_erf_node():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Erf" for n in q.graph.node)
    assert any(n.op_type == "Sign" for n in q.graph.node)


def test_ibert_gelu_polynomial_approximates_real_erf_closely():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu(model)

    x = np.linspace(-4.0, 4.0, 401).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    # This module's own numeric min-max fit achieves ~0.021 worst-case
    # absolute error against real erf over this exact range (verified
    # directly against math.erf) -- allow a little headroom for float32
    # rounding in the ONNX graph vs. the fit's own float64 computation.
    assert np.max(np.abs(float_out - approx_out)) < 0.025


def test_ibert_gelu_exact_at_zero_and_saturates_at_extremes():
    model = _erf_model()
    q = onnxsim.apply_ibert_gelu(model)

    x = np.array([0.0, 5.0, -5.0], dtype=np.float32)
    (approx_out,) = _run(q, {"X": x})
    assert approx_out[0] == pytest.approx(0.0, abs=1e-6)
    # Far from zero the polynomial saturates at +-1, matching erf's own
    # asymptotic behavior (a*(-b+b)**2 + c == c == 1.0).
    assert approx_out[1] == pytest.approx(1.0, abs=1e-6)
    assert approx_out[2] == pytest.approx(-1.0, abs=1e-6)


def test_ibert_gelu_end_to_end_on_decomposed_gelu():
    model = _gelu_decomposed_model()
    q = onnxsim.apply_ibert_gelu(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Erf" for n in q.graph.node)

    x = np.linspace(-3.0, 3.0, 61).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    # GELU(x) = 0.5*x*(1+erf(x/sqrt2)); erf's own approximation error
    # scales by 0.5*|x|, so allow a correspondingly larger absolute bound.
    assert np.max(np.abs(float_out - approx_out)) < 0.5 * 3.0 * 0.025 + 1e-3

    # Sanity: GELU is (approximately) monotonic and passes through the
    # origin the same way the exact function does.
    zero_idx = len(x) // 2
    assert x[zero_idx] == pytest.approx(0.0, abs=1e-6)
    assert approx_out[zero_idx] == pytest.approx(0.0, abs=1e-5)


def test_ibert_gelu_skip_names_leaves_matched_node_untouched():
    model = _erf_model()
    erf_name = "my_erf_node"
    for n in model.graph.node:
        if n.op_type == "Erf":
            n.name = erf_name
    q = onnxsim.apply_ibert_gelu(model, skip_names={erf_name})
    assert q.SerializeToString() == model.SerializeToString()


def test_ibert_gelu_noop_when_no_erf_present():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_ibert_gelu(model)
    assert q.SerializeToString() == model.SerializeToString()
