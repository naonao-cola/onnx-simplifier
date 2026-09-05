"""Tests for ``onnxsim.apply_ibert_softmax`` (I-BERT's own integer-only
Softmax exp-approximation piece, see ``onnxsim/ibert_softmax.py``) --
replaces a standalone ``Softmax`` node with the paper's own
polynomial-plus-power-of-two-rescale approximation of ``exp``, then a
plain division for the normalization (the integer-only iterative
reciprocal itself is not ported -- see the module's own docstring).
"""

import math

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, opset=18, ir_version=9):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _softmax_model(shape="N,K", axis=-1):
    return _model(
        f"""
        g (float[{shape}] X) => (float[{shape}] Y)
        {{
          Y = Softmax<axis = {axis}>(X)
        }}
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def test_ibert_softmax_replaces_softmax_node():
    model = _softmax_model()
    q = onnxsim.apply_ibert_softmax(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Softmax" for n in q.graph.node)
    assert any(n.op_type == "ReduceSum" for n in q.graph.node)


def test_ibert_softmax_polynomial_approximates_real_exp_closely():
    # Directly verifies this module's own numeric fit (A ~= 0.36118,
    # B ~= 0.9701, see the module docstring) against math.exp, independent
    # of the ONNX graph -- the empirical basis for the module's claimed
    # ~0.22% max relative error over p in [-ln2, 0].
    from onnxsim.ibert_softmax import (
        _IBERT_SOFTMAX_QUAD_A,
        _IBERT_SOFTMAX_QUAD_B,
        _LN2,
    )

    p = np.linspace(-_LN2, 0.0, 2001)
    approx = _IBERT_SOFTMAX_QUAD_A * p**2 + _IBERT_SOFTMAX_QUAD_B * p + 1.0
    true_exp = np.exp(p)
    rel_err = np.max(np.abs(approx - true_exp) / true_exp)
    assert rel_err < 0.01
    # Exact at p=0 by construction (the z=0 boundary, exp(0) == 1).
    assert approx[-1] == pytest.approx(1.0, abs=1e-9)


def test_ibert_softmax_output_matches_real_softmax_closely():
    model = _softmax_model(shape="8,16")
    q = onnxsim.apply_ibert_softmax(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 16)) * 5.0).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    assert np.max(np.abs(float_out - approx_out)) < 0.01


def test_ibert_softmax_output_sums_to_one_and_stays_in_unit_range():
    model = _softmax_model(shape="4,32")
    q = onnxsim.apply_ibert_softmax(model)

    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 32)) * 10.0).astype(np.float32)
    (approx_out,) = _run(q, {"X": x})

    assert np.all(approx_out >= 0.0)
    assert np.all(approx_out <= 1.0)
    row_sums = approx_out.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-4)


def test_ibert_softmax_handles_large_negative_logits_without_nan_or_inf():
    # A logit far below the row max drives z large -- 2**(-z) should
    # underflow smoothly toward 0.0, not overflow/NaN.
    model = _softmax_model(shape="1,4")
    q = onnxsim.apply_ibert_softmax(model)

    x = np.array([[0.0, -50.0, -200.0, 10.0]], dtype=np.float32)
    (approx_out,) = _run(q, {"X": x})
    assert np.all(np.isfinite(approx_out))
    assert np.all(approx_out >= 0.0)
    np.testing.assert_allclose(approx_out.sum(), 1.0, atol=1e-4)


def test_ibert_softmax_respects_non_default_axis():
    model = _softmax_model(shape="4,8", axis=0)
    q = onnxsim.apply_ibert_softmax(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal((4, 8)) * 3.0).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})
    assert np.max(np.abs(float_out - approx_out)) < 0.01
    col_sums = approx_out.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-4)


def test_ibert_softmax_skip_names_leaves_matched_node_untouched():
    model = _softmax_model()
    softmax_name = "my_softmax_node"
    for n in model.graph.node:
        if n.op_type == "Softmax":
            n.name = softmax_name
    q = onnxsim.apply_ibert_softmax(model, skip_names={softmax_name})
    assert q.SerializeToString() == model.SerializeToString()


def test_ibert_softmax_noop_when_opset_below_18():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Softmax<axis = -1>(X)
        }
        """,
        opset=13,
        ir_version=8,
    )
    q = onnxsim.apply_ibert_softmax(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_ibert_softmax_noop_when_no_softmax_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_ibert_softmax(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_ln2_constant_matches_math_log():
    from onnxsim.ibert_softmax import _LN2

    assert _LN2 == pytest.approx(math.log(2.0))
