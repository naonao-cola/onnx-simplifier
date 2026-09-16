"""Tests for ``onnxsim.apply_ibert_softmax_cpp`` -- the C++-backed port of
``onnxsim.apply_ibert_softmax`` (I-BERT's own integer-only Softmax
exp-approximation piece, see ``onnxsim/passes/ibert_softmax.h``). Unlike
this repo's other data-free ``*_cpp`` ports, this is not a weight
quantizer -- it replaces a standalone ``Softmax`` node with the paper's
own polynomial-plus-power-of-two-rescale approximation of ``exp``, then a
plain division for the normalization (the integer-only iterative
reciprocal itself is not ported -- see ``ibert_softmax.py``'s own
docstring, a scope narrowing this port inherits unchanged).
"""

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


def test_cpp_ibert_softmax_replaces_softmax_node():
    model = _softmax_model()
    q = onnxsim.apply_ibert_softmax_cpp(model)
    onnx.checker.check_model(q)
    assert not any(n.op_type == "Softmax" for n in q.graph.node)
    assert any(n.op_type == "ReduceSum" for n in q.graph.node)


def test_cpp_ibert_softmax_output_matches_real_softmax_closely():
    model = _softmax_model(shape="8,16")
    q = onnxsim.apply_ibert_softmax_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 16)) * 5.0).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})

    assert np.max(np.abs(float_out - approx_out)) < 0.01


def test_cpp_ibert_softmax_output_sums_to_one_and_stays_in_unit_range():
    model = _softmax_model(shape="4,32")
    q = onnxsim.apply_ibert_softmax_cpp(model)

    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 32)) * 10.0).astype(np.float32)
    (approx_out,) = _run(q, {"X": x})

    assert np.all(approx_out >= 0.0)
    assert np.all(approx_out <= 1.0)
    row_sums = approx_out.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-4)


def test_cpp_ibert_softmax_handles_large_negative_logits_without_nan_or_inf():
    # A logit far below the row max drives z large -- 2**(-z) should
    # underflow smoothly toward 0.0, not overflow/NaN.
    model = _softmax_model(shape="1,4")
    q = onnxsim.apply_ibert_softmax_cpp(model)

    x = np.array([[0.0, -50.0, -200.0, 10.0]], dtype=np.float32)
    (approx_out,) = _run(q, {"X": x})
    assert np.all(np.isfinite(approx_out))
    assert np.all(approx_out >= 0.0)
    np.testing.assert_allclose(approx_out.sum(), 1.0, atol=1e-4)


def test_cpp_ibert_softmax_respects_non_default_axis():
    model = _softmax_model(shape="4,8", axis=0)
    q = onnxsim.apply_ibert_softmax_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(2)
    x = (rng.standard_normal((4, 8)) * 3.0).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (approx_out,) = _run(q, {"X": x})
    assert np.max(np.abs(float_out - approx_out)) < 0.01
    col_sums = approx_out.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-4)


def test_cpp_ibert_softmax_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical -- both sides build the
    # exact same closed-form op sequence with no accumulation step, so
    # they are expected to agree far more closely than this repo's own
    # "comparable, not identical" contract requires; checked here via
    # onnxruntime rather than a direct initializer/node diff, since the
    # two sides may order/name intermediate nodes differently.
    model = _softmax_model(shape="6,20")
    py_q = onnxsim.apply_ibert_softmax(model)
    cpp_q = onnxsim.apply_ibert_softmax_cpp(model)

    rng = np.random.default_rng(3)
    x = (rng.standard_normal((6, 20)) * 4.0).astype(np.float32)
    (py_out,) = _run(py_q, {"X": x})
    (cpp_out,) = _run(cpp_q, {"X": x})
    np.testing.assert_allclose(py_out, cpp_out, rtol=1e-5, atol=1e-6)


def test_cpp_ibert_softmax_noop_when_opset_below_18():
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
    q = onnxsim.apply_ibert_softmax_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_ibert_softmax_noop_when_no_softmax_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_ibert_softmax_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()
