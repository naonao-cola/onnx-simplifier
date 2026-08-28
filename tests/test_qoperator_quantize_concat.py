"""Tests for ``onnxsim.quantize_qoperator_concat`` (the
``qoperator_quantize_concat`` C++ pass) -- the variadic analogue of
``test_qoperator_quantize_elementwise.py``'s ``QLinearAdd``/``QLinearMul``
coverage, using ONNX Runtime's "com.microsoft" contrib op ``QLinearConcat``
instead.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_concat_two_inputs():
    rng = np.random.default_rng(0)
    model = _model(
        """
        g (float[4,8] A, float[4,4] B) => (float[4,12] C)
        {
          C = Concat<axis = 1>(A, B)
        }
        """
    )

    quant = onnxsim.quantize_qoperator_concat(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Concat"] == 0
    assert ops["QLinearConcat"] == 1
    assert ops["QuantizeLinear"] == 2  # one for A, one for B
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    _assert_close(_run(model, {"A": a, "B": b}), _run(quant, {"A": a, "B": b}))


def test_quantize_concat_three_inputs():
    rng = np.random.default_rng(1)
    model = _model(
        """
        g (float[2,8] A, float[3,8] B, float[1,8] C) => (float[6,8] D)
        {
          D = Concat<axis = 0>(A, B, C)
        }
        """
    )

    quant = onnxsim.quantize_qoperator_concat(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["QLinearConcat"] == 1
    assert ops["QuantizeLinear"] == 3
    assert ops["DequantizeLinear"] == 1

    a = rng.standard_normal((2, 8)).astype(np.float32)
    b = rng.standard_normal((3, 8)).astype(np.float32)
    c = rng.standard_normal((1, 8)).astype(np.float32)
    feeds = {"A": a, "B": b, "C": c}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_constant_operand():
    # A constant operand is left alone -- it should be quantized from its
    # own static values, not force-fed through the calibration harness.
    const = _f32(np.random.default_rng(2).standard_normal((1, 8)), "B")
    model = _model(
        """
        g (float[4,8] A) => (float[5,8] C)
        {
          C = Concat<axis = 0>(A, B)
        }
        """,
        initializer=[const],
    )

    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_concat_quantizable_tensors(model.SerializeToString())
    assert names == []

    quant = onnxsim.quantize_qoperator_concat(model)
    assert _op_counts(quant)["Concat"] == 1
    assert _op_counts(quant)["QLinearConcat"] == 0


def test_quantize_skips_non_float():
    model = _model(
        """
        g (int64[2] A, int64[2] B) => (int64[4] C)
        {
          C = Concat<axis = 0>(A, B)
        }
        """
    )
    quant = onnxsim.quantize_qoperator_concat(model)
    assert _op_counts(quant)["Concat"] == 1
    assert _op_counts(quant)["QLinearConcat"] == 0


def test_list_qoperator_concat_quantizable_tensors():
    model = _model(
        """
        g (float[4,8] A, float[4,4] B) => (float[4,12] C)
        {
          C = Concat<axis = 1>(A, B)
        }
        """
    )
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_concat_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"A", "B", "C"}
