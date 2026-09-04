"""Tests for ``onnxsim.apply_owq`` (OWQ, see ``onnxsim/owq.py``) -- restores
a small number of the most quantization-sensitive input columns (the
classic Optimal Brain Surgeon saliency metric, reusing GPTQ's own Hessian
machinery) of an already-INT4-quantized MatMul/Gemm layer to exact float32
precision via an inserted correction term, leaving every other column's
INT4 codes untouched.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model_with_outlier_column(K=64, N=16, outlier_col=3, seed=0):
    # One column with much larger magnitude than the rest of its block --
    # round-to-nearest's block-shared scale is dominated by this column, so
    # every *other* column in the same block rounds coarsely, and this
    # column's own relative error is large too (its extreme values sit far
    # from any of the 15 grid points a 4-bit code offers). OWQ's own
    # motivating scenario: this is exactly the kind of column worth
    # rescuing to full precision rather than quantizing.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    weight[outlier_col, :] = rng.standard_normal(N).astype(np.float32) * 20.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _calibration(K=64, num_samples=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def test_owq_reduces_reconstruction_error_by_rescuing_outlier_column():
    model = _matmul_model_with_outlier_column(K=64, N=16, outlier_col=3, seed=0)
    x = _calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    (rtn_y,) = _run(quant, {"X": x})
    rtn_err = np.linalg.norm(y_float - rtn_y.astype(np.float64))

    owq_model = onnxsim.apply_owq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(owq_model)
    assert any(n.op_type == "Gather" for n in owq_model.graph.node)

    (owq_y,) = _run(owq_model, {"X": x})
    owq_err = np.linalg.norm(y_float - owq_y.astype(np.float64))

    assert owq_err < rtn_err


def test_owq_restores_selected_column_to_exact_precision():
    # Whichever column(s) OWQ actually selects (a data-dependent choice --
    # see the module's own OBS-based sensitivity metric), a probe that
    # isolates just one of them (every other input channel zero) should
    # come back essentially exact, since that column's whole contribution
    # is now the *correction* term, computed directly from the float
    # weight, not from any INT4 code.
    K, N, outlier_col = 32, 8, 5
    model = _matmul_model_with_outlier_column(K=K, N=N, outlier_col=outlier_col, seed=2)
    x = _calibration(K=K, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.1
    )
    onnx.checker.check_model(owq_model)

    gather_node = next(n for n in owq_model.graph.node if n.op_type == "Gather")
    idx_init = next(
        t for t in owq_model.graph.initializer if t.name == gather_node.input[1]
    )
    selected_col = int(onnx.numpy_helper.to_array(idx_init)[0])

    probe = np.zeros((1, K), dtype=np.float32)
    probe[0, selected_col] = 3.7

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = probe.astype(np.float64) @ w_float

    (owq_y,) = _run(owq_model, {"X": probe})
    owq_err = np.linalg.norm(y_float - owq_y.astype(np.float64))
    # OWQ's correction restores the selected column's contribution exactly
    # (computed directly from the float weight, independent of the INT4
    # code), so isolating it should reproduce the float output almost
    # exactly, regardless of how well or poorly plain RTN happened to
    # handle that same column on its own.
    assert owq_err < 1e-3


def test_owq_leaves_int4_codes_untouched():
    model = _matmul_model_with_outlier_column(K=32, N=8, outlier_col=1, seed=4)
    x = _calibration(K=32, num_samples=16, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    before = next(
        t for t in quant.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.1
    )
    assert any(n.op_type == "Gather" for n in owq_model.graph.node)
    after = next(
        t for t in owq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert before.raw_data == after.raw_data


def test_owq_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    weight[:, 7] = rng.standard_normal(N).astype(np.float32) * 20.0
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _calibration(K=K, num_samples=32, seed=7)
    calibration_data = [{"X": x}]

    owq_model = onnxsim.apply_owq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(owq_model)

    (float_y,) = _run(model, {"X": x})
    (owq_y,) = _run(owq_model, {"X": x})
    assert _rel_l2(float_y, owq_y) < 0.25


def test_owq_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_owq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_owq_noop_when_outlier_fraction_rounds_to_zero():
    model = _matmul_model_with_outlier_column(K=32, N=8, outlier_col=2, seed=8)
    x = _calibration(K=32, num_samples=16, seed=9)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    owq_model = onnxsim.apply_owq(
        model, quant, calibration_data=calibration_data, outlier_fraction=0.001
    )
    assert owq_model.SerializeToString() == quant.SerializeToString()
