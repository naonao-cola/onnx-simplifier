"""Tests for ``onnxsim.apply_adaquant_cpp`` -- the C++-backed port of
``onnxsim.apply_adaquant`` (AdaQuant, see ``onnxsim/adaquant.py``). Like
``test_adaround_cpp.py``, this is an iterative Adam optimization (here over
THREE parameter groups jointly: the weight-rounding relaxation plus the
activation's own log-scale and zero-point), not a closed-form computation --
floating-point summation-order differences between this port's own scalar
dense-matmul kernels and numpy's own can compound across iterations (see
``onnxsim/adaquant_entry.h``'s own accepted numerical scope note). Measured
empirically here rather than assumed.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.adaquant import apply_adaquant
from onnxsim.onnx_simplifier import apply_adaquant_cpp

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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _structured_calibration(K=64, num_samples=64, salient_channels=(3, 7), seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] = x[:, c] * 15.0 + 8.0
    return x


def _codes_scale_zp(model):
    q_by_output = {}
    for n in model.graph.node:
        if n.output:
            q_by_output[n.output[0]] = n
    mm = next(n for n in model.graph.node if n.op_type == "MatMul")
    wdq = q_by_output[mm.input[1]]
    xdq = q_by_output[mm.input[0]]
    xq = q_by_output[xdq.input[0]]
    init = {t.name: t for t in model.graph.initializer}
    wq = init[wdq.input[0]]
    x_scale = init[xq.input[1]]
    x_zp = init[xq.input[2]]
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    return (
        codes,
        float(onnx.numpy_helper.to_array(x_scale)),
        int(onnx.numpy_helper.to_array(x_zp)),
    )


def test_adaquant_cpp_matches_python_reference_closely():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _structured_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    onnx.checker.check_model(quant)
    assert any(n.op_type == "QuantizeLinear" for n in quant.graph.node)

    py_result = apply_adaquant(model, quant, calibration_data=calibration_data)
    cpp_result = apply_adaquant_cpp(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(py_result)
    onnx.checker.check_model(cpp_result)

    py_codes, py_scale, py_zp = _codes_scale_zp(py_result)
    cpp_codes, cpp_scale, cpp_zp = _codes_scale_zp(cpp_result)

    # An iterative joint Adam optimization -- close agreement, not
    # necessarily bit-exact (see this file's own module docstring).
    mismatch_frac = float(np.mean(py_codes != cpp_codes))
    assert mismatch_frac < 0.1
    assert abs(py_scale - cpp_scale) / max(abs(py_scale), 1e-6) < 0.1
    assert abs(py_zp - cpp_zp) <= 3


def test_adaquant_cpp_reduces_reconstruction_error_with_structured_calibration():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _structured_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    adaquant_model = apply_adaquant_cpp(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (quant_y,) = _run(quant, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})

    float_y = float_y.astype(np.float64)
    quant_err = np.linalg.norm(float_y - quant_y.astype(np.float64))
    adaquant_err = np.linalg.norm(float_y - adaquant_y.astype(np.float64))
    assert adaquant_err < quant_err


def test_adaquant_cpp_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _structured_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    adaquant_model = apply_adaquant_cpp(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})
    assert np.all(np.isfinite(adaquant_y))
    assert _rel_l2(float_y, adaquant_y) < 0.25


def test_adaquant_cpp_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 48, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    x = _structured_calibration(K=K, num_samples=32, salient_channels=(5, 20), seed=7)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    onnx.checker.check_model(quant)

    adaquant_model = apply_adaquant_cpp(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})
    assert _rel_l2(float_y, adaquant_y) < 0.25


def test_adaquant_cpp_activation_scale_and_zero_point_stay_sane():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _structured_calibration(K=32, num_samples=32, salient_channels=(1,), seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    adaquant_model = apply_adaquant_cpp(model, quant, calibration_data=calibration_data)

    _, x_scale, x_zp = _codes_scale_zp(adaquant_model)
    assert x_scale > 0.0
    assert 0 <= x_zp <= 255


def test_adaquant_cpp_noop_when_no_matching_layer_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_adaquant_cpp(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_adaquant_cpp_missing_calibration_input_raises():
    model = _matmul_model(K=16, N=4, seed=9)
    quant = onnxsim.quantize_static(
        model, calibration_data=[{"X": np.zeros((4, 16), dtype=np.float32)}]
    )
    with pytest.raises((RuntimeError, ValueError)):
        apply_adaquant_cpp(
            model, quant, calibration_data=[{"WrongName": np.zeros((4, 16))}]
        )
