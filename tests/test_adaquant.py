"""Tests for ``onnxsim.apply_adaquant`` (AdaQuant -- the layer-wise
calibration half of Hubara et al. 2020/2021's "Improving Post Training
Neural Quantization: Layer-wise Calibration and Integer Programming", see
``onnxsim/adaquant.py``) -- jointly optimizes a W8A8-quantized MatMul/Gemm
layer's weight rounding *and* its activation's (scale, zero_point) against
real calibration data, targeting :func:`onnxsim.quantize_static`'s QDQ
output.
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
    # A handful of large-magnitude, positively-shifted channels: like AWQ's
    # own salient-channel scenario, but also asymmetric (nonzero mean) so
    # the activation's calibrated zero-point isn't trivially 0 or 128 --
    # AdaQuant's own joint optimization over the clip range actually has
    # something to do here, not just the weight rounding.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] = x[:, c] * 15.0 + 8.0
    return x


def _quantize_linear_node(model):
    return next(n for n in model.graph.node if n.op_type == "QuantizeLinear")


def test_adaquant_reduces_reconstruction_error_with_structured_calibration():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _structured_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    onnx.checker.check_model(quant)
    # quantize_static must have actually produced the QDQ MatMul shape this
    # module targets -- otherwise this is vacuously comparing a model to
    # itself.
    assert any(n.op_type == "QuantizeLinear" for n in quant.graph.node)

    adaquant_model = onnxsim.apply_adaquant(
        model, quant, calibration_data=calibration_data
    )
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (quant_y,) = _run(quant, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})

    float_y = float_y.astype(np.float64)
    quant_err = np.linalg.norm(float_y - quant_y.astype(np.float64))
    adaquant_err = np.linalg.norm(float_y - adaquant_y.astype(np.float64))
    assert adaquant_err < quant_err


def test_adaquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _structured_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    adaquant_model = onnxsim.apply_adaquant(
        model, quant, calibration_data=calibration_data
    )
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})
    assert np.all(np.isfinite(adaquant_y))
    assert _rel_l2(float_y, adaquant_y) < 0.25


def test_adaquant_activation_scale_and_zero_point_stay_sane():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _structured_calibration(K=32, num_samples=32, salient_channels=(1,), seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    adaquant_model = onnxsim.apply_adaquant(
        model, quant, calibration_data=calibration_data
    )

    ql = _quantize_linear_node(adaquant_model)
    init_by_name = {t.name: t for t in adaquant_model.graph.initializer}
    x_scale = init_by_name[ql.input[1]]
    x_zp = init_by_name[ql.input[2]]

    assert x_scale.data_type == onnx.TensorProto.FLOAT
    assert float(onnx.numpy_helper.to_array(x_scale)) > 0.0
    assert x_zp.data_type == onnx.TensorProto.UINT8
    zp_val = int(onnx.numpy_helper.to_array(x_zp))
    assert 0 <= zp_val <= 255


def test_adaquant_gemm_transb():
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

    adaquant_model = onnxsim.apply_adaquant(
        model, quant, calibration_data=calibration_data
    )
    onnx.checker.check_model(adaquant_model)

    (float_y,) = _run(model, {"X": x})
    (adaquant_y,) = _run(adaquant_model, {"X": x})
    assert _rel_l2(float_y, adaquant_y) < 0.25


def test_adaquant_noop_when_no_matching_layer_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_adaquant(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
