"""Tests for ``onnxsim.apply_quantease`` (QuantEase, see
``onnxsim/quantease.py``) -- refines each INT4-quantized MatMul/Gemm layer's
rounding via cyclic coordinate descent on the same Hessian-weighted layer
reconstruction objective GPTQ minimizes, revisiting every column across
several sweeps instead of GPTQ's own single greedy left-to-right pass.
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


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # Same motivating scenario GPTQ's own tests use: input channels that are
    # linear combinations of a handful of latent factors, so independent
    # per-element/per-channel rounding can't compensate for one channel's
    # error using another's, but the shared Hessian-weighted objective can.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def test_quantease_reduces_reconstruction_error_with_correlated_channels():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    qe_model = onnxsim.apply_quantease(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(qe_model)
    w_qe = _dequantize_int4(qe_model)
    y_qe = x.astype(np.float64) @ w_qe
    qe_err = np.linalg.norm(y_float - y_qe)

    assert qe_err < rtn_err


def test_quantease_more_epochs_never_increases_reconstruction_error():
    # Every cyclic sweep only ever decreases the shared quadratic objective
    # -- more epochs should never make the layer reconstruction error worse.
    model = _matmul_model(K=64, N=12, seed=11)
    x = _correlated_calibration(K=64, num_samples=48, rank=5, seed=13)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float

    def _err(num_epochs):
        m = onnxsim.apply_quantease(
            model, quant, calibration_data=calibration_data, num_epochs=num_epochs
        )
        w = _dequantize_int4(m)
        return np.linalg.norm(y_float - x.astype(np.float64) @ w)

    err_1 = _err(1)
    err_8 = _err(8)
    assert err_8 <= err_1 + 1e-9


def test_quantease_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    qe_model = onnxsim.apply_quantease(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(qe_model)

    (float_y,) = _run(model, {"X": x})
    (qe_y,) = _run(qe_model, {"X": x})
    assert np.all(np.isfinite(qe_y))
    assert _rel_l2(float_y, qe_y) < 0.25


def test_quantease_preserves_scale_and_shape():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    )
    qe_model = onnxsim.apply_quantease(model, quant, calibration_data=calibration_data)
    qe_dq = next(n for n in qe_model.graph.node if n.op_type == "DequantizeLinear")
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in qe_model.graph.initializer if t.name == qe_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t for t in qe_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_quantease_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=6)
    x = _correlated_calibration(K=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    qe_model = onnxsim.apply_quantease(model, quant, calibration_data=calibration_data)
    wq = next(
        t for t in qe_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_quantease_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
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
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _correlated_calibration(K=K, num_samples=32, rank=8, seed=9)
    calibration_data = [{"X": x}]

    qe_model = onnxsim.apply_quantease(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(qe_model)

    (float_y,) = _run(model, {"X": x})
    (qe_y,) = _run(qe_model, {"X": x})
    assert _rel_l2(float_y, qe_y) < 0.25


def test_quantease_handles_dead_input_channel():
    # A channel with zero variance in the calibration data (H's whole row/
    # column at that index is exactly 0) must not blow up (NaN/inf) the
    # coordinate-descent update.
    model = _matmul_model(K=32, N=8, seed=10)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=12)
    x[:, 5] = 0.0  # dead channel
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    qe_model = onnxsim.apply_quantease(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(qe_model)

    (qe_y,) = _run(qe_model, {"X": x})
    assert np.all(np.isfinite(qe_y))


def test_quantease_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_quantease(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
