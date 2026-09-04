"""Tests for ``onnxsim.apply_flexround`` (FlexRound -- Learnable Rounding
based on Element-wise Division, see ``onnxsim/flexround.py``) -- reparametrizes
each INT4-quantized MatMul/Gemm layer's per-element quantization divisor
(rather than AdaRound's additive per-element perturbation) and optimizes it
by gradient descent to minimize that layer's real reconstruction error.
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


def _mixed_magnitude_weight_model(K=64, N=16, seed=0, outlier_scale=25.0):
    # FlexRound's own motivating scenario: within a single quantization
    # block, one element's magnitude dominates the rest -- round-to-nearest
    # scales that whole block off the outlier's own magnitude, so every
    # non-outlier element in the block is quantized far coarser than its own
    # value warrants. FlexRound's per-element multiplicative divisor
    # (unlike AdaRound's fixed-range additive delta) can locally rescale
    # each element's own effective step size, which should help disproportionately
    # here (K // block_size lets each block get exactly one outlier).
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.05).astype(np.float32)
    block_size = 32
    for n in range(N):
        for block_start in range(0, K, block_size):
            outlier_k = block_start + int(rng.integers(0, block_size))
            weight[outlier_k, n] = outlier_scale * (1 if rng.random() < 0.5 else -1)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    # Fetch Wq/Ws by the DequantizeLinear node's own input names, not by
    # scanning for "some tensor of this dtype": quantize_weight_only_int4
    # never prunes the original (now-dead) float32 weight initializer, so a
    # dtype-only scan can silently grab that instead of the real scale.
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


def test_flexround_reduces_reconstruction_error_with_mixed_magnitude_blocks():
    model = _mixed_magnitude_weight_model(K=64, N=16, seed=0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    flexround_model = onnxsim.apply_flexround(
        model, quant, calibration_data=calibration_data, num_iterations=200
    )
    onnx.checker.check_model(flexround_model)
    w_flexround = _dequantize_int4(flexround_model)
    y_flexround = x.astype(np.float64) @ w_flexround
    flexround_err = np.linalg.norm(y_float - y_flexround)

    assert flexround_err < rtn_err


def test_flexround_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    flexround_model = onnxsim.apply_flexround(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
        num_iterations=200,
    )
    onnx.checker.check_model(flexround_model)

    (float_y,) = _run(model, {"X": x})
    (flexround_y,) = _run(flexround_model, {"X": x})
    assert np.all(np.isfinite(flexround_y))
    assert _rel_l2(float_y, flexround_y) < 0.25


def test_flexround_gemm_transb():
    rng = np.random.default_rng(6)
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

    x = rng.standard_normal((32, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    flexround_model = onnxsim.apply_flexround(
        model, quant, calibration_data=calibration_data, num_iterations=200
    )
    onnx.checker.check_model(flexround_model)

    (float_y,) = _run(model, {"X": x})
    (flexround_y,) = _run(flexround_model, {"X": x})
    assert _rel_l2(float_y, flexround_y) < 0.25


def test_flexround_codes_stay_in_range():
    model = _mixed_magnitude_weight_model(K=32, N=8, seed=8)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((16, 32)).astype(np.float32)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    flexround_model = onnxsim.apply_flexround(
        model, quant, calibration_data=calibration_data, num_iterations=200
    )
    wq = next(
        t
        for t in flexround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
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


def test_flexround_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_flexround(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
