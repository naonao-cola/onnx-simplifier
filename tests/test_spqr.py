"""Tests for ``onnxsim.quantize_weight_only_spqr`` -- see ``onnxsim/spqr.py``
for the technique (outlier-aware block-wise INT4 quantization using a
diagonal-Hessian OBQ sensitivity score to pick outliers, reconstructed via a
sparse ``ScatterND`` correction).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip

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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
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
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_spqr_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_spqr(
        model, block_size=8, outlier_fraction=0.02, num_samples=16, seed=1
    )
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert "DequantizeLinear" in op_types
    assert "ScatterND" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_spqr_outlier_positions_reconstruct_exactly():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 0.1
    weight[0, 0] = 50.0
    model = _matmul_model(K=32, N=8, weight=weight)
    q = onnxsim.quantize_weight_only_spqr(
        model, block_size=8, outlier_fraction=0.05, num_samples=16, seed=4
    )
    onnx.checker.check_model(q)

    onehot = np.zeros((1, 32), dtype=np.float32)
    onehot[0, 0] = 1.0
    (q_y,) = _run(q, {"X": onehot})
    assert abs(float(q_y[0, 0]) - 50.0) < 1e-3


def test_spqr_reduces_error_vs_plain_blockwise_quantization():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 0.1
    weight[0, 0] = 40.0
    weight[5, 3] = -35.0

    codes_nk, scale_nk = _quantize_blockwise_int4_with_clip(
        weight.T.astype(np.float64), 8, 1.0
    )
    plain_dequant = (codes_nk * np.repeat(scale_nk, 8, axis=1)).T
    plain_err = np.abs(weight.astype(np.float64) - plain_dequant)

    model = _matmul_model(K=32, N=8, weight=weight)
    q = onnxsim.quantize_weight_only_spqr(
        model, block_size=8, outlier_fraction=0.02, num_samples=16, seed=7
    )

    probe = np.eye(32, dtype=np.float32)
    (q_y,) = _run(q, {"X": probe})
    spqr_dequant = q_y  # [32, 8] -- row k = reconstructed W[k, :]
    spqr_err = np.abs(weight.astype(np.float64) - spqr_dequant.astype(np.float64))

    assert spqr_err.max() <= plain_err.max() + 1e-6
    assert spqr_err.sum() < plain_err.sum()


def test_spqr_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_spqr(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_spqr_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_spqr(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_spqr_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_spqr(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_spqr_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.quantize_weight_only_spqr(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_spqr_zero_outlier_fraction_skips_sparse_correction():
    model = _matmul_model(K=32, N=8, seed=10)
    q = onnxsim.quantize_weight_only_spqr(
        model, block_size=8, outlier_fraction=0.0, num_samples=16, seed=11
    )
    onnx.checker.check_model(q)
    op_types = [n.op_type for n in q.graph.node]
    assert "ScatterND" not in op_types
    assert "DequantizeLinear" in op_types
